#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pandas",
#     "rich",
# ]
# ///
"""Letterboxd Set Analyzer

Analyze your Letterboxd history to discover completed and near-complete "sets"
of titles by directors or actors. Uses IMDb datasets to match watched titles and
calculate completion percentages.

This tool helps you find directors or actors where you've watched most of their
filmography, making it easy to complete sets and discover new titles from creators
you already enjoy.
"""

import argparse
import gzip
import re
import sqlite3
import sys
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Literal, NamedTuple, TextIO, TypedDict

import pandas as pd
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    track,
)
from rich.prompt import Confirm
from rich.table import Table

console = Console()


def make_progress(**kwargs) -> Progress:
    """Create a Progress instance with standard columns"""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        **kwargs,
    )


CURRENT_YEAR = datetime.now().year

TITLE_TYPE_MAP = {
    "movie": 1,
    "video": 2,
    "tvMiniSeries": 3,
    "tvMovie": 4,
}

IMDB_DATASETS: dict[str, str] = {
    "basics": "https://datasets.imdbws.com/title.basics.tsv.gz",
    "crew": "https://datasets.imdbws.com/title.crew.tsv.gz",
    "principals": "https://datasets.imdbws.com/title.principals.tsv.gz",
    "names": "https://datasets.imdbws.com/name.basics.tsv.gz",
}


class Film(NamedTuple):
    """A film with its title and release year"""

    title: str
    year: int


class Person(TypedDict):
    """Data for a person (director/actor) being analyzed"""

    name: str
    watched: set[Film]


class PersonResult(TypedDict):
    """Result dictionary for director/actor completion analysis"""

    name: str
    watched: int
    total: int
    completion: float
    missing: set[Film]
    type: Literal["director", "actor"]


class ImdbMatch(NamedTuple):
    """IMDb movie match result from database query"""

    tconst: str
    title: str
    year: int


def normalize_title(title: str) -> str:
    """Normalize a title for consistent matching across different sources.

    Handles:
    - Case normalization (lowercase)
    - Em/en dashes to hyphens
    - Curly quotes to straight quotes
    - Superscript numbers to regular numbers
    - Ampersand to "and"
    - Plus sign spacing
    - Subtitle separator normalization
    """
    normalized = title.lower()
    normalized = normalized.replace("–", "-").replace("—", "-")  # em/en dash → hyphen
    normalized = normalized.replace("'", "'").replace("'", "'")  # curly quotes → straight
    normalized = normalized.replace(""", '"').replace(""", '"')  # curly quotes → straight
    normalized = normalized.replace("²", " 2").replace("³", " 3")
    normalized = normalized.replace(" & ", " and ")
    normalized = normalized.replace(" + ", "+")
    normalized = normalized.replace(" - ", ": ").replace("- ", ": ")
    return normalized.strip()


def slugify(text: str) -> str:
    """Convert text to a URL-friendly slug (lowercase, hyphens, alphanumeric only)"""
    slug = text.lower().replace(" ", "-")
    slug = "".join(c if c.isalnum() or c == "-" else "" for c in slug)
    slug = re.sub(r"-+", "-", slug)  # Replace consecutive hyphens
    return slug.strip("-")


def linkify(url: str, text: str) -> str:
    """Wrap text in Rich clickable link markup"""
    return f"[link={url}]{text}[/link]"


def download_imdb_data(data_dir: Path, replace: bool = False) -> None:
    """Download IMDb datasets with progress bars"""
    data_dir.mkdir(exist_ok=True)

    console.print("\n[bold cyan]Downloading IMDb Datasets[/bold cyan]")
    console.print("Total size: ~1GB | This is a one-time setup\n")

    with make_progress() as progress:
        for name, url in IMDB_DATASETS.items():
            gz_file = data_dir / f"{name}.tsv.gz"
            tsv_file = data_dir / f"{name}.tsv"

            if tsv_file.exists() and not replace:
                console.print(f"✓ [green]{name}.tsv[/green] already exists")
                continue

            try:
                download_task = progress.add_task(f"[cyan]Downloading {name}...", total=100)

                def reporthook(block_num: int, block_size: int, total_size: int) -> None:
                    if total_size > 0:
                        percent = min(100, (block_num * block_size / total_size) * 100)
                        progress.update(download_task, completed=percent)

                urllib.request.urlretrieve(url, gz_file, reporthook)
                progress.update(download_task, completed=100)

                extract_task = progress.add_task(f"[cyan]Extracting {name}...", total=None)
                with gzip.open(gz_file, "rb") as f_in:
                    with open(tsv_file, "wb") as f_out:
                        f_out.write(f_in.read())
                progress.remove_task(extract_task)

                gz_file.unlink()
                progress.remove_task(download_task)

                file_size_mb = tsv_file.stat().st_size / (1024 * 1024)
                console.print(f"→ [green]{name}[/green] ({file_size_mb:.1f} MB)")

            except Exception as e:
                console.print(f"[red]Error downloading {name}: {e}[/red]")
                sys.exit(1)


def convert_to_sqlite(db_path: Path) -> None:
    """Convert TSV files to optimized SQLite database using pandas"""
    data_dir = db_path.parent

    console.print("\n[bold cyan]Converting IMDb data to SQLite database[/bold cyan]")

    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    with make_progress(transient=True) as progress:
        # Load and process titles (basics) - 5 steps total
        task = progress.add_task("[cyan]Loading titles...", total=5)

        # Step 1: Read CSV
        basics_df = pd.read_csv(
            data_dir / "basics.tsv",
            sep="\t",
            na_values="\\N",
            usecols=[  # type: ignore[call-overload]
                "tconst",
                "titleType",
                "primaryTitle",
                "originalTitle",
                "startYear",
            ],
            dtype={
                "tconst": "string",
                "titleType": "category",
                "primaryTitle": "string",
                "originalTitle": "string",
            },
        )
        progress.update(task, advance=1)

        # Step 2: Filter and clean data
        basics_df = basics_df[basics_df["titleType"].isin(TITLE_TYPE_MAP)].copy()
        basics_df = basics_df[basics_df["primaryTitle"].notna()].copy()
        basics_df["startYear"] = pd.to_numeric(basics_df["startYear"], errors="coerce").astype(  # type: ignore[union-attr]
            "Int32"
        )
        progress.update(
            task, advance=1, description=f"[cyan]Processing {len(basics_df):,} titles..."
        )

        # Step 3: Create normalized columns
        basics_df["title_lower"] = basics_df["primaryTitle"].apply(normalize_title)
        basics_df["original_title_lower"] = basics_df["originalTitle"].apply(normalize_title)
        basics_df["title_type_int"] = basics_df["titleType"].map(TITLE_TYPE_MAP).astype("Int32")
        progress.update(task, advance=1)

        # Step 4: Prepare for SQL insertion
        movies_df = basics_df[
            [
                "tconst",
                "title_type_int",
                "primaryTitle",
                "title_lower",
                "originalTitle",
                "original_title_lower",
                "startYear",
            ]
        ].rename(
            columns={
                "title_type_int": "title_type",
                "primaryTitle": "title",
                "originalTitle": "original_title",
                "startYear": "year",
            }
        )
        progress.update(
            task, advance=1, description=f"[cyan]Inserting {len(movies_df):,} titles..."
        )

        # Step 5: Insert to database and create index
        movies_df.to_sql("titles", conn, if_exists="replace", index=False)
        cursor.execute("CREATE UNIQUE INDEX idx_titles_pk ON titles(tconst)")
        progress.update(task, advance=1)

    console.print(f"✓ Inserted [cyan]{len(movies_df):,}[/cyan] titles")

    with make_progress(transient=True) as progress:
        # Load and process directors - 4 steps total
        task = progress.add_task("[cyan]Loading directors...", total=4)

        # Step 1: Read CSV
        crew_df = pd.read_csv(
            data_dir / "crew.tsv",
            sep="\t",
            na_values="\\N",
            usecols=["tconst", "directors"],  # type: ignore[call-overload]
            dtype={"tconst": "string", "directors": "string"},
        )
        progress.update(task, advance=1)

        # Step 2: Filter to non-null directors
        crew_df = crew_df[crew_df["directors"].notna()].copy()
        progress.update(task, advance=1, description="[cyan]Denormalizing directors...")

        # Step 3: Explode comma-separated director IDs into separate rows (vectorized pandas)
        crew_df["directors"] = crew_df["directors"].str.split(",")
        directors_df = crew_df.explode("directors").reset_index(drop=True)
        directors_df = directors_df.rename(
            columns={"directors": "director_id", "tconst": "title_id"}
        )
        directors_df = directors_df[["director_id", "title_id"]].drop_duplicates()
        progress.update(
            task, advance=1, description=f"[cyan]Inserting {len(directors_df):,} pairs..."
        )

        # Step 4: Insert to database
        directors_df.to_sql("directors", conn, if_exists="replace", index=False)
        progress.update(task, advance=1)

    console.print(f"✓ Inserted [cyan]{len(directors_df):,}[/cyan] director-title relationships")

    with make_progress(transient=True) as progress:
        # Load and process actors (chunked due to size) - with progress tracking
        # Principals file has ~95M rows (500k chunk size = ~191 chunks)
        chunk_size = 500_000
        expected_chunks = 191  # Based on known file size, avoids expensive line counting
        total_steps = expected_chunks + 5  # Add buffer for combine + insert steps

        task = progress.add_task("[cyan]Loading actors...", total=total_steps)
        actors_chunks = []

        for chunk_num, chunk in enumerate(
            pd.read_csv(
                data_dir / "principals.tsv",
                sep="\t",
                na_values="\\N",
                usecols=["tconst", "nconst", "category"],  # type: ignore[call-overload]
                dtype={"tconst": "string", "nconst": "string", "category": "category"},
                chunksize=chunk_size,
            ),
            start=1,
        ):
            progress.update(
                task,
                completed=chunk_num,
                description=f"[cyan]Processing chunk {chunk_num}...",
            )

            # Filter to actors/actresses only
            filtered = chunk[chunk["category"].isin(["actor", "actress"])].copy()
            filtered = filtered[["nconst", "tconst"]].rename(
                columns={"nconst": "actor_id", "tconst": "title_id"}
            )
            actors_chunks.append(filtered)

        progress.update(task, completed=expected_chunks, description="[cyan]Combining actors...")
        actors_df = pd.concat(actors_chunks, ignore_index=True).drop_duplicates()

        progress.update(
            task,
            completed=expected_chunks + 3,
            description=f"[cyan]Inserting {len(actors_df):,} pairs...",
        )
        actors_df.to_sql("actors", conn, if_exists="replace", index=False)
        progress.update(task, completed=total_steps)

    console.print(f"✓ Inserted [cyan]{len(actors_df):,}[/cyan] actor-title relationships")

    with make_progress(transient=True) as progress:
        # Load and process names - 4 steps total
        task = progress.add_task("[cyan]Loading names...", total=4)

        # Step 1: Read CSV
        names_df = pd.read_csv(
            data_dir / "names.tsv",
            sep="\t",
            na_values="\\N",
            usecols=["nconst", "primaryName"],  # type: ignore[call-overload]
            dtype={"nconst": "string", "primaryName": "string"},
        )
        progress.update(task, advance=1)

        # Step 2: Filter and rename columns
        names_df = names_df[names_df["primaryName"].notna()].copy()
        names_df = names_df.rename(columns={"nconst": "name_id", "primaryName": "name"})
        progress.update(task, advance=1, description=f"[cyan]Inserting {len(names_df):,} names...")

        # Step 3: Insert to database
        names_df.to_sql("names", conn, if_exists="replace", index=False)
        progress.update(task, advance=1)

        # Step 4: Create primary key index
        cursor.execute("CREATE UNIQUE INDEX idx_names_pk ON names(name_id)")
        progress.update(task, advance=1)

    console.print(f"✓ Inserted [cyan]{len(names_df):,}[/cyan] names")

    # Create indexes with transient progress bar
    with make_progress(transient=True) as progress:
        task = progress.add_task("[cyan]Creating indexes (2-3 minutes)...", total=10)

        # Composite index for the most common query pattern (title + year range)
        cursor.execute("CREATE INDEX idx_titles_title_year ON titles(title_lower, year)")
        progress.update(task, advance=1)

        cursor.execute(
            "CREATE INDEX idx_titles_original_title_year ON titles(original_title_lower, year)"
        )
        progress.update(task, advance=1)

        # Single-column indexes for other query patterns
        cursor.execute("CREATE INDEX idx_titles_title_lower ON titles(title_lower)")
        progress.update(task, advance=1)

        cursor.execute(
            "CREATE INDEX idx_titles_original_title_lower ON titles(original_title_lower)"
        )
        progress.update(task, advance=1)

        cursor.execute("CREATE INDEX idx_titles_year ON titles(year)")
        progress.update(task, advance=1)

        # Composite index for filmography queries (type + year filtering)
        cursor.execute("CREATE INDEX idx_titles_type_year ON titles(title_type, year)")
        progress.update(task, advance=1)

        cursor.execute("CREATE INDEX idx_directors_title ON directors(title_id)")
        progress.update(task, advance=1)

        cursor.execute("CREATE INDEX idx_directors_director ON directors(director_id)")
        progress.update(task, advance=1)

        cursor.execute("CREATE INDEX idx_actors_actor ON actors(actor_id)")
        progress.update(task, advance=1)

        cursor.execute("CREATE INDEX idx_actors_title ON actors(title_id)")
        progress.update(task, advance=1)

    console.print("✓ Created indexes")

    # Gather statistics for query optimizer
    with console.status("[cyan]Gathering statistics for query optimizer..."):
        cursor.execute("ANALYZE")

    conn.commit()
    conn.close()

    console.print(
        f"\n[bold green]✓ SQLite database ({db_path}) created successfully![/bold green]\n"
    )


def get_db_connection(db_path: Path) -> sqlite3.Connection:
    """Get SQLite database connection"""
    return sqlite3.connect(db_path)


def _try_exact_match(
    cursor: sqlite3.Cursor,
    title_lower: str,
    year: int,
    year_tolerance: int,
    title_types: set[int],
) -> ImdbMatch | None:
    """Try exact match on title_lower and original_title_lower with year range"""
    type_placeholders = ",".join("?" * len(title_types))
    cursor.execute(
        f"""
        SELECT tconst, title, year
        FROM titles
        WHERE (title_lower = ? OR original_title_lower = ?)
          AND year BETWEEN ? AND ?
          AND title_type IN ({type_placeholders})
        LIMIT 1
    """,
        (title_lower, title_lower, year - year_tolerance, year + year_tolerance, *title_types),
    )
    if result := cursor.fetchone():
        return ImdbMatch(result[0], result[1], result[2])

    return None


def match_movie_sqlite(
    conn: sqlite3.Connection,
    title: str,
    year: int | None,
    title_types: set[int],
    year_tolerance: int = 1,
) -> ImdbMatch | None:
    """Match a title using SQLite indexed queries with multiple fallback strategies"""
    title_lower = normalize_title(title)
    cursor = conn.cursor()
    type_placeholders = ",".join("?" * len(title_types))

    # Try exact match with year
    if year is not None and (
        result := _try_exact_match(cursor, title_lower, year, year_tolerance, title_types)
    ):
        return result

    # Try without year constraint
    cursor.execute(
        f"""
        SELECT tconst, title, year
        FROM titles
        WHERE title_lower = ?
          AND title_type IN ({type_placeholders})
        ORDER BY ABS(year - ?) ASC
        LIMIT 1
    """,
        (title_lower, *title_types, year if year is not None else 0),
    )
    if result := cursor.fetchone():
        return ImdbMatch(result[0], result[1], result[2])

    # Fallback strategies (only with year)
    if year is None:
        return None

    # Try alternative title variations
    alternative_titles = []

    # Strategy: Try parts around colon (for "Series: Subtitle" or "Title: Subtitle")
    if ":" in title_lower:
        parts = title_lower.split(":")
        after_colon = parts[-1].strip()
        before_colon = parts[0].strip()

        # Try part after colon (for "Black Holes: The Edge of All We Know" → "The Edge of All We Know")
        if len(after_colon) >= 5:  # Avoid matching too-short fragments
            alternative_titles.append(after_colon)

        # Try part before colon (for "A Disturbance in the Force: How..." → "A Disturbance in the Force")
        if len(before_colon) >= 5:
            alternative_titles.append(before_colon)

    # Strategy: Remove possessive prefix (for "Director's Title" → "Title")
    if possessive_match := re.match(r"^[\w\s]+['']s\s+(.+)$", title_lower):
        alternative_titles.append(possessive_match.group(1))

    # Try each alternative with exact match
    for alt_title in alternative_titles:
        if result := _try_exact_match(cursor, alt_title, year, year_tolerance, title_types):
            return result

    # Prefix match: "Mission: Impossible" matches "Mission: Impossible - Part One"
    fuzzy_patterns = [title_lower + "%"]

    # Add substring match for longer titles (>= 10 chars)
    if len(title_lower) >= 10:
        fuzzy_patterns.append("%" + title_lower + "%")

    for pattern in fuzzy_patterns:
        cursor.execute(
            f"""
            SELECT tconst, title, year
            FROM titles
            WHERE title_lower LIKE ?
              AND year BETWEEN ? AND ?
              AND title_type IN ({type_placeholders})
            ORDER BY LENGTH(title_lower) ASC
            LIMIT 1
        """,
            (pattern, year - year_tolerance, year + year_tolerance, *title_types),
        )
        if result := cursor.fetchone():
            return ImdbMatch(result[0], result[1], result[2])

    return None


def fetch_min_years_from_db(cursor: sqlite3.Cursor, titles: set[str]) -> dict[str, int]:
    """Query database for the minimum year for each title (for URL disambiguation)"""
    if not titles:
        return {}

    placeholders = ",".join("?" * len(titles))
    cursor.execute(
        f"""
        SELECT title, MIN(year) as min_year
        FROM titles
        WHERE title IN ({placeholders})
        GROUP BY title
    """,
        list(titles),
    )

    return dict(cursor.fetchall())


def should_include_year_in_url(title: str, year: int, min_year_for_title: int) -> bool:
    """Determine if year should be included in Letterboxd URL for disambiguation.

    Letterboxd only adds the year to URLs when there are multiple films with the
    same title, and only for films that are NOT the earliest chronologically.
    """
    return year > min_year_for_title


def film_to_letterboxd_url(
    title: str, year: int | None = None, min_year_for_title: int | None = None
) -> str:
    """Convert a film title to a Letterboxd URL with clickable link markup"""
    slug = slugify(title)
    if year and min_year_for_title and should_include_year_in_url(title, year, min_year_for_title):
        slug = f"{slug}-{year}"
    url = f"https://letterboxd.com/film/{slug}/"
    return linkify(url, title)


def format_title_list(
    films: list[Film],
    available_width: int,
    more_url: str,
    min_years: dict[str, int],
    watchlist_films: set[Film],
) -> str:
    """Format a list of films with links, fitting as many as possible in available width.

    Films on the user's watchlist are shown first and in bold.
    """
    if not films:
        return ""

    # Partition films: watchlist first, then others (both sorted by year descending)
    on_watchlist = [f for f in films if f in watchlist_films]
    not_on_watchlist = [f for f in films if f not in watchlist_films]
    sorted_films = on_watchlist + not_on_watchlist

    # Calculate how many films fit (using plain text lengths)
    separator = ", "
    current_length = len(sorted_films[0].title)
    count = 1

    for i in range(1, len(sorted_films)):
        next_length = current_length + len(separator) + len(sorted_films[i].title)

        # Reserve space for overflow indicator if we're not at the end
        if i < len(sorted_films) - 1:
            overflow_text = f" (+{len(sorted_films) - i - 1} more)"
            if next_length + len(overflow_text) > available_width:
                break
        elif next_length > available_width:
            break

        current_length = next_length
        count += 1

    # Apply linkification to the films we're showing, with bold for watchlist films
    film_links = []
    for film in sorted_films[:count]:
        url = film_to_letterboxd_url(film.title, film.year, min_years.get(film.title))
        if film in watchlist_films:
            # Wrap in bold tags
            url = f"[bold]{url}[/bold]"
        film_links.append(url)

    formatted = "[dim], [/dim]".join(film_links)

    if len(sorted_films) > count:
        more_text = f"(+{len(sorted_films) - count} more)"
        formatted += f" [dim]{linkify(more_url, more_text)}[/dim]"

    return formatted


def collect_people_from_watched_movies(
    cursor: sqlite3.Cursor,
    watched_tconsts: set[str],
    table_name: Literal["directors", "actors"],
    person_type: Literal["director", "actor"],
    min_watched: int,
    min_year: int,
    max_year: int,
    title_types: set[int],
) -> dict[str, Person]:
    """
    Collect directors or actors from watched titles with progress tracking.

    Args:
        cursor: Database cursor
        watched_tconsts: Set of watched title IDs
        table_name: "directors" or "actors"
        person_type: "director" or "actor" (for display)
        min_watched: Minimum watched titles to include person
        min_year: Minimum release year to include
        max_year: Maximum release year to include
        title_types: Set of title type IDs to include

    Returns:
        Dictionary mapping person_id to {name, watched} data
    """
    tconsts = list(watched_tconsts)
    placeholders = ",".join("?" * len(tconsts))
    type_placeholders = ",".join("?" * len(title_types))
    id_column = f"{person_type}_id"

    # Get count for progress bar
    with console.status(f"[cyan]Counting {person_type}s in watched titles..."):
        cursor.execute(
            f"""
            SELECT COUNT(*)
            FROM {table_name} t
            JOIN titles m ON t.title_id = m.tconst
            WHERE t.title_id IN ({placeholders})
              AND m.year >= ?
              AND (m.year IS NULL OR m.year <= ?)
              AND m.title_type IN ({type_placeholders})
        """,
            [*tconsts, min_year, max_year, *title_types],
        )
        total_entries = cursor.fetchone()[0]

    # Fetch with progress bar (transient - disappears after completion)
    with make_progress(transient=True) as progress:
        task = progress.add_task(
            f"[cyan]Collecting {total_entries:,} {person_type} entries...",
            total=total_entries,
        )

        cursor.execute(
            f"""
            SELECT t.{id_column}, n.name, m.title, m.year
            FROM {table_name} t
            JOIN names n ON t.{id_column} = n.name_id
            JOIN titles m ON t.title_id = m.tconst
            WHERE t.title_id IN ({placeholders})
              AND m.year >= ?
              AND (m.year IS NULL OR m.year <= ?)
              AND m.title_type IN ({type_placeholders})
        """,
            [*tconsts, min_year, max_year, *title_types],
        )

        # Build dict of people by ID directly
        people_by_id: dict[str, Person] = {}
        people_names: set[str] = set()
        processed = 0
        while True:
            rows = cursor.fetchmany(1000)
            if not rows:
                break
            for person_id, person_name, movie_title, movie_year in rows:
                people_names.add(person_name)
                if person_id not in people_by_id:
                    people_by_id[person_id] = {"name": person_name, "watched": set()}
                people_by_id[person_id]["watched"].add(Film(movie_title, movie_year))
            processed += len(rows)
            progress.update(task, completed=processed)

    console.print(f"✓ Found [cyan]{len(people_names)}[/cyan] {person_type}s in your watched titles")

    # Filter to people worth analyzing
    candidates = {
        person_id: person
        for person_id, person in people_by_id.items()
        if len(person["watched"]) >= min_watched
    }

    console.print(
        f"✓ Analyzing [cyan]{len(candidates)}[/cyan] {person_type}s with {min_watched}+ watched titles"
    )

    return candidates


def fetch_filmographies_bulk(
    cursor: sqlite3.Cursor,
    candidates: dict[str, Person],
    table_name: Literal["directors", "actors"],
    person_type: Literal["director", "actor"],
    min_year: int,
    title_types: set[int],
) -> dict[str, set[Film]]:
    """Fetch filmographies for multiple people in a single bulk query."""
    with console.status(f"[cyan]Fetching {person_type} filmographies..."):
        person_placeholders = ",".join("?" * len(candidates))
        type_placeholders = ",".join("?" * len(title_types))
        id_column = f"{person_type}_id"

        cursor.execute(
            f"""
            SELECT t.{id_column}, m.title, m.year
            FROM {table_name} t
            JOIN titles m ON t.title_id = m.tconst
            WHERE t.{id_column} IN ({person_placeholders})
              AND m.year >= ?
              AND (m.year IS NULL OR m.year <= ?)
              AND m.title_type IN ({type_placeholders})
        """,
            [*candidates.keys(), min_year, CURRENT_YEAR, *title_types],
        )

        filmographies = defaultdict(set)
        for person_id, title, year in cursor.fetchall():
            filmographies[person_id].add(Film(title, year))

    console.print(f"✓ Fetched filmographies for {len(filmographies)} {person_type}s")
    return filmographies


def calculate_completion_results(
    candidates: dict[str, Person],
    filmographies: dict[str, set[Film]],
    person_type: Literal["director", "actor"],
    min_set_size: int,
) -> list[PersonResult]:
    """
    Calculate completion percentages and build results list.

    Args:
        candidates: Dictionary mapping person_id to {name, watched}
        filmographies: Dictionary mapping person_id to set of Film tuples
        person_type: "Director" or "Actor" (capitalized, for results)
        min_set_size: Minimum total titles to include in results

    Returns:
        List of result dictionaries
    """
    results: list[PersonResult] = []
    with console.status("[cyan]Calculating completion percentages..."):
        for person_id, person_data in candidates.items():
            total_films = filmographies.get(person_id, set())

            if len(total_films) < min_set_size:
                continue

            watched = person_data["watched"]
            completion = len(watched) / len(total_films)
            missing = total_films - watched

            results.append(
                {
                    "name": person_data["name"],
                    "watched": len(watched),
                    "total": len(total_films),
                    "completion": completion,
                    "missing": missing,
                    "type": person_type,
                }
            )

    return results


def analyze_sets(
    watched_file: TextIO,
    min_set_size: int,
    threshold: int,
    only: Literal["directors", "actors"] | None,
    max_results: int,
    min_year: int,
    debug: bool,
    types: list[str],
    db_path: Path,
    watchlist_file: TextIO | None = None,
) -> list[PersonResult]:
    analyze_directors = only is None or only == "directors"
    analyze_actors = only is None or only == "actors"
    title_types = {TITLE_TYPE_MAP[name] for name in types}

    console.print("\n[bold magenta]🎬 Letterboxd Set Analyzer[/bold magenta]\n")

    watched_df = pd.read_csv(watched_file)
    watchlist_df = pd.read_csv(watchlist_file) if watchlist_file else None

    watchlist_msg = (
        f" and [cyan]{len(watchlist_df)}[/cyan] watchlisted" if watchlist_df is not None else ""
    )
    console.print(f"✓ Found [cyan]{len(watched_df)}[/cyan] watched{watchlist_msg} titles\n")

    # Connect to SQLite database (instant!)
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    # Match titles using SQLite indexed queries (FAST!)
    watched_tconsts: set[str] = set()
    watchlist_films: set[Film] = set()
    unmatched: list[str] = []

    for _idx, row in track(
        watched_df.iterrows(),
        description="[cyan]Matching titles to IMDb...",
        total=len(watched_df),
        console=console,
        transient=True,
    ):
        title = str(row["Name"])
        year = int(row["Year"]) if pd.notna(row["Year"]) else None  # type: ignore[arg-type]

        if match := match_movie_sqlite(conn, title, year, title_types):
            watched_tconsts.add(match.tconst)
        else:
            year_str = f" ({year})" if year is not None else ""
            unmatched.append(f"{title}{year_str}")

    # Get IMDb dataset date from database file modification time
    db_modified_time = datetime.fromtimestamp(Path(db_path).stat().st_mtime)
    dataset_date = db_modified_time.strftime("%Y-%m-%d")

    console.print(
        f"✓ Matched [green]{len(watched_tconsts)}/{len(watched_df)}[/green] titles "
        f"([cyan]{len(watched_tconsts) / len(watched_df) * 100:.1f}%[/cyan]) "
        f"[dim]({','.join(types)} • {min_year}–{CURRENT_YEAR} • IMDb {dataset_date})[/dim]"
    )

    if unmatched and debug:
        console.print(f"\n[yellow]⚠ Unmatched titles ({len(unmatched)}):[/yellow]")
        for title in sorted(unmatched):
            console.print(f"  [dim]•[/dim] {title}")
        console.print()

    # Match watchlist films to IMDb if provided
    if watchlist_df is not None:
        for _idx, row in watchlist_df.iterrows():
            title = str(row["Name"])
            year = int(row["Year"]) if pd.notna(row["Year"]) else None  # type: ignore[arg-type]
            if match := match_movie_sqlite(conn, title, year, title_types):
                watchlist_films.add(Film(match.title, match.year))

    director_results: list[PersonResult] = []
    if analyze_directors:
        director_candidates = collect_people_from_watched_movies(
            cursor,
            watched_tconsts,
            "directors",
            "director",
            min_watched=3,
            min_year=min_year,
            max_year=CURRENT_YEAR,
            title_types=title_types,
        )
        director_filmographies = fetch_filmographies_bulk(
            cursor, director_candidates, "directors", "director", min_year, title_types
        )
        director_results = calculate_completion_results(
            director_candidates, director_filmographies, "director", min_set_size
        )
        console.print("✓ Completed director analysis")

    actor_results: list[PersonResult] = []
    if analyze_actors:
        console.print()
        actor_candidates = collect_people_from_watched_movies(
            cursor,
            watched_tconsts,
            "actors",
            "actor",
            min_watched=5,
            min_year=min_year,
            max_year=CURRENT_YEAR,
            title_types=title_types,
        )
        actor_filmographies = fetch_filmographies_bulk(
            cursor, actor_candidates, "actors", "actor", min_year, title_types
        )
        actor_results = calculate_completion_results(
            actor_candidates, actor_filmographies, "actor", min_set_size
        )
        console.print("✓ Completed actor analysis")

    # Combine and sort results by descending completion
    all_results = director_results + actor_results
    all_results.sort(key=lambda x: x["completion"], reverse=True)

    if debug:
        console.print(
            f"\n[dim]Debug: Total results: {len(all_results)} "
            f"({len(director_results)} directors, {len(actor_results)} actors)[/dim]"
        )
        if director_results:
            director_completions = [r["completion"] * 100 for r in director_results]
            console.print(
                f"[dim]Debug: Director completion range: "
                f"{min(director_completions):.1f}% - {max(director_completions):.1f}%[/dim]"
            )
        if actor_results:
            actor_completions = [r["completion"] * 100 for r in actor_results]
            console.print(
                f"[dim]Debug: Actor completion range: "
                f"{min(actor_completions):.1f}% - {max(actor_completions):.1f}%[/dim]"
            )

    # Filter results by threshold
    console.print()
    threshold_decimal = threshold / 100
    filtered_results = [r for r in all_results if r["completion"] >= threshold_decimal]

    if debug:
        completed = [r for r in filtered_results if r["completion"] == 1.0]
        near_complete = [r for r in filtered_results if r["completion"] < 1.0]
        completed_directors = [r for r in completed if r["type"] == "director"]
        completed_actors = [r for r in completed if r["type"] == "actor"]
        near_complete_directors = [r for r in near_complete if r["type"] == "director"]
        near_complete_actors = [r for r in near_complete if r["type"] == "actor"]
        console.print(
            f"[dim]Debug: Completed: {len(completed_directors)} directors, "
            f"{len(completed_actors)} actors[/dim]"
        )
        console.print(
            f"[dim]Debug: Near-complete: {len(near_complete_directors)} directors, "
            f"{len(near_complete_actors)} actors[/dim]"
        )

    if filtered_results:
        display_results = filtered_results[:max_results]

        table = Table(
            box=box.ROUNDED,
            border_style="dim",
            expand=True,
        )
        # Calculate available width for Titles column
        # Terminal width - (Name: 30 + Progress: 13 + borders/padding: ~10)
        terminal_width = console.width
        fixed_columns_width = 30 + 13 + 10  # Name + Progress + overhead
        available_width = max(40, terminal_width - fixed_columns_width)  # Minimum 40 chars

        table.add_column("Name", style="yellow", max_width=30, no_wrap=False)
        table.add_column("Progress", width=13, no_wrap=True)
        table.add_column("Unwatched Titles", width=available_width, no_wrap=False)

        # Build min_years lookup for URL disambiguation by querying database
        unique_film_titles = set()
        for result in display_results:
            for film in result["missing"]:
                unique_film_titles.add(film.title)
        min_years = fetch_min_years_from_db(cursor, unique_film_titles)

        for result in display_results:
            percent = result["completion"] * 100
            watched = result["watched"]
            total = result["total"]

            # Color-code percentage based on completion level
            colors = [(90, "green"), (80, "cyan"), (70, "yellow"), (50, "magenta"), (0, "red")]
            color = next(c for score, c in colors if percent >= score)

            fraction = f"({watched}/{total})"
            progress = f"[{color}][bold]{int(percent):>3}%[/bold] {fraction:>9}[/{color}]"

            person_url = f"https://letterboxd.com/{result['type']}/{slugify(result['name'])}/"

            # For 100% complete: leave blank; otherwise show unwatched titles
            titles_preview = (
                ""
                if result["completion"] == 1.0
                else format_title_list(
                    sorted(result["missing"], key=lambda film: film.year, reverse=True),
                    available_width,
                    person_url,
                    min_years,
                    watchlist_films,
                )
            )

            name_link = linkify(person_url, result["name"])
            emoji = "🎬" if result["type"] == "director" else "👤"
            name_display = f"[dim]{emoji}[/dim] {name_link}"

            table.add_row(
                name_display,
                progress,
                titles_preview,
            )

        console.print(table)
    elif all_results:
        max_completion = max(r["completion"] for r in all_results) * 100
        suggested_threshold = int(max_completion * 0.9)  # Suggest 90% of max
        console.print(
            Panel(
                f"[yellow]No sets found at {threshold}% threshold.[/yellow]\n"
                f"[dim]Highest completion: {max_completion:.0f}%[/dim]\n"
                f"[dim]Try: --threshold {suggested_threshold}[/dim]",
                title=f"📊 Sets ({threshold}%+)",
                border_style="yellow",
            )
        )
    else:
        console.print(
            Panel(
                f"[yellow]No results found with minimum {min_set_size} titles.[/yellow]\n"
                "[dim]Try lowering --min-titles[/dim]",
                title=f"📊 Sets ({threshold}%+)",
                border_style="yellow",
            )
        )

    console.print()
    return all_results


def main() -> None:
    class CustomFormatter(
        argparse.RawDescriptionHelpFormatter,
        argparse.ArgumentDefaultsHelpFormatter,
    ):
        pass

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=CustomFormatter,
        epilog="""
Export your data: https://letterboxd.com/settings/data/
Use watched.csv (and optionally watchlist.csv) from the downloaded archive.

Examples:
  # First time: rebuild database from IMDb datasets
  %(prog)s --rebuild

  # Analyze your watch history
  %(prog)s watched.csv
  %(prog)s watched.csv --threshold 80 --limit 20
  %(prog)s watched.csv --only directors
  %(prog)s watched.csv --only actors --min-titles 10
  %(prog)s watched.csv --watchlist watchlist.csv
        """,
    )

    parser.add_argument(
        "file",
        nargs="?",
        type=argparse.FileType("r"),
        help="path to watched.csv from your Letterboxd data export",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="show debug information",
    )

    filter_group = parser.add_argument_group("filters")
    filter_group.add_argument(
        "-n",
        "--limit",
        type=int,
        metavar="int",
        default=20,
        help="maximum number of near-complete sets to display",
    )
    filter_group.add_argument(
        "-t",
        "--threshold",
        type=int,
        metavar="int",
        default=50,
        help="completion threshold percentage for near-complete sets (0-100)",
    )
    filter_group.add_argument(
        "-m",
        "--min-titles",
        type=int,
        metavar="int",
        default=5,
        help="minimum number of titles in a filmography to consider",
    )
    filter_group.add_argument(
        "-y",
        "--min-year",
        type=int,
        metavar="int",
        default=1930,
        help="minimum year for titles to include",
    )
    filter_group.add_argument(
        "--only",
        choices=["directors", "actors"],
        help="only analyze directors or actors",
    )
    filter_group.add_argument(
        "--types",
        type=str,
        nargs="+",
        choices=TITLE_TYPE_MAP,
        default=["movie"],
        help="title types to consider",
    )
    filter_group.add_argument(
        "--watchlist",
        type=argparse.FileType("r"),
        metavar="file",
        help="path to watchlist.csv to prioritize unwatched films",
    )

    data_group = parser.add_argument_group("data")
    data_group.add_argument(
        "--data-dir",
        type=Path,
        metavar="path",
        default=Path("imdb_data"),
        help="directory for IMDb data files",
    )
    data_group.add_argument(
        "--rebuild",
        action="store_true",
        help="rebuild database from IMDb datasets",
    )

    args = parser.parse_args()

    # Handle data setup
    db_path = args.data_dir / "imdb.db"
    dataset_files = {name: args.data_dir / f"{name}.tsv" for name in IMDB_DATASETS}
    has_all_data = all(f.exists() for f in dataset_files.values())

    if args.rebuild:
        if has_all_data:
            oldest_file = min(dataset_files.values(), key=lambda f: f.stat().st_mtime)
            file_date = datetime.fromtimestamp(oldest_file.stat().st_mtime).strftime("%Y-%m-%d")
            if Confirm.ask(f"Download fresh data? (current: {file_date})", default=False):
                download_imdb_data(args.data_dir, replace=True)
        else:
            download_imdb_data(args.data_dir)

    if args.rebuild or (has_all_data and not db_path.exists()):
        convert_to_sqlite(db_path)
        if args.rebuild:
            console.print(
                "[bold green]✓ Setup complete![/bold green] You can now analyze your watch history.\n"
            )
            sys.exit(0)

    if not db_path.exists():
        console.print("[red]Error: IMDb database not found.[/red]")
        console.print("\nPlease run with [cyan]--rebuild[/cyan] first:")
        console.print(f"  [dim]python {sys.argv[0]} --rebuild[/dim]\n")
        sys.exit(1)

    if not args.file:
        parser.error("the following arguments are required: file")

    if args.threshold < 0 or args.threshold > 100:
        parser.error("Threshold must be between 0 and 100")

    if args.min_titles < 1:
        parser.error("Minimum titles must be at least 1")

    try:
        analyze_sets(
            watched_file=args.file,
            min_set_size=args.min_titles,
            threshold=args.threshold,
            only=args.only,
            max_results=args.limit,
            min_year=args.min_year,
            debug=args.debug,
            types=args.types,
            db_path=db_path,
            watchlist_file=args.watchlist,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Analysis interrupted by user.[/yellow]")
        sys.exit(0)
    except Exception:
        console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    main()
