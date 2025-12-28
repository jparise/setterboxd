#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pandas",
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
import enum
import gzip
import re
import shutil
import sqlite3
import sys
import traceback
import urllib.request
from collections import defaultdict
from collections.abc import Sized
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, NamedTuple, Self, TextIO, TypedDict

import pandas as pd


# ANSI escape codes for terminal formatting
def green(text: str) -> str:
    """Format text in green"""
    return f"\033[32m{text}\033[0m"


def red(text: str) -> str:
    """Format text in red"""
    return f"\033[31m{text}\033[0m"


def yellow(text: str) -> str:
    """Format text in yellow"""
    return f"\033[33m{text}\033[0m"


def cyan(text: str) -> str:
    """Format text in cyan"""
    return f"\033[36m{text}\033[0m"


def magenta(text: str) -> str:
    """Format text in magenta"""
    return f"\033[35m{text}\033[0m"


def bold(text: str) -> str:
    """Format text in bold"""
    return f"\033[1m{text}\033[0m"


def dim(text: str) -> str:
    """Format text dimmed"""
    return f"\033[2m{text}\033[0m"


def linkify(url: str, text: str) -> str:
    """Create clickable terminal hyperlink using OSC 8 escape sequences"""
    return f"\033]8;;{url}\033\\{text}\033]8;;\033\\"


def confirm(prompt: str, default: bool = True) -> bool:
    """Ask yes/no question with default value"""
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        response = input(f"{prompt} {suffix}: ").strip().lower()
        if not response:
            return default
        if response in ("y", "yes"):
            return True
        if response in ("n", "no"):
            return False
        print(yellow("Please answer 'y' or 'n'"))


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text for length calculation"""
    # Remove ANSI color codes and OSC 8 hyperlinks
    text = re.sub(r"\033\[[0-9;]*m", "", text)
    return re.sub(r"\033\]8;;[^\033]*\033\\", "", text)


def print_table(headers: list[str], rows: list[list[str]], col_widths: list[int]) -> None:
    """Print a simple table with aligned columns"""
    # Header row
    header_parts = []
    for header, width in zip(headers, col_widths, strict=True):
        visual_len = len(strip_ansi(header))
        padding = width - visual_len
        header_parts.append(header + " " * padding)
    print("  ".join(header_parts))

    # Separator
    total_width = sum(col_widths) + (len(col_widths) - 1) * 2  # 2 spaces between columns
    print(dim("─" * total_width))

    # Data rows
    for row in rows:
        row_parts = []
        for cell, width in zip(row, col_widths, strict=True):
            visual_len = len(strip_ansi(cell))
            if visual_len > width:
                # Truncate - we lose ANSI styling but keep it readable
                plain = strip_ansi(cell)
                truncated = plain[: width - 1] + "…"
                row_parts.append(truncated)
            else:
                # Pad
                padding = width - visual_len
                row_parts.append(cell + " " * padding)
        print("  ".join(row_parts))


@enum.unique
class TitleType(enum.IntEnum):
    movie = 1
    video = 2
    tvMiniSeries = 3  # noqa: N815
    tvMovie = 4  # noqa: N815


TitleType.default = frozenset({TitleType.movie})  # type: ignore[attr-defined]


@enum.unique
class PersonType(enum.Enum):
    director = "director"
    actor = "actor"

    @property
    def table_name(self) -> Literal["directors", "actors"]:
        """Return the database table name for this person type."""
        return "directors" if self == PersonType.director else "actors"


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


class Range(NamedTuple):
    """Numeric range with min and max bounds (inclusive)"""

    min: int
    max: int

    def __str__(self) -> str:
        if self.min == self.max:
            return str(self.min)
        return f"{self.min}-{self.max}"

    @classmethod
    def parse(cls, value: str, min_bound: int, max_bound: int) -> Self:
        """Parse range string like '80' or '80-99'.

        Single values (e.g., '80') default to max_bound as the upper limit.
        """
        try:
            if "-" in value:
                parts = value.split("-")
                if len(parts) != 2:
                    raise argparse.ArgumentTypeError(
                        f"Invalid range format '{value}'. Use 'N' or 'MIN-MAX'"
                    )
                min_val = int(parts[0])
                max_val = int(parts[1])
            else:
                min_val = int(value)
                max_val = max_bound

            # Validation
            if not (min_bound <= min_val <= max_bound):
                raise argparse.ArgumentTypeError(
                    f"Min value {min_val} out of bounds [{min_bound}, {max_bound}]"
                )
            if not (min_bound <= max_val <= max_bound):
                raise argparse.ArgumentTypeError(
                    f"Max value {max_val} out of bounds [{min_bound}, {max_bound}]"
                )
            if min_val > max_val:
                raise argparse.ArgumentTypeError(f"Min {min_val} cannot exceed max {max_val}")

            return cls(min_val, max_val)
        except ValueError as e:
            raise argparse.ArgumentTypeError(f"Invalid number in range '{value}'") from e


@dataclass(frozen=True)
class Filters:
    """Filtering parameters applied consistently across the pipeline."""

    # Year range for titles to include in analysis. Allows focusing on specific eras
    # (e.g., 1970-1990 for classic cinema) or filtering out very old/new titles.
    # Supports single year (e.g., Range(1980, 2025)) or specific range (Range(1980, 2000)).
    years: Range
    # Year tolerance for fuzzy matching: allows ±N years to handle year discrepancies
    # between Letterboxd and IMDb data. Testing shows year_tolerance=1 rescues ~0.5% of
    # matches with minimal false positive risk; larger values provide diminishing returns.
    year_tolerance: int = 1
    # Set of title types to consider. Allows filtering by content type such as
    # theatrical movies only, or including TV movies and miniseries.
    # Defaults to movie only.
    title_types: frozenset[TitleType] = TitleType.default  # type: ignore[attr-defined]

    def to_sql(self) -> tuple[str, list]:
        """Returns (WHERE clause fragment, params) for filtering."""
        where_clause = f"""m.year >= ?
              AND (m.year IS NULL OR m.year <= ?)
              AND m.title_type IN ({sql_placeholders(self.title_types)})"""
        params = [self.years.min, self.years.max, *self.title_types]
        return (where_clause, params)


def sql_placeholders(collection: Sized) -> str:
    """Generate SQL placeholders for a collection: '?,?,?'"""
    return ",".join("?" * len(collection))


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


def download_imdb_data(data_dir: Path, replace: bool = False) -> None:
    """Download IMDb datasets"""
    data_dir.mkdir(exist_ok=True)

    print(f"\n{bold(cyan('Downloading IMDb Datasets'))}")
    print("Total size: ~1GB | One-time setup\n")

    for name, url in IMDB_DATASETS.items():
        gz_file = data_dir / f"{name}.tsv.gz"
        tsv_file = data_dir / f"{name}.tsv"

        if tsv_file.exists() and not replace:
            print(f"✓ {green(name + '.tsv')} already exists")
            continue

        try:
            print(f"→ Downloading {name}...", end="", flush=True)
            urllib.request.urlretrieve(url, gz_file)
            print(" ✓")

            print(f"  Extracting {name}...", end="", flush=True)
            with gzip.open(gz_file, "rb") as f_in:
                with open(tsv_file, "wb") as f_out:
                    f_out.write(f_in.read())

            gz_file.unlink()
            file_size_mb = tsv_file.stat().st_size / (1024 * 1024)
            print(f" ✓ {green(f'({file_size_mb:.0f} MB)')}")

        except Exception as e:
            print(f"{red(f'Error downloading {name}: {e}')}")
            sys.exit(1)


def convert_to_sqlite(db_path: Path) -> None:
    """Convert TSV files to optimized SQLite database using pandas"""
    data_dir = db_path.parent

    print(f"\n{bold(cyan('Converting IMDb data to SQLite database'))}")

    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Load and process titles (basics)
    print("→ Loading titles...", end="", flush=True)
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
    print(" ✓")

    print("  Processing titles...", end="", flush=True)
    basics_df = basics_df[
        basics_df["titleType"].isin(TitleType.__members__) & basics_df["primaryTitle"].notna()
    ].copy()
    basics_df["titleType"] = basics_df["titleType"].cat.remove_unused_categories()
    basics_df["startYear"] = pd.to_numeric(basics_df["startYear"], errors="coerce").astype(  # type: ignore[union-attr]
        "Int32"
    )

    basics_df["title_lower"] = basics_df["primaryTitle"].apply(normalize_title)
    basics_df["original_title_lower"] = basics_df["originalTitle"].apply(normalize_title)
    basics_df["title_type_int"] = basics_df["titleType"].map(lambda x: TitleType[x]).astype("Int32")

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
    print(" ✓")

    print("  Inserting titles...", end="", flush=True)
    movies_df.to_sql("titles", conn, if_exists="replace", index=False)
    cursor.execute("CREATE UNIQUE INDEX idx_titles_pk ON titles(tconst)")
    print(f" ✓ {green(f'({len(movies_df):,} titles)')}")

    # Load and process directors
    print("→ Loading directors...", end="", flush=True)
    crew_df = pd.read_csv(
        data_dir / "crew.tsv",
        sep="\t",
        na_values="\\N",
        usecols=["tconst", "directors"],  # type: ignore[call-overload]
        dtype={"tconst": "string", "directors": "string"},
    )
    crew_df = crew_df[crew_df["directors"].notna()].copy()
    crew_df["directors"] = crew_df["directors"].str.split(",")
    directors_df = crew_df.explode("directors").reset_index(drop=True)
    directors_df = directors_df.rename(columns={"directors": "director_id", "tconst": "title_id"})
    directors_df = directors_df[["director_id", "title_id"]].drop_duplicates()
    print(" ✓")

    print("  Inserting directors...", end="", flush=True)
    directors_df.to_sql("directors", conn, if_exists="replace", index=False)
    print(f" ✓ {green(f'({len(directors_df):,} relationships)')}")

    # Load and process actors (chunked due to size)
    print("\n→ Loading actors (large file, this may take a minute)...")
    chunk_size = 500_000
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
        # Print progress every 25 chunks for better feedback
        if chunk_num % 25 == 0:
            rows_processed = chunk_num * chunk_size
            print(
                f"  {dim(f'Processing actors... ({rows_processed:,} rows)')}", end="\r", flush=True
            )

        # Filter to actors/actresses only
        filtered = chunk[chunk["category"].isin(["actor", "actress"])].copy()
        filtered = filtered[["nconst", "tconst"]].rename(
            columns={"nconst": "actor_id", "tconst": "title_id"}
        )
        actors_chunks.append(filtered)

    print("  Combining and deduplicating...", end="", flush=True)
    actors_df = pd.concat(actors_chunks, ignore_index=True).drop_duplicates()
    print(" ✓")

    print("  Inserting actors...", end="", flush=True)
    actors_df.to_sql("actors", conn, if_exists="replace", index=False)
    print(f" ✓ {green(f'({len(actors_df):,} relationships)')}")

    # Load and process names
    print("→ Loading names...", end="", flush=True)
    names_df = pd.read_csv(
        data_dir / "names.tsv",
        sep="\t",
        na_values="\\N",
        usecols=["nconst", "primaryName"],  # type: ignore[call-overload]
        dtype={"nconst": "string", "primaryName": "string"},
    )
    names_df = names_df[names_df["primaryName"].notna()].copy()
    names_df = names_df.rename(columns={"nconst": "name_id", "primaryName": "name"})
    print(" ✓")

    print("  Inserting names...", end="", flush=True)
    names_df.to_sql("names", conn, if_exists="replace", index=False)
    cursor.execute("CREATE UNIQUE INDEX idx_names_pk ON names(name_id)")
    print(f" ✓ {green(f'({len(names_df):,} names)')}")

    # Create indexes
    indexes = [
        # Composite indexes for the most common query patterns
        ("idx_titles_title_year", "titles(title_lower, year)"),
        ("idx_titles_original_title_year", "titles(original_title_lower, year)"),
        # Single-column indexes for other query patterns
        ("idx_titles_title_lower", "titles(title_lower)"),
        ("idx_titles_original_title_lower", "titles(original_title_lower)"),
        ("idx_titles_year", "titles(year)"),
        # Composite index for filmography queries (type + year filtering)
        ("idx_titles_type_year", "titles(title_type, year)"),
        # Junction table indexes
        ("idx_directors_title", "directors(title_id)"),
        ("idx_directors_director", "directors(director_id)"),
        ("idx_actors_actor", "actors(actor_id)"),
        ("idx_actors_title", "actors(title_id)"),
    ]

    print("\n→ Creating indexes (2-3 minutes)...", end="", flush=True)
    for i, (idx_name, idx_def) in enumerate(indexes, 1):
        print(f"  {dim(f'Creating index {i}/{len(indexes)}...')}", end="\r", flush=True)
        cursor.execute(f"CREATE INDEX {idx_name} ON {idx_def}")
    # Clear the progress line and show completion
    print(f"{'→ Creating indexes (2-3 minutes)...'}{' ' * 20} ✓")

    # Gather statistics for query optimizer
    print("  Gathering statistics...", end="", flush=True)
    cursor.execute("ANALYZE")
    print(" ✓")

    conn.commit()
    conn.close()

    print(f"\n{bold(green(f'✓ SQLite database ({db_path}) created successfully!'))}\n")


def get_db_connection(db_path: Path) -> sqlite3.Connection:
    """Get SQLite database connection"""
    return sqlite3.connect(db_path)


def _try_exact_match(
    cursor: sqlite3.Cursor,
    title_lower: str,
    year: int,
    filters: Filters,
) -> ImdbMatch | None:
    """Try exact match on title_lower and original_title_lower with year range"""
    cursor.execute(
        f"""
        SELECT tconst, title, year
        FROM titles
        WHERE (title_lower = ? OR original_title_lower = ?)
          AND year BETWEEN ? AND ?
          AND title_type IN ({sql_placeholders(filters.title_types)})
        ORDER BY ABS(year - ?) ASC
        LIMIT 1
    """,
        (
            title_lower,
            title_lower,
            year - filters.year_tolerance,
            year + filters.year_tolerance,
            *filters.title_types,
            year,
        ),
    )
    if result := cursor.fetchone():
        return ImdbMatch(result[0], result[1], result[2])

    return None


def match_movie_sqlite(
    conn: sqlite3.Connection,
    title: str,
    year: int | None,
    filters: Filters,
) -> ImdbMatch | None:
    """Match a title using SQLite indexed queries with multiple fallback strategies"""
    title_lower = normalize_title(title)
    cursor = conn.cursor()

    # Try exact match with year
    if year is not None and (result := _try_exact_match(cursor, title_lower, year, filters)):
        return result

    # Try without year constraint
    cursor.execute(
        f"""
        SELECT tconst, title, year
        FROM titles
        WHERE title_lower = ?
          AND title_type IN ({sql_placeholders(filters.title_types)})
        ORDER BY ABS(year - ?) ASC
        LIMIT 1
    """,
        (title_lower, *filters.title_types, year if year is not None else 0),
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
        if result := _try_exact_match(cursor, alt_title, year, filters):
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
              AND title_type IN ({sql_placeholders(filters.title_types)})
            ORDER BY LENGTH(title_lower) ASC
            LIMIT 1
        """,
            (
                pattern,
                year - filters.year_tolerance,
                year + filters.year_tolerance,
                *filters.title_types,
            ),
        )
        if result := cursor.fetchone():
            return ImdbMatch(result[0], result[1], result[2])

    return None


def fetch_min_years_from_db(cursor: sqlite3.Cursor, titles: set[str]) -> dict[str, int]:
    """Query database for the minimum year for each title (for URL disambiguation)"""
    if not titles:
        return {}

    cursor.execute(
        f"""
        SELECT title, MIN(year) as min_year
        FROM titles
        WHERE title IN ({sql_placeholders(titles)})
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

    # Sort films: watchlist first, then by year descending
    sorted_films = sorted(films, key=lambda f: (f not in watchlist_films, -f.year))

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
            # Wrap in bold
            url = bold(url)
        film_links.append(url)

    # Join with dimmed separators
    separator = dim(", ")
    formatted = separator.join(film_links)

    if len(sorted_films) > count:
        more_text = f"(+{len(sorted_films) - count})"
        formatted += f"{dim(', ')}{dim(linkify(more_url, more_text))}"

    return formatted


def collect_people_from_watched_movies(
    cursor: sqlite3.Cursor,
    watched_tconsts: set[str],
    table_name: Literal["directors", "actors"],
    person_type: Literal["director", "actor"],
    min_watched: int,
    filters: Filters,
) -> dict[str, Person]:
    """
    Collect directors or actors from watched titles with progress tracking.

    Args:
        cursor: Database cursor
        watched_tconsts: Set of watched title IDs
        table_name: "directors" or "actors"
        person_type: "director" or "actor" (for display)
        min_watched: Minimum watched titles to include person
        filters: Filtering parameters (year range, title types)

    Returns:
        Dictionary mapping person_id to {name, watched} data
    """
    tconsts = list(watched_tconsts)
    id_column = f"{person_type}_id"
    where_clause, filter_params = filters.to_sql()

    # Fetch people from watched titles
    print(f"Collecting {person_type}s from watched titles...", end="", flush=True)
    cursor.execute(
        f"""
        SELECT t.{id_column}, n.name, m.title, m.year
        FROM {table_name} t
        JOIN names n ON t.{id_column} = n.name_id
        JOIN titles m ON t.title_id = m.tconst
        WHERE t.title_id IN ({sql_placeholders(tconsts)})
          AND {where_clause}
    """,
        [*tconsts, *filter_params],
    )

    # Build dict of people by ID directly
    people_by_id: dict[str, Person] = {}
    people_names: set[str] = set()
    for person_id, person_name, movie_title, movie_year in cursor.fetchall():
        people_names.add(person_name)
        if person_id not in people_by_id:
            people_by_id[person_id] = {"name": person_name, "watched": set()}
        people_by_id[person_id]["watched"].add(Film(movie_title, movie_year))
    print(f" ✓ {green(f'({len(people_names)} {person_type}s)')}")

    # Filter to people worth analyzing
    return {
        person_id: person
        for person_id, person in people_by_id.items()
        if len(person["watched"]) >= min_watched
    }


def fetch_filmographies_bulk(
    cursor: sqlite3.Cursor,
    candidates: dict[str, Person],
    table_name: Literal["directors", "actors"],
    person_type: Literal["director", "actor"],
    filters: Filters,
    min_watched: int,
) -> dict[str, set[Film]]:
    """Fetch filmographies for multiple people in a single bulk query."""
    print(f"Fetching {person_type} filmographies...", end="", flush=True)
    id_column = f"{person_type}_id"
    where_clause, filter_params = filters.to_sql()

    cursor.execute(
        f"""
        SELECT t.{id_column}, m.title, m.year
        FROM {table_name} t
        JOIN titles m ON t.title_id = m.tconst
        WHERE t.{id_column} IN ({sql_placeholders(candidates)})
          AND {where_clause}
    """,
        [*candidates.keys(), *filter_params],
    )

    filmographies = defaultdict(set)
    for person_id, title, year in cursor.fetchall():
        filmographies[person_id].add(Film(title, year))
    print(f" ✓ {green(f'({len(filmographies)} {person_type}s with {min_watched}+ watched)')}")

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


def analyze_person_type(
    cursor: sqlite3.Cursor,
    person_type: PersonType,
    watched_tconsts: set[str],
    min_watched: int,
    min_set_size: int,
    filters: Filters,
) -> list[PersonResult]:
    """Analyze a person type (directors or actors) for completion."""
    candidates = collect_people_from_watched_movies(
        cursor,
        watched_tconsts,
        person_type.table_name,
        person_type.value,
        min_watched=min_watched,
        filters=filters,
    )
    filmographies = fetch_filmographies_bulk(
        cursor, candidates, person_type.table_name, person_type.value, filters, min_watched
    )
    return calculate_completion_results(candidates, filmographies, person_type.value, min_set_size)


def analyze_sets(
    watched_file: TextIO,
    min_set_size: int,
    threshold: Range,
    only: Literal["directors", "actors"] | None,
    max_results: int,
    debug: bool,
    filters: Filters,
    db_path: Path,
    watchlist_file: TextIO | None = None,
    filter_names: list[str] | None = None,
) -> list[PersonResult]:
    analyze_directors = only is None or only == "directors"
    analyze_actors = only is None or only == "actors"

    print(f"\n{bold(magenta('🎬 Letterboxd Set Analyzer'))}\n")

    watched_df = pd.read_csv(watched_file)
    watchlist_df = pd.read_csv(watchlist_file) if watchlist_file else None

    # Connect to SQLite database to get dataset date
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    # Get IMDb dataset date from database file modification time
    db_modified_time = datetime.fromtimestamp(Path(db_path).stat().st_mtime)
    dataset_date = db_modified_time.strftime("%Y-%m-%d")

    # Build and print filter info line
    type_names = [t.name for t in filters.title_types]
    types_str = ", ".join(type_names)
    filter_parts = [
        types_str,
        str(filters.years),
        f"≥{threshold}% complete",
        f"≥{min_set_size} titles",
    ]

    # Add person type filter if specified
    if only:
        filter_parts.append(f"{only} only")

    # Add name filter if specified
    if filter_names:
        names_str = ", ".join(filter_names)
        filter_parts.append(f"names: {names_str}")

    filter_parts.append(f"IMDb {dataset_date}")
    filter_info = f" {dim('•')} ".join(filter_parts)
    print(f"{magenta('▶')} {filter_info}")

    # Match titles using SQLite indexed queries (FAST!)
    watched_tconsts: set[str] = set()
    watchlist_films: set[Film] = set()
    unmatched: list[str] = []

    for _idx, row in watched_df.iterrows():
        title = str(row["Name"])
        year = int(row["Year"]) if pd.notna(row["Year"]) else None  # type: ignore[arg-type]

        if match := match_movie_sqlite(conn, title, year, filters):
            watched_tconsts.add(match.tconst)
        else:
            year_str = f" ({year})" if year is not None else ""
            unmatched.append(f"{title}{year_str}")

    # Print match results
    watchlist_msg = (
        f" and {bold(str(len(watchlist_df)))} watchlisted" if watchlist_df is not None else ""
    )
    print(
        f"{magenta('↳')} matched {bold(f'{len(watched_tconsts)}/{len(watched_df)}')} watched{watchlist_msg} titles\n"
    )

    if unmatched and debug:
        print(f"\n{yellow(f'⚠ Unmatched titles ({len(unmatched)}):')}")
        for title in sorted(unmatched):
            print(f"  {dim('•')} {title}")
        print()

    # Match watchlist films to IMDb if provided
    if watchlist_df is not None:
        for _idx, row in watchlist_df.iterrows():
            title = str(row["Name"])
            year = int(row["Year"]) if pd.notna(row["Year"]) else None  # type: ignore[arg-type]
            if match := match_movie_sqlite(conn, title, year, filters):
                watchlist_films.add(Film(match.title, match.year))

    director_results = (
        analyze_person_type(cursor, PersonType.director, watched_tconsts, 3, min_set_size, filters)
        if analyze_directors
        else []
    )

    if analyze_actors:
        print()

    actor_results = (
        analyze_person_type(cursor, PersonType.actor, watched_tconsts, 5, min_set_size, filters)
        if analyze_actors
        else []
    )

    # Combine and sort results by descending completion
    all_results = director_results + actor_results
    all_results.sort(key=lambda x: x["completion"], reverse=True)

    if debug:
        print(
            f"\n{
                dim(
                    f'Debug: Total results: {len(all_results)} '
                    f'({len(director_results)} directors, {len(actor_results)} actors)'
                )
            }"
        )
        if director_results:
            director_completions = [r["completion"] * 100 for r in director_results]
            print(
                dim(
                    f"Debug: Director completion range: "
                    f"{min(director_completions):.1f}% - {max(director_completions):.1f}%"
                )
            )
        if actor_results:
            actor_completions = [r["completion"] * 100 for r in actor_results]
            print(
                dim(
                    f"Debug: Actor completion range: "
                    f"{min(actor_completions):.1f}% - {max(actor_completions):.1f}%"
                )
            )

    # Filter results by threshold or names
    print()
    if filter_names:
        name_patterns = [
            re.compile(rf"\b{re.escape(name.strip())}\b", re.IGNORECASE) for name in filter_names
        ]
        filtered_results = [
            r for r in all_results if any(pattern.search(r["name"]) for pattern in name_patterns)
        ]
    else:
        threshold_min = threshold.min / 100
        threshold_max = threshold.max / 100
        filtered_results = [
            r for r in all_results if threshold_min <= r["completion"] <= threshold_max
        ]

    if debug:
        completed = [r for r in filtered_results if r["completion"] == 1.0]
        near_complete = [r for r in filtered_results if r["completion"] < 1.0]
        completed_directors = [r for r in completed if r["type"] == "director"]
        completed_actors = [r for r in completed if r["type"] == "actor"]
        near_complete_directors = [r for r in near_complete if r["type"] == "director"]
        near_complete_actors = [r for r in near_complete if r["type"] == "actor"]
        print(
            dim(
                f"Debug: Completed: {len(completed_directors)} directors, "
                f"{len(completed_actors)} actors"
            )
        )
        print(
            dim(
                f"Debug: Near-complete: {len(near_complete_directors)} directors, "
                f"{len(near_complete_actors)} actors"
            )
        )

    if filtered_results:
        display_results = filtered_results[:max_results]

        # Calculate available width for Titles column
        col_widths = [1, 29, 20, 0]  # Type, Name, Progress, Titles (calculated)
        terminal_width = shutil.get_terminal_size((80, 24)).columns
        separator_width = (len(col_widths) - 1) * 2  # 2 spaces between each column
        fixed_width = sum(col_widths[:3]) + separator_width
        available_width = max(40, terminal_width - fixed_width)
        col_widths[3] = available_width

        headers = ["", "Name", "Progress", "Unwatched Titles"]
        table_rows = []

        # Build min_years lookup for URL disambiguation by querying database
        unique_film_titles = set()
        for result in display_results:
            for film in result["missing"]:
                unique_film_titles.add(film.title)
        min_years = fetch_min_years_from_db(cursor, unique_film_titles)

        # Color map for completion percentages
        color_map = [(90, green), (80, cyan), (70, yellow), (50, magenta), (0, red)]

        for result in display_results:
            percent = result["completion"] * 100
            watched = result["watched"]
            total = result["total"]

            # Color-code percentage based on completion level
            color_func = next(c for score, c in color_map if percent >= score)

            fraction = f"({watched}/{total})"
            progress = f"{color_func(bold(f'{int(percent):>3}%'))} {dim(f'{fraction:>9}')}"

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
            symbol = "●" if result["type"] == "director" else "○"
            type_icon = dim(cyan(symbol))

            table_rows.append([type_icon, name_link, progress, titles_preview])

        print_table(headers, table_rows, col_widths)
    elif all_results:
        filter_desc = "filtered by name" if filter_names else f"{threshold}%"
        max_completion = max(r["completion"] for r in all_results) * 100
        suggested_threshold = int(max_completion * 0.9)  # Suggest 90% of max
        print(f"\n📊 {yellow(f'No sets found ({filter_desc}).')}")
        print(f"  {dim(f'Highest completion: {max_completion:.0f}%')}")
        print(f"  {dim(f'Try: --threshold {suggested_threshold}')}\n")
    else:
        print(f"\n📊 {yellow(f'No results found with minimum {min_set_size} titles.')}")
        print(f"  {dim('Try lowering --min-titles')}\n")

    print()
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
        "-m",
        "--min-titles",
        type=int,
        metavar="int",
        default=5,
        help="minimum number of titles in a filmography to consider",
    )
    filter_group.add_argument(
        "-t",
        "--threshold",
        type=lambda s: Range.parse(s, 0, 100),
        default=Range(50, 100),
        metavar="N or MIN-MAX",
        help="completion threshold (e.g., 80 for >=80%%, 80-99 to exclude 100%%)",
    )
    filter_group.add_argument(
        "-y",
        "--years",
        type=lambda s: Range.parse(s, 1900, datetime.now().year),
        default=Range(1930, datetime.now().year),
        metavar="YEAR or MIN-MAX",
        help="year range (e.g., 1980 for >=1980, 1980-2000 for specific era)",
    )
    filter_group.add_argument(
        "--name",
        action="append",
        metavar="str",
        help="limit to specific person names (repeatable, case-insensitive)",
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
        choices=list(TitleType.__members__),
        default=[t.name for t in TitleType.default],  # type: ignore[attr-defined]
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
            if confirm(f"Download fresh data? (current: {file_date})", default=False):
                download_imdb_data(args.data_dir, replace=True)
        else:
            download_imdb_data(args.data_dir)

    if args.rebuild or (has_all_data and not db_path.exists()):
        convert_to_sqlite(db_path)
        if args.rebuild:
            print(f"{bold(green('✓ Setup complete!'))} You can now analyze your watch history.\n")
            sys.exit(0)

    if not db_path.exists():
        print(red("Error: IMDb database not found."))
        print("\nPlease run with --rebuild first:")
        print(f"  {dim(f'python {sys.argv[0]} --rebuild')}\n")
        sys.exit(1)

    if not args.file:
        parser.error("the following arguments are required: file")

    if args.min_titles < 1:
        parser.error("Minimum titles must be at least 1")

    filters = Filters(
        years=args.years,
        title_types=frozenset(TitleType[name] for name in args.types),
    )

    try:
        analyze_sets(
            watched_file=args.file,
            min_set_size=args.min_titles,
            threshold=args.threshold,
            only=args.only,
            max_results=args.limit,
            debug=args.debug,
            filters=filters,
            db_path=db_path,
            watchlist_file=args.watchlist,
            filter_names=args.name,
        )
    except KeyboardInterrupt:
        print(f"\n{yellow('Analysis interrupted by user.')}")
        sys.exit(0)
    except Exception:
        print(red("Error:"), file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
