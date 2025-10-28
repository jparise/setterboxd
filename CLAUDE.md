# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**User-facing documentation:** See [README.md](README.md) for installation instructions, usage examples, and end-user documentation.

## Documentation Maintenance

**IMPORTANT:** When making code changes that affect user-facing functionality, **always update both files:**
- **CLAUDE.md** - Update technical details, architecture notes, and development guidance
- **README.md** - Update user-facing instructions, examples, and feature descriptions

This includes changes to:
- Command-line arguments and options
- Feature additions or removals
- Database schema or data handling
- Performance characteristics
- Setup or installation steps

## Project Overview

Setterboxd is a Python CLI tool that analyzes Letterboxd watchlists to discover completed and near-complete "sets" of movies by directors or actors. It downloads IMDb datasets and converts them to a SQLite database for fast, efficient querying.

## Development Principles

- **Performance first**: Optimize for runtime efficiency (bulk queries, indexed lookups, minimal memory)
- **Modern Python**: Use idiomatic patterns, type hints, and modern syntax (Python 3.11+)
- **Clear naming**: Prefer descriptive, contextual names over abbreviations

## Dependencies

The project requires Python 3.11+ and two external packages:
- **pandas**: For efficient CSV parsing and data processing
- **rich**: For terminal UI (progress bars, tables, clickable links)

Dependencies are declared using PEP 723 inline script metadata at the top of `setterboxd.py`, allowing the script to be run with tools like `uv` without a separate requirements file.

## Environment Setup

Activate the virtual environment:

```bash
source .venv/bin/activate
```

## Key Commands

**First-time setup:**
```bash
# Download IMDb datasets and convert to SQLite (~1GB download, 2-5 min conversion)
python setterboxd.py --rebuild
```

**Standard usage:**
```bash
# Analyze a Letterboxd watched.csv export
python setterboxd.py watched.csv

# With custom thresholds and limits
python setterboxd.py watched.csv --threshold 80 --limit 20

# Exclude 100% complete sets
python setterboxd.py watched.csv --threshold 80-99

# Analyze only directors or actors
python setterboxd.py watched.csv --only directors
python setterboxd.py watched.csv --only actors

# Look up specific directors or actors
python setterboxd.py watched.csv --name Hitchcock --name "John Ford"

# Adjust set parameters
python setterboxd.py watched.csv --min-titles 10 --years 1970

# Analyze a specific era
python setterboxd.py watched.csv --years 1980-2000

# Include watchlist to prioritize unwatched films
python setterboxd.py watched.csv --watchlist watchlist.csv

# Include TV content
python setterboxd.py watched.csv --types movie tvMovie tvMiniSeries
```

## Architecture

**Single-file design:** The entire application is `setterboxd.py`.

**Core modules (by function):**

1. **IMDb Dataset Management** (`download_imdb_data`): Downloads and extracts 4 IMDb TSV datasets from datasets.imdbws.com into `imdb_data/`

2. **SQLite Conversion** (`convert_to_sqlite`): One-time conversion of TSV files to optimized SQLite database with 4 tables:
   - `titles`: Movies/TV shows with normalized title columns for matching
     - `tconst` (TEXT, unique): IMDb title identifier (primary key)
     - `title_type` (INT): Mapped integer from TITLE_TYPE_MAP (1=movie, 2=video, 3=tvMiniSeries, 4=tvMovie)
     - `title` (TEXT): Primary title from IMDb
     - `title_lower` (TEXT, indexed): Normalized lowercase title for matching
     - `original_title` (TEXT): Original title (may differ from primary)
     - `original_title_lower` (TEXT, indexed): Normalized lowercase original title
     - `year` (INT, indexed): Release year
   - `directors`: Denormalized junction table
     - `director_id` (TEXT, indexed): IMDb name identifier (nconst)
     - `title_id` (TEXT, indexed): IMDb title identifier (tconst)
   - `actors`: Denormalized junction table
     - `actor_id` (TEXT, indexed): IMDb name identifier (nconst)
     - `title_id` (TEXT, indexed): IMDb title identifier (tconst)
   - `names`: Person lookup table
     - `name_id` (TEXT, unique): IMDb name identifier (nconst)
     - `name` (TEXT): Person's name

3. **Title Matching** (`match_movie_sqlite`, `normalize_title`): Multi-strategy fuzzy matching with the following fallback chain:
   - **Normalization** (`normalize_title`): Converts titles to lowercase, normalizes punctuation (em/en dashes → hyphens, curly quotes → straight quotes), converts ampersands to "and", handles superscript numbers, normalizes subtitle separators
   - **Strategy 1**: Exact match on normalized title with year tolerance (±1 year by default)
   - **Strategy 2**: Exact match on normalized title without year constraint (picks closest year)
   - **Strategy 3**: Subtitle variation matching (for "Series: Subtitle" formats)
     - Try part after colon: "Black Holes: The Edge of All We Know" → "The Edge of All We Know"
     - Try part before colon: "A Disturbance in the Force: How..." → "A Disturbance in the Force"
   - **Strategy 4**: Possessive prefix removal (for "Director's Title" → "Title")
   - **Strategy 5**: Prefix matching ("Mission: Impossible" matches "Mission: Impossible - Part One")
   - **Strategy 6**: Substring matching for longer titles (≥10 chars)

4. **Set Analysis** (`analyze_sets`): Main orchestration function that:
   - Loads Letterboxd CSV exports (watched + optional watchlist)
   - Matches titles to IMDb using indexed queries
   - Collects directors/actors from watched movies
   - Fetches complete filmographies in bulk
   - Calculates completion percentages using set operations

5. **Display** (`format_title_list`, `film_to_letterboxd_url`): Rich terminal output with clickable Letterboxd links, watchlist prioritization, and color-coded results

**Performance optimizations:**
- **Bulk queries**: Single query fetches all filmographies (not per-person queries)
- **Indexed lookups**: Composite indexes on frequently-queried columns
- **Denormalized tables**: Pre-split many-to-many relationships for fast JOINs
- **Minimal memory**: Streaming CSV parsing and batched SQL fetches

## Key Parameters

**Filters:**
- `--threshold` / `-t`: Completion % or range (e.g., 80, 80-99 to exclude 100%, default: 50)
- `--limit` / `-n`: Maximum number of sets to display (default 20)
- `--min-titles` / `-m`: Minimum films in a filmography to consider (default 5)
- `--years`: Year or range (e.g., 1980, 1980-2000 for specific era, default: 1930-present)
- `--name`: Filter results to specific person names (repeatable, case-insensitive word matching)
- `--only`: Only analyze `directors` or `actors` (default: analyze both)
- `--types`: Title types to consider: `movie`, `video`, `tvMovie`, `tvMiniSeries` (default: `movie`)
- `--watchlist`: Path to watchlist.csv to prioritize unwatched films in results

**Data:**
- `--data-dir`: Location for IMDb datasets and database (default `imdb_data/`)
- `--rebuild`: Rebuild database from IMDb datasets (downloads fresh data if requested)

**Other:**
- `--debug`: Show debug information including unmatched titles and statistics

## Data Files

**IMDb datasets** (stored in `imdb_data/`):
- Downloaded from https://datasets.imdbws.com/
- TSV files: ~5GB total
- SQLite database: `imdb.db` (~2-3GB, created automatically)

**Input format:**
Letterboxd CSV export with columns: `Name`, `Year` (at minimum)

## Performance

**Expected runtimes:**
- First-time setup: 2-5 minutes (download + SQLite conversion)
- Subsequent runs: 5-15 seconds total for 500+ movie watchlist
  - Database connection: <0.1s
  - Movie matching: 1-3s
  - Director/actor analysis: 3-10s

## Code Quality

The project uses **ruff** for linting and formatting, and **pyright** for type checking.

**Automatic formatting**: `ruff format` runs automatically via `.claude/settings.json` hooks after any Python file edit. No manual formatting needed.

**Manual checks**:

```bash
# Lint and check code quality
ruff check setterboxd.py

# Auto-fix linting issues
ruff check --fix setterboxd.py

# Type checking
pyright setterboxd.py
```

**Ruff** is configured in `pyproject.toml` with a comprehensive rule set for code quality, formatting, and modernization.

**Type checking**: The project uses type hints throughout (`NamedTuple`, `TypedDict`, `Literal`, etc.) and pyright validates type correctness.

## Database Schema Details

**Key indexes for performance:**
- `idx_titles_pk`: Unique index on `titles(tconst)`
- `idx_titles_title_year`: Composite index on `titles(title_lower, year)` - primary matching query
- `idx_titles_original_title_year`: Composite index on `titles(original_title_lower, year)`
- `idx_titles_type_year`: Composite index on `titles(title_type, year)` - filmography queries
- `idx_directors_director`: Index on `directors(director_id)` - filmography lookups
- `idx_directors_title`: Index on `directors(title_id)` - reverse lookups
- `idx_actors_actor`: Index on `actors(actor_id)` - filmography lookups
- `idx_actors_title`: Index on `actors(title_id)` - reverse lookups
- `idx_names_pk`: Unique index on `names(name_id)`

The database uses `ANALYZE` to gather statistics for the query optimizer.

## Development Workflow

**Testing changes:**
- Run against a small test CSV to verify functionality quickly
- Use `--debug` flag to see detailed matching statistics and unmatched titles
- Compare results before/after changes to verify correctness

**Database regeneration during development:**
- Only regenerate when modifying `convert_to_sqlite()` or `download_imdb_data()`
- Use `--rebuild` to force fresh database creation
- Database creation takes 2-5 minutes; avoid regenerating unnecessarily

**Verifying performance:**
- Track total runtime with `time python setterboxd.py watched.csv`
- Expected: 5-15 seconds for 500+ movie watchlist on modern hardware
- If significantly slower, check query patterns and index usage
- Use SQLite's `EXPLAIN QUERY PLAN` to verify indexes are being used

**Adding new functionality:**
- Place new helper functions near related functions
- Public API: `main()`, `analyze_sets()`, command-line interface
- Internal: All other functions (prefixed with `_` if purely internal)
- Follow existing patterns for progress bars (use `make_progress()`) and console output (use `console.print()`)

## Common Modifications

**Adding new title types:**
1. Add to `TITLE_TYPE_MAP` dictionary (e.g., `"tvSeries": 5`)
2. Update `--types` choices in argument parser
3. Rebuild database to include new type in filtering

**Modifying matching logic:**
- Edit `match_movie_sqlite()` to add new fallback strategies
- Edit `normalize_title()` to handle new normalization patterns
- Test with `--debug` to see unmatched titles and verify improvements
- No database rebuild required for matching changes

**Adding new filtering options:**
1. Add argument in `filter_group` section of argument parser
2. Pass parameter to `analyze_sets()`
3. Apply filter in appropriate location (e.g., in `fetch_filmographies_bulk()` for filmography filters)

**Extending output format:**
- Modify table creation in `analyze_sets()` for new columns
- Edit `format_title_list()` for custom title formatting
- Use Rich markup for styling: `[bold]`, `[cyan]`, `[dim]`, `[link=url]text[/link]`

## Troubleshooting

**Corrupted or outdated database:**
```bash
# Delete and rebuild from scratch
rm imdb_data/imdb.db
python setterboxd.py --rebuild
```

**Verifying database integrity:**
```bash
sqlite3 imdb_data/imdb.db "PRAGMA integrity_check;"
sqlite3 imdb_data/imdb.db "SELECT COUNT(*) FROM titles;"  # Should be ~10M titles
sqlite3 imdb_data/imdb.db "SELECT COUNT(*) FROM names;"   # Should be ~13M names
```

**Many unmatched titles:**
- Use `--debug` to see which titles aren't matching
- Check year accuracy in Letterboxd export
- Verify title spelling matches IMDb (some Letterboxd titles use alternative spellings)
- Consider adding new normalization rules to `normalize_title()` or matching strategies to `match_movie_sqlite()`

**Slow performance:**
- Verify indexes exist: `sqlite3 imdb_data/imdb.db ".indexes titles"`
- Check database statistics: `sqlite3 imdb_data/imdb.db "PRAGMA stats;"`
- If indexes missing, rebuild database with `--rebuild`
- Large watchlists (1000+) may take longer but should still complete in <30s

**Memory issues:**
- The tool uses streaming/chunked processing to minimize memory
- If running out of memory, reduce pandas chunk size in `convert_to_sqlite()` (currently 500k rows)
- SQLite queries use batched fetching (1000 rows at a time) to avoid loading full result sets

## Performance Benchmarks

**Expected database sizes:**
- TSV files: ~5GB total
- SQLite database: ~2-3GB (better compression than raw TSV)
- Memory usage during conversion: ~1-2GB peak
- Memory usage during analysis: ~200-500MB

**Query performance targets:**
- Title matching (500 titles): 1-3 seconds
- Director collection: 2-5 seconds
- Actor collection: 3-8 seconds (larger dataset)
- Filmography bulk fetch: 1-2 seconds per category

**Match rate expectations:**
- Well-maintained watchlist: 95-99% match rate
- Older or less common films: 85-95% match rate
- If below 80%, investigate with `--debug` flag
