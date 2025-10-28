# Setterboxd

Discover completed and near-complete "sets" of movies in your Letterboxd watchlist. Find directors and actors where you've seen most of their work, and get recommendations for the films you're missing.

## What it does

Setterboxd analyzes your Letterboxd watched films to find:
- **Directors** whose filmography you've nearly completed
- **Actors** whose movies you've mostly seen
- **Missing titles** to complete each set

Perfect for cinephiles who want to explore filmographies systematically or discover which directors/actors they've been unconsciously following.

## Quick Start

**Prerequisites:** Python 3.11+

**Installation options:**

**Option 1: Using `uv` (recommended)** - automatically handles dependencies

```bash
# Install uv if you don't have it: https://docs.astral.sh/uv/
# Clone or download this repository
cd setterboxd

# Download IMDb data and build database (~1GB download, takes 2-5 minutes)
uv run setterboxd.py --rebuild
```

**Option 2: Using Python directly** - requires manual dependency installation

```bash
# Clone or download this repository
cd setterboxd

# Install dependencies
pip install pandas rich

# Download IMDb data and build database (~1GB download, takes 2-5 minutes)
python setterboxd.py --rebuild
```

**Export your Letterboxd data:**

1. Go to [letterboxd.com/settings/data](https://letterboxd.com/settings/data)
2. Export your data
3. Extract `watched.csv` from the downloaded zip file

**Run the analysis:**

```bash
# With uv:
uv run setterboxd.py watched.csv

# Or with Python:
python setterboxd.py watched.csv
```

This shows directors and actors where you've seen at least 50% of their films (minimum 5 films each).

## Examples

All examples work with either `uv run setterboxd.py` or `python setterboxd.py`.

**Find directors you've almost completed:**

```bash
uv run setterboxd.py watched.csv --only directors --threshold 80
```

**Look up specific directors or actors:**

```bash
uv run setterboxd.py watched.csv --name Hitchcock --name "John Ford"
```

**Include your watchlist to prioritize films you've already added:**

```bash
uv run setterboxd.py watched.csv --watchlist watchlist.csv
```

**Focus on recent cinema (1990+) with deeper filmographies:**

```bash
uv run setterboxd.py watched.csv --min-year 1990 --min-titles 10
```

**Analyze a specific era (e.g., 1980s-2000s cinema):**

```bash
uv run setterboxd.py watched.csv --min-year 1980 --max-year 2000
```

**Include TV movies and miniseries:**

```bash
uv run setterboxd.py watched.csv --types movie tvMovie tvMiniSeries
```

**See debug info for unmatched titles:**

```bash
uv run setterboxd.py watched.csv --debug
```

## All Options

```
--threshold, -t    Completion percentage (0-100, default: 50)
--limit, -n        Max sets to display (default: 20)
--min-titles, -m   Min films in filmography (default: 5)
--min-year         Filter by minimum year (default: 1930)
--max-year         Filter by maximum year (default: current year)
--name             Filter to specific person names (repeatable)
--only             Analyze only "directors" or "actors"
--types            Title types: movie, video, tvMovie, tvMiniSeries
--watchlist        Path to watchlist.csv to prioritize unwatched films
--data-dir         Location for IMDb data (default: imdb_data/)
--rebuild          Rebuild database from fresh IMDb data
--debug            Show unmatched titles and statistics
```

## How it works

Setterboxd uses IMDb's public datasets to match your watched films and build complete filmographies. The first run downloads ~1GB of data and converts it to a local SQLite database. Subsequent analyses are fast (5-15 seconds for 500+ films).

The matching algorithm handles title variations, year discrepancies, and international titles to maximize match accuracy (typically 95%+).

## Data & Privacy

All analysis happens locally on your machine. No data is sent anywhere. IMDb datasets are from [datasets.imdbws.com](https://datasets.imdbws.com/) (updated daily by IMDb).

## License

MIT
