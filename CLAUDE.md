# CLAUDE.md

Guidance for Claude Code when working with this repository. See [README.md](README.md) for user documentation.

## Project Overview

Python CLI that analyzes Letterboxd exports to find nearly-complete director/actor "sets". Uses IMDb datasets in a SQLite database. Single-file design: `setterboxd.py`.

**Key principles:** Optimize for runtime performance (bulk queries, indexes). Use modern Python (3.11+, type hints). Update README.md when changing user-facing features.

## Code Quality

Always run after changes:
```bash
ruff format setterboxd.py  # Format first
ruff check --fix setterboxd.py
pyright setterboxd.py
```

## Architecture

**Database (4 tables, see `convert_to_sqlite()`):**
- `titles`: Movies/TV with `title_lower`, `original_title_lower`, `year`, `title_type` (mapped int from TITLE_TYPE_MAP)
- `directors/actors`: Denormalized junctions (`person_id`, `title_id`)
- `names`: Person lookup (`name_id`, `name`)

**Key indexes:** `idx_titles_title_year`, `idx_titles_type_year`, `idx_directors_director`, `idx_actors_actor`

**Core functions:**
- `match_movie_sqlite()`: 6-strategy fuzzy matching (exact → subtitle variants → prefix → substring)
- `normalize_title()`: Lowercase, punctuation normalization, ampersands → "and"
- `analyze_sets()`: Main orchestration (load CSV → match → collect people → fetch filmographies → calculate %)
- `format_title_list()`: Rich output with Letterboxd links

## Development Notes

- Test with `--debug` to see unmatched titles and stats
- Only rebuild database (`--rebuild`) when changing `convert_to_sqlite()` or dataset handling (takes 2-5 min)
- Expected runtime: 5-15s for 500 movies
- Use `make_progress()` for progress bars, `console.print()` for output
