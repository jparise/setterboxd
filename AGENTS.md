# AGENTS.md

Python CLI analyzing Letterboxd exports against IMDb datasets. Single file: `setterboxd.py`. See [README.md](README.md) for usage.

## Code Quality

Always run after changes:
```bash
ruff format setterboxd.py  # Format first
ruff check --fix setterboxd.py
pyright setterboxd.py
```

## Notes

- `--rebuild` takes 2-5 min; only needed when changing `convert_to_sqlite()`.
- Update README.md when changing user-facing features.
