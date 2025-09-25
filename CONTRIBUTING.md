# Contributing

Thanks for contributing! This project aims to keep a clean main branch with CI checks.

## Dev setup
- Python >= 3.10
- Create a venv and install dev tools:
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install pre-commit ruff black mypy
pre-commit install
```

## Running
```
python ollama_manager_gui.py
```

## Style
- Ruff + Black
- Mypy (non-strict)
- Keep network/IO off the UI thread; use worker threads and Qt signals.

## Releases
- Tag using semantic versioning (vMAJOR.MINOR.PATCH)
- GitHub Actions will build and publish release notes.