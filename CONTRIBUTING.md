# Contributing

Thanks for contributing! This project aims to keep a clean main branch with CI checks.

## Dev setup
For the Rust/Tauri application, use `npm ci --ignore-scripts` and
`npm run tauri dev`. Run `npm test`, `npm run build`, `cargo fmt --manifest-path
src-tauri/Cargo.toml -- --check`, `cargo test --manifest-path src-tauri/Cargo.toml`,
and `cargo clippy --manifest-path src-tauri/Cargo.toml --all-targets -- -D warnings`.
Worker checks are `python3 -m unittest discover -s worker -p test_worker.py -v`,
`ruff check worker`, and `black --check worker`. CUDA and native WebView integration
test instructions are in README; those tests use temporary generated fixtures.
Do not execute repository code during ingestion, bypass source review, enable
remote model code, or perform destructive cleanup of unfamiliar model stores.

The setup below is for the preserved legacy PyQt launcher:
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