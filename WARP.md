# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

Project overview
- A PyQt6 desktop GUI to manage local Ollama models: list/update/delete, perform maintenance on the models directory, and discover/download models from the Ollama Library and Hugging Face (including TheBloke).
- Requires Ollama running locally (default API http://localhost:11434) and Python 3.10+.
- Core dependencies: PyQt6, requests. Optional: HF_API_TOKEN/HUGGINGFACE_API_TOKEN to reduce HF rate limiting.

Common commands
- Create and activate a virtualenv
  - python -m venv .venv
  - source .venv/bin/activate

- Install runtime deps
  - pip install -r requirements.txt

- Run the app
  - python ollama_manager_gui.py

- Install dev tools and pre-commit hooks (recommended for contributors)
  - pip install pre-commit ruff black mypy
  - pre-commit install
  - pre-commit run --all-files

- Lint/format locally
  - ruff --fix .
  - black .

- Type-check
  - mypy --ignore-missing-imports --follow-untyped-imports .

Notes
- No test suite is present in this repo at the time of writing.
- A console entry point is declared (pyproject [project.scripts]: ollama-manager -> ollama_manager_gui:main). Packaging may require adding py_modules configuration if building wheels; local development runs the app directly with python ollama_manager_gui.py.

Configuration and environment
- Ollama host: the app assumes http://localhost:11434 by default.
- Models directory selection precedence (Maintenance tab)
  1) User selection in the GUI (persisted)
  2) $OLLAMA_MODELS if set
  3) ~/.ollama/models
- Persisted config path: ~/.config/ollama-manager-gui/config.json
  - Keys observed: models_dir, hf_api_token
- Hugging Face authentication (optional): set HF_API_TOKEN (or HUGGINGFACE_API_TOKEN) in the environment before launch to reduce rate limiting.

High-level architecture
- GUI framework: PyQt6, single MainWindow with a QTabWidget providing three functional areas.
  - Models tab
    - Lists installed models via OllamaClient.list_models() (GET /api/tags)
    - Update/Delete actions call OllamaClient.pull_model_stream() (POST /api/pull) and DELETE /api/delete
    - Pull progress is streamed and surfaced in UI; operations run on background threads
  - Maintenance tab
    - External models directory picker, with status of blobs/ and manifests/
    - Scan: walks manifests/ and blobs/ under the selected models dir
      - Parses manifests to extract sha256 digests; collects present blob digests from filenames
      - Computes missing, unused, and lists .partial files
      - Implemented by ScanWorker (QObject) emitting status/finished signals
    - Verify Hashes: VerifyWorker recomputes SHA256 for blobs and reports corrupt files
    - Cleanup actions: delete unused blobs and .partial files, with re-scan afterwards
    - Repair Installed: iterates installed models and re-pulls sequentially to restore missing blobs
  - Discover tab
    - Provider selector: Ollama Library, Hugging Face (GGUF), HF (TheBloke GGUF)
    - Background workers fetch and prepare result lists:
      - OllamaLibraryDiscoverWorker scrapes names/metadata from https://ollama.com/library with simple HTML parsing and description augmentation
      - HuggingFaceDiscoverWorker and HuggingFaceTheBlokeWorker call the HF REST API, enumerate GGUF assets, and compose a descriptive summary with quant/params/task/license/tags
    - UI supports filtering by instruct/base, minimum params (e.g., "7B" or "8x7B"), quant substring (e.g., Q4_K), and task substring
    - Actions: copy suggested command, open source page, or Download Selected
      - For HF providers, Download Selected streams the .gguf into ~/.cache/ollama-manager-gui/downloads and then creates an Ollama model by generating a temporary Modelfile (FROM <path>) and calling OllamaClient.create_model()
- Concurrency model
  - Long-running operations run on Python threads and communicate back via Qt signals (pyqtSignal) to keep the UI responsive
  - Progress dialogs (QProgressDialog) are used for verification, pulls, and downloads
- Networking
  - All outbound HTTP uses requests with timeouts and simple retry logic for HF API calls (network_client.get_json_with_retry)
  - User-agent headers are set to avoid provider blocks; HF requests optionally include a Bearer token when present in env
- Configuration utilities
  - Config is read/written as JSON under ~/.config/ollama-manager-gui
  - GUI keeps the external models directory setting synchronized with config and updates status labels accordingly

CI and repo hygiene
- Pre-commit is configured with ruff (with --fix), black, and mypy (non-strict). Use pre-commit run --all-files locally before pushing.
- CONTRIBUTING.md: follow Python 3.10+, keep network/IO off the UI thread (use worker threads + Qt signals), tag releases with vMAJOR.MINOR.PATCH. GitHub Actions will build and publish release notes on tags.

Key files
- ollama_manager_gui.py: main application, UI layout, tabs, workers and their orchestration, app entry point (main())
- network_client.py: HTTP helpers and HF headers/token handling
- pyproject.toml: PEP 621 metadata, dependencies, and a console script declaration
- .pre-commit-config.yaml: ruff, black, mypy hooks
- README.md: feature overview and quick-start instructions
