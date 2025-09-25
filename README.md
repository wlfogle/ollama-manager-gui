# Ollama Manager GUI

A simple GUI to manage local Ollama models: list, delete, update (pull latest), and download new models.

## Features
- List installed models
- Delete selected model
- Update (re‑pull) selected model
- Download a model by name (e.g., `llama3.1:8b`)
- Pick an external models directory (with Browse…) and persist the selection

## Requirements
- Ollama running locally (default API: http://localhost:11434)
- Python 3.10+
- `PyQt6` and `requests`

## Quick start
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python ollama_manager_gui.py
```

## Notes
- Long‑running pulls run in a background thread with a progress dialog.
- The app talks to Ollama’s HTTP API; ensure the Ollama service is running.
