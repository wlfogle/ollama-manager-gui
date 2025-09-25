# Ollama Manager GUI

A full‑GUI manager for local Ollama models: list/update/delete/local maintenance and internet discovery/download.

## Features
- Models tab
  - List installed models, delete, and re‑pull (update)
  - Download by explicit name (e.g., `llama3.1:8b`)
- Maintenance tab
  - External models directory picker (uses your selection, else `$OLLAMA_MODELS`, else `~/.ollama/models`)
  - Scan: parses manifests/blobs, finds missing/unused/incomplete (.partial) files
  - Verify: computes SHA‑256 and checks against the digest encoded in filenames
  - Cleanup: delete unused blobs and .partial files
  - Repair: re‑pulls all installed models via Ollama API
- Discover tab
  - Source picker with auto‑reload and pagination (Prev/Next):
    - [OLLAMA] Ollama Library
    - [HF] Hugging Face (GGUF)
    - [HF‑TB] Hugging Face (TheBloke GGUF)
  - Each entry is tagged with its provider and shows a brief description (params, quant, instruct/base, task, license, downloads where available)
  - One‑click Download Selected

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

## Tips
- If Discover shows few entries on HF: you may be rate limited; try switching sources or paging. I can add HF token support on request.
- Maintenance works best if your models directory is writable by your user. If you store models on external drives, ensure ownership is correct for your user.
