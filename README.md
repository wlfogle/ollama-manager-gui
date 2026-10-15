# Ollama Manager GUI

## Ollama Workshop — Rust/Tauri desktop app

A local-first desktop app for model management, grounded repository questions,
reviewed repository datasets, and real LoRA/QLoRA fine-tuning. The original PyQt
manager remains available below; it has not been removed or silently migrated.

### Run the desktop app

Requirements: Linux graphical session, Rust, Node.js 20.19+ (or a supported newer
LTS), Python 3.10+, Git, GTK 3 and WebKitGTK 4.1 development libraries. Ollama
must be running for model operations and grounded answers. Repository indexing
and dataset editing work without CUDA training dependencies.

From this repository (Fish-compatible; activation is not required):

```fish
npm ci --ignore-scripts
npm run tauri dev
```

The development app is served only on loopback. Build an installable Debian
package with `npm run tauri build`; output is under
`src-tauri/target/release/bundle/deb/`. `npm run tauri build -- --debug --no-bundle`
creates a native debug binary with embedded frontend assets.

If native headers are missing on Pop!_OS/Ubuntu:

```fish
sudo nala install build-essential pkg-config libwebkit2gtk-4.1-dev libgtk-3-dev libssl-dev librsvg2-dev libxdo-dev libayatana-appindicator3-dev
```

### Storage and model management

- **Models** lists models from the running local Ollama daemon, with inspection,
  explicit pull/update, confirmed deletion, and inference.
- **Maintenance** inspects the configured model store, finds missing references,
  verifies SHA256, and can re-pull installed models for repair.
- The initial model-store path is
  `/media/loufogle/73cf9511-0af0-4ac4-9d83-ee21eb17ff5d/models`.
  Selecting this path does **not** reconfigure the Ollama daemon or relocate files.
  `manifests-v2` and `metadata` are reported as unfamiliar reference layouts:
  destructive blob cleanup is disabled rather than risking referenced models.
- **Settings** configures the model-store path, a separate workspace, a local
  Ollama origin, and an absolute Python executable.
- Linux settings live at `~/.local/share/dev.ollama.workshop/settings.json`.
  The default workspace is the adjacent `workspace/` directory. Custom
  `XDG_DATA_HOME` settings may change the base path.
- The workspace contains `repositories.sqlite3`, `datasets/`, `downloads/`,
  and `jobs/<uuid>/`. Keep it private: it contains code, training examples,
  checkpoints, and adapters. None are uploaded by this application.
- **Discover** searches Hugging Face, shows published file sizes, and explicitly
  downloads a pinned training snapshot or a selected GGUF. Snapshots include
  top-level safetensors/tokenizer/configuration files, not Python model code or
  pickle weights. Interrupted files restart safely; previously completed files
  are reused only after checking their published checksum. No blind Range appends.
- Authentication is inherited from `HF_TOKEN`, `HF_API_TOKEN`, or
  `HUGGINGFACE_API_TOKEN`; tokens are not saved in UI state or job configuration.
  Launch the app from an appropriately configured environment for gated models.
  Authorization headers are removed on redirects away from Hugging Face.

### Use your repositories

1. Open **Repositories**, enter a local Git repository root, and index it.
   Only tracked working-tree files are considered; ignored paths, symlinks,
   unsupported/binary/oversized files, common generated folders, credential
   filenames, and suspected secrets are excluded. The report shows paths and
   reasons, never the detected secret values. Git filesystem-monitor hooks are
   disabled during ingestion; repository code is not executed.
2. Search the local SQLite full-text index, or select an installed Ollama model
   and ask a question. Answers receive bounded retrieved chunks and source
   records with file/line citations. This is lexical RAG, not embedding search.
   Refresh after code changes: the index is a snapshot, not a live filesystem.
3. Review indexed files using the preview buttons, check the approved files,
   and export a named dataset. Confirm your right to train on their contents.
   Secret filtering is heuristic, **not a guarantee**; human review is required.
4. Open **Datasets** to inspect the JSONL. Exported rows contain `text` and
   `_group` (repository/file identity). A `.provenance.json` sidecar records the
   repository commit, source paths, line ranges, hashes, and ingestion policy.
   Raw source produces continued-language-modeling examples, not fabricated
   instruction/answer pairs.
5. Train/validation splits are deterministic and grouped by source file.
   Keep `_group` when editing source datasets to avoid same-file chunk leakage.
   Editor saves create a new dataset name; they do not overwrite exports or
   claim that edited content still matches the original provenance.

The editor also imports JSON arrays or JSONL containing `{"text":"..."}`,
`{"instruction":"...","input":"...","output":"..."}`, or chat `messages`.
Chat examples require an assistant response and a model tokenizer with a chat
template. The editor/import size limit is 32 MiB; source files are limited to
512 KiB and chunked at 80 lines.

### Set up local GPU training

Rust supervises a Python CUDA worker using argument arrays and a JSON protocol.
It does not run training through a shell or install dependencies automatically.
The following installs into an isolated environment; CUDA packages require
several GB of disk space and download bandwidth.

```fish
python3 -m venv .training-venv
.training-venv/bin/python -m pip install 'torch>=2.10,<3' --index-url https://download.pytorch.org/whl/cu128
.training-venv/bin/python -m pip install -r worker/requirements.txt
```

In **Settings**, set Python to the absolute path of
`.training-venv/bin/python` (for this checkout:
`/media/loufogle/Data/Repos/ollama-manager-gui/.training-venv/bin/python`).
`worker/requirements-tested.txt` records the tested core Linux/CUDA package versions.
Choose a CUDA build supported by your driver if installing on other hardware.

In **Training**:

1. Run the environment check. It reports actual PyTorch/CUDA availability,
   library versions, GPU free memory, and disk space.
2. Select a local Transformers safetensors snapshot and a reviewed dataset.
   Ollama inference blobs and GGUF files are **not** training base weights.
   Download original weights explicitly in Discover first; the training worker
   uses `local_files_only=True` and `trust_remote_code=False`.
3. Choose LoRA or NF4 QLoRA, rank, sequence length, batch size, gradient
   accumulation, learning rate, maximum steps, and checkpoint interval.
   A positive maximum step count takes precedence over epochs; use `-1` to
   train for the configured number of epochs instead.
   Start with a small 0.5B–1.5B coding model; 12 GB VRAM does not guarantee that
   every larger model fits. Validation metrics are full-sequence language-model
   loss, not a benchmark proving coding capability.
4. Approve review and start. The jobs panel reports actual worker events, loss,
   step/epoch, evaluation metrics, and checkpoints. Click a job for its event
   history and workspace-relative checkpoint paths.
5. Cancel requests stop at a safe step and save a checkpoint; cancellation
   during model loading can take longer. A crash/closed desktop kills its Linux
   worker; restarted jobs are marked interrupted, not falsely running. Resume
   from a listed `jobs/<uuid>/checkpoint-N` with compatible model/dataset/config.
6. Completed runs save PEFT adapters, tokenizer files, and a training manifest.
   Export to a **new** absolute directory. This does not overwrite existing
   directories, merge full model weights, convert to GGUF, or promise universal
   Ollama adapter compatibility. Use an appropriate PEFT-capable runtime.

Training/download/pull/repair jobs are serialized. Other programs (including
Ollama inference) may still occupy your GPU: stop/unload their models before
large training runs. There is no from-scratch pretraining or remote-server
execution in this version.

### Desktop environment fixes

- Startup removes `xapp-gtk3-module` from `GTK3_MODULES`/`GTK_MODULES` only when
  its shared library is absent. A leftover XApp X-session hook can request it
  even when its optional package is not installed. This fixes the startup
  warning without muting GTK diagnostics, removing accessibility modules,
  changing system-wide settings, or requiring a reboot.
- Worker processes do not inherit `LD_PRELOAD`. On this system, NoMachine's
  injected graphics library causes `cudnnGetVersion` symbol-loading crashes.
  Isolation fixes CUDA without changing the desktop session.

### Verification

```fish
python3 -m unittest discover -s worker -p test_worker.py -v
npm test
npm run build
cargo test --manifest-path src-tauri/Cargo.toml
cargo clippy --manifest-path src-tauri/Cargo.toml --all-targets -- -D warnings
```

Opt-in real CUDA tests generate a tiny random Llama model and tokenizer locally;
they download no pretrained weights. They verify nonzero adapter updates,
checkpoint resume, cancellation, and adapter export—not useful model quality.

```fish
env -u LD_PRELOAD WORKSHOP_CUDA_TEST=1 .training-venv/bin/python -m unittest discover -s worker -p test_cuda.py -v
```

The native WebView smoke test uses a temporary repository/workspace and a real
Ollama service; it exercises UI controls, IPC, retrieval, and GPU training. It
requires a graphical session and an already installed chat model. The opt-in
feature is inactive in normal builds.

```fish
npm run tauri build -- --debug --no-bundle --features desktop-smoke
env -u LD_PRELOAD .training-venv/bin/python worker/desktop_smoke.py --binary src-tauri/target/debug/ollama-workshop
```

To verify the actual release package without installing it (requires Python
GObject/AT-SPI bindings and a graphical session):

```fish
python3 worker/package_smoke.py --deb 'src-tauri/target/release/bundle/deb/Ollama Workshop_0.2.0_amd64.deb'
```

### Installed-package verification (0.2.0)
The Debian package was installed and its files passed
`dpkg --verify ollama-workshop`. The production executable at `/usr/bin/ollama-workshop`
ingested and reviewed four generated source files, exported their dataset,
and completed a real two-step CUDA QLoRA run with checkpoints.

Resume and export were then verified directly through the installed worker at
`/usr/lib/Ollama Workshop/worker/main.py`, continuing the same checkpoint to
optimizer step 3. The exported adapter had nonzero updated LoRA weights,
reloaded successfully on CUDA, produced finite logits, and generated tokens.
This validates the training pipeline, not useful model quality: the base was
a tiny randomly initialized local fixture, with no pretrained weight download.

The full production-GUI resume/export automation did **not** pass: remote
desktop typing and WebKit accessibility/modal behavior were unreliable.
Do not describe the worker-level follow-up as an all-GUI end-to-end pass.
Original automation failures are retained alongside the successful
`installed-final-report.json`; no failed reports were overwritten.

To repeat artifact verification after a completed installed-GUI fixture run:

```fish
python3 worker/verify_installed_training.py \
  --artifact-root /path/to/retained/fixture-run \
  --training-python /absolute/path/to/.training-venv/bin/python
```

The verifier checks the installed binary's hash, source provenance, dataset
checksum, checkpoint state, resume progress, exported adapter bytes, and
CUDA reload/generation. It writes a report and retains the exported adapter.

## Legacy PyQt application
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
- Optional: set `HF_API_TOKEN` to your Hugging Face token to reduce rate limiting

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
