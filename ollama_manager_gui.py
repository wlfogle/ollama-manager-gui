import sys
import os
import re
import json
import hashlib
import threading
from pathlib import Path
import requests
from typing import Optional, Iterable, Dict, List, Set

from PyQt6.QtCore import pyqtSignal, QObject, Qt
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressDialog,
    QStatusBar,
    QVBoxLayout,
    QWidget,
    QPlainTextEdit,
    QTabWidget,
    QComboBox,
)


DEFAULT_OLLAMA_HOST = "http://localhost:11434"

# Simple headers for outbound requests to avoid being blocked by providers
HF_HEADERS = {"User-Agent": "ollama-manager-gui/1.0 (+https://github.com/wlfogle/ollama-manager-gui)", "Accept": "application/json"}
OLLAMA_LIB_HEADERS = {"User-Agent": "ollama-manager-gui/1.0", "Accept": "text/html"}

# Simple config helpers
CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".config", "ollama-manager-gui")
CONFIG_PATH = os.path.join(CONFIG_DIR, "config.json")

def load_config() -> dict:
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_config(cfg: dict) -> None:
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def default_models_dir() -> Path:
    # Respect OLLAMA_MODELS or fall back to ~/.ollama/models
    env = os.environ.get("OLLAMA_MODELS", "").strip()
    if env:
        return Path(env).expanduser()
    return Path.home() / ".ollama" / "models"


def find_all_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file():
            yield p


def extract_sha256_digests_from_text(text: str) -> Set[str]:
    # Find occurrences like sha256:<64hex>
    return set(m.group(1) for m in re.finditer(r"sha256:([0-9a-f]{64})", text))


def parse_manifest_file(path: Path) -> Set[str]:
    try:
        data = path.read_text(encoding="utf-8", errors="ignore")
        digests = extract_sha256_digests_from_text(data)
        if not digests and path.suffix.lower() == ".json":
            # Try JSON parse and walk values
            try:
                obj = json.loads(data)
                text_blob = json.dumps(obj)
                digests = extract_sha256_digests_from_text(text_blob)
            except Exception:
                pass
        return digests
    except Exception:
        return set()


def compute_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


class ScanResult:
    def __init__(self):
        self.models_dir: Path = Path("")
        self.manifest_files: List[Path] = []
        self.blob_files: List[Path] = []
        self.partial_files: List[Path] = []
        self.used_digests: Set[str] = set()
        self.present_digests: Set[str] = set()
        self.unused_blob_files: List[Path] = []
        self.missing_digests: Set[str] = set()
        self.corrupt_blob_files: List[Path] = []


class ScanWorker(QObject):
    finished = pyqtSignal(object)  # ScanResult
    status = pyqtSignal(str)

    def __init__(self, models_dir: Path):
        super().__init__()
        self.models_dir = models_dir

    def start(self):
        res = ScanResult()
        res.models_dir = self.models_dir
        manifests_dir = self.models_dir / "manifests"
        blobs_dir = self.models_dir / "blobs"

        # Collect manifests and digests referenced
        manifest_files: List[Path] = []
        if manifests_dir.exists():
            for p in find_all_files(manifests_dir):
                # consider text/json-like files only (heuristic): size < 10MB
                try:
                    if p.is_file() and p.stat().st_size < 10 * 1024 * 1024:
                        manifest_files.append(p)
                except Exception:
                    pass
        res.manifest_files = manifest_files

        used: Set[str] = set()
        for mf in manifest_files:
            self.status.emit(f"Parsing manifest: {mf}")
            used |= parse_manifest_file(mf)
        res.used_digests = used

        # Collect blobs
        blob_files: List[Path] = []
        present: Set[str] = set()
        if blobs_dir.exists():
            for p in find_all_files(blobs_dir):
                if p.is_file():
                    blob_files.append(p)
                    # try to extract sha256 from filename patterns
                    m = re.search(r"([0-9a-f]{64})", p.name)
                    if m:
                        present.add(m.group(1))
        res.blob_files = blob_files
        res.present_digests = present

        # .partial files anywhere under models_dir
        partials: List[Path] = []
        for p in find_all_files(self.models_dir):
            if p.suffix == ".partial" or p.name.endswith(".partial"):
                partials.append(p)
        res.partial_files = partials

        # Compute missing and unused sets
        res.missing_digests = used - present
        # Unused blobs are those whose digest isn't referenced in any manifest
        # Map digest->paths
        digest_to_paths: Dict[str, List[Path]] = {}
        for p in blob_files:
            m = re.search(r"([0-9a-f]{64})", p.name)
            if not m:
                continue
            digest_to_paths.setdefault(m.group(1), []).append(p)
        unused_paths: List[Path] = []
        for d, paths in digest_to_paths.items():
            if d not in used:
                unused_paths.extend(paths)
        res.unused_blob_files = unused_paths

        self.finished.emit(res)


class VerifyWorker(QObject):
    progress = pyqtSignal(str)
    finished = pyqtSignal(list)  # corrupt files list

    def __init__(self, blob_files: List[Path]):
        super().__init__()
        self.blob_files = blob_files

    def start(self):
        corrupt: List[Path] = []
        for p in self.blob_files:
            m = re.search(r"([0-9a-f]{64})", p.name)
            if not m:
                continue
            expected = m.group(1)
            try:
                actual = compute_sha256(p)
                if actual != expected:
                    corrupt.append(p)
                    self.progress.emit(f"Corrupt: {p.name}")
                else:
                    self.progress.emit(f"OK: {p.name}")
            except Exception as e:
                corrupt.append(p)
                self.progress.emit(f"Error reading {p}: {e}")
        self.finished.emit(corrupt)


class OllamaClient:
    def __init__(self, base_url: str = DEFAULT_OLLAMA_HOST):
        self.base_url = base_url.rstrip("/")

    def list_models(self) -> list[dict]:
        r = requests.get(f"{self.base_url}/api/tags", timeout=10)
        r.raise_for_status()
        data = r.json()
        return data.get("models", [])

    def delete_model(self, name: str) -> None:
        r = requests.delete(f"{self.base_url}/api/delete", json={"name": name}, timeout=30)
        r.raise_for_status()

    def pull_model_stream(self, name: str, stream: bool = True):
        # Streaming JSON lines. Yield dict events.
        r = requests.post(
            f"{self.base_url}/api/pull",
            json={"name": name, "stream": stream},
            stream=True,
            timeout=None,
        )
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            try:
                yield json.loads(line.decode("utf-8"))
            except Exception:
                pass

    def create_model(self, name: str, modelfile: str) -> None:
        # Create a model from a Modelfile text
        r = requests.post(
            f"{self.base_url}/api/create",
            json={"name": name, "modelfile": modelfile},
            timeout=None,
        )
        r.raise_for_status()


class PullWorker(QObject):
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)  # success, message

    def __init__(self, client: OllamaClient, model_name: str):
        super().__init__()
        self.client = client
        self.model_name = model_name

    def start(self):
        ok = True
        msg = ""
        try:
            for ev in self.client.pull_model_stream(self.model_name, stream=True):
                # Common fields: status, completed, total, digest, error
                if "error" in ev:
                    ok = False
                    msg = ev.get("error", "Error during pull")
                    self.progress.emit(f"Error: {msg}")
                    break
                status = ev.get("status") or ev.get("status_code") or "..."
                completed = ev.get("completed")
                total = ev.get("total")
                if completed and total:
                    self.progress.emit(f"{status} {completed}/{total}")
                else:
                    self.progress.emit(str(status))
        except Exception as e:
            ok = False
            msg = str(e)
        finally:
            if ok and not msg:
                msg = "Completed"
            self.finished.emit(ok, msg)


class HuggingFaceDiscoverWorker(QObject):
    progress = pyqtSignal(str)
    finished = pyqtSignal(list)  # list of result dicts

    def __init__(self, installed_names: Set[str], limit: int = 50):
        super().__init__()
        self.installed_names = installed_names
        self.limit = max(1, min(limit, 200))

    def start(self):
        results: List[dict] = []
        try:
            # Get popular models (sorted by downloads)
            params = {"limit": self.limit, "sort": "downloads"}
            r = requests.get("https://huggingface.co/api/models", params=params, headers=HF_HEADERS, timeout=30)
            r.raise_for_status()
            repos = r.json()
            if not isinstance(repos, list):
                self.progress.emit("Unexpected response from Hugging Face")
                self.finished.emit(results)
                return
            for repo in repos:
                repo_id = repo.get("modelId") or repo.get("id") or repo.get("_id")
                if not repo_id:
                    continue
                rd = requests.get(f"https://huggingface.co/api/models/{repo_id}", headers=HF_HEADERS, timeout=30)
                if rd.status_code != 200:
                    continue
                info = rd.json()
                card = info.get("cardData") or {}
                # Prefer a short summary-like field if present
                summary = card.get("summary") or card.get("description") or card.get("title") or info.get("pipeline_tag") or repo_id
                siblings = info.get("siblings") or []
                ggufs = [s for s in siblings if isinstance(s, dict) and str(s.get("rfilename", "")).lower().endswith(".gguf")]
                for s in ggufs:
                    gguf_path = s.get("rfilename")
                    if not gguf_path:
                        continue
                    url = f"https://huggingface.co/{repo_id}/resolve/main/{gguf_path}"
                    suggested_name = f"hf/{repo_id}:{Path(gguf_path).stem}"
                    if suggested_name in self.installed_names:
                        continue
                    results.append({
                        "repo_id": repo_id,
                        "gguf_path": gguf_path,
                        "url": url,
                        "suggested_name": suggested_name,
                        "desc": str(summary)[:160],
                    })
            self.finished.emit(results)
        except Exception as e:
            self.progress.emit(f"Discover error: {e}")
            self.finished.emit(results)


class HuggingFaceTheBlokeWorker(QObject):
    progress = pyqtSignal(str)
    finished = pyqtSignal(list)

    def __init__(self, installed_names: Set[str], limit: int = 50):
        super().__init__()
        self.installed_names = installed_names
        self.limit = max(1, min(limit, 200))

    def start(self):
        results: List[dict] = []
        try:
            params = {"limit": self.limit, "sort": "downloads", "search": "TheBloke"}
            r = requests.get("https://huggingface.co/api/models", params=params, headers=HF_HEADERS, timeout=30)
            r.raise_for_status()
            repos = r.json()
            if not isinstance(repos, list):
                self.finished.emit(results)
                return
            for repo in repos:
                repo_id = repo.get("modelId") or repo.get("id") or repo.get("_id")
                if not repo_id:
                    continue
                rd = requests.get(f"https://huggingface.co/api/models/{repo_id}", headers=HF_HEADERS, timeout=30)
                if rd.status_code != 200:
                    continue
                info = rd.json()
                card = info.get("cardData") or {}
                summary = card.get("summary") or card.get("description") or card.get("title") or info.get("pipeline_tag") or repo_id
                siblings = info.get("siblings") or []
                ggufs = [s for s in siblings if isinstance(s, dict) and str(s.get("rfilename", "")).lower().endswith(".gguf")]
                for s in ggufs:
                    gguf_path = s.get("rfilename")
                    if not gguf_path:
                        continue
                    url = f"https://huggingface.co/{repo_id}/resolve/main/{gguf_path}"
                    suggested_name = f"hf/{repo_id}:{Path(gguf_path).stem}"
                    if suggested_name in self.installed_names:
                        continue
                    results.append({
                        "repo_id": repo_id,
                        "gguf_path": gguf_path,
                        "url": url,
                        "suggested_name": suggested_name,
                        "desc": str(summary)[:160],
                    })
            self.finished.emit(results)
        except Exception:
            self.finished.emit(results)


class DownloadAndCreateWorker(QObject):
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def __init__(self, url: str, dest: Path, client: OllamaClient, model_name: str):
        super().__init__()
        self.url = url
        self.dest = dest
        self.client = client
        self.model_name = model_name

    def start(self):
        try:
            # Download file
            with requests.get(self.url, stream=True, timeout=None) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                done = 0
                tmp = self.dest.with_suffix(self.dest.suffix + ".part")
                with tmp.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        f.write(chunk)
                        done += len(chunk)
                        if total:
                            self.progress.emit(f"Downloading {done}/{total} bytes")
                        else:
                            self.progress.emit(f"Downloading {done} bytes")
                tmp.rename(self.dest)
            self.progress.emit("Download complete. Importing into Ollama…")
            # Create a temporary Modelfile
            modelfile = f"FROM {self.dest}\n"
            self.client.create_model(self.model_name, modelfile)
            self.finished.emit(True, "ok")
        except Exception as e:
            self.finished.emit(False, str(e))


class OllamaLibraryDiscoverWorker(QObject):
    progress = pyqtSignal(str)
    finished = pyqtSignal(list)  # list of dicts {name, desc}

    def __init__(self, installed_names: Set[str], limit: int = 200):
        super().__init__()
        self.installed_names = installed_names
        self.limit = max(1, min(limit, 500))

    def fetch_desc(self, name: str) -> str:
        try:
            d = requests.get(f"https://ollama.com/library/{name}", headers=OLLAMA_LIB_HEADERS, timeout=30)
            if d.status_code != 200:
                return "Ollama Library model"
            page = d.text
            m = re.search(r'<meta\s+name="description"\s+content="([^"]+)', page, re.IGNORECASE)
            if m:
                return m.group(1)[:160]
            # fallback: first paragraph
            m2 = re.search(r"<p>(.*?)</p>", page, re.IGNORECASE | re.DOTALL)
            if m2:
                # strip tags crudely
                text = re.sub(r"<[^>]+>", " ", m2.group(1))
                return re.sub(r"\s+", " ", text).strip()[:160]
        except Exception:
            pass
        return "Ollama Library model"

    def start(self):
        results: List[dict] = []
        try:
            r = requests.get("https://ollama.com/library", headers=OLLAMA_LIB_HEADERS, timeout=30)
            r.raise_for_status()
            html = r.text
            names = sorted(set(m.group(1) for m in re.finditer(r"/library/([a-zA-Z0-9_.:-]+)", html)))
            count = 0
            for name in names:
                if name in self.installed_names:
                    continue
                desc = self.fetch_desc(name)
                results.append({"name": name, "desc": desc})
                count += 1
                if count >= self.limit:
                    break
            self.finished.emit(results)
        except Exception as e:
            self.progress.emit(f"Discover error: {e}")
            self.finished.emit(results)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ollama Manager GUI")
        self.client = OllamaClient()

        # UI
        self.model_list = QListWidget()

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_models)

        delete_btn = QPushButton("Delete Selected")
        delete_btn.clicked.connect(self.delete_selected)

        update_btn = QPushButton("Update Selected")
        update_btn.clicked.connect(self.update_selected)

        self.download_input = QLineEdit()
        self.download_input.setPlaceholderText("model name (e.g., llama3.1:8b)")
        download_btn = QPushButton("Download")
        download_btn.clicked.connect(self.download_model)

        top_bar = QHBoxLayout()
        top_bar.addWidget(refresh_btn)
        top_bar.addWidget(delete_btn)
        top_bar.addWidget(update_btn)
        top_bar.addStretch(1)

        dl_bar = QHBoxLayout()
        dl_bar.addWidget(QLabel("Download:"))
        dl_bar.addWidget(self.download_input)
        dl_bar.addWidget(download_btn)

        # Maintenance group: external models directory picker
        maint_group = QGroupBox("Maintenance")
        maint_layout = QVBoxLayout()

        dir_row = QHBoxLayout()
        dir_row.addWidget(QLabel("Models dir:"))
        self.dir_input = QLineEdit()
        self.dir_input.setPlaceholderText("optional path to external Ollama models directory")
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self.browse_models_dir)
        dir_row.addWidget(self.dir_input)
        dir_row.addWidget(browse_btn)
        maint_layout.addLayout(dir_row)

        self.dir_status = QLabel("No external models directory set.")
        self.dir_status.setWordWrap(True)
        maint_layout.addWidget(self.dir_status)

        # Maintenance actions
        actions_row = QHBoxLayout()
        scan_btn = QPushButton("Scan")
        scan_btn.clicked.connect(self.scan_models_dir)
        verify_btn = QPushButton("Verify Hashes")
        verify_btn.clicked.connect(self.verify_hashes)
        del_unused_btn = QPushButton("Delete Unused Blobs")
        del_unused_btn.clicked.connect(self.delete_unused_blobs)
        del_partials_btn = QPushButton("Delete .partial Files")
        del_partials_btn.clicked.connect(self.delete_partials)
        repair_btn = QPushButton("Repair Installed (Re-pull)")
        repair_btn.clicked.connect(self.repair_installed)
        actions_row.addWidget(scan_btn)
        actions_row.addWidget(verify_btn)
        actions_row.addWidget(del_unused_btn)
        actions_row.addWidget(del_partials_btn)
        actions_row.addWidget(repair_btn)
        maint_layout.addLayout(actions_row)

        self.maint_output = QPlainTextEdit()
        self.maint_output.setReadOnly(True)
        self.maint_output.setPlaceholderText("Maintenance results will appear here…")
        maint_layout.addWidget(self.maint_output)

        maint_group.setLayout(maint_layout)

        # ========== Build Models tab ==========
        models_page = QWidget()
        models_layout = QVBoxLayout()
        models_layout.addLayout(top_bar)
        models_layout.addWidget(self.model_list)
        models_layout.addLayout(dl_bar)
        models_page.setLayout(models_layout)

        # ========== Build Maintenance tab ==========
        maintenance_page = QWidget()
        maintenance_layout = QVBoxLayout()
        maintenance_layout.addWidget(maint_group)
        maintenance_page.setLayout(maintenance_layout)

# ========== Build Discovery tab ==========
        discover_page = QWidget()
        discover_layout = QVBoxLayout()
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Source:"))
        self.site_combo = QComboBox()
        self.site_combo.addItem("Ollama Library", "ollama")
        self.site_combo.addItem("Hugging Face (GGUF)", "hf_gguf")
        self.site_combo.addItem("Hugging Face (TheBloke GGUF)", "hf_thebloke")
        row1.addWidget(self.site_combo)
        load_btn = QPushButton("Load")
        load_btn.clicked.connect(self.discover_search)
        row1.addWidget(load_btn)
        discover_layout.addLayout(row1)
        self.discover_list = QListWidget()
        discover_layout.addWidget(self.discover_list)
        row2 = QHBoxLayout()
        download_sel_btn = QPushButton("Download Selected")
        download_sel_btn.clicked.connect(self.discover_download_selected)
        row2.addStretch(1)
        row2.addWidget(download_sel_btn)
        discover_layout.addLayout(row2)
        discover_page.setLayout(discover_layout)

        tabs = QTabWidget()
        tabs.addTab(models_page, "Models")
        tabs.addTab(maintenance_page, "Maintenance")
        tabs.addTab(discover_page, "Discover")
        self.setCentralWidget(tabs)

        self.status = QStatusBar()
        self.setStatusBar(self.status)

        self.resize(900, 600)

        # Load/save config for external models dir
        self.config = load_config()
        self.dir_input.setText(self.config.get("models_dir", ""))
        self.update_models_dir_status()
        self.dir_input.editingFinished.connect(self.on_models_dir_changed)

        # Initial load
        self.refresh_models()

    def refresh_models(self):
        try:
            self.model_list.clear()
            models = self.client.list_models()
            for m in models:
                # Example fields: name, size, modified_at, digest
                name = m.get("name", "<unknown>")
                size = m.get("size", 0)
                item = QListWidgetItem(f"{name}  —  {size} bytes")
                item.setData(Qt.ItemDataRole.UserRole, m)
                self.model_list.addItem(item)
            self.status.showMessage(f"Loaded {len(models)} models", 3000)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load models:\n{e}")

    def selected_model_name(self) -> Optional[str]:
        item = self.model_list.currentItem()
        if not item:
            return None
        m = item.data(Qt.ItemDataRole.UserRole)
        return m.get("name") if isinstance(m, dict) else None

    def delete_selected(self):
        name = self.selected_model_name()
        if not name:
            QMessageBox.information(self, "Delete", "Select a model first.")
            return
        if QMessageBox.question(self, "Delete", f"Delete model '{name}'? This cannot be undone.") != QMessageBox.StandardButton.Yes:
            return
        try:
            self.client.delete_model(name)
            self.status.showMessage(f"Deleted {name}", 3000)
            self.refresh_models()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to delete model:\n{e}")

    def update_selected(self):
        name = self.selected_model_name()
        if not name:
            QMessageBox.information(self, "Update", "Select a model first.")
            return
        self._pull_with_progress(name)

    def download_model(self):
        name = (self.download_input.text() or "").strip()
        if not name:
            QMessageBox.information(self, "Download", "Enter a model name.")
            return
        self._pull_with_progress(name)

    # External models directory handling
    def on_models_dir_changed(self):
        path = (self.dir_input.text() or "").strip()
        if not isinstance(getattr(self, "config", None), dict):
            self.config = {}
        self.config["models_dir"] = path
        try:
            save_config(self.config)
        except Exception as e:
            self.status.showMessage(f"Failed to save config: {e}", 5000)
        self.update_models_dir_status()

    def browse_models_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select models directory", self.dir_input.text() or "")
        if path:
            self.dir_input.setText(path)
            self.on_models_dir_changed()

    def update_models_dir_status(self):
        path = (self.dir_input.text() or "").strip()
        if not path:
            # If no external selection, show dynamic default
            d = default_models_dir()
            exists = d.is_dir()
            blobs = (d / "blobs").is_dir()
            manifests = (d / "manifests").is_dir()
            status_parts = ["default path", "exists" if exists else "missing"]
            if exists:
                status_parts.append("blobs/ ok" if blobs else "no blobs/")
                status_parts.append("manifests/ ok" if manifests else "no manifests/")
            self.dir_status.setText(f"Using default: {d}\n{' • '.join(status_parts)}")
            return
        exists = os.path.isdir(path)
        blobs = os.path.isdir(os.path.join(path, "blobs"))
        manifests = os.path.isdir(os.path.join(path, "manifests"))
        status_parts = []
        status_parts.append("exists" if exists else "missing")
        if exists:
            status_parts.append("blobs/ ok" if blobs else "no blobs/")
            status_parts.append("manifests/ ok" if manifests else "no manifests/")
        status = " • ".join(status_parts)
        self.dir_status.setText(f"{path}\n{status}")

    def models_dir_path(self) -> Path:
        # Use chosen dir if set, else OLLAMA_MODELS or ~/.ollama/models
        t = (self.dir_input.text() or "").strip()
        if t:
            return Path(t).expanduser()
        return default_models_dir()

    def scan_models_dir(self):
        md = self.models_dir_path()
        if not md.exists():
            QMessageBox.warning(self, "Scan", f"Models directory not found:\n{md}")
            return
        self.maint_output.clear()
        worker = ScanWorker(md)

        def run():
            worker.start()

        def on_status(msg: str):
            self.maint_output.appendPlainText(msg)

        def on_finished(res: ScanResult):
            self._last_scan = res
            summary = [
                f"Scanned: {res.models_dir}",
                f"Manifests: {len(res.manifest_files)}",
                f"Blobs: {len(res.blob_files)}",
                f"Partials: {len(res.partial_files)}",
                f"Referenced digests: {len(res.used_digests)}",
                f"Present digests: {len(res.present_digests)}",
                f"Missing digests: {len(res.missing_digests)}",
                f"Unused blobs: {len(res.unused_blob_files)}",
            ]
            self.maint_output.appendPlainText("\n".join(summary))
            if res.missing_digests:
                self.maint_output.appendPlainText("\nMissing digests:\n" + "\n".join(sorted(res.missing_digests)))
            if res.unused_blob_files:
                self.maint_output.appendPlainText("\nUnused blobs:\n" + "\n".join(str(p) for p in res.unused_blob_files))
            if res.partial_files:
                self.maint_output.appendPlainText("\nPartial files:\n" + "\n".join(str(p) for p in res.partial_files))

        worker.status.connect(on_status)
        worker.finished.connect(on_finished)
        threading.Thread(target=run, daemon=True).start()

    def verify_hashes(self):
        res: ScanResult = getattr(self, "_last_scan", None)
        if not res or not res.blob_files:
            # Run a quick scan implicitly
            self.scan_models_dir()
            QMessageBox.information(self, "Verify", "Ran scan. Click Verify again to start hash verification.")
            return

        dlg = QProgressDialog("Verifying…", "Cancel", 0, 0, self)
        dlg.setWindowTitle("Verify Blobs")
        dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
        dlg.setAutoClose(True)
        dlg.setAutoReset(True)
        dlg.setMinimumDuration(0)

        worker = VerifyWorker(res.blob_files)

        def run():
            worker.start()

        def on_progress(text: str):
            self.maint_output.appendPlainText(text)

        def on_finished(corrupt: List[Path]):
            if corrupt:
                self.maint_output.appendPlainText("\nCorrupt blobs:\n" + "\n".join(str(p) for p in corrupt))
                self._corrupt_paths = corrupt
            else:
                self.maint_output.appendPlainText("\nAll blobs verified OK.")
            dlg.reset()

        worker.progress.connect(on_progress)
        worker.finished.connect(on_finished)
        threading.Thread(target=run, daemon=True).start()
        dlg.exec()

    def delete_unused_blobs(self):
        res: ScanResult = getattr(self, "_last_scan", None)
        if not res or not res.unused_blob_files:
            QMessageBox.information(self, "Cleanup", "No unused blobs detected. Run Scan first.")
            return
        if QMessageBox.question(self, "Delete unused", f"Delete {len(res.unused_blob_files)} unused blob file(s)?") != QMessageBox.StandardButton.Yes:
            return
        deleted = 0
        for p in res.unused_blob_files:
            try:
                p.unlink(missing_ok=True)
                deleted += 1
            except Exception as e:
                self.maint_output.appendPlainText(f"Failed to delete {p}: {e}")
        self.maint_output.appendPlainText(f"Deleted {deleted} unused blob file(s).")
        # Re-scan to update state
        self.scan_models_dir()

    def delete_partials(self):
        res: ScanResult = getattr(self, "_last_scan", None)
        if not res or not res.partial_files:
            QMessageBox.information(self, "Cleanup", "No .partial files detected. Run Scan first.")
            return
        if QMessageBox.question(self, "Delete .partial", f"Delete {len(res.partial_files)} .partial file(s)?") != QMessageBox.StandardButton.Yes:
            return
        deleted = 0
        for p in res.partial_files:
            try:
                p.unlink(missing_ok=True)
                deleted += 1
            except Exception as e:
                self.maint_output.appendPlainText(f"Failed to delete {p}: {e}")
        self.maint_output.appendPlainText(f"Deleted {deleted} .partial file(s).")
        self.scan_models_dir()

    def repair_installed(self):
        # Re-pull all installed models to restore any missing blobs
        try:
            models = self.client.list_models()
        except Exception as e:
            QMessageBox.critical(self, "Repair", f"Failed to list models: {e}")
            return
        names = [m.get("name") for m in models if m.get("name")]
        if not names:
            QMessageBox.information(self, "Repair", "No installed models found to repair.")
            return
        if QMessageBox.question(self, "Repair", f"Re-pull {len(names)} installed model(s)?") != QMessageBox.StandardButton.Yes:
            return

        # Pull sequentially with progress dialog
        dlg = QProgressDialog("Repairing…", "Cancel", 0, len(names), self)
        dlg.setWindowTitle("Repair Installed Models")
        dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
        dlg.setAutoClose(True)
        dlg.setAutoReset(True)
        dlg.setMinimumDuration(0)

        def pull_one(model_name: str):
            worker = PullWorker(self.client, model_name)
            # inline run, but update output
            for ev in self.client.pull_model_stream(model_name, stream=True):
                if "error" in ev:
                    self.maint_output.appendPlainText(f"{model_name}: Error: {ev.get('error')}")
                    return
                status = ev.get("status") or "..."
                completed = ev.get("completed")
                total = ev.get("total")
                if completed and total:
                    self.maint_output.appendPlainText(f"{model_name}: {status} {completed}/{total}")
                else:
                    self.maint_output.appendPlainText(f"{model_name}: {status}")

        def run():
            for i, n in enumerate(names, start=1):
                if dlg.wasCanceled():
                    break
                pull_one(n)
                dlg.setValue(i)

        t = threading.Thread(target=run, daemon=True)
        t.start()
        dlg.exec()

    def _pull_with_progress(self, name: str):
        dlg = QProgressDialog("Starting…", "Cancel", 0, 0, self)
        dlg.setWindowTitle(f"Pulling {name}")
        dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
        dlg.setAutoClose(True)
        dlg.setAutoReset(True)
        dlg.setMinimumDuration(0)

        worker = PullWorker(self.client, name)

        def run():
            worker.start()

        def on_progress(text: str):
            dlg.setLabelText(text)

        def on_finished(ok: bool, msg: str):
            dlg.reset()
            if ok:
                self.status.showMessage(f"Pull finished: {name}", 5000)
                self.refresh_models()
            else:
                QMessageBox.critical(self, "Pull failed", msg or "Unknown error")

        worker.progress.connect(on_progress)
        worker.finished.connect(on_finished)
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        dlg.exec()

    # ========== Discover tab logic ==========
    def discover_search(self):
        self.discover_list.clear()
        try:
            installed = {m.get("name") for m in self.client.list_models() if m.get("name")}
        except Exception:
            installed = set()
        provider = self.site_combo.currentData() if hasattr(self, 'site_combo') else 'hf_gguf'
        if provider == "hf_gguf":
            worker = HuggingFaceDiscoverWorker(installed_names=installed, limit=50)
        elif provider == "hf_thebloke":
            worker = HuggingFaceTheBlokeWorker(installed_names=installed, limit=50)
        else:
            worker = OllamaLibraryDiscoverWorker(installed_names=installed)

        def run():
            worker.start()

        def on_progress(msg: str):
            self.status.showMessage(msg, 2000)

        def on_finished(results: List[dict]):
            for r in results:
                if "suggested_name" in r:
                    text = f"{r['suggested_name']} — {r.get('desc','')}"
                else:
                    text = f"{r.get('name','<unknown>')} — {r.get('desc','')}"
                item = QListWidgetItem(text)
                item.setData(Qt.ItemDataRole.UserRole, r)
                self.discover_list.addItem(item)
            if not results:
                QMessageBox.information(self, "Discover", "No models found from the selected source (or all are already installed).")
            self.status.showMessage(f"Found {len(results)} models not installed", 5000)

        worker.progress.connect(on_progress)
        worker.finished.connect(on_finished)
        threading.Thread(target=run, daemon=True).start()

    def discover_download_selected(self):
        item = self.discover_list.currentItem()
        if not item:
            QMessageBox.information(self, "Download", "Select a model in the Discover list.")
            return
        data = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(data, dict):
            return
        provider = self.site_combo.currentData() if hasattr(self, 'site_combo') else 'hf_gguf'
        if provider == "hf_gguf":
            suggested_name = data.get("suggested_name")
            url = data.get("url")
            gguf_path = data.get("gguf_path")
            repo_id = data.get("repo_id")
            if not (suggested_name and url and gguf_path and repo_id):
                QMessageBox.critical(self, "Download", "Invalid selection data.")
                return
            cache_dir = Path.home() / ".cache" / "ollama-manager-gui" / "downloads"
            cache_dir.mkdir(parents=True, exist_ok=True)
            target_file = cache_dir / f"{repo_id.replace('/', '_')}__{Path(gguf_path).name}"
            dlg = QProgressDialog("Downloading…", "Cancel", 0, 0, self)
            dlg.setWindowTitle("Download and Import")
            dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
            dlg.setAutoClose(True)
            dlg.setAutoReset(True)
            dlg.setMinimumDuration(0)
            worker = DownloadAndCreateWorker(url=url, dest=target_file, client=self.client, model_name=suggested_name)
            def run():
                worker.start()
            def on_progress(msg: str):
                self.status.showMessage(msg, 2000)
                self.maint_output.appendPlainText(msg)
            def on_finished(ok: bool, message: str):
                dlg.reset()
                if ok:
                    self.status.showMessage(f"Imported model: {suggested_name}", 5000)
                    self.refresh_models()
                else:
                    QMessageBox.critical(self, "Import failed", message or "Unknown error")
            worker.progress.connect(on_progress)
            worker.finished.connect(on_finished)
            threading.Thread(target=run, daemon=True).start()
            dlg.exec()
        else:
            name = data.get("name")
            if not name:
                QMessageBox.critical(self, "Download", "Invalid selection data.")
                return
            self._pull_with_progress(name)


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
