import sys
import os
import re
import json
import hashlib
import threading
import time
from pathlib import Path
import requests
from typing import Optional, Iterable, Dict, List, Set
from PyQt6.QtWidgets import QDialog, QFormLayout, QSpinBox, QDialogButtonBox, QCheckBox
from network_client import hf_headers, get_json_with_retry
from download_manager import DownloadManager, DownloadJob, DownloadStatus

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

# ---- Utilities for better descriptions ----
def _extract_size_from_repo(repo_id: str) -> str:
    m = re.search(r"(\d+(?:x\d+)?B)", repo_id, re.IGNORECASE)
    return (m.group(1) + " params") if m else ""

def _extract_quant_from_filename(fname: str) -> str:
    # Common GGUF quant patterns like Q4_K_M, Q5_0, Q8_0, Q6_K
    m = re.search(r"(Q\d[_A-Z0-9]+)", fname.upper())
    return m.group(1) if m else ""

def _extract_ctx(fname_or_id: str) -> str:
    s = fname_or_id.lower()
    # Try patterns like 8k, 32k, 128k, or ctx-8192
    m = re.search(r"(\d+)[ ]*k", s)
    if m:
        try:
            v = int(m.group(1)) * 1024
            return f"ctx={v}"
        except Exception:
            pass
    m2 = re.search(r"ctx[-_ ]?(\d+)", s)
    if m2:
        return f"ctx={m2.group(1)}"
    return ""

def _fmt_int(n: Optional[int]) -> str:
    try:
        return f"{int(n):,}"
    except Exception:
        return ""

def _instruct_flag(repo_id: str, fname: str) -> str:
    s = (repo_id + " " + fname).lower()
    return "instruct" if "instruct" in s else "base"

def _compose_hf_desc(repo: dict, info: dict, gguf_path: str) -> str:
    repo_id = repo.get("modelId") or repo.get("id") or repo.get("_id") or ""
    card = info.get("cardData") or {}
    task = info.get("pipeline_tag") or card.get("pipeline_tag") or "text-generation"
    license_ = info.get("license") or card.get("license") or "license:unknown"
    downloads = repo.get("downloads") or repo.get("downloadsAllTime")
    size = _extract_size_from_repo(repo_id)
    quant = _extract_quant_from_filename(gguf_path)
    ctx = _extract_ctx(gguf_path)
    ib = _instruct_flag(repo_id, gguf_path)
    parts = []
    if size: parts.append(size)
    if quant: parts.append(quant)
    if ib: parts.append(ib)
    if ctx: parts.append(ctx)
    parts.append(f"task={task}")
    parts.append(license_)
    if downloads: parts.append(f"dl={_fmt_int(downloads)}")
    return ", ".join(parts)

def _augment_ollama_desc(txt: str) -> str:
    size = None
    m = re.search(r"(\d+(?:x\d+)?)B", txt, re.IGNORECASE)
    if m: size = m.group(1) + " params"
    quant = None
    m2 = re.search(r"(Q\d[_A-Z0-9]+)", txt.upper())
    if m2: quant = m2.group(1)
    extras = ", ".join([p for p in [size, quant] if p])
    base = re.sub(r"\s+", " ", txt).strip()[:160]
    return f"{base} ({extras})" if extras else base

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
            repos = get_json_with_retry("https://huggingface.co/api/models", params=params, headers=hf_headers(), retries=3, backoff=1.0)
            if not isinstance(repos, list):
                self.progress.emit("Unexpected response from Hugging Face")
                self.finished.emit(results)
                return
            for repo in repos:
                repo_id = repo.get("modelId") or repo.get("id") or repo.get("_id")
                if not repo_id:
                    continue
                info = get_json_with_retry(f"https://huggingface.co/api/models/{repo_id}", headers=hf_headers(), retries=3, backoff=1.0)
                if not isinstance(info, dict):
                    continue
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
                    desc = _compose_hf_desc(repo, info, gguf_path)
                    size = s.get("size") or (s.get("lfs") or {}).get("size")
                    sha256 = s.get("sha256") or (s.get("lfs") or {}).get("sha256")
                    tags = info.get("tags") or (info.get("cardData") or {}).get("tags") or []
                    license_ = info.get("license") or (info.get("cardData") or {}).get("license")
                    results.append({
                        "provider": "hf_gguf",
                        "repo_id": repo_id,
                        "gguf_path": gguf_path,
                        "url": url,
                        "suggested_name": suggested_name,
                        "desc": desc,
                        "size": size,
                        "sha256": sha256,
                        "tags": tags,
                        "license": license_,
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
            repos = get_json_with_retry("https://huggingface.co/api/models", params=params, headers=hf_headers(), retries=3, backoff=1.0)
            if not isinstance(repos, list):
                self.finished.emit(results)
                return
            for repo in repos:
                repo_id = repo.get("modelId") or repo.get("id") or repo.get("_id")
                if not repo_id:
                    continue
                info = get_json_with_retry(f"https://huggingface.co/api/models/{repo_id}", headers=hf_headers(), retries=3, backoff=1.0)
                if not isinstance(info, dict):
                    continue
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
                    desc = _compose_hf_desc(repo, info, gguf_path)
                    size = s.get("size") or (s.get("lfs") or {}).get("size")
                    sha256 = s.get("sha256") or (s.get("lfs") or {}).get("sha256")
                    tags = info.get("tags") or (info.get("cardData") or {}).get("tags") or []
                    license_ = info.get("license") or (info.get("cardData") or {}).get("license")
                    results.append({
                        "provider": "hf_thebloke",
                        "repo_id": repo_id,
                        "gguf_path": gguf_path,
                        "url": url,
                        "suggested_name": suggested_name,
                        "desc": desc,
                        "size": size,
                        "sha256": sha256,
                        "tags": tags,
                        "license": license_,
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
                desc = _augment_ollama_desc(self.fetch_desc(name))
                results.append({"provider": "ollama", "name": name, "desc": desc})
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
        settings_btn = QPushButton("Settings…")
        settings_btn.clicked.connect(self.open_settings)
        row1.addWidget(load_btn)
        row1.addWidget(settings_btn)
        discover_layout.addLayout(row1)
        # Filter row
        filter_row = QHBoxLayout()
        self.filter_instruct = QCheckBox("Instruct only")
        self.filter_min_params = QLineEdit()
        self.filter_min_params.setPlaceholderText(">= params (e.g., 7B)")
        self.filter_quant = QLineEdit()
        self.filter_quant.setPlaceholderText("quant contains (e.g., Q4_K)")
        self.filter_task = QLineEdit()
        self.filter_task.setPlaceholderText("task contains (e.g., text-generation)")
        self.filter_apply_btn = QPushButton("Apply Filters")
        self.filter_apply_btn.clicked.connect(self.apply_discover_filters)
        filter_row.addWidget(self.filter_instruct)
        filter_row.addWidget(self.filter_min_params)
        filter_row.addWidget(self.filter_quant)
        filter_row.addWidget(self.filter_task)
        filter_row.addWidget(self.filter_apply_btn)
        discover_layout.addLayout(filter_row)

        self.discover_list = QListWidget()
        self.discover_list.currentItemChanged.connect(self.update_discover_details)
        discover_layout.addWidget(self.discover_list)

        # Details pane for the selected item + actions
        self.discover_details = QPlainTextEdit()
        self.discover_details.setReadOnly(True)
        self.discover_details.setPlaceholderText("Details for selected item…")
        discover_layout.addWidget(self.discover_details)
        # Actions row below details
        actions_row = QHBoxLayout()
        self.copy_btn = QPushButton("Copy Command")
        self.copy_btn.clicked.connect(self.copy_discover_command)
        self.open_btn = QPushButton("Open Source Page")
        self.open_btn.clicked.connect(self.open_discover_url)
        actions_row.addWidget(self.copy_btn)
        actions_row.addWidget(self.open_btn)
        actions_row.addStretch(1)
        discover_layout.addLayout(actions_row)
        row2 = QHBoxLayout()
        self.prev_btn = QPushButton("Prev")
        self.prev_btn.clicked.connect(self.discover_prev_page)
        self.page_label = QLabel("Page 1/1")
        self.next_btn = QPushButton("Next")
        self.next_btn.clicked.connect(self.discover_next_page)
        download_sel_btn = QPushButton("Download Selected")
        download_sel_btn.clicked.connect(self.discover_download_selected)
        row2.addWidget(self.prev_btn)
        row2.addWidget(self.page_label)
        row2.addWidget(self.next_btn)
        row2.addStretch(1)
        row2.addWidget(download_sel_btn)
        discover_layout.addLayout(row2)

        # Download Queue section
        queue_group = QGroupBox("Download Queue")
        queue_v = QVBoxLayout()
        self.queue_list = QListWidget()
        queue_buttons = QHBoxLayout()
        self.queue_pause_btn = QPushButton("Pause")
        self.queue_resume_btn = QPushButton("Resume")
        self.queue_cancel_btn = QPushButton("Cancel")
        self.queue_clear_btn = QPushButton("Clear Completed")
        self.queue_pause_btn.clicked.connect(self.queue_pause_selected)
        self.queue_resume_btn.clicked.connect(self.queue_resume_selected)
        self.queue_cancel_btn.clicked.connect(self.queue_cancel_selected)
        self.queue_clear_btn.clicked.connect(self.queue_clear_completed)
        queue_buttons.addWidget(self.queue_pause_btn)
        queue_buttons.addWidget(self.queue_resume_btn)
        queue_buttons.addWidget(self.queue_cancel_btn)
        queue_buttons.addStretch(1)
        queue_buttons.addWidget(self.queue_clear_btn)
        queue_v.addWidget(self.queue_list)
        queue_v.addLayout(queue_buttons)
        queue_group.setLayout(queue_v)
        discover_layout.addWidget(queue_group)

        discover_page.setLayout(discover_layout)
        # Auto-load when switching source so the list matches the picker
        self.site_combo.currentIndexChanged.connect(self.on_site_changed)

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

        # Pagination state
        self._discover_cache: Dict[str, List[dict]] = {}
        self._discover_page: int = 0
        self._discover_page_size: int = 50

        # Download manager
        def import_into_ollama(name: str, path: Path):
            modelfile = f"FROM {path}\n"
            self.client.create_model(name, modelfile)
        self.dl_manager = DownloadManager(
            max_concurrency=2,
            on_progress=self.on_download_progress,
            on_done=self.on_download_done,
        )
        self._queue_items: Dict[str, QListWidgetItem] = {}
        self._import_cb = import_into_ollama

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
        self.status.showMessage(f"Loading from source: {provider}", 3000)
        if provider == "hf_gguf":
            worker = HuggingFaceDiscoverWorker(installed_names=installed, limit=max(50, self._discover_page_size * 5))
        elif provider == "hf_thebloke":
            worker = HuggingFaceTheBlokeWorker(installed_names=installed, limit=max(50, self._discover_page_size * 5))
        else:
            worker = OllamaLibraryDiscoverWorker(installed_names=installed, limit=max(200, self._discover_page_size * 10))

        def run():
            worker.start()

        def on_progress(msg: str):
            self.status.showMessage(msg, 2000)

        def on_finished(results: List[dict]):
            # cache and render page 0
            self._discover_cache[provider] = results
            self._discover_page = 0
            self.render_discover_page(provider)

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
        # Use the item's provider so switching the combo later doesn't break downloads
        provider = data.get("provider") or (self.site_combo.currentData() if hasattr(self, 'site_combo') else 'hf_gguf')
        if provider in ("hf_gguf", "hf_thebloke"):
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
            # create job
            job_id = f"{provider}:{suggested_name}:{int(threading.get_ident())}:{int(time.time())}"
            size = data.get('size') or data.get('head_size')
            sha256 = data.get('sha256')
            job = DownloadJob(
                job_id=job_id,
                url=url,
                dest=target_file,
                provider=provider,
                suggested_name=suggested_name,
                repo_id=repo_id,
                gguf_path=gguf_path,
                size=size if isinstance(size, int) else None,
                sha256=sha256,
                ollama_import=self._import_cb,
            )
            self._enqueue_job(job)
        else:
            name = data.get("name")
            if not name:
                QMessageBox.critical(self, "Download", "Invalid selection data.")
                return
            # simpler: queue a pull job by wrapping a small gguf-less job that only imports via ollama pull
            # Here we directly call pull as before
            self._pull_with_progress(name)


    def _provider_tag(self, provider: str) -> str:
        return {"ollama": "OLLAMA", "hf_gguf": "HF", "hf_thebloke": "HF-TB"}.get(provider or "", "?")

    def on_site_changed(self, _index: int):
        # Auto-fetch or render cache for the selected provider
        provider = self.site_combo.currentData() if hasattr(self, 'site_combo') else 'hf_gguf'
        self._discover_page = 0
        if provider in self._discover_cache and self._discover_cache[provider]:
            self.render_discover_page(provider)
        else:
            self.discover_list.clear()
            self.status.showMessage(f"Source changed to {provider}. Loading…", 2000)
            self.discover_search()

    def _parse_params_to_number(self, text: str) -> int:
        # Convert "7B", "13B", "8x7B" into an integer number of parameters
        s = text.upper()
        m = re.search(r"(\d+)X(\d+)B", s)
        if m:
            try:
                return int(m.group(1)) * int(m.group(2)) * 1_000_000_000
            except Exception:
                pass
        m2 = re.search(r"(\d+)B", s)
        if m2:
            try:
                return int(m2.group(1)) * 1_000_000_000
            except Exception:
                pass
        return 0

    def get_filtered_results(self, provider: str) -> List[dict]:
        results = list(self._discover_cache.get(provider, []))
        if not results:
            return results
        # Filters
        instruct_only = self.filter_instruct.isChecked() if hasattr(self, 'filter_instruct') else False
        min_params_txt = self.filter_min_params.text().strip() if hasattr(self, 'filter_min_params') else ""
        quant_sub = (self.filter_quant.text() or "").strip().upper() if hasattr(self, 'filter_quant') else ""
        task_sub = (self.filter_task.text() or "").strip().lower() if hasattr(self, 'filter_task') else ""
        min_params = 0
        if min_params_txt:
            min_params = self._parse_params_to_number(min_params_txt)
        out = []
        for r in results:
            desc = (r.get('desc') or '').lower()
            # instruct
            if instruct_only:
                if r.get('provider') in ('hf_gguf','hf_thebloke'):
                    if 'instruct' not in (r.get('suggested_name') or '').lower() and 'instruct' not in desc:
                        continue
                else:
                    if 'instruct' not in (r.get('name') or '').lower() and 'instruct' not in desc:
                        continue
            # min params
            if min_params > 0:
                # attempt from suggested_name or repo_id
                ref = r.get('suggested_name') or r.get('repo_id') or r.get('name') or ''
                pnum = self._parse_params_to_number(ref)
                if pnum < min_params:
                    continue
            # quant substring
            if quant_sub:
                q = _extract_quant_from_filename(r.get('gguf_path','')) if r.get('gguf_path') else ''
                if quant_sub not in q.upper():
                    continue
            # task
            if task_sub and task_sub not in desc:
                continue
            out.append(r)
        return out

    def render_discover_page(self, provider: str):
        results = self.get_filtered_results(provider)
        page_size = self._discover_page_size
        total = len(results)
        pages = max(1, (total + page_size - 1) // page_size)
        # Clamp page index
        if self._discover_page >= pages:
            self._discover_page = pages - 1
        start = self._discover_page * page_size
        end = min(start + page_size, total)
        self.discover_list.clear()
        self.discover_details.clear()
        tag = self._provider_tag(provider)
        for r in results[start:end]:
            p = r.get("provider")
            itag = self._provider_tag(p) if p else tag
            if "suggested_name" in r:
                text = f"[{itag}] {r['suggested_name']} — {r.get('desc','')}"
            else:
                text = f"[{itag}] {r.get('name','<unknown>')} — {r.get('desc','')}"
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, r)
            self.discover_list.addItem(item)
        self.page_label.setText(f"Page {self._discover_page+1}/{pages}")
        self.prev_btn.setEnabled(self._discover_page > 0)
        self.next_btn.setEnabled(self._discover_page+1 < pages)
        # Update details for the first item on this page
        if self.discover_list.count() > 0:
            self.discover_list.setCurrentRow(0)

    def discover_prev_page(self):
        provider = self.site_combo.currentData() if hasattr(self, 'site_combo') else 'hf_gguf'
        if self._discover_page > 0:
            self._discover_page -= 1
            self.render_discover_page(provider)

    def discover_next_page(self):
        provider = self.site_combo.currentData() if hasattr(self, 'site_combo') else 'hf_gguf'
        self._discover_page += 1
        self.render_discover_page(provider)

    def apply_discover_filters(self):
        provider = self.site_combo.currentData() if hasattr(self, 'site_combo') else 'hf_gguf'
        self._discover_page = 0
        self.render_discover_page(provider)

    def open_settings(self):
        class SettingsDialog(QDialog):
            def __init__(self, parent, cfg):
                super().__init__(parent)
                self.setWindowTitle("Settings")
                self.cfg = cfg.copy()
                form = QFormLayout()
                self.token = QLineEdit()
                self.token.setEchoMode(QLineEdit.EchoMode.Password)
                self.token.setText(self.cfg.get('hf_api_token',''))
                self.page_size = QSpinBox()
                self.page_size.setRange(10, 500)
                self.page_size.setValue(int(parent._discover_page_size))
                form.addRow("Hugging Face token:", self.token)
                form.addRow("Discover page size:", self.page_size)
                buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
                buttons.accepted.connect(self.accept)
                buttons.rejected.connect(self.reject)
                layout = QVBoxLayout()
                layout.addLayout(form)
                layout.addWidget(buttons)
                self.setLayout(layout)
        d = SettingsDialog(self, self.config)
        if d.exec() == QDialog.DialogCode.Accepted:
            tok = d.token.text().strip()
            self.config['hf_api_token'] = tok
            if tok:
                os.environ['HF_API_TOKEN'] = tok
            self._discover_page_size = int(d.page_size.value())
            try:
                save_config(self.config)
            except Exception:
                pass

    def _human_size(self, n: Optional[int]) -> str:
        try:
            n = int(n)
        except Exception:
            return "?"
        units = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        x = float(n)
        while x >= 1024 and i < len(units) - 1:
            x /= 1024.0
            i += 1
        return f"{x:.1f} {units[i]}"

    def _details_text(self, data: dict) -> str:
        if not data:
            return ""
        prov = data.get("provider", "?")
        lines = [f"provider: {prov}"]
        if prov in ("hf_gguf", "hf_thebloke"):
            size = data.get('size')
            sha = data.get('sha256')
            tags = data.get('tags') or []
            license_ = data.get('license') or ''
            lines += [
                f"repo_id: {data.get('repo_id','')}",
                f"gguf: {data.get('gguf_path','')}",
                f"suggested: {data.get('suggested_name','')}",
                f"desc: {data.get('desc','')}",
                f"size: {self._human_size(size) if size else (self._human_size(data.get('head_size')) if data.get('head_size') else '?')}",
                f"checksum: {sha or 'n/a'}",
                f"tags: {', '.join(tags[:10])}{'…' if len(tags)>10 else ''}",
                f"license: {license_}",
                f"url: {data.get('url','')}",
            ]
            if data.get('readme_preview'):
                lines += ["", "README:", data['readme_preview']]
        else:
            lines += [
                f"name: {data.get('name','')}",
                f"desc: {data.get('desc','')}",
                f"pull: {data.get('name','')}",
            ]
        return "\n".join(lines)

    def update_discover_details(self, _cur, _prev=None):
        item = self.discover_list.currentItem()
        if not item:
            self.discover_details.clear()
            return
        data = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(data, dict):
            # kick off background enrich for HF providers
            prov = data.get('provider')
            if prov in ('hf_gguf','hf_thebloke'):
                # Copy to avoid mutating original directly while thread runs
                enriched = dict(data)
                def enrich():
                    try:
                        # Disk space estimate via HEAD
                        try:
                            h = requests.head(enriched.get('url',''), headers=hf_headers(), timeout=20, allow_redirects=True)
                            sz = int(h.headers.get('Content-Length','0')) if h.status_code < 400 else 0
                            if sz > 0 and not enriched.get('size'):
                                enriched['head_size'] = sz
                        except Exception:
                            pass
                        # README preview
                        repo_id = enriched.get('repo_id')
                        readme_txt = ''
                        for path in ('README.md','README.MD','Readme.md'):
                            try:
                                r = requests.get(f"https://huggingface.co/{repo_id}/raw/main/{path}", headers=hf_headers(), timeout=20)
                                if r.status_code == 200 and r.text:
                                    readme_txt = r.text.strip()
                                    break
                            except Exception:
                                continue
                        if readme_txt:
                            # take first ~1200 chars
                            enriched['readme_preview'] = (readme_txt[:1200] + ('…' if len(readme_txt) > 1200 else ''))
                    finally:
                        # Update UI on main thread
                        def apply():
                            # Replace stored data on the item for future use
                            item.setData(Qt.ItemDataRole.UserRole, enriched)
                            self.discover_details.setPlainText(self._details_text(enriched))
                        QApplication.instance().postEvent(self.discover_details, type('Dummy', (), {})())
                        # Use direct call since we're in Python thread; just set text safely
                        self.discover_details.setPlainText(self._details_text(enriched))
                threading.Thread(target=enrich, daemon=True).start()
            self.discover_details.setPlainText(self._details_text(data))
        else:
            self.discover_details.clear()

    def copy_discover_command(self):
        item = self.discover_list.currentItem()
        if not item:
            return
        data = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(data, dict):
            return
        prov = data.get("provider")
        if prov in ("hf_gguf", "hf_thebloke"):
            cmd = f"Modelfile:\nFROM {data.get('url','')}\n"
        else:
            name = data.get("name", "")
            cmd = f"ollama pull {name}" if name else ""
        if cmd:
            # put in details pane tail as a visible copy source
            self.discover_details.appendPlainText("\n---\n" + cmd)
    
    def open_discover_url(self):
        # Best-effort: open HF or Ollama page in default browser if available
        try:
            import webbrowser
        except Exception:
            return
        item = self.discover_list.currentItem()
        if not item:
            return
        data = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(data, dict):
            return
        prov = data.get("provider")
        url = None
        if prov in ("hf_gguf", "hf_thebloke"):
            rid = data.get("repo_id")
            if rid:
                url = f"https://huggingface.co/{rid}"
        else:
            name = data.get("name")
            if name:
                url = f"https://ollama.com/library/{name}"
        if url:
            try:
                webbrowser.open(url)
            except Exception:
                pass

    # ======== Download queue integration ========
    def _enqueue_job(self, job: DownloadJob):
        self.dl_manager.add_job(job)
        it = QListWidgetItem(self._queue_item_text(job))
        it.setData(Qt.ItemDataRole.UserRole, job.job_id)
        self.queue_list.addItem(it)
        self._queue_items[job.job_id] = it

    def _queue_item_text(self, job: DownloadJob) -> str:
        # build a status line
        name = job.suggested_name or job.dest.name
        st = job.status
        total = job.total or 0
        done = job.downloaded
        pct = (done * 100 // total) if total else 0
        sz = f"{self._human_size(done)}/{self._human_size(total) if total else '?'}"
        return f"[{st}] {name} — {pct}% ({sz})"

    def on_download_progress(self, job: DownloadJob):
        it = self._queue_items.get(job.job_id)
        if it:
            it.setText(self._queue_item_text(job))

    def on_download_done(self, job: DownloadJob):
        it = self._queue_items.get(job.job_id)
        if it:
            it.setText(self._queue_item_text(job))
        # on complete import, refresh models
        if job.status == DownloadStatus.DONE:
            self.status.showMessage(f"Imported model: {job.suggested_name}", 5000)
            try:
                self.refresh_models()
            except Exception:
                pass
        elif job.status in (DownloadStatus.FAILED, DownloadStatus.CANCELED):
            self.status.showMessage(f"Download {job.status}: {job.error or ''}", 7000)

    def _queue_selected_job_id(self) -> Optional[str]:
        it = self.queue_list.currentItem()
        if not it:
            return None
        job_id = it.data(Qt.ItemDataRole.UserRole)
        return job_id

    def queue_pause_selected(self):
        jid = self._queue_selected_job_id()
        if jid:
            self.dl_manager.pause_job(jid)

    def queue_resume_selected(self):
        jid = self._queue_selected_job_id()
        if jid:
            self.dl_manager.resume_job(jid)

    def queue_cancel_selected(self):
        jid = self._queue_selected_job_id()
        if jid:
            self.dl_manager.cancel_job(jid)

    def queue_clear_completed(self):
        # remove DONE/FAILED/CANCELED items from list and map
        remove_rows: List[int] = []
        for row in range(self.queue_list.count()):
            it = self.queue_list.item(row)
            jid = it.data(Qt.ItemDataRole.UserRole)
            job = self.dl_manager.find_job(jid) if jid else None
            if job and job.status in (DownloadStatus.DONE, DownloadStatus.FAILED, DownloadStatus.CANCELED):
                remove_rows.append(row)
        for row in reversed(remove_rows):
            it = self.queue_list.takeItem(row)
            try:
                jid = it.data(Qt.ItemDataRole.UserRole)
                if jid in self._queue_items:
                    del self._queue_items[jid]
            except Exception:
                pass


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
