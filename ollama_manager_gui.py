import sys
import os
import json
import threading
import requests
from typing import Optional

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
)


DEFAULT_OLLAMA_HOST = "http://localhost:11434"

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

        maint_group.setLayout(maint_layout)

        layout = QVBoxLayout()
        layout.addLayout(top_bar)
        layout.addWidget(self.model_list)
        layout.addLayout(dl_bar)
        layout.addWidget(maint_group)

        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

        self.status = QStatusBar()
        self.setStatusBar(self.status)

        self.resize(800, 540)

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
            self.dir_status.setText("No external models directory set.")
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

        thread = threading.Thread(target=run, daemon=True)

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
        thread.start()
        dlg.exec()


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
