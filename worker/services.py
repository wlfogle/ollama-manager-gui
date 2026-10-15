"""Explicit network downloads and immutable artifact exports."""

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import urllib.parse
import urllib.request

from data import contained


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urllib.parse.urlsplit(newurl).scheme != "https":
            raise ValueError("Refusing a non-HTTPS download redirect")
        result = super().redirect_request(req, fp, code, msg, headers, newurl)
        if result and urllib.parse.urlsplit(newurl).netloc != "huggingface.co":
            result.remove_header("Authorization")
        return result


OPENER = urllib.request.build_opener(SafeRedirect())


def hf_request(path):
    if not path.startswith("/") or path.startswith("//"):
        raise ValueError("Invalid Hugging Face API path")
    headers = {"User-Agent": "ollama-workshop/0.2"}
    token = next(
        (
            os.environ.get(k)
            for k in ("HF_TOKEN", "HF_API_TOKEN", "HUGGINGFACE_API_TOKEN")
            if os.environ.get(k)
        ),
        None,
    )
    if token:
        headers["Authorization"] = "Bearer " + token
    return urllib.request.Request("https://huggingface.co" + path, headers=headers)


def hf_json(path):
    with OPENER.open(hf_request(path), timeout=30) as response:
        return json.load(response)


def repository_id(value):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", value):
        raise ValueError("Expected owner/model repository ID")
    return value


def hf_search(query):
    rows = hf_json(
        "/api/models?"
        + urllib.parse.urlencode(
            {
                "search": query,
                "limit": 30,
                "sort": "downloads",
                "direction": -1,
            }
        )
    )
    return [
        {
            "id": row["id"],
            "downloads": row.get("downloads", 0),
            "pipeline_tag": row.get("pipeline_tag"),
        }
        for row in rows
    ]


def hf_info(repo, revision=None):
    repository_id(repo)
    if revision is not None and not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("Expected an inspected immutable commit revision")
    endpoint = f"/api/models/{repo}" + (f"/revision/{revision}" if revision else "")
    info = hf_json(endpoint + "?blobs=true")
    return {
        "id": repo,
        "sha": info["sha"],
        "files": [
            {
                "name": f["rfilename"],
                "size": f.get("size"),
                "lfs": f.get("lfs"),
                "blob_id": f.get("blobId"),
            }
            for f in info.get("siblings", [])
        ],
    }


def download(root, config, job_dir, emit):
    repo = repository_id(config["repo"])
    info = hf_info(repo, config.get("revision"))
    if config.get("kind") == "snapshot":
        allowed = {".json", ".safetensors", ".model", ".txt", ".tiktoken", ".jinja"}
        files = [
            f
            for f in info["files"]
            if Path(f["name"]).suffix in allowed and "/" not in f["name"]
        ]
        if not any(f["name"].endswith(".safetensors") for f in files):
            raise ValueError("Repository does not contain safetensors training weights")
    else:
        files = [
            f
            for f in info["files"]
            if f["name"] == config.get("file") and f["name"].lower().endswith(".gguf")
        ]
    if not files:
        raise ValueError("No eligible model files")
    base = contained(root, f"downloads/{repo}/{info['sha']}")
    base.mkdir(parents=True, exist_ok=True)
    known_size = sum(f.get("size") or 0 for f in files)
    if known_size and shutil.disk_usage(base).free < known_size + 1024**3:
        raise ValueError("Insufficient disk space for selected download")
    for f in files:
        if (job_dir / "cancel").exists():
            emit({"stage": "cancelled"})
            return
        destination = contained(base, f["name"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            expected = (f.get("lfs") or {}).get("sha256")
            blob = f.get("blob_id")
            if not expected and not blob:
                raise ValueError(
                    "Existing file has no published checksum; inspect it before retrying"
                )
            h = hashlib.sha256() if expected else hashlib.sha1()
            size = destination.stat().st_size
            if not expected:
                h.update(f"blob {size}\0".encode())
            with destination.open("rb") as saved:
                for chunk in iter(lambda: saved.read(1024 * 1024), b""):
                    if (job_dir / "cancel").exists():
                        emit({"stage": "cancelled"})
                        return
                    h.update(chunk)
            if h.hexdigest() != (expected or blob):
                raise ValueError(
                    "Existing file checksum differs from pinned snapshot; refusing to overwrite"
                )
            emit({"stage": "verified_existing", "file": f["name"], "bytes": size})
            continue
        temporary = contained(base, f["name"] + ".partial")
        sha = hashlib.sha256()
        # Stream fresh downloads; never append a Range response without validating it.
        with (
            OPENER.open(
                hf_request(
                    f"/{repo}/resolve/{info['sha']}/{urllib.parse.quote(f['name'], safe='/')}"
                ),
                timeout=60,
            ) as response,
            temporary.open("wb") as output,
        ):
            done = 0
            last_progress = 0
            while True:
                if (job_dir / "cancel").exists():
                    emit({"stage": "cancelled"})
                    return
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                sha.update(chunk)
                done += len(chunk)
                if done - last_progress >= 8 * 1024 * 1024:
                    emit(
                        {
                            "stage": "download",
                            "file": f["name"],
                            "bytes": done,
                            "total": f.get("size"),
                        }
                    )
                    last_progress = done
        if f.get("size") is not None and done != f["size"]:
            raise ValueError("Download size mismatch; partial file retained")
        expected = (f.get("lfs") or {}).get("sha256")
        if expected and sha.hexdigest() != expected:
            raise ValueError("Download SHA256 mismatch; partial file retained")
        temporary.replace(destination)
    emit({"stage": "complete", "path": str(base), "revision": info["sha"]})


def export_adapter(root, job_id, destination):
    if not re.fullmatch(r"[a-f0-9-]{36}", job_id):
        raise ValueError("Invalid job ID")
    source = contained(root, f"jobs/{job_id}/adapter")
    if not (source / "adapter_config.json").is_file():
        raise ValueError("Completed PEFT adapter not found")
    destination = Path(destination).expanduser()
    if not destination.is_absolute() or destination.exists():
        raise ValueError("Choose a new absolute destination directory")
    if any(p.is_symlink() for p in source.rglob("*")):
        raise ValueError("Refusing symlinked adapter contents")
    shutil.copytree(source, destination)
    return {"path": str(destination)}
