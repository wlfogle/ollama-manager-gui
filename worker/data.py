"""Stdlib-only local repository and dataset services. Never executes repository code."""

import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import stat
import subprocess
import time

MAX_FILE = 512 * 1024
CHUNK_LINES = 80
EXCLUDED_DIRS = {
    ".git",
    "node_modules",
    "vendor",
    "target",
    "dist",
    "build",
    "__pycache__",
    ".venv",
    "venv",
    ".next",
    "coverage",
    ".idea",
    ".vscode",
}
EXTENSIONS = {
    ".py",
    ".rs",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".json",
    ".toml",
    ".yaml",
    ".yml",
    ".md",
    ".txt",
    ".sh",
    ".fish",
    ".css",
    ".html",
    ".go",
    ".c",
    ".h",
    ".cpp",
    ".java",
    ".sql",
    ".svelte",
    ".vue",
    ".ini",
    ".cfg",
}
SECRET = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----|"
    r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
    r"hf_[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16}|sk-[A-Za-z0-9_-]{20,})\b|"
    r"""^\s*["']?(?:api[_-]?key|access[_-]?token|secret|password|"""
    r"""private[_-]?key|auth[_-]?token)["']?\s*[:=]\s*["']?[^\s"'${}<>]{8,}""",
    re.MULTILINE | re.IGNORECASE,
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def contained(root, relative):
    """For app-managed artifacts only; reject traversal and symlink escapes."""
    root = Path(root).resolve()
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Expected a relative artifact path")
    result = root / relative
    if not result.resolve().is_relative_to(root):
        raise ValueError("Artifact path escapes storage")
    for ancestor in [result, *result.parents]:
        if ancestor == root:
            break
        if ancestor.is_symlink():
            raise ValueError("Symlink artifacts are not supported")
    return result


def git(repo, *args, input_data=None):
    result = subprocess.run(
        ["git", "-c", "core.fsmonitor=false", "-C", str(repo), *args],
        input=input_data,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        check=False,
    )
    if result.returncode:
        raise ValueError("Git operation failed; select an accessible Git repository")
    return result.stdout


def connect(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(root / "repositories.sqlite3", timeout=30)
    db.row_factory = sqlite3.Row
    db.executescript("""
        CREATE TABLE IF NOT EXISTS repos (
            id TEXT PRIMARY KEY, path TEXT NOT NULL, commit_id TEXT, indexed REAL
        );
        CREATE TABLE IF NOT EXISTS files (
            repo TEXT, path TEXT, hash TEXT, PRIMARY KEY(repo,path)
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(
            repo UNINDEXED, path UNINDEXED, start UNINDEXED, end UNINDEXED,
            content, tokenize='unicode61'
        );
    """)
    return db


def exclusion(relative):
    p = Path(relative)
    name = p.name.lower()
    if set(p.parts) & EXCLUDED_DIRS:
        return "generated, vendor, or internal directory"
    if (
        name.startswith(".env")
        or any(
            x in name
            for x in (
                "credential",
                "secret",
                "id_rsa",
                "id_ed25519",
                "token",
                ".pem",
                ".key",
                ".p12",
                ".pfx",
                "lock",
            )
        )
        or name in {".netrc", ".npmrc", ".pypirc"}
    ):
        return "sensitive or lock file"
    if p.suffix.lower() not in EXTENSIONS and name not in {"dockerfile", "makefile"}:
        return "unsupported text format"
    return None


def read_source(repo, relative):
    p = repo / relative
    if not p.resolve().is_relative_to(repo):
        raise ValueError("path escapes repository")
    for part in [p, *p.parents]:
        if part == repo:
            break
        if part.is_symlink():
            raise ValueError("symlink")
    if not p.is_file() or p.stat().st_size > MAX_FILE:
        raise ValueError("missing, special, or oversized file")
    # O_NOFOLLOW prevents following a final-component link switched during scanning.
    # Walk from an open repository FD so swapped parent symlinks cannot escape it.
    current = os.open(repo, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for component in Path(relative).parts[:-1]:
            child = os.open(
                component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=current
            )
            os.close(current)
            current = child
        fd = os.open(
            Path(relative).name,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=current,
        )
    finally:
        os.close(current)
    with os.fdopen(fd, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("Not a regular source file")
        data = stream.read(MAX_FILE + 1)
    if len(data) > MAX_FILE or b"\0" in data:
        raise ValueError("binary or oversized file")
    text = data.decode("utf-8")
    if SECRET.search(text):
        raise ValueError("suspected secret (content withheld)")
    return text, digest(data)


def ingest(root, path):
    repo = Path(path).expanduser().resolve(strict=True)
    top = Path(os.fsdecode(git(repo, "rev-parse", "--show-toplevel")).strip()).resolve()
    if top != repo:
        raise ValueError("Select the repository root, not a subdirectory")
    commit = git(repo, "rev-parse", "HEAD").decode().strip()
    repo_id = digest(os.fsencode(repo))[:16]
    tracked = git(repo, "ls-files", "-z", "--cached")
    ignored_proc = subprocess.run(
        [
            "git",
            "-c",
            "core.fsmonitor=false",
            "-C",
            str(repo),
            "check-ignore",
            "--no-index",
            "-z",
            "--stdin",
        ],
        input=tracked,
        capture_output=True,
        timeout=60,
        check=False,
    )
    if ignored_proc.returncode not in (0, 1):
        raise ValueError("Could not determine ignored paths; ingestion stopped")
    ignored = set(ignored_proc.stdout.split(b"\0"))
    accepted, excluded = {}, []
    for raw in sorted(set(tracked.split(b"\0")) - {b""}):
        relative = os.fsdecode(raw)
        reason = "ignored by Git rules" if raw in ignored else exclusion(relative)
        if reason:
            excluded.append({"path": relative, "reason": reason})
            continue
        try:
            text, sha = read_source(repo, relative)
            accepted[relative] = (text, sha)
        except (OSError, ValueError, UnicodeError):
            excluded.append(
                {"path": relative, "reason": "unsafe, secret, unreadable, or non-text"}
            )
    with connect(root) as db:
        existing = {
            r["path"]: r["hash"]
            for r in db.execute("SELECT path,hash FROM files WHERE repo=?", (repo_id,))
        }
        changed = 0
        for relative in existing.keys() - accepted.keys():
            db.execute(
                "DELETE FROM chunks WHERE repo=? AND path=?", (repo_id, relative)
            )
            db.execute("DELETE FROM files WHERE repo=? AND path=?", (repo_id, relative))
        for relative, (text, sha) in accepted.items():
            if existing.get(relative) == sha:
                continue
            changed += 1
            db.execute(
                "DELETE FROM chunks WHERE repo=? AND path=?", (repo_id, relative)
            )
            lines = text.splitlines()
            for start in range(0, len(lines), CHUNK_LINES):
                end = min(start + CHUNK_LINES, len(lines))
                db.execute(
                    "INSERT INTO chunks VALUES(?,?,?,?,?)",
                    (repo_id, relative, start + 1, end, "\n".join(lines[start:end])),
                )
            db.execute(
                "INSERT OR REPLACE INTO files VALUES(?,?,?)", (repo_id, relative, sha)
            )
        db.execute(
            "INSERT OR REPLACE INTO repos VALUES(?,?,?,?)",
            (repo_id, str(repo), commit, time.time()),
        )
    return {
        "id": repo_id,
        "commit": commit,
        "files": len(accepted),
        "changed": changed,
        "excluded": excluded,
        "warning": "Tracked working-tree contents indexed; secret detection is heuristic. Review before export.",
    }


def repositories(root):
    with connect(root) as db:
        return [
            dict(r)
            for r in db.execute(
                "SELECT repos.*, (SELECT count(*) FROM files WHERE repo=repos.id) AS files FROM repos"
            )
        ]


def search(root, query, repo_id=None):
    words = re.findall(r"\w+", query, flags=re.UNICODE)[:20]
    if not words:
        return []
    match = " OR ".join('"' + w + '"' for w in words)
    with connect(root) as db:
        return [
            dict(r)
            for r in db.execute(
                """SELECT repo,path,start,end,content FROM chunks
               WHERE chunks MATCH ? AND (? IS NULL OR repo=?)
               ORDER BY bm25(chunks) LIMIT 6""",
                (match, repo_id, repo_id),
            )
        ]


def normalize(row):
    if not isinstance(row, dict):
        raise ValueError("Each row must be an object")
    if isinstance(row.get("text"), str) and row["text"].strip():
        return {"text": row["text"]}
    if isinstance(row.get("instruction"), str) and isinstance(row.get("output"), str):
        if not row["instruction"].strip() or not row["output"].strip():
            raise ValueError("Instruction and output must be nonempty")
        extra = row.get("input", "")
        if not isinstance(extra, str):
            raise ValueError("Input must be text")
        return {
            "messages": [
                {
                    "role": "user",
                    "content": row["instruction"] + ("\n" + extra if extra else ""),
                },
                {"role": "assistant", "content": row["output"]},
            ]
        }
    messages = row.get("messages")
    if isinstance(messages, list) and messages:
        for message in messages:
            if not isinstance(message, dict) or message.get("role") not in {
                "system",
                "user",
                "assistant",
            }:
                raise ValueError("Messages need system/user/assistant roles")
            if (
                not isinstance(message.get("content"), str)
                or not message["content"].strip()
            ):
                raise ValueError("Message content must be nonempty text")
        if not any(m["role"] == "assistant" for m in messages):
            raise ValueError("Chat examples need an assistant response")
        return {
            "messages": [{"role": m["role"], "content": m["content"]} for m in messages]
        }
    raise ValueError("Expected text, instruction/output, or messages")


def parse_dataset(text):
    if len(text.encode()) > 32 * 1024 * 1024:
        raise ValueError("Dataset editor limit is 32 MiB")
    if text.lstrip().startswith("["):
        rows = json.loads(text)
        if not isinstance(rows, list):
            raise ValueError("Expected a JSON array")
    else:
        rows = []
        for i, line in enumerate(text.splitlines(), 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except ValueError as exc:
                    raise ValueError(f"Invalid JSON on line {i}") from exc
    if not rows:
        raise ValueError("Dataset is empty")
    result = []
    for i, row in enumerate(rows, 1):
        try:
            clean = normalize(row)
            contents = (
                [clean["text"]]
                if "text" in clean
                else [m["content"] for m in clean["messages"]]
            )
            if any(SECRET.search(text) for text in contents):
                raise ValueError("Suspected secret; content withheld")
            # Group is kept for split isolation but never used as model input.
            if isinstance(row.get("_group"), str):
                clean["_group"] = row["_group"]
            result.append(clean)
        except ValueError as exc:
            raise ValueError(f"Row {i}: {exc}") from exc
    return result


def dataset_save(root, name, text):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", name):
        raise ValueError(
            "Dataset name must use letters, digits, underscores, or hyphens"
        )
    rows = parse_dataset(text)
    target = contained(root, f"datasets/{name}.jsonl")
    if target.exists():
        raise ValueError(
            "Dataset already exists; save with a new name to preserve provenance"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    with target.open("x", encoding="utf-8") as f:
        f.write(data)
    return {"path": str(target), "rows": len(rows), "sha256": digest(data.encode())}


def export_repo(root, repo_id, name, approved_paths):
    if not approved_paths:
        raise ValueError("Select source files after reviewing their contents")
    with connect(root) as db:
        repo = db.execute("SELECT * FROM repos WHERE id=?", (repo_id,)).fetchone()
        if repo is None:
            raise ValueError("Repository is not indexed")
        rows, sources = [], []
        for path in sorted(set(approved_paths)):
            stored = db.execute(
                "SELECT hash FROM files WHERE repo=? AND path=?", (repo_id, path)
            ).fetchone()
            if not stored:
                raise ValueError("Selected file is not in the safe index")
            text, sha = read_source(Path(repo["path"]), path)
            if sha != stored["hash"]:
                raise ValueError(
                    "Source changed since indexing; refresh and review again"
                )
            for chunk in db.execute(
                "SELECT start,end,content FROM chunks WHERE repo=? AND path=? ORDER BY CAST(start AS INTEGER)",
                (repo_id, path),
            ):
                rows.append({"text": chunk["content"], "_group": f"{repo_id}/{path}"})
                sources.append(
                    {
                        "repo": repo_id,
                        "path": path,
                        "start": chunk["start"],
                        "end": chunk["end"],
                        "sha256": sha,
                    }
                )
        result = dataset_save(root, name, "\n".join(json.dumps(r) for r in rows))
        atomic_json(
            Path(result["path"]).with_suffix(".provenance.json"),
            {
                "repository": dict(repo),
                "sources": sources,
                "dataset_sha256": result["sha256"],
                "policy": "tracked-only; git-ignore; no symlinks; bounded UTF-8; secret heuristic",
                "objective": "causal language modeling on reviewed source (not instruction tuning)",
            },
        )
        return result


def repo_files(root, repo_id):
    with connect(root) as db:
        return [
            dict(r)
            for r in db.execute(
                "SELECT path,hash FROM files WHERE repo=? ORDER BY path", (repo_id,)
            )
        ]


def split_rows(rows, fraction=0.1, seed=42):
    if not 0 <= fraction < 0.5:
        raise ValueError("Validation fraction must be at least 0 and below 0.5")
    groups = {}
    for row in rows:
        key = row.get("_group") or digest(json.dumps(row, sort_keys=True).encode())
        groups.setdefault(key, []).append(row)
    ordered = sorted(groups, key=lambda key: digest(f"{seed}:{key}".encode()))
    count = (
        max(1, round(len(ordered) * fraction)) if fraction and len(ordered) > 1 else 0
    )
    validation_keys = set(ordered[:count])
    train, validation = [], []
    for key, group in groups.items():
        (validation if key in validation_keys else train).extend(group)
    return train, validation


def scan_models(path, verify=False):
    root = Path(path).expanduser()
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Select a real models directory")
    used, present, partial, corrupt, warnings = set(), set(), [], [], []
    # Newer stores may have additional reference formats. Inspection is safe;
    # destructive cleanup is intentionally disabled in this case.
    unknown = [p.name for p in root.iterdir() if p.name not in {"blobs", "manifests"}]
    if unknown:
        warnings.append("Additional store layouts detected: " + ", ".join(unknown))
    manifests = root / "manifests"
    if manifests.is_symlink() or not manifests.is_dir():
        warnings.append("Missing or symlinked manifests")
    else:
        for base, dirs, files in os.walk(manifests, followlinks=False):
            for d in list(dirs):
                if (Path(base) / d).is_symlink():
                    warnings.append("Symlinked manifest directory")
                    dirs.remove(d)
            for name in files:
                p = Path(base) / name
                try:
                    if p.is_symlink() or p.stat().st_size > MAX_FILE:
                        raise ValueError()
                    value = json.loads(p.read_text())
                    refs = [value["config"], *value["layers"]]
                    for ref in refs:
                        sha = ref["digest"]
                        if not re.fullmatch(r"sha256:[0-9a-f]{64}", sha):
                            raise ValueError()
                        used.add(sha[7:])
                except (OSError, ValueError, KeyError, TypeError):
                    warnings.append(
                        "Malformed or unsafe manifest: " + str(p.relative_to(root))
                    )
    blobs = root / "blobs"
    if blobs.is_symlink() or not blobs.is_dir():
        warnings.append("Missing or symlinked blobs")
    else:
        for p in blobs.iterdir():
            if p.is_symlink() or not p.is_file():
                warnings.append("Unsafe blob entry: " + p.name)
            elif ".partial" in p.name:
                partial.append(p.name)
            elif re.fullmatch(r"sha256-[0-9a-f]{64}", p.name):
                sha = p.name[7:]
                present.add(sha)
                if verify:
                    h = hashlib.sha256()
                    with p.open("rb") as f:
                        for chunk in iter(lambda: f.read(1024 * 1024), b""):
                            h.update(chunk)
                    if h.hexdigest() != sha:
                        corrupt.append(p.name)
            else:
                warnings.append("Unrecognized blob entry: " + p.name)
    return {
        "blobs": len(present),
        "referenced": len(used),
        "missing": sorted(used - present),
        "unreferenced": sorted(present - used),
        "partial": partial,
        "corrupt": corrupt,
        "warnings": warnings,
        "cleanup_enabled": False,
        "note": "Inspection only. Cleanup disabled to protect active downloads and unfamiliar store metadata.",
    }
