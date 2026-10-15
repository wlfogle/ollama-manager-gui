"""Line-delimited JSON IPC. No dependency installation or implicit training downloads."""

import argparse
import ctypes
import fcntl
import json
import os
from pathlib import Path
import sys

from data import (
    connect,
    contained,
    dataset_save,
    export_repo,
    ingest,
    parse_dataset,
    repo_files,
    repositories,
    scan_models,
    search,
)
from services import download, export_adapter, hf_info, hf_search
from training import preflight, run_training


def dispatch(root, request):
    op, p = request["op"], request.get("payload") or {}
    if op == "ingest":
        return ingest(root, p["path"])
    if op == "repositories":
        return repositories(root)
    if op == "repo_files":
        return repo_files(root, p["repo_id"])
    if op == "repo_preview":
        with connect(root) as db:
            rows = db.execute(
                "SELECT content FROM chunks WHERE repo=? AND path=? ORDER BY CAST(start AS INTEGER)",
                (p["repo_id"], p["path"]),
            )
            return "\n".join(row["content"] for row in rows)
    if op == "search":
        return search(root, p["query"], p.get("repo_id") or None)
    if op == "repo_export":
        return export_repo(root, p["repo_id"], p["name"], p["approved_paths"])
    if op == "dataset_validate":
        rows = parse_dataset(p["text"])
        return {"rows": len(rows), "preview": rows[:3]}
    if op == "dataset_save":
        return dataset_save(root, p["name"], p["text"])
    if op == "dataset_list":
        folder = contained(root, "datasets")
        return [f"datasets/{f.name}" for f in sorted(folder.glob("*.jsonl"))]
    if op == "dataset_read":
        path = contained(root, p["path"])
        if path.parent != root / "datasets" or path.suffix != ".jsonl":
            raise ValueError("Select an app dataset")
        if path.stat().st_size > 32 * 1024 * 1024:
            raise ValueError("Dataset too large for the editor")
        return path.read_text(encoding="utf-8")
    if op == "scan":
        return scan_models(p["path"], p.get("verify", False))
    if op == "preflight":
        return preflight(root)
    if op == "hf_search":
        return hf_search(p["query"])
    if op == "hf_info":
        return hf_info(p["repo"])
    if op == "adapter_export":
        return export_adapter(root, p["job_id"], p["destination"])
    raise ValueError("Unsupported worker operation")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--job")
    parser.add_argument("--supervisor-pid", type=int)
    args = parser.parse_args()
    if args.supervisor_pid and sys.platform == "linux":
        # Linux PR_SET_PDEATHSIG: a crashed/closed desktop cannot leave an
        # orphan GPU worker running. Check the race after installing the signal.
        if ctypes.CDLL(None).prctl(1, 9, 0, 0, 0) != 0:
            raise RuntimeError("Cannot establish worker supervision")
        if os.getppid() != args.supervisor_pid:
            return 1
    root = Path(args.root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    try:
        if args.job:
            directory = contained(root, f"jobs/{args.job}")
            request = json.loads((directory / "request.json").read_text())

            def emit(value):
                print(json.dumps(value, allow_nan=False), flush=True)

            if request["op"] == "train":
                with (root / "training.lock").open("a") as lock:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError as exc:
                        raise ValueError(
                            "Another training worker is active in this workspace"
                        ) from exc
                    run_training(root, directory, request["payload"], emit)
            elif request["op"] == "download":
                download(root, request["payload"], directory, emit)
            else:
                raise ValueError("Unsupported job type")
        else:
            request = json.loads(sys.stdin.readline(40 * 1024 * 1024))
            print(
                json.dumps(
                    {"ok": True, "result": dispatch(root, request)}, allow_nan=False
                )
            )
    except Exception as exc:
        # Do not expose network exception URLs, tokens, dataset content, or tracebacks.
        safe = (
            str(exc)
            if isinstance(exc, (ValueError, FileNotFoundError, ModuleNotFoundError))
            else type(exc).__name__
        )
        print(json.dumps({"ok": False, "stage": "failed", "error": safe}), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
