import json
import hashlib
import io
import fcntl
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import urllib.request

from data import (
    contained,
    dataset_save,
    export_repo,
    ingest,
    parse_dataset,
    repo_files,
    scan_models,
    search,
    split_rows,
)
from services import SafeRedirect, download
from training import validate_config


class WorkspaceTests(unittest.TestCase):
    def test_second_training_worker_is_rejected_before_loading_a_model(self):
        job = self.root / "jobs" / "test-job"
        job.mkdir(parents=True)
        (job / "request.json").write_text(json.dumps({"op": "train", "payload": {}}))
        with (self.root / "training.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("main.py")),
                    "--root",
                    str(self.root),
                    "--job",
                    "test-job",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(result.returncode, 1)
        self.assertIn("Another training worker", json.loads(result.stdout)["error"])

    def test_download_verifies_size_and_hash_and_reuses_verified_files(self):
        job = self.root / "job"
        job.mkdir()
        content = b"GGUF test fixture, not real model data"
        info = {
            "sha": "b" * 40,
            "files": [
                {
                    "name": "tiny.gguf",
                    "size": len(content),
                    "lfs": {"sha256": hashlib.sha256(content).hexdigest()},
                }
            ],
        }
        cfg = {"repo": "owner/model", "kind": "gguf", "file": "tiny.gguf"}
        with (
            patch("services.hf_info", return_value=info),
            patch(
                "services.OPENER.open",
                return_value=io.BytesIO(content),
            ),
        ):
            events = []
            download(self.root, cfg, job, events.append)
        target = self.root / "downloads" / "owner/model" / info["sha"] / "tiny.gguf"
        self.assertEqual(target.read_bytes(), content)
        self.assertEqual(events[-1]["stage"], "complete")
        with (
            patch("services.hf_info", return_value=info),
            patch("services.OPENER.open") as network,
        ):
            download(self.root, cfg, job, events.append)
        network.assert_not_called()
        self.assertEqual(events[-2]["stage"], "verified_existing")

    def test_download_rejects_checksum_mismatch_and_partial_symlink(self):
        job = self.root / "job"
        job.mkdir()
        info = {
            "sha": "b" * 40,
            "files": [
                {
                    "name": "tiny.gguf",
                    "size": 3,
                    "lfs": {"sha256": "a" * 64},
                }
            ],
        }
        cfg = {"repo": "owner/model", "kind": "gguf", "file": "tiny.gguf"}
        with (
            patch("services.hf_info", return_value=info),
            patch(
                "services.OPENER.open",
                return_value=io.BytesIO(b"bad"),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                download(self.root, cfg, job, lambda e: None)
        base = self.root / "downloads" / "owner/model" / info["sha"]
        self.assertFalse((base / "tiny.gguf").exists())
        (base / "tiny.gguf.partial").unlink()
        (base / "tiny.gguf.partial").symlink_to(self.root / "escape")
        with patch("services.hf_info", return_value=info):
            with self.assertRaises(ValueError):
                download(self.root, cfg, job, lambda e: None)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "workspace"
        self.root.mkdir()
        self.repo = self.base / "repo"
        self.repo.mkdir()
        self.git("init", "-q")
        self.git("config", "user.email", "tests@example.invalid")
        self.git("config", "user.name", "Tests")

    def git(self, *args):
        return subprocess.run(
            ["git", "-C", str(self.repo), *args], check=True, capture_output=True
        ).stdout

    def commit_files(self, files):
        for name, content in files.items():
            p = self.repo / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
        self.git("add", "-f", ".")
        self.git("-c", "core.hooksPath=/dev/null", "commit", "-qm", "Fixture")

    def test_ignore_secrets_symlinks_and_untracked_are_excluded(self):
        self.commit_files(
            {
                "main.rs": "fn workshop_unique() {}\n",
                ".gitignore": "ignored.py\n",
                "ignored.py": "ignored_marker = True",
                ".env": "sensitive settings",
                "settings.py": "password = 'fixture-not-a-real-secret'\n",
                "node_modules/vendor.js": "vendor_marker",
                "image.py": "\0binary",
            }
        )
        (self.repo / "untracked.py").write_text("untracked_marker")
        (self.repo / "linked.rs").symlink_to(self.repo / "main.rs")
        self.git("add", "linked.rs")
        result = ingest(self.root, self.repo)
        self.assertEqual(result["files"], 1)
        paths = [f["path"] for f in repo_files(self.root, result["id"])]
        self.assertEqual(paths, ["main.rs"])
        self.assertEqual(search(self.root, "workshop_unique")[0]["path"], "main.rs")
        self.assertFalse(search(self.root, "ignored_marker"))
        self.assertFalse(search(self.root, '") OR *'))

    def test_refresh_removes_stale_chunks_and_updates_changes(self):
        self.commit_files({"a.py": "old_marker = True", "b.py": "remove_marker = True"})
        first = ingest(self.root, self.repo)
        (self.repo / "a.py").write_text("new_marker = True")
        (self.repo / "b.py").unlink()
        second = ingest(self.root, self.repo)
        self.assertEqual(first["id"], second["id"])
        self.assertEqual(second["changed"], 1)
        self.assertFalse(search(self.root, "old_marker remove_marker"))
        self.assertTrue(search(self.root, "new_marker"))

    def test_export_provenance_and_reject_changed_source(self):
        self.commit_files({"a.py": "print('reviewed')\n" * 100})
        result = ingest(self.root, self.repo)
        exported = export_repo(self.root, result["id"], "source", ["a.py"])
        path = Path(exported["path"])
        self.assertEqual(exported["rows"], 2)
        metadata = json.loads(path.with_suffix(".provenance.json").read_text())
        self.assertEqual(metadata["dataset_sha256"], exported["sha256"])
        self.assertEqual(metadata["repository"]["commit_id"], result["commit"])
        (self.repo / "a.py").write_text("print('changed')")
        with self.assertRaisesRegex(ValueError, "changed"):
            export_repo(self.root, result["id"], "other", ["a.py"])

    def test_grouped_split_has_no_file_leakage(self):
        rows = [{"text": str(i), "_group": f"file-{i // 4}"} for i in range(40)]
        train, valid = split_rows(rows, 0.2)
        self.assertTrue(valid)
        self.assertFalse({r["_group"] for r in train} & {r["_group"] for r in valid})
        self.assertEqual((train, valid), split_rows(rows, 0.2))
        self.assertEqual(len(train) + len(valid), len(rows))

    def test_formats_and_secret_detection(self):
        rows = parse_dataset(
            json.dumps(
                [
                    {"text": "source text"},
                    {"instruction": "Explain", "input": "code", "output": "answer"},
                    {
                        "messages": [
                            {"role": "user", "content": "Hi"},
                            {"role": "assistant", "content": "Hello"},
                        ]
                    },
                ]
            )
        )
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[1]["messages"][0]["content"], "Explain\ncode")
        for text in [
            "[]",
            '{"text":""}',
            '{"messages":[{"role":"user","content":"Hi"}]}',
            json.dumps({"text": "line\npassword = 'fixture-not-a-real-secret'"}),
        ]:
            with self.assertRaises(ValueError):
                parse_dataset(text)

    def test_storage_rejects_traversal_and_symlinks(self):
        for name in ("../escape", "/tmp/escape"):
            with self.assertRaises(ValueError):
                contained(self.root, name)
        (self.root / "link").symlink_to(self.base)
        with self.assertRaises(ValueError):
            contained(self.root, "link/escape")
        dataset_save(self.root, "immutable", '{"text":"safe training text"}')
        with self.assertRaises(ValueError):
            dataset_save(self.root, "immutable", '{"text":"replacement"}')

    def test_model_scan_fail_closed_and_verifies(self):
        store = self.base / "models"
        (store / "manifests").mkdir(parents=True)
        (store / "blobs").mkdir()
        sha = "a" * 64
        (store / "manifests" / "model").write_text(
            json.dumps(
                {
                    "config": {"digest": "sha256:" + sha},
                    "layers": [],
                }
            )
        )
        (store / "blobs" / ("sha256-" + sha)).write_bytes(b"corrupt fixture")
        (store / "metadata").mkdir()
        report = scan_models(store, True)
        self.assertEqual(len(report["corrupt"]), 1)
        self.assertFalse(report["cleanup_enabled"])
        self.assertTrue(report["warnings"])

    def test_training_requires_explicit_review_and_local_safe_weights(self):
        model = self.base / "model"
        model.mkdir()
        (model / "config.json").write_text("{}")
        (model / "model.safetensors").write_bytes(b"validation fixture only")
        cfg = dict(
            rank=8,
            sequence_length=512,
            batch_size=1,
            gradient_accumulation=8,
            max_steps=2,
            save_steps=1,
            epochs=1,
            learning_rate=0.0002,
            validation_fraction=0.1,
            method="qlora",
            model=str(model),
            reviewed=True,
        )
        self.assertEqual(validate_config(cfg), model)
        self.assertEqual(validate_config({**cfg, "max_steps": -1}), model)
        with self.assertRaises(ValueError):
            validate_config({**cfg, "max_steps": 0})
        with self.assertRaises(ValueError):
            validate_config({**cfg, "reviewed": False})
        with self.assertRaises(ValueError):
            validate_config({**cfg, "rank": float("nan")})
        with self.assertRaises(ValueError):
            validate_config({**cfg, "sequence_length": 512.5})

    def test_worker_protocol(self):
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("main.py")),
                "--root",
                str(self.root),
            ],
            input=json.dumps(
                {"op": "dataset_validate", "payload": {"text": '{"text":"example"}'}}
            ),
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(json.loads(result.stdout)["result"]["rows"], 1)

    def test_redirect_does_not_leak_token_to_cdn(self):
        req = urllib.request.Request(
            "https://huggingface.co/a",
            headers={"Authorization": "Bearer test-placeholder"},
        )
        redirect = SafeRedirect().redirect_request(
            req, None, 302, "", {}, "https://cdn.example.invalid/file"
        )
        self.assertFalse(redirect.has_header("Authorization"))
        with self.assertRaises(ValueError):
            SafeRedirect().redirect_request(
                req, None, 302, "", {}, "http://cdn.example.invalid/file"
            )

    def test_cancelled_download_does_not_transfer_data(self):
        job = self.root / "job"
        job.mkdir()
        (job / "cancel").touch()
        events = []
        info = {"sha": "a" * 40, "files": [{"name": "model.gguf", "size": 1}]}
        with (
            patch("services.hf_info", return_value=info),
            patch("services.OPENER.open") as network,
        ):
            download(
                self.root,
                {"repo": "owner/model", "kind": "gguf", "file": "model.gguf"},
                job,
                events.append,
            )
        network.assert_not_called()
        self.assertEqual(events[-1]["stage"], "cancelled")


if __name__ == "__main__":
    unittest.main()
