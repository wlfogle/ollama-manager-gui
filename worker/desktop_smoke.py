"""Launch an opt-in native UI smoke build with isolated generated fixtures."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from test_cuda import CudaTrainingTests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    args = parser.parse_args()
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        parser.error("A graphical session is required for native WebView smoke testing")
    fixture = CudaTrainingTests("test_qlora_updates_adapter_and_exports")
    fixture.setUp()
    try:
        root = fixture.root
        repository = root / "fixture-repository"
        repository.mkdir()
        (repository / "main.py").write_text(
            "def workshop_fixture():\n    return 'local source'\n"
        )
        for command in [
            ["init", "-q"],
            ["add", "."],
            [
                "-c",
                "user.name=Workshop Test",
                "-c",
                "user.email=test@example.invalid",
                "-c",
                "core.hooksPath=/dev/null",
                "commit",
                "-qm",
                "Fixture",
            ],
        ]:
            subprocess.run(
                ["git", "-C", str(repository), *command],
                check=True,
                capture_output=True,
            )
        data = root / "desktop"
        data.mkdir()
        (data / "fixture.json").write_text(
            json.dumps(
                {
                    "repo": str(repository),
                    "python": sys.executable,
                    "model": str(fixture.model),
                    "export": str(root / "exported"),
                }
            )
        )
        # The UI uses its isolated workspace; copy just the generated training fixture.
        datasets = data / "workspace" / "datasets"
        datasets.mkdir(parents=True)
        (datasets / "tiny.jsonl").write_bytes(
            (root / "datasets" / "tiny.jsonl").read_bytes()
        )
        result = subprocess.run(
            [str(Path(args.binary).resolve())],
            timeout=240,
            check=False,
            env={**os.environ, "WORKSHOP_SMOKE_ROOT": str(data)},
        )
        report_path = data / "report.json"
        if not report_path.exists():
            raise RuntimeError(
                f"Native smoke exited {result.returncode} without a report"
            )
        report = json.loads(report_path.read_text())
        print(json.dumps(report, indent=2))
        return 0 if report.get("ok") and result.returncode == 0 else 1
    finally:
        fixture.doCleanups()


if __name__ == "__main__":
    sys.exit(main())
