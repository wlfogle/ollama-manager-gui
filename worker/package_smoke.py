"""Verify an extracted Debian release through its accessibility tree; no install."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time


def main():
    import gi

    gi.require_version("Atspi", "2.0")
    from gi.repository import Atspi

    parser = argparse.ArgumentParser()
    parser.add_argument("--deb", required=True)
    args = parser.parse_args()
    if not os.environ.get("DISPLAY"):
        parser.error("Set DISPLAY to an accessible X11 graphical session")
    with tempfile.TemporaryDirectory(prefix="workshop-package-") as temporary:
        root = Path(temporary)
        subprocess.run(
            ["dpkg-deb", "--extract", str(Path(args.deb).resolve()), str(root)],
            check=True,
        )
        worker = root / "usr/lib/Ollama Workshop/worker/main.py"
        response = subprocess.run(
            ["/usr/bin/python3", str(worker), "--root", str(root / "worker-data")],
            input=json.dumps(
                {
                    "op": "dataset_validate",
                    "payload": {"text": '{"text":"package fixture"}'},
                }
            ),
            capture_output=True,
            text=True,
            check=True,
        )
        assert json.loads(response.stdout)["result"]["rows"] == 1, response.stdout
        process = subprocess.Popen(
            [str(root / "usr/bin/ollama-workshop")],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, "XDG_DATA_HOME": str(root / "data")},
        )
        try:
            desktop = Atspi.get_desktop(0)

            def nodes(parent, depth=0):
                if depth > 30:
                    return
                yield parent
                for i in range(parent.get_child_count()):
                    child = parent.get_child_at_index(i)
                    if child:
                        yield from nodes(child, depth + 1)

            def own_app():
                for i in range(desktop.get_child_count()):
                    app = desktop.get_child_at_index(i)
                    if app and app.get_process_id() == process.pid:
                        return app
                return None

            def wait_for(name):
                until = time.monotonic() + 30
                last_names = []
                while time.monotonic() < until:
                    if process.poll() is not None:
                        raise RuntimeError("Release app exited unexpectedly")
                    app = own_app()
                    if app:
                        entries = list(nodes(app))
                        last_names = [entry.get_name() for entry in entries]
                        for entry in entries:
                            if entry.get_name() == name:
                                return entry
                    time.sleep(0.2)
                raise RuntimeError(
                    f"Release UI control not found: {name}; tree={last_names}"
                )

            wait_for("Repositories").get_action_iface().do_action(0)
            wait_for("Index / refresh")
            wait_for("Datasets").get_action_iface().do_action(0)
            wait_for("Validate")
            wait_for("Maintenance").get_action_iface().do_action(0)
            wait_for("Scan references").get_action_iface().do_action(0)
            # Reading the store in the release invokes its bundled worker; the
            # Repositories and Datasets screens also require successful worker IPC.
            time.sleep(2)
        finally:
            process.terminate()
            stdout, stderr = process.communicate(timeout=15)
        if "Failed to load module" in stderr:
            raise RuntimeError(stderr)
        print(
            json.dumps(
                {
                    "ok": True,
                    "checks": [
                        "extracted runtime worker validates data",
                        "optimized release app opens",
                        "packaged worker IPC opens repository and dataset screens",
                        "maintenance scan invoked",
                        "no missing GTK module warnings",
                    ],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
