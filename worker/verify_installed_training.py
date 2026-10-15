"""Finish verification from a real completed installed-GUI training run."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import uuid

INSTALLED_BINARY = Path("/usr/bin/ollama-workshop")
INSTALLED_WORKER = Path("/usr/lib/Ollama Workshop/worker/main.py")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--training-python", type=Path, required=True)
    args = parser.parse_args()
    root = args.artifact_root.expanduser().resolve(strict=True)
    workspace = root / "workspace"
    previous_report = json.loads((root / "report.json").read_text())
    previous = previous_report["runs"][0]
    previous_id = str(uuid.UUID(previous["id"]))
    previous_dir = workspace / "jobs" / previous_id
    previous_state = json.loads((previous_dir / "status.json").read_text())
    if previous_state["status"] != "complete":
        raise ValueError("The original installed-GUI run did not complete")
    if previous_report["binary_sha256"] != sha256(INSTALLED_BINARY):
        raise ValueError("Installed binary differs from the GUI-tested binary")
    config = json.loads((previous_dir / "request.json").read_text())["payload"]
    if (
        config["method"] != "qlora"
        or config["dataset"] != "datasets/installed-repository.jsonl"
    ):
        raise ValueError("Expected the reviewed repository QLoRA fixture")
    checkpoint = previous_dir / f"checkpoint-{previous['global_step']}"
    trainer_state = json.loads((checkpoint / "trainer_state.json").read_text())
    if trainer_state["global_step"] != previous["global_step"]:
        raise ValueError("GUI checkpoint state does not match the report")
    provenance = json.loads((previous_dir / "training-manifest.json").read_text())[
        "provenance"
    ]
    if provenance["repository"]["path"] != str(root / "repository"):
        raise ValueError(
            "Training provenance does not reference the ingested fixture repository"
        )
    dataset_path = workspace / config["dataset"]
    if provenance["dataset_sha256"] != sha256(dataset_path):
        raise ValueError("Reviewed dataset changed since the GUI run")

    verification_id = str(uuid.uuid4())
    directory = workspace / "jobs" / verification_id
    directory.mkdir(mode=0o700)
    resumed_config = {
        **config,
        "max_steps": previous["global_step"] + 1,
        "resume": f"jobs/{previous_id}/checkpoint-{previous['global_step']}",
    }
    (directory / "request.json").write_text(
        json.dumps({"op": "train", "payload": resumed_config}, indent=2)
    )
    environment = {
        **{key: value for key, value in os.environ.items() if key != "LD_PRELOAD"},
        "HF_HUB_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    report = {
        "ok": False,
        "tested_at": datetime.now(timezone.utc).isoformat(),
        "installed_version": subprocess.check_output(
            ["dpkg-query", "-W", "-f=${Version}", "ollama-workshop"], text=True
        ),
        "installed_binary": str(INSTALLED_BINARY),
        "installed_worker": str(INSTALLED_WORKER),
        "binary_sha256": sha256(INSTALLED_BINARY),
        "gui_training": {
            "job": previous_id,
            "global_step": trainer_state["global_step"],
            "losses": previous["losses"],
            "status": previous_state["status"],
            "dataset_sha256": sha256(dataset_path),
        },
        "artifact_root": str(root),
        "scope": "Initial repository ingestion/export/training via installed GUI; resume/export via installed worker.",
    }
    try:
        result = subprocess.run(
            [
                str(args.training_python),
                "-u",
                str(INSTALLED_WORKER),
                "--root",
                str(workspace),
                "--job",
                verification_id,
            ],
            capture_output=True,
            text=True,
            env=environment,
            timeout=180,
        )
        (directory / "verification.stdout.log").write_text(result.stdout)
        (directory / "verification.stderr.log").write_text(result.stderr)
        events = []
        for line in result.stdout.splitlines():
            try:
                events.append(json.loads(line))
            except ValueError:
                continue
        if result.returncode or not events or events[-1].get("stage") != "complete":
            raise RuntimeError(f"Installed worker resume failed; inspect {directory}")
        actual_steps = json.loads(
            (
                directory
                / f"checkpoint-{resumed_config['max_steps']}"
                / "trainer_state.json"
            ).read_text()
        )["global_step"]
        if actual_steps != resumed_config["max_steps"]:
            raise RuntimeError("Resume did not reach the expected optimizer step")
        exported = root / f"exported-adapter-{verification_id}"
        response = subprocess.run(
            [
                str(args.training_python),
                str(INSTALLED_WORKER),
                "--root",
                str(workspace),
            ],
            input=json.dumps(
                {
                    "op": "adapter_export",
                    "payload": {
                        "job_id": verification_id,
                        "destination": str(exported),
                    },
                }
            ),
            capture_output=True,
            text=True,
            env=environment,
            timeout=60,
            check=True,
        )
        if not json.loads(response.stdout)["ok"]:
            raise RuntimeError("Installed worker adapter export failed")
        trained_weights = directory / "adapter/adapter_model.safetensors"
        exported_weights = exported / "adapter_model.safetensors"
        if sha256(trained_weights) != sha256(exported_weights):
            raise RuntimeError("Export changed the trained adapter")
        verification_code = """
import json, sys, torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM
from peft import PeftModel
weights = load_file(sys.argv[2] + "/adapter_model.safetensors")
updated = any(t.abs().sum().item() > 0 for k, t in weights.items() if "lora_B" in k)
base = AutoModelForCausalLM.from_pretrained(
    sys.argv[1], local_files_only=True, use_safetensors=True, trust_remote_code=False
)
model = PeftModel.from_pretrained(base, sys.argv[2], local_files_only=True).to("cuda").eval()
inputs = torch.tensor([[3, 4, 5]], device="cuda")
with torch.no_grad():
    finite = bool(torch.isfinite(model(inputs).logits).all().item())
    output = model.generate(inputs, max_new_tokens=2, do_sample=False, pad_token_id=1)
generated = int(output.shape[1] - inputs.shape[1])
assert updated and finite and generated > 0
print(json.dumps({"adapter_weights_updated": updated, "finite_logits": finite,
                  "generated_tokens": generated, "device": str(model.device)}))
"""
        verified = subprocess.run(
            [
                str(args.training_python),
                "-c",
                verification_code,
                config["model"],
                str(exported),
            ],
            capture_output=True,
            text=True,
            env=environment,
            timeout=60,
            check=True,
        )
        report["worker_resume"] = {
            "job": verification_id,
            "global_step": actual_steps,
            "losses": [event["loss"] for event in events if "loss" in event],
        }
        report["exported_adapter"] = str(exported)
        report["adapter_sha256"] = sha256(exported_weights)
        report["reload_verification"] = json.loads(verified.stdout.splitlines()[-1])
        report["ok"] = True
    except Exception as exc:
        report["error"] = str(exc)
        raise
    finally:
        report_path = root / "installed-final-report.json"
        report_path.write_text(json.dumps(report, indent=2))
        print(json.dumps({**report, "report": str(report_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
