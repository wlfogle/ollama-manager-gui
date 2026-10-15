"""Optional GPU stack is imported only when explicitly requested."""

import importlib.util
import json
import math
from pathlib import Path
import shutil
import signal
import subprocess

from data import atomic_json, contained, digest, parse_dataset, split_rows


def preflight(root):
    packages = {
        name: importlib.util.find_spec(name) is not None
        for name in (
            "torch",
            "transformers",
            "peft",
            "accelerate",
            "bitsandbytes",
            "safetensors",
        )
    }
    gpu = ""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        gpu = result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    runtime = {}
    if all(packages.values()):
        try:
            import torch
            import transformers
            import peft
            import bitsandbytes

            runtime = {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "peft": peft.__version__,
                "bitsandbytes": bitsandbytes.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda": torch.version.cuda,
            }
        except Exception as exc:
            runtime = {"error": type(exc).__name__, "cuda_available": False}
    return {
        "packages": packages,
        "gpu": gpu,
        "disk_free": shutil.disk_usage(root).free,
        "runtime": runtime,
        "ready": all(packages.values())
        and bool(gpu)
        and runtime.get("cuda_available", False),
        "warning": "VRAM fit depends on architecture and sequence length; no estimate guarantees fit.",
    }


def validate_config(config):
    bounds = {
        "rank": (1, 256),
        "sequence_length": (32, 8192),
        "batch_size": (1, 16),
        "gradient_accumulation": (1, 256),
        "max_steps": (-1, 1000000),
        "save_steps": (1, 100000),
        "epochs": (0.1, 100),
        "learning_rate": (1e-8, 0.1),
        "validation_fraction": (0, 0.49),
    }
    for key, (low, high) in bounds.items():
        value = config.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
        ):
            raise ValueError(f"Invalid {key}")
        if not low <= value <= high:
            raise ValueError(f"{key} must be between {low} and {high}")
        if (
            key not in {"epochs", "learning_rate", "validation_fraction"}
            and int(value) != value
        ):
            raise ValueError(f"{key} must be an integer")
    if config.get("method") not in {"lora", "qlora"}:
        raise ValueError("Choose lora or qlora")
    if config["max_steps"] == 0:
        raise ValueError(
            "Use -1 for epoch-based training or a positive maximum step count"
        )
    if config.get("reviewed") is not True:
        raise ValueError("Review the dataset and explicitly approve training")
    model = Path(config.get("model", "")).expanduser()
    if not model.is_absolute() or not (model / "config.json").is_file():
        raise ValueError(
            "Select a downloaded local Transformers model directory, not an Ollama blob"
        )
    if not list(model.glob("*.safetensors")):
        raise ValueError("Safe training requires local safetensors weights")
    return model


def run_training(root, job_dir, config, emit):
    model_path = validate_config(config)
    dataset = contained(root, config["dataset"])
    if (
        dataset.suffix != ".jsonl"
        or dataset.parent != Path(root).resolve() / "datasets"
    ):
        raise ValueError("Select an app-managed dataset")
    raw = dataset.read_text(encoding="utf-8")
    rows = parse_dataset(raw)
    train_rows, eval_rows = split_rows(rows, config["validation_fraction"])
    if not train_rows:
        raise ValueError("No training examples")
    emit(
        {
            "stage": "loading",
            "train_rows": len(train_rows),
            "validation_rows": len(eval_rows),
        }
    )
    import torch
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if not torch.cuda.is_available():
        raise ValueError(
            "A CUDA-enabled PyTorch installation and accessible GPU are required"
        )
    if shutil.disk_usage(root).free < 2 * 1024**3:
        raise ValueError("At least 2 GiB free space is required for checkpoints")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define an EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    use_bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    kwargs = {
        "local_files_only": True,
        "trust_remote_code": False,
        "use_safetensors": True,
        "torch_dtype": dtype,
        "device_map": {"": 0},
    }
    if config["method"] == "qlora":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    model = AutoModelForCausalLM.from_pretrained(str(model_path), **kwargs)
    model.config.use_cache = False
    if config["method"] == "qlora":
        model = prepare_model_for_kbit_training(model)
    else:
        model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=config["rank"],
            lora_alpha=2 * config["rank"],
            lora_dropout=0.05,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
            bias="none",
        ),
    )

    class Examples(Dataset):
        def __init__(self, source):
            self.source = source

        def __len__(self):
            return len(self.source)

        def __getitem__(self, index):
            row = self.source[index]
            if "text" in row:
                text = row["text"]
            else:
                if not tokenizer.chat_template:
                    raise ValueError("Chat datasets require a tokenizer chat template")
                text = tokenizer.apply_chat_template(
                    row["messages"],
                    tokenize=False,
                    add_generation_prompt=False,
                )
            tokens = tokenizer(
                text,
                truncation=True,
                max_length=config["sequence_length"] - 1,
                add_special_tokens=True,
            )["input_ids"]
            if not tokens or tokens[-1] != tokenizer.eos_token_id:
                tokens.append(tokenizer.eos_token_id)
            return {"input_ids": tokens, "attention_mask": [1] * len(tokens)}

    def collate(examples):
        batch = tokenizer.pad(examples, padding=True, return_tensors="pt")
        labels = batch["input_ids"].clone()
        # Mask actual padding positions, not all EOS IDs when EOS is the pad token.
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch

    cancelled = False

    def request_stop(signum, frame):
        nonlocal cancelled
        cancelled = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    class Progress(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            clean = {
                k: v
                for k, v in (logs or {}).items()
                if isinstance(v, (int, float, str))
            }
            emit(
                {
                    "stage": "training",
                    "step": state.global_step,
                    "total_steps": state.max_steps,
                    **clean,
                }
            )

        def on_step_end(self, args, state, control, **kwargs):
            if cancelled or (job_dir / "cancel").exists():
                control.should_save = True
                control.should_training_stop = True
            return control

        def on_save(self, args, state, control, **kwargs):
            emit(
                {
                    "stage": "checkpoint",
                    "path": str(job_dir / f"checkpoint-{state.global_step}"),
                }
            )

    fingerprint = {
        **config,
        "dataset_sha256": digest(raw.encode()),
        "model_config_sha256": digest((model_path / "config.json").read_bytes()),
    }
    provenance = dataset.with_suffix(".provenance.json")
    if provenance.exists():
        fingerprint["provenance"] = json.loads(provenance.read_text())
    resume = config.get("resume")
    if resume:
        resume_path = contained(root, resume)
        if (
            not resume_path.is_dir()
            or not (resume_path / "trainer_state.json").exists()
        ):
            raise ValueError("Checkpoint must contain Trainer state")
        previous = json.loads(
            (resume_path.parent / "training-manifest.json").read_text()
        )
        for key in (
            "model",
            "model_config_sha256",
            "dataset_sha256",
            "method",
            "rank",
            "sequence_length",
            "batch_size",
            "gradient_accumulation",
            "validation_fraction",
        ):
            if previous.get(key) != fingerprint.get(key):
                raise ValueError(f"Checkpoint is incompatible: {key}")
        resume = str(resume_path)
    atomic_json(job_dir / "training-manifest.json", fingerprint)
    args = TrainingArguments(
        output_dir=str(job_dir),
        num_train_epochs=config["epochs"],
        max_steps=config["max_steps"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=config["gradient_accumulation"],
        learning_rate=config["learning_rate"],
        gradient_checkpointing=True,
        bf16=use_bf16,
        fp16=not use_bf16,
        logging_steps=1,
        save_steps=config["save_steps"],
        save_total_limit=2,
        eval_strategy="steps" if eval_rows else "no",
        eval_steps=config["save_steps"],
        report_to="none",
        seed=42,
        data_seed=42,
        dataloader_num_workers=0,
        optim="paged_adamw_8bit" if config["method"] == "qlora" else "adamw_torch",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=Examples(train_rows),
        eval_dataset=Examples(eval_rows) if eval_rows else None,
        data_collator=collate,
        callbacks=[Progress()],
    )
    trainer.train(resume_from_checkpoint=resume or None)
    if cancelled or (job_dir / "cancel").exists():
        emit(
            {
                "stage": "cancelled",
                "message": "Stopped at a training step; checkpoint saved",
            }
        )
        return
    target = job_dir / "adapter"
    model.save_pretrained(target, safe_serialization=True)
    tokenizer.save_pretrained(target)
    atomic_json(target / "training-manifest.json", fingerprint)
    emit({"stage": "complete", "adapter": str(target)})
