"""Opt-in integration tests: real CUDA training, no pretrained model downloads.

WORKSHOP_CUDA_TEST=1 .training-venv/bin/python -m unittest discover -s worker -p test_cuda.py -v
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import uuid


@unittest.skipUnless(
    os.environ.get("WORKSHOP_CUDA_TEST") == "1", "Opt-in CUDA integration test"
)
class CudaTrainingTests(unittest.TestCase):
    def test_epoch_based_training_completes(self):
        directory, events = self.worker("lora", max_steps=-1)
        self.assertEqual(events[-1]["stage"], "complete")
        self.assertTrue((directory / "adapter" / "adapter_config.json").exists())
        progress = [e for e in events if e.get("stage") == "training" and "epoch" in e]
        self.assertTrue(progress)
        self.assertAlmostEqual(progress[-1]["epoch"], 1.0)

    def setUp(self):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

        self.assertTrue(
            torch.cuda.is_available(), "CUDA is required for this integration test"
        )
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / "model"
        vocabulary = {
            "[UNK]": 0,
            "[PAD]": 1,
            "[EOS]": 2,
            **{
                word: i + 3
                for i, word in enumerate(
                    "def return code local model training repository source test hello world function value".split()
                )
            },
        }
        backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            unk_token="[UNK]",
            pad_token="[PAD]",
            eos_token="[EOS]",
        )
        tokenizer.save_pretrained(self.model)
        torch.manual_seed(42)
        model = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=len(vocabulary),
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=128,
                eos_token_id=2,
                pad_token_id=1,
            )
        )
        model.save_pretrained(self.model, safe_serialization=True)
        datasets = self.root / "datasets"
        datasets.mkdir()
        (datasets / "tiny.jsonl").write_text(
            "\n".join(
                json.dumps(
                    {
                        "text": "def function return value code local model training repository source test",
                        "_group": f"file-{i}",
                    }
                )
                for i in range(12)
            )
        )

    def worker(self, method, max_steps=2, resume=None, cancel=False):
        job_id = str(uuid.uuid4())
        directory = self.root / "jobs" / job_id
        directory.mkdir(parents=True)
        config = dict(
            model=str(self.model),
            dataset="datasets/tiny.jsonl",
            method=method,
            rank=4,
            sequence_length=32,
            batch_size=1,
            gradient_accumulation=1,
            learning_rate=0.001,
            epochs=1,
            max_steps=max_steps,
            save_steps=1,
            validation_fraction=0.2,
            reviewed=True,
            resume=resume,
        )
        (directory / "request.json").write_text(
            json.dumps({"op": "train", "payload": config})
        )
        process = subprocess.Popen(
            [
                sys.executable,
                "-u",
                str(Path(__file__).with_name("main.py")),
                "--root",
                str(self.root),
                "--job",
                job_id,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={
                **{k: v for k, v in os.environ.items() if k != "LD_PRELOAD"},
                "HF_HUB_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
            },
        )
        # Tiny fixtures produce bounded stderr; communicate avoids pipe deadlocks.
        if cancel:
            (directory / "cancel").touch()
        stdout, stderr = process.communicate(timeout=180)
        self.assertEqual(process.returncode, 0, stdout + "\n" + stderr)
        events = []
        for line in stdout.splitlines():
            try:
                events.append(json.loads(line))
            except ValueError:
                pass
        self.assertTrue(events, stdout)
        return directory, events

    def test_lora_updates_weights_and_resumes_checkpoint(self):
        from safetensors.torch import load_file

        directory, events = self.worker("lora")
        self.assertEqual(events[-1]["stage"], "complete")
        self.assertTrue(any("loss" in event for event in events))
        weights = load_file(str(directory / "adapter" / "adapter_model.safetensors"))
        self.assertTrue(
            any(t.abs().sum().item() > 0 for k, t in weights.items() if "lora_B" in k)
        )
        checkpoint = f"jobs/{directory.name}/checkpoint-2"
        resumed, events = self.worker("lora", max_steps=3, resume=checkpoint)
        self.assertEqual(events[-1]["stage"], "complete")
        self.assertTrue((resumed / "checkpoint-3" / "trainer_state.json").exists())

    def test_qlora_updates_adapter_and_exports(self):
        from safetensors.torch import load_file
        from services import export_adapter

        directory, events = self.worker("qlora")
        self.assertEqual(events[-1]["stage"], "complete")
        weights = load_file(str(directory / "adapter" / "adapter_model.safetensors"))
        self.assertTrue(
            any(t.abs().sum().item() > 0 for k, t in weights.items() if "lora_B" in k)
        )
        target = self.root / "exported"
        export_adapter(self.root, directory.name, target)
        self.assertTrue((target / "adapter_config.json").exists())
        self.assertTrue((target / "training-manifest.json").exists())

    def test_cancellation_saves_checkpoint_not_completed_adapter(self):
        directory, events = self.worker("qlora", cancel=True)
        self.assertEqual(events[-1]["stage"], "cancelled")
        self.assertFalse((directory / "adapter").exists())
        self.assertTrue(list(directory.glob("checkpoint-*/trainer_state.json")))
