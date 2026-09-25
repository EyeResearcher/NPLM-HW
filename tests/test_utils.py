import json
import math
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from nplm.utils import (
    Timer,
    create_tensorboard_writer,
    evaluate_metrics,
    evaluate_nll,
    load_checkpoint,
    load_config,
    perplexity,
    resolve_device,
    save_checkpoint,
    set_seed,
    write_json,
)


class IdentityModel(nn.Module):
    def forward(self, contexts: torch.Tensor) -> torch.Tensor:
        return contexts.float()


class UtilsTests(unittest.TestCase):
    def _valid_config(self) -> dict[str, object]:
        return {
            "tokenizer_path": "artifacts/vocab.json",
            "context_size": 5,
            "embedding_dim": 8,
            "hidden_dim": 16,
            "lr": 0.001,
            "batch_size": 4,
            "max_steps": 10,
            "eval_every": 2,
        }

    def _write_config(self, path: Path, config: object) -> None:
        path.write_text(yaml.safe_dump(config), encoding="utf-8")

    def test_load_config_applies_defaults_and_rejects_bad_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            self._write_config(path, self._valid_config())
            config = load_config(path)

            self.assertEqual(config["seed"], 1337)
            self.assertEqual(config["tokenizer_type"], "word")
            self.assertEqual(config["activation"], "tanh")
            self.assertEqual(config["dropout"], 0.0)
            self.assertEqual(config["optimizer"], "adam")
            self.assertIsNone(config["clip_grad_norm"])
            self.assertEqual(config["device"], "auto")
            self.assertEqual(config["num_workers"], 0)
            self.assertEqual(config["log_every"], 100)
            self.assertIsNone(config["run_name"])

            invalid = self._valid_config()
            invalid["eval_evey"] = invalid.pop("eval_every")
            self._write_config(path, invalid)
            with self.assertRaisesRegex(ValueError, "Unknown configuration"):
                load_config(path)

            invalid = self._valid_config()
            invalid["batch_size"] = True
            self._write_config(path, invalid)
            with self.assertRaisesRegex(ValueError, "batch_size"):
                load_config(path)

            for key, value in (("log_every", 0), ("run_name", "")):
                with self.subTest(key=key, value=value):
                    invalid = self._valid_config()
                    invalid[key] = value
                    self._write_config(path, invalid)
                    with self.assertRaisesRegex(ValueError, key):
                        load_config(path)

            named = self._valid_config()
            named["run_name"] = "relu-large"
            named["log_every"] = 3
            self._write_config(path, named)
            config = load_config(path)
            self.assertEqual(config["run_name"], "relu-large")
            self.assertEqual(config["log_every"], 3)

    def test_set_seed_reproduces_python_numpy_and_torch(self) -> None:
        set_seed(123)
        first = (random.random(), np.random.rand(), torch.rand(3))
        set_seed(123)
        second = (random.random(), np.random.rand(), torch.rand(3))

        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])
        torch.testing.assert_close(first[2], second[2])

    def test_device_and_perplexity_validation(self) -> None:
        self.assertEqual(resolve_device("cpu"), torch.device("cpu"))
        self.assertAlmostEqual(perplexity(math.log(5)), 5.0)
        for value in (-1.0, float("nan"), float("inf"), 1000.0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                perplexity(value)

    def test_evaluate_nll_weights_uneven_batches_and_restores_mode(self) -> None:
        logits = torch.tensor(
            [
                [5.0, 0.0],
                [4.0, 0.0],
                [0.0, 0.0],
            ]
        )
        targets = torch.tensor([0, 0, 1])
        loader = DataLoader(TensorDataset(logits, targets), batch_size=2)
        model = IdentityModel()
        model.train()

        total_nll, count = evaluate_nll(model, loader, torch.device("cpu"))

        expected = torch.nn.functional.cross_entropy(
            logits, targets, reduction="sum"
        ).item()
        batch_means = [
            torch.nn.functional.cross_entropy(logits[:2], targets[:2]).item(),
            torch.nn.functional.cross_entropy(logits[2:], targets[2:]).item(),
        ]
        self.assertAlmostEqual(total_nll, expected, places=6)
        self.assertEqual(count, 3)
        self.assertNotAlmostEqual(total_nll / count, sum(batch_means) / 2)
        self.assertTrue(model.training)

        model.eval()
        evaluate_nll(model, loader, torch.device("cpu"))
        self.assertFalse(model.training)

    def test_evaluate_metrics_matches_uneven_batch_fixture_and_has_no_state_leak(
        self,
    ) -> None:
        logits = torch.tensor(
            [
                [5.0, 0.0],
                [0.0, 4.0],
                [0.0, 1.0],
            ]
        )
        targets = torch.tensor([0, 0, 1])
        loader = DataLoader(TensorDataset(logits, targets), batch_size=2)
        model = IdentityModel()
        model.train()

        expected_loss = torch.nn.functional.cross_entropy(logits, targets).item()
        expected_accuracy = 2 / 3
        first = evaluate_metrics(
            model, loader, torch.device("cpu"), num_classes=2
        )
        second = evaluate_metrics(
            model, loader, torch.device("cpu"), num_classes=2
        )

        self.assertEqual(set(first), {"loss", "perplexity", "accuracy", "target_count"})
        self.assertAlmostEqual(first["loss"], expected_loss, places=6)
        self.assertAlmostEqual(first["perplexity"], math.exp(expected_loss), places=6)
        self.assertAlmostEqual(first["accuracy"], expected_accuracy, places=6)
        self.assertEqual(first["target_count"], 3)
        self.assertEqual(first, second)
        self.assertTrue(model.training)

        empty_loader = DataLoader(
            TensorDataset(torch.empty((0, 2)), torch.empty(0, dtype=torch.long)),
            batch_size=2,
        )
        with self.assertRaisesRegex(ValueError, "empty"):
            evaluate_metrics(
                model, empty_loader, torch.device("cpu"), num_classes=2
            )

    def test_tensorboard_writer_creates_event_with_scalar_tags(self) -> None:
        from tensorboard.backend.event_processing import event_accumulator

        with tempfile.TemporaryDirectory() as directory:
            log_dir = Path(directory) / "nested" / "tensorboard"
            writer = create_tensorboard_writer(log_dir)
            writer.add_scalar("train/loss", 1.25, 4)
            writer.add_scalar("train/accuracy", 0.5, 4)
            writer.flush()
            writer.close()

            event_files = list(log_dir.glob("events.out.tfevents.*"))
            self.assertTrue(event_files)
            events = event_accumulator.EventAccumulator(str(log_dir))
            events.Reload()
            self.assertIn("train/loss", events.Tags()["scalars"])
            self.assertIn("train/accuracy", events.Tags()["scalars"])

    def test_checkpoint_round_trip_and_validation(self) -> None:
        model = nn.Linear(2, 3)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        model_config = {
            "vocab_size": 3,
            "context_size": 1,
            "embedding_dim": 2,
            "hidden_dim": 4,
            "activation": "tanh",
            "dropout": 0.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "checkpoint.pt"
            save_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                step=7,
                best_val_loss=1.25,
                model_config=model_config,
                training_config=self._valid_config(),
                tokenizer_type="word",
                tokenizer_path="vocab.json",
                train_time_sec=2.5,
            )
            loaded = load_checkpoint(path)
            self.assertEqual(loaded["format_version"], 1)
            self.assertEqual(loaded["step"], 7)
            self.assertEqual(loaded["model_config"], model_config)
            self.assertIsInstance(loaded["optimizer_state_dict"], dict)

            missing = dict(loaded)
            missing.pop("step")
            torch.save(missing, path)
            with self.assertRaisesRegex(ValueError, "missing required"):
                load_checkpoint(path)

            wrong_version = dict(loaded)
            wrong_version["format_version"] = 2
            torch.save(wrong_version, path)
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                load_checkpoint(path)

    def test_atomic_writers_preserve_destination_on_replace_failure(self) -> None:
        model = nn.Linear(1, 1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_path = root / "metrics.json"
            json_path.write_text('{"old": true}\n', encoding="utf-8")
            checkpoint_path = root / "checkpoint.pt"
            checkpoint_path.write_bytes(b"old-checkpoint")
            model_config = {
                "vocab_size": 2,
                "context_size": 1,
                "embedding_dim": 1,
                "hidden_dim": 1,
                "activation": "relu",
                "dropout": 0.0,
            }

            with patch("nplm.utils.os.replace", side_effect=OSError("simulated")):
                with self.assertRaisesRegex(OSError, "simulated"):
                    write_json(json_path, {"new": True})
                with self.assertRaisesRegex(OSError, "simulated"):
                    save_checkpoint(
                        checkpoint_path,
                        model=model,
                        optimizer=None,
                        step=0,
                        best_val_loss=1.0,
                        model_config=model_config,
                        training_config={},
                        tokenizer_type="word",
                        tokenizer_path="vocab.json",
                        train_time_sec=0.0,
                    )

            self.assertEqual(json.loads(json_path.read_text()), {"old": True})
            self.assertEqual(checkpoint_path.read_bytes(), b"old-checkpoint")
            self.assertEqual(list(root.glob(".*.tmp")), [])

    def test_write_json_is_strict_utf8_and_timer_advances(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "metrics.json"
            write_json(path, {"label": "café", "value": 1.0})
            self.assertIn("café", path.read_text(encoding="utf-8"))
            with self.assertRaises(ValueError):
                write_json(path, {"bad": float("nan")})

        with patch("nplm.utils.time.perf_counter", side_effect=[10.0, 12.5]):
            timer = Timer()
            self.assertEqual(timer.elapsed(), 2.5)


if __name__ == "__main__":
    unittest.main()
