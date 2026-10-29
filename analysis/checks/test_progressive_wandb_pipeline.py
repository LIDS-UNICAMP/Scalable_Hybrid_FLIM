"""Real offline W&B + annotated SPiFiL + two-generation CPU integration.

No SDK, initializer, metrics or training computations are mocked. The only
injection checks that a telemetry infrastructure exception escapes evaluation.
"""
from __future__ import annotations

import json
import math
import os
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from ga_search.progressive import (ExperimentLoggingError, ProgressiveExperiment,
                                   file_digest, read_config, run_config)


CONFIG = Path(__file__).resolve().parents[2] / "ga_search/config/progressive_nas_synthetic.yaml"


class WandbPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def config(self, root, enabled):
        return read_config(CONFIG, {
            "output_dir": str(root), "device": "cpu", "mode": "search", "batch_size": 1,
            "synthetic_train_samples": 4, "synthetic_val_samples": 4,
            "latency_warmup": 0, "latency_repeats": 1,
            "epochs_a": 1, "epochs_b": 1,
            "genetic": {"population_size": 2, "generations": 2, "min_depth": 2, "max_depth": 2,
                        "elitism": 1, "mutation_rate": 1.0, "patience": 2},
            "wandb": {"enabled": enabled, "mode": "offline", "entity": "ophira-ai",
                      "project": "progressive_nas_ga_tests", "upload_artifacts": True},
        })

    def events(self, root):
        return [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]

    def test_two_generations_real_offline_sdk_and_finished_resume(self):
        with tempfile.TemporaryDirectory(prefix="nas-wandb-pipeline-", dir="/tmp") as tmp:
            base = Path(tmp)
            root = base / "experiment"
            with patch.dict(os.environ, {
                "WANDB_CACHE_DIR": str(base / "wandb-cache"),
                "WANDB_CONFIG_DIR": str(base / "wandb-settings"),
                "WANDB_DATA_DIR": str(base / "wandb-data"),
                "WANDB_SILENT": "true",
            }):
                config = self.config(root, True)
                experiment = ProgressiveExperiment(config)
                result = experiment.run()
                self.assertEqual(result["status"], "finished")
                self.assertEqual(result["generations"], 2)
                events = self.events(root)
                names = [row["event"] for row in events]
                self.assertEqual(names.count("generation_start"), 2)
                self.assertEqual(names.count("generation_end"), 2)
                self.assertEqual(names.count("individual_result"), 4)
                self.assertIn("offspring", names)
                self.assertIn("search_end", names)
                self.assertTrue(any(row.get("cache") for row in events if row["event"] == "individual_result"))
                evaluations = [row for row in events if row["event"] == "individual_result"]
                for row in evaluations:
                    self.assertIn("requested_genome", row)
                    self.assertTrue(math.isfinite(row["result"]["J"]))
                    self.assertIn(row["result"]["status"], ("ok", "invalid", "infeasible"))
                    if row["result"]["status"] != "ok":
                        self.assertTrue(row["result"]["reason"])
                epochs = [row for row in events if row["event"] == "epoch_end"]
                self.assertGreaterEqual(len(epochs), 4)
                self.assertEqual({row["phase"] for row in epochs}, {"A", "B"})
                self.assertEqual({row["depth"] for row in epochs}, {1, 2})
                for row in epochs:
                    metrics = row["result"]
                    self.assertTrue(math.isfinite(metrics["J"]))
                    self.assertTrue(math.isfinite(metrics["reconstruction_loss"]))
                    self.assertIn("status", metrics)
                    self.assertIn("reason", metrics)
                    self.assertEqual(metrics["requested_genome"], row["requested_genome"])
                    self.assertIn("effective_genome", row)
                initializations = [row for row in events if row["event"] == "initialization"]
                self.assertGreaterEqual(len(initializations), 2)
                for row in initializations:
                    self.assertEqual(row["result"]["origin"], "spifil")
                    self.assertTrue(row["result"]["sample_ids"])
                    self.assertGreater(row["result"]["actual_channels"], 0)
                state = json.loads((root / "logger_state.json").read_text())
                self.assertTrue(state["enabled"])
                self.assertEqual(state["run_id"], experiment.logger.run_id)
                self.assertIsNone(state["url"])
                self.assertTrue(list((root / "wandb").glob("offline-run-*/run-*.wandb")))
                # Snapshot logs are actual SDK artifact inputs, not model/data uploads.
                snapshots = [path for directory in (root / "artifact_snapshots", root / "logger_artifact")
                             if directory.exists() for path in directory.rglob("events.jsonl")]
                self.assertTrue(snapshots)
                self.assertTrue(all(path.stat().st_size > 0 for path in snapshots))
                self.assertFalse(any(path.suffix in {".pt", ".ckpt", ".png"}
                                     for directory in (root / "artifact_snapshots", root / "logger_artifact")
                                     if directory.exists() for path in directory.rglob("*")))
                search_path = root / "search/search_state.json"
                search_hash = file_digest(search_path)
                selected_hashes = {str(path): file_digest(path) for path in (root / "individuals").glob("*/depth_*/selected.ckpt")}
                sequence = state["sequence"]
                epoch_count = len(epochs)
                # A completed search resumes telemetry without training or changing its checkpoint/RNG state.
                python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
                resumed = ProgressiveExperiment(config, resume=True)
                self.assertEqual(resumed.logger.run_id, state["run_id"])
                self.assertEqual(resumed.run(), result)
                self.assertEqual(file_digest(search_path), search_hash)
                self.assertEqual(selected_hashes, {str(path): file_digest(path) for path in (root / "individuals").glob("*/depth_*/selected.ckpt")})
                self.assertEqual(random.getstate(), python_state)
                self.assertEqual(np.random.get_state()[0], numpy_state[0])
                np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
                self.assertEqual(np.random.get_state()[2:], numpy_state[2:])
                self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
                after = self.events(root)
                self.assertEqual(len([row for row in after if row["event"] == "epoch_end"]), epoch_count)
                self.assertEqual([row["sequence"] for row in after], list(range(len(after))))
                self.assertGreater(after[-1]["sequence"], sequence)
                self.assertIn("search_resume", [row["event"] for row in after[len(events):]])
                self.assertEqual(json.loads((root / "logger_state.json").read_text())["run_id"], state["run_id"])

    def test_logging_infrastructure_failure_is_not_an_infeasible_candidate(self):
        with tempfile.TemporaryDirectory(prefix="nas-log-failure-", dir="/tmp") as tmp:
            root = Path(tmp)
            experiment = ProgressiveExperiment(self.config(root, False))
            try:
                with patch.object(experiment.logger, "emit", side_effect=RuntimeError("injected telemetry outage")):
                    with self.assertRaisesRegex(ExperimentLoggingError, "telemetry outage"):
                        experiment.evaluate(experiment.config["genome"])
                self.assertFalse(list((root / "individuals").glob("*/result.json")))
            finally:
                experiment.logger.finish(status="failed")


if __name__ == "__main__":
    unittest.main()
