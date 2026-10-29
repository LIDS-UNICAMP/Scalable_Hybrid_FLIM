"""Real CPU integration, no mocked SPiFiL, datasets or training computations.

Run with the project's Python 3.11 environment, optionally adding the contrib
HDBSCAN installation to PYTHONPATH. All artifacts live in a unique /tmp folder.
"""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from ga_search.progressive import ProgressiveExperiment, file_digest, read_config, run_config
from ga_search.progressive_data import load_data
from methods.autoencoder.progressive import EvolutionaryCNN, ProgressiveTrainer


CONFIG = Path(__file__).resolve().parents[2] / "ga_search/config/progressive_nas_synthetic.yaml"


class ProgressivePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory(prefix="nas-d1-unique-", dir="/tmp")
        cls.root = Path(cls.temp.name)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def overrides(self, name, **extra):
        return dict(output_dir=str(self.root / name), device="cpu", batch_size=1,
                    synthetic_train_samples=4, synthetic_val_samples=4,
                    latency_warmup=0, latency_repeats=1,
                    **extra)

    def test_two_depth_real_fit_interrupt_resume_and_inference(self):
        overrides = self.overrides("train", mode="train")
        original = ProgressiveTrainer.train_epoch
        calls = 0

        def interrupt_before_second_epoch(trainer, loader):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise KeyboardInterrupt("simulated process interruption after saved epoch")
            return original(trainer, loader)

        with patch.object(ProgressiveTrainer, "train_epoch", interrupt_before_second_epoch):
            with self.assertRaises(KeyboardInterrupt):
                run_config(CONFIG, overrides=overrides)
        candidate = next((Path(overrides["output_dir"]) / "individuals").iterdir())
        interrupted = torch.load(candidate / "last.ckpt", map_location="cpu", weights_only=False)
        self.assertEqual(interrupted["model"].depth, 1)
        self.assertEqual(interrupted["metadata"]["phase"], "A")
        self.assertEqual(interrupted["metadata"]["epoch_done"], 1)
        self.assertTrue(interrupted["optimizer"]["state"])

        first_checkpoint_hash = []
        phase_checks = []

        def observe_real_training(trainer, loader):
            depth, phase = trainer.model.depth, trainer.phase
            prefix_before = {k: v.clone() for k, v in trainer.model.encoder[0].state_dict().items()}
            if depth == 2 and phase == "A":
                first_checkpoint_hash.append(file_digest(candidate / "depth_1/selected.ckpt"))
            result = original(trainer, loader)
            if depth == 2 and phase == "A":
                for key, value in prefix_before.items():
                    self.assertTrue(torch.equal(value, trainer.model.encoder[0].state_dict()[key]), key)
                self.assertTrue(any(p.grad is not None for p in trainer.model.encoder[1].parameters()))
                phase_checks.append("frozen")
            if depth == 2 and phase == "B":
                self.assertTrue(any(p.grad is not None for p in trainer.model.encoder[0].parameters()))
                self.assertTrue(any(not torch.equal(v, trainer.model.encoder[0].state_dict()[k])
                                    for k, v in prefix_before.items()))
                phase_checks.append("unfrozen")
            return result

        with patch.object(ProgressiveTrainer, "train_epoch", observe_real_training):
            result = run_config(CONFIG, resume=True, overrides=overrides)
        self.assertIn(result["status"], ("ok", "invalid"), result)
        self.assertEqual(result["depth"], 2)
        self.assertEqual(phase_checks, ["frozen", "unfrozen"])
        self.assertEqual(first_checkpoint_hash, [file_digest(candidate / "depth_1/selected.ckpt")])
        history = json.loads((candidate / "history.json").read_text())
        self.assertEqual([(r["depth"], r["phase"]) for r in history], [(1, "A"), (1, "B"), (2, "A"), (2, "B")])
        self.assertTrue(all(np.isfinite(r["reconstruction_loss"]) for r in history))
        self.assertTrue(all(record["origin"] == "spifil" for record in result["initialization_records"]))
        self.assertEqual(len(result["initialization_records"][1]["prefix_id"]), 64)
        self.assertNotEqual(result["initialization_records"][1]["prefix_id"], "empty_prefix")
        with np.load(result["embedding_path"]) as artifact:
            self.assertEqual(artifact["embeddings"].shape, (4, 1280))
            self.assertEqual(artifact["checkpoint_id"].item(), result["checkpoint_id"])
        restored = ProgressiveTrainer(EvolutionaryCNN(read_config(CONFIG, overrides)), read_config(CONFIG, overrides))
        restored.load_checkpoint(result["checkpoint"])
        restored.model.eval()
        x = torch.zeros(1, 3, 200, 200)
        inference = torch.load(result["inference_path"], map_location="cpu", weights_only=False)
        with torch.no_grad():
            self.assertEqual(restored.model(x).shape, x.shape)
            self.assertEqual(inference(x).shape, (1, 1280))
            torch.testing.assert_close(inference(x), restored.model.embed(x))
        self.assertEqual(run_config(CONFIG, resume=True, overrides=overrides), result)

    def test_complete_genetic_generation_and_resume(self):
        overrides = self.overrides("search", mode="search")
        result = run_config(CONFIG, overrides=overrides)
        self.assertEqual(result["status"], "finished")
        self.assertEqual(result["generations"], 2)
        state_path = Path(result["history_path"])
        state = json.loads(state_path.read_text())
        self.assertEqual(len(state["history"]), 2)
        self.assertTrue(state["finished"])
        for generation in state["history"]:
            self.assertEqual(len(generation["population"]), 2)
            self.assertTrue(all(np.isfinite(individual["result"]["J"])
                                for individual in generation["population"]))
        before = file_digest(state_path)
        self.assertEqual(run_config(CONFIG, resume=True, overrides=overrides), result)
        self.assertEqual(file_digest(state_path), before)

    def test_constraints_and_training_only_donors(self):
        overrides = self.overrides("constraints", mode="train", max_parameters=1)
        result = run_config(CONFIG, overrides=overrides)
        self.assertEqual(result["status"], "infeasible")
        self.assertEqual(result["J"], 1e12)
        self.assertIn("constraint", result["reason"])
        self.assertEqual(len(result["initialization_records"]), 1)
        config = read_config(CONFIG, self.overrides("splits"))
        train, validation, donors, identity = load_data(config, self.root / "splits")
        training_ids = {sample.name for sample in train.dataset.samples}
        validation_ids = {sample.name for sample in validation.dataset.samples}
        self.assertFalse(training_ids & validation_ids)
        self.assertLessEqual({sample.name for sample in donors}, training_ids)
        self.assertEqual(set(identity["selected_sample_ids"]), training_ids)
        self.assertFalse({sample.image_path.resolve() for sample in train.dataset.samples} &
                         {sample.image_path.resolve() for sample in validation.dataset.samples})
        self.assertNotIn("test", identity["samples"])

    def test_mismatched_resume_does_not_overwrite_data_or_identity(self):
        from ga_search.progressive import ProgressiveExperiment
        from PIL import Image
        overrides = self.overrides("identity_safety", mode="train")
        config = read_config(CONFIG, overrides)
        experiment = ProgressiveExperiment(config)
        image = experiment.train.dataset.samples[0].image_path
        identity = Path(config["output_dir"]) / "data_identity.json"
        before = (file_digest(image), file_digest(identity))
        with self.assertRaisesRegex(ValueError, "changed"):
            ProgressiveExperiment({**config, "seed": config["seed"] + 1}, resume=True)
        self.assertEqual(before, (file_digest(image), file_digest(identity)))
        # Altered data must invalidate identity, not be regenerated over the edit.
        Image.fromarray(np.zeros((200, 200, 3), dtype=np.uint8)).save(image)
        changed = file_digest(image)
        with self.assertRaisesRegex(ValueError, "changed"):
            ProgressiveExperiment(config, resume=True)
        self.assertEqual(file_digest(image), changed)
        self.assertEqual(file_digest(identity), before[1])

    def test_effective_width_cache_and_missing_checkpoint(self):
        for missing in (False, True):
            with self.subTest(missing_checkpoint=missing):
                config = read_config(CONFIG, self.overrides(f"cache-{missing}"))
                experiment = ProgressiveExperiment(config)
                genome = {"blocks": [copy.deepcopy(config["genome"]["blocks"][0])]}
                first = experiment.evaluate(genome)
                self.assertIn(first["status"], ("ok", "invalid"), first)
                cache_paths = list((experiment.root / "cache").glob("*.json"))
                self.assertEqual(len(cache_paths), 1)
                if missing:
                    cached = json.loads(cache_paths[0].read_text())
                    cached["checkpoint"] = str(experiment.root / "absent-checkpoint.ckpt")
                    cache_paths[0].write_text(json.dumps(cached))
                changed = copy.deepcopy(genome)
                changed["blocks"][0]["channels"] = 5  # SPiFiL rounds 5 requested to 4 effective.
                second = experiment.evaluate(changed)
                self.assertIn(second["status"], ("ok", "invalid"), second)
                self.assertEqual(second["effective_genome"]["blocks"][0]["channels"], 4)
                history = json.loads(Path(second["history_path"]).read_text())
                self.assertEqual(len(history), 2)
                self.assertTrue(all(row["individual_id"] == second["individual_id"] for row in history))
                self.assertTrue(all(row["requested_genome"] == changed for row in history))
                if missing:
                    self.assertFalse(any("cache_source_checkpoint" in row for row in history))
                else:
                    self.assertTrue(all(row["cache_source_checkpoint"] == first["checkpoint_id"]
                                        for row in history))
                checkpoint = torch.load(second["checkpoint"], map_location="cpu", weights_only=False)
                provenance = checkpoint["model"].provenance["encoder1"]
                self.assertEqual(provenance["requested_channels"], 5)
                self.assertEqual(provenance["actual_channels"], 4)


if __name__ == "__main__":
    unittest.main()
