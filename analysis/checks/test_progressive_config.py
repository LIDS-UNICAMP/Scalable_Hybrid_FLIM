"""Configuration/launcher isolation and the existing complementary SVM protocol."""
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from experiments.ray.schema import Experiment
from ga_search.progressive import complementary_probe, read_config
from methods.autoencoder.progressive import auxiliary_init


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "ga_search/config/progressive_nas_synthetic.yaml"


class ProgressiveConfigurationTests(unittest.TestCase):
    def test_configuration_rejections(self):
        for overrides in ({"auxiliary_init": "trunc_normal"}, {"unknown_knob": True},
                          {"embedding_dim": 128}, {"in_channels": 1}, {"max_depth": 11},
                          {"spifil_in_feature": True}, {"weight_inheritance": True},
                          {"epochs_b": 0}, {"refine_lr": .1}, {"batch_size": True}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                read_config(CONFIG, overrides)
        self.assertEqual(read_config(CONFIG)["embedding_dim"], 1280)

    def test_auxiliary_policy_is_explicit(self):
        conv = nn.Conv2d(3, 4, 1)
        with self.assertRaises(ValueError):
            auxiliary_init(conv, "trunc_normal")
        for policy in ("he", "xavier"):
            auxiliary_init(conv, policy)
            self.assertTrue(torch.equal(conv.bias, torch.zeros_like(conv.bias)))

    def test_launcher_opt_in_does_not_change_legacy_selection(self):
        data = dict(name="nas_check", method="autoencoder", runner="growth",
                    resources={"gpu_ids": [0]}, output={"work_dir": "/tmp/nas-check"},
                    evaluate={"SVM": False})
        with self.assertRaises(ValueError):
            Experiment.from_dict(data)
        exp = Experiment.from_dict({**data, "runner_args": {"progressive_nas": str(CONFIG)}})
        from experiments.ray.runners.growth import run_arm
        result = run_arm(exp, dict(dataset="protozoan-cysts", split=1, percentage=5), 0, dry_run=True)
        self.assertEqual(len(result["plan"]), 1)
        self.assertIn("--progressive-nas", result["plan"][0])
        with self.assertRaises(ValueError):
            Experiment.from_dict({**data, "runner_args": {"progressive_nas": str(CONFIG), "max_rounds": 1}})

    def test_complementary_probe_reuses_canonical_svm(self):
        class FixtureEmbedding(nn.Module):
            def embed(self, x):
                return x.mean((1, 2, 3))[:, None].repeat(1, 1280)
        x = torch.stack([torch.full((3, 2, 2), v) for v in (.1, .12, .8, .82)])
        # Original annotation IDs need not be zero-indexed; metric caller remaps.
        batch = [(x, torch.tensor([1, 1, 7, 7]), ["a", "b", "c", "d"])]
        metrics = complementary_probe(FixtureEmbedding(), batch, batch, "cpu")
        self.assertEqual(metrics["svm_kappa"], 1.0)
        self.assertEqual(metrics["svm_fit_status"], 0)
        self.assertIn("kernel=linear;C=100", metrics["svm_protocol"])


if __name__ == "__main__":
    unittest.main()
