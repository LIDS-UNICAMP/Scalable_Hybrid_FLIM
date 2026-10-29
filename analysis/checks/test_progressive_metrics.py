"""CPU checks: python -m unittest analysis.checks.test_progressive_metrics -v."""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import davies_bouldin_score
from unittest.mock import patch

from ga_search.progressive_metrics import (
    evaluate_embeddings, extract_embeddings, hoyer_sparsity, scalar_fitness,
)


class MetricsTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(19)
        self.x = np.r_[rng.normal(-5, .1, (20, 4)), rng.normal(5, .1, (20, 4))]

    def test_hoyer_extremes_and_scale(self):
        self.assertAlmostEqual(float(hoyer_sparsity([[1, 0, 0, 0]])[0]), 1)
        self.assertAlmostEqual(float(hoyer_sparsity([[1, -1, 1, -1]])[0]), 0)
        np.testing.assert_allclose(hoyer_sparsity(self.x), hoyer_sparsity(self.x * 1e200))
        for x in ([[0, 0]], [[np.inf, 1]], [[np.nan, 1]], [[1]]):
            with self.assertRaises(ValueError):
                hoyer_sparsity(x)

    def test_real_hdbscan_and_fitness(self):
        a = evaluate_embeddings(self.x, {"min_cluster_size": 5, "sparsity_lambda": .004})
        b = evaluate_embeddings(self.x, {"min_cluster_size": 5, "sparsity_lambda": .04})
        self.assertEqual(a["status"], "ok", a)
        self.assertEqual(a["n_clusters"], 2)
        self.assertEqual(a["coverage"], 1)
        self.assertGreaterEqual(a["dbi"], 0)
        self.assertAlmostEqual(a["dbi_norm"], a["dbi"] / (1 + a["dbi"]))
        self.assertAlmostEqual(a["J"], .996 * a["dbi_norm"] + .004 * (1 - a["hoyer_mean"]))
        self.assertAlmostEqual(b["J"], .96 * b["dbi_norm"] + .04 * (1 - b["hoyer_mean"]))
        self.assertLess(scalar_fitness(1, .9), scalar_fitness(1, .1))
        self.assertIn(a["dbcv_status"], ("ok", "unavailable", "invalid"))

    def test_invalid_penalty(self):
        for x in (np.ones((20, 4)), np.zeros((20, 4)), np.full((20, 4), np.nan), [[1, 2]]):
            result = evaluate_embeddings(x, {"penalty": 123.0})
            self.assertEqual(result["status"], "invalid", result)
            self.assertEqual(result["J"], 123)
            self.assertTrue(result["reason"])

    @unittest.skipUnless(importlib.util.find_spec("hdbscan"), "optional contrib hdbscan absent")
    def test_real_dbcv_and_backend_parameter_equivalence(self):
        contrib = evaluate_embeddings(self.x, {"min_cluster_size": 5})
        self.assertEqual(contrib["dbcv_status"], "ok", contrib)
        self.assertTrue(np.isfinite(contrib["dbcv"]))
        self.assertGreaterEqual(contrib["dbcv"], -1)
        self.assertLessEqual(contrib["dbcv"], 1)
        self.assertEqual(contrib["clustering_backend"], "hdbscan")
        self.assertEqual(contrib["min_samples_including_self"], 5)
        self.assertEqual(contrib["min_samples_backend"], 4)
        # Import masking selects the installed sklearn implementation; its
        # clustering remains real and receives the equivalent neighbor count.
        with patch.dict(sys.modules, {"hdbscan": None}):
            sklearn = evaluate_embeddings(self.x, {"min_cluster_size": 5})
        self.assertEqual(sklearn["status"], "ok", sklearn)
        self.assertEqual(sklearn["min_samples_backend"], 5)
        self.assertEqual(sklearn["n_clusters"], contrib["n_clusters"])
        self.assertAlmostEqual(sklearn["J"], contrib["J"])
        result = evaluate_embeddings(self.x, {"min_samples": 1})
        self.assertEqual(result["status"], "invalid")
        self.assertIn("min_samples", result["reason"])

    def test_noise_excluded_from_dbi_and_coverage(self):
        # Controlled assignments isolate DBI/noise arithmetic; HDBSCAN itself
        # is exercised without mocks in test_real_hdbscan_and_fitness.
        import ga_search.progressive_metrics as metrics
        labels = np.r_[np.zeros(20, dtype=int), np.ones(20, dtype=int), [-1] * 10]
        noisy = np.r_[self.x, np.full((10, 4), 1000)]
        module = "hdbscan.HDBSCAN.fit_predict" if importlib.util.find_spec("hdbscan") else "sklearn.cluster.HDBSCAN.fit_predict"
        with patch(module, return_value=labels):
            good = metrics.evaluate_embeddings(noisy, {"min_coverage": .8})
            bad = metrics.evaluate_embeddings(noisy, {"min_coverage": .81})
        self.assertEqual(good["status"], "ok", good)
        self.assertAlmostEqual(good["coverage"], .8)
        self.assertAlmostEqual(good["dbi"], davies_bouldin_score(self.x, labels[:40]))
        self.assertEqual(bad["status"], "invalid")
        self.assertEqual(bad["J"], 1e12)
        self.assertIn("coverage", bad["reason"])

    def test_extract_artifact_and_modes(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = torch.nn.BatchNorm1d(4)

            def embed(self, images):
                if self.training or torch.is_grad_enabled():
                    raise AssertionError("extraction needs eval and no_grad")
                return images.mean((1, 2, 3))[:, None].expand(-1, 1280)

        model = Model()
        model.train()
        model.bn.eval()
        batch = torch.ones(2, 3, 200, 200)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "embeddings.npz"
            z, ids = extract_embeddings(model, [(batch, ["a", "b"])], path, "sha256:example")
            self.assertEqual(z.shape, (2, 1280))
            self.assertEqual(ids.tolist(), ["a", "b"])
            with np.load(path) as artifact:
                self.assertEqual(artifact["checkpoint_id"].item(), "sha256:example")
                np.testing.assert_array_equal(artifact["embeddings"], z)
        self.assertTrue(model.training)
        self.assertFalse(model.bn.training)
        z, ids = extract_embeddings(model, [(batch, torch.tensor([1, 2]), ["a", "b"])])
        self.assertEqual(z.shape, (2, 1280))
        with self.assertRaises(ValueError):
            extract_embeddings(model, [(batch, ["a", "a"])])


if __name__ == "__main__":
    unittest.main()
