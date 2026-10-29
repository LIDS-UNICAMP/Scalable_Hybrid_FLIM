"""Real SPiFiL integration on annotated synthetic 200x200 training images."""
import tempfile
import unittest
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn

from flim.progressive import fit_new_block, InfeasibleCandidate
from spifil.data import Sample


class ProgressiveSPiFiLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory(prefix="progressive-spifil-")
        root = Path(cls.temp.name)
        cls.samples = []
        rng = np.random.default_rng(23)
        for i in range(4):
            label = i // 2
            image = rng.integers(0, 130, (200, 200, 3), dtype=np.uint8)
            image[..., label] += 100
            path = root / f"{label}_{i}.png"
            mask_path = root / f"mask_{i}.png"
            Image.fromarray(image).save(path)
            Image.fromarray(np.full((200, 200), 255, np.uint8)).save(mask_path)
            cls.samples.append(Sample(path.stem, path, mask_path, label))
        cls.config = {"image_size": 200, "device": "cpu", "seed": 19,
                      "n_superpixels": 100, "spifil_in_image": True,
                      "budget_check": True}
        cls.spec = {"op": "conv", "kernel": 3, "channels": 5,
                    "pool_stride": 2, "connection": "sequential", "bn": True,
                    "activation": "silu", "dropout": 0.1}
        cls.block, cls.provenance = fit_new_block(nn.Identity(), cls.samples, cls.spec, cls.config)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_real_fit_actual_width_provenance_and_auxiliary_bn(self):
        self.assertEqual(self.provenance["origin"], "spifil")
        self.assertEqual(self.provenance["actual_channels"], 4)
        self.assertEqual(self.provenance["requested_channels"], 5)
        self.assertGreater(self.provenance["patch_count"], self.provenance["patch_dimension"])
        self.assertEqual(self.provenance["sample_ids"], [s.name for s in self.samples])
        self.assertTrue(torch.equal(self.block.bn.weight, torch.ones(4)))
        self.assertTrue(torch.equal(self.block.bn.bias, torch.zeros(4)))
        self.assertTrue(all(p.requires_grad for p in self.block.parameters()))
        with torch.no_grad():
            self.assertEqual(self.block(torch.randn(1, 3, 200, 200)).shape, (1, 4, 100, 100))

    def test_same_samples_seed_and_config_reproduce_real_filters(self):
        other, metadata = fit_new_block(nn.Identity(), self.samples, self.spec, self.config)
        self.assertTrue(torch.equal(self.block.conv.weight, other.conv.weight))
        self.assertTrue(torch.equal(self.block.conv.bias, other.conv.bias))
        self.assertEqual(metadata["prefix_id"], self.provenance["prefix_id"])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_first_cuda_fit_initializes_memory_counters(self):
        # A fresh process reproduces an empty encoder before its first allocation.
        payload = dict(samples=[dict(name=s.name, image_path=str(s.image_path),
                                    mask_path=str(s.mask_path), label=int(s.label)) for s in self.samples],
                       spec=self.spec, config=dict(self.config, device="cuda:0"))
        code = """
import json, sys, torch
from pathlib import Path
from torch import nn
from spifil.data import Sample
from flim.progressive import fit_new_block
data = json.loads(sys.argv[1])
samples = [Sample(s['name'], Path(s['image_path']), Path(s['mask_path']), s['label']) for s in data['samples']]
assert not torch.cuda.is_initialized(), 'regression requires a fresh CUDA process'
block, provenance = fit_new_block(nn.Identity(), samples, data['spec'], data['config'])
assert next(block.parameters()).is_cuda
assert provenance['fit_peak_cuda_bytes'] > 0
print('Fresh-process CUDA SPiFiL fitting passed')
"""
        result = subprocess.run([sys.executable, "-c", code, json.dumps(payload)],
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_kernel5_real_filter_and_shape_contract(self):
        block, metadata = fit_new_block(nn.Identity(), self.samples,
                                       dict(self.spec, kernel=5), self.config)
        self.assertEqual(block.conv.groups, 1)
        self.assertEqual(block.conv.weight.shape, (4, 3, 5, 5))
        self.assertEqual(metadata["patch_dimension"], 75)

    def test_prefix_weights_bn_statistics_and_eval_are_preserved(self):
        prefix = nn.Sequential(nn.Conv2d(3, 2, 3, padding=1), nn.BatchNorm2d(2), nn.ReLU())
        before = {k: v.clone() for k, v in prefix.state_dict().items()}
        block, provenance = fit_new_block(prefix, self.samples, dict(self.spec, channels=4),
                                          self.config, prefix_id="trained-checkpoint-sha")
        self.assertEqual(block.conv.in_channels, 2)
        self.assertEqual(provenance["prefix_id"], "trained-checkpoint-sha")
        self.assertFalse(prefix.training)
        self.assertFalse(prefix[1].training)
        self.assertTrue(all(not p.requires_grad for p in prefix.parameters()))
        for k, value in prefix.state_dict().items():
            self.assertTrue(torch.equal(value, before[k]), k)

    def test_budget_exhaustion_is_explicit_no_random_fallback(self):
        with self.assertRaisesRegex(InfeasibleCandidate, "covariance budget"):
            fit_new_block(nn.Identity(), self.samples, self.spec,
                          dict(self.config, n_superpixels=1))

    def test_unsupported_ops_and_exclusive_modes_are_rejected(self):
        for changes in ({"op": "depthwise"}, {"kernel": 1}, {"groups": 3},
                        {"connection": "residual"}, {"pool_stride": 3}):
            with self.subTest(changes=changes), self.assertRaises(InfeasibleCandidate):
                fit_new_block(nn.Identity(), self.samples, dict(self.spec, **changes), self.config)
        with self.assertRaisesRegex(InfeasibleCandidate, "exclusive"):
            fit_new_block(None, self.samples, self.spec, dict(self.config, spifil_in_feature=True))


if __name__ == "__main__":
    unittest.main()
