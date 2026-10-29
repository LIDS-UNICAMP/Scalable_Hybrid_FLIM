"""Small CPU-only checks for parallel preprocessing and RNG-compatible loading."""
import unittest

import torch
from torch.utils.data import DataLoader, Dataset

from ga_search.progressive_cpu import cache_loader


class FixtureDataset(Dataset):
    def __init__(self):
        self.samples = [f"sample-{i}" for i in range(5)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return torch.arange(3 * 4 * 4, dtype=torch.float32).reshape(3, 4, 4) / (index + 1), index % 2, self.samples[index]


class CacheLoaderTests(unittest.TestCase):
    def test_parallel_values_metadata_batches_and_rng(self):
        original = DataLoader(FixtureDataset(), batch_size=2, num_workers=0)
        torch.manual_seed(57)
        rng_before = torch.get_rng_state().clone()
        cached, stats = cache_loader(original, workers=2, device="cpu")
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))
        self.assertIs(cached.dataset.samples, original.dataset.samples)
        self.assertEqual(cached.batch_size, original.batch_size)
        self.assertEqual(cached.num_workers, 0)
        self.assertEqual(stats["samples"], 5)
        self.assertEqual(stats["bytes"], 5 * 3 * 4 * 4 * 4)
        self.assertEqual(stats["workers"], 2)
        self.assertGreaterEqual(stats["load_seconds"], 0)
        torch.set_rng_state(rng_before)
        expected = list(original)
        original_rng_after = torch.get_rng_state().clone()
        torch.set_rng_state(rng_before)
        actual = list(cached)
        self.assertTrue(torch.equal(torch.get_rng_state(), original_rng_after))
        for (x, y, ids), (cx, cy, cids) in zip(expected, actual):
            self.assertTrue(torch.equal(x, cx))
            self.assertTrue(torch.equal(y, cy))
            self.assertEqual(ids, cids)

    def test_single_process_restores_thread_limit_and_drop_last(self):
        original = DataLoader(FixtureDataset(), batch_size=2, drop_last=True)
        threads = torch.get_num_threads()
        cached, _ = cache_loader(original, workers=1, device="cpu")
        self.assertEqual(torch.get_num_threads(), threads)
        self.assertTrue(cached.drop_last)
        self.assertEqual(len(list(cached)), 2)

    def test_invalid_workers(self):
        with self.assertRaisesRegex(ValueError, "workers"):
            cache_loader(DataLoader(FixtureDataset()), workers=0, device="cpu")


if __name__ == "__main__":
    unittest.main()
