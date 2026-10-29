"""Exact MST and DBCV equivalence checks; optional contrib hdbscan required."""
import importlib.util
import unittest

import numpy as np
from scipy.spatial.distance import cdist


@unittest.skipUnless(importlib.util.find_spec("hdbscan"), "optional contrib hdbscan absent")
class ParallelDBCVTests(unittest.TestCase):
    def test_mst_nodes_and_edges_identical_with_ties(self):
        from hdbscan.validity import internal_minimum_spanning_tree
        from ga_search.progressive_dbcv import fast_internal_minimum_spanning_tree
        rng = np.random.default_rng(51)
        matrices = []
        for n in (2, 3, 16, 100):
            x = rng.normal(size=(n, 8))
            matrices.append(cdist(x, x))
        matrices.append(cdist(np.arange(12)[:, None], np.arange(12)[:, None]).astype(float))
        matrices.append(np.ones((14, 14)) - np.eye(14))
        for matrix in matrices:
            expected = internal_minimum_spanning_tree(matrix)
            actual = fast_internal_minimum_spanning_tree(matrix)
            np.testing.assert_array_equal(actual[0], expected[0])
            np.testing.assert_array_equal(actual[1], expected[1])

    def test_default_dbcv_scalar_identical_multiple_dimensions_and_noise(self):
        from hdbscan.validity import validity_index
        from ga_search.progressive_dbcv import parallel_validity_index
        rng = np.random.default_rng(82)
        for dimension in (2, 16, 1280):
            x = np.r_[rng.normal(-5, .5, (24, dimension)),
                      rng.normal(5, .5, (26, dimension)),
                      rng.normal(15, .5, (22, dimension)),
                      rng.normal(30, 2, (3, dimension))]
            labels = np.r_[np.zeros(24, int), np.ones(26, int), np.full(22, 2), [-1] * 3]
            expected = validity_index(x, labels)
            for workers in (1, 3, 24):
                actual = parallel_validity_index(x, labels, workers=workers)
                np.testing.assert_equal(actual, expected)

    def test_precomputed_gapped_labels_and_fallback(self):
        from hdbscan.validity import validity_index
        from ga_search.progressive_dbcv import parallel_validity_index
        rng = np.random.default_rng(19)
        x = np.r_[rng.normal(-5, .1, (15, 4)), rng.normal(5, .1, (17, 4))]
        labels = np.r_[np.zeros(15, int), np.full(17, 2)]
        distances = cdist(x, x)
        np.testing.assert_equal(parallel_validity_index(distances, labels, metric="precomputed", d=4),
                                validity_index(distances, labels, metric="precomputed", d=4))
        for option in ("mst_raw_dist", "per_cluster_scores"):
            expected = validity_index(x, labels, **{option: True})
            actual = parallel_validity_index(x, labels, **{option: True})
            if option == "per_cluster_scores":
                self.assertEqual(actual[0], expected[0])
                # Gapped label slot is uninitialized upstream; compare actual clusters.
                np.testing.assert_array_equal(actual[1][[0, 2]], expected[1][[0, 2]])
            else:
                self.assertEqual(actual, expected)
        with self.assertRaises(ValueError):
            parallel_validity_index(x, labels, workers=0)


if __name__ == "__main__":
    unittest.main()
