import unittest

import numpy as np

from swid_retrieval.experiments.rq1_native import native_ce_from_cache, native_ce_macro


class NativeCETest(unittest.TestCase):
    def test_ordered_class_map(self):
        logits = [[9, 0], [9, 0], [0, 9]]
        self.assertEqual(native_ce_macro(logits, ["b", "a"], ["b", "b", "a"])["mean"], 1)
        self.assertEqual(native_ce_macro(logits, ["a", "b"], ["b", "b", "a"])["mean"], 0)

    def test_missing_bound_map_fails(self):
        with self.assertRaises(ValueError):
            native_ce_from_cache({"logits_id_ce_full": np.eye(2)}, ["a", "b"])

    def test_invalid_shapes_or_labels_fail(self):
        for logits, classes, labels in [([[1, 0]], ["a"], ["a"]),
                                         ([[1, 0]], ["a", "b"], ["c"]),
                                         ([[1, 0]], ["a", "a"], ["a"]),
                                         ([[float("nan"), 0]], ["a", "b"], ["a"])]:
            with self.assertRaises(ValueError):
                native_ce_macro(logits, classes, labels)

    def test_same_length_reordered_queries_fail(self):
        cache = {"logits_id_ce_full": np.eye(2), "ce_species_list": ["a", "b"],
                 "ce_full_checkpoint_sha256": np.asarray("a" * 64),
                 "labels_id_ce_full": ["a", "b"], "paths_id": ["one", "two"],
                 "paths_id_ce_full": ["two", "one"]}
        with self.assertRaisesRegex(ValueError, "paths"):
            native_ce_from_cache(cache, ["a", "b"])
        cache["paths_id_ce_full"] = ["one", "two"]
        self.assertEqual(native_ce_from_cache(cache, ["a", "b"])["mean"], 1)
        with self.assertRaisesRegex(ValueError, "labels"):
            native_ce_from_cache(cache, ["b", "a"])


if __name__ == "__main__":
    unittest.main()
