"""Reference-budget selection and scorer invariants."""

import importlib.util
import unittest


@unittest.skipUnless(all(importlib.util.find_spec(name) is not None
                         for name in ("torch", "numpy", "pandas")),
                     "Score dependencies unavailable")
class ReferenceBudgetTest(unittest.TestCase):
    @staticmethod
    def _items():
        return [(f"/tmp/scale_{scale}/patch_{scale}_{i}_from_Tw{scan}.jpg", label)
                for label in ("a", "b")
                for scan in (1, 2)
                for scale in (256, 512)
                for i in range(2)]

    def test_selection_is_deterministic_nested_and_covers_classes(self):
        from swid_retrieval.wood_reference_budget import select_indices

        items = self._items()
        for policy in ("scan_hash", "scan_scale"):
            small = select_indices(items, 1, policy)
            medium = select_indices(items, 3, policy)
            large = select_indices(items, 5, policy)
            self.assertEqual(len(small), 2)
            self.assertEqual(len(medium), 6)
            self.assertEqual(len(large), 10)
            self.assertEqual(small, select_indices(items, 1, policy))
            self.assertEqual(small, [medium[0], medium[3]])
            self.assertEqual(medium[:3], large[:3])
            self.assertEqual(medium[3:], large[5:8])
            self.assertEqual({items[i][1] for i in small}, {"a", "b"})

    def test_scale_policy_covers_scales_and_scans(self):
        from swid_retrieval.wood_reference_budget import (
            _image_scale, _source_scan_id, select_indices)

        items = self._items()
        picked = select_indices(items, 4, "scan_scale")[:4]
        self.assertEqual({_image_scale(items[i][0]) for i in picked}, {256, 512})
        self.assertEqual({_source_scan_id(items[i][0]) for i in picked},
                         {"tw1", "tw2"})

    def test_scale_scoring_and_prototype_identity_at_one_reference(self):
        import torch
        from swid_retrieval.wood_reference_budget import score_references

        queries = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        references = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        labels = torch.tensor([0, 1])
        scores = score_references(queries, references, labels, [256, 512], 2)
        for name in ("scale_top2", "prototype_scale_half"):
            torch.testing.assert_close(scores[name], scores["prototype"])

    def test_scale_prototypes_do_not_drop_sparse_class(self):
        import torch
        from swid_retrieval.wood_reference_budget import score_references

        queries = torch.tensor([[1.0, 0.0]])
        references = torch.tensor([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]])
        labels = torch.tensor([0, 0, 1])
        scores = score_references(queries, references, labels, [256, 512, 256], 2)
        self.assertEqual(scores["prototype"].shape, (1, 2))
        self.assertTrue(all(torch.isfinite(value).all() for value in scores.values()))
        torch.testing.assert_close(scores["scale_top2"][:, 1],
                                   scores["prototype"][:, 1])
        torch.testing.assert_close(scores["prototype_scale_half"],
                                   (scores["prototype"] + scores["scale_top2"]) / 2)

    def test_invalid_reference_budget_rejected(self):
        from swid_retrieval.wood_reference_budget import select_indices

        with self.assertRaises(ValueError):
            select_indices(self._items(), 0, "scan_hash")
        with self.assertRaises(ValueError):
            select_indices(self._items(), 3, "unknown")

    def test_query_scan_is_excluded_only_for_its_target_species(self):
        from swid_retrieval.wood_reference_budget import _reference_pool

        manifest = {"meta-train": [
            ("/tmp/scale_256/patch_256_0_from_Tw1.jpg", "b")],
            "meta-val": [
                ("/tmp/scale_256/patch_256_0_from_Tw1.jpg", "a"),
                ("/tmp/scale_512/patch_512_1_from_Tw1.jpg", "a"),
                ("/tmp/scale_256/patch_256_2_from_Tw2.jpg", "a")]}
        queries = [("/tmp/scale_256/patch_256_0_from_Tw1.jpg", "a")]
        result = _reference_pool(manifest, queries)
        self.assertEqual(len(result), 2)
        self.assertEqual({label for _, label in result}, {"a", "b"})
        self.assertEqual([path for path, label in result if label == "a"],
                         ["/tmp/scale_256/patch_256_2_from_Tw2.jpg"])

    def test_recipe_selection_uses_both_validation_folds(self):
        import pandas as pd
        from swid_retrieval.wood_reference_budget import (
            BUDGETS, POLICIES, SCORERS, _winning_recipe)

        rows = [{"fold": fold, "policy": policy, "budget": budget,
                 "scorer": scorer, "macro_r1": 0.1, "macro_r5": 0.2,
                 "macro_recall128": 0.3}
                for fold in (0, 1) for policy in POLICIES
                for budget in BUDGETS for scorer in SCORERS]
        for row in rows:
            if (row["policy"], row["budget"], row["scorer"]) == (
                    "scan_scale", 10, "scale_top2"):
                row["macro_r1"] = 0.4 if row["fold"] == 0 else 0.2
        selected = _winning_recipe(pd.DataFrame(rows)).iloc[0]
        self.assertEqual((selected.policy, selected.budget, selected.scorer),
                         ("scan_scale", 10, "scale_top2"))
        self.assertAlmostEqual(selected.macro_r1, 0.3)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            _winning_recipe(pd.DataFrame(rows[:-1]))
