"""Focused checks for inference-only DINOv2-S gallery diagnostics."""

import importlib.util
import unittest


DEPS = all(importlib.util.find_spec(name) is not None for name in
           ("torch", "numpy", "pandas", "cv2", "albumentations"))


@unittest.skipUnless(DEPS, "Gallery dependencies are unavailable")
class GalleryDiagnosticsTest(unittest.TestCase):
    def setUp(self):
        import numpy as np
        import pandas as pd
        from swid_retrieval import gallery_diagnostics as diagnostics
        self.np, self.pd, self.diagnostics = np, pd, diagnostics

    def test_gallery_plan_preserves_balanced_reference_and_stress_sizes(self):
        labels = [f"genus_{index:03d}" for index in range(57)]
        references = [(f"/scale_256/patch_{label}_{j}_from_Tw{index:03d}.jpg", label)
                      for index, label in enumerate(labels) for j in range(5)]
        queries = [(f"/scale_512/patch_{label}_query_from_Tw{index + 100:03d}.jpg", label)
                   for index, label in enumerate(labels) for _ in range(5)]
        extras = [(f"/scale_256/patch_extra_{index}_from_Tw999.jpg", f"extra_{index:03d}")
                  for index in range(580)]
        plan = self.diagnostics.gallery_plan(references, queries, extras)
        self.assertEqual(set(plan), {"24x5", "57x5", "57x1", "128x1", "256x1", "637x1"})
        for key, expected in (("24x5", 120), ("57x5", 285), ("57x1", 57),
                              ("128x1", 128), ("256x1", 256), ("637x1", 637)):
            self.assertEqual(len(plan[key][0]), expected)
        self.assertEqual(len(plan["24x5"][1]), 120)
        self.assertEqual(len(plan["637x1"][1]), 285)
        self.assertEqual(len(set(plan["637x1"][0])), 637)

    def test_query_ranks_margin_genus_and_scan_guard(self):
        refs = [("/scale_256/patch_acer_alpha_0_from_Tw001.jpg", "acer_alpha"),
                ("/scale_512/patch_acer_beta_0_from_Tw002.jpg", "acer_beta")]
        queries = [("/scale_256/patch_acer_alpha_1_from_Tw003.jpg", "acer_alpha"),
                   ("/scale_512/patch_acer_beta_1_from_Tw004.jpg", "acer_beta")]
        rows = self.diagnostics.query_diagnostics(
            self.np.asarray([[7., 6.], [9., 8.]]),
            self.np.asarray(["acer_alpha", "acer_beta"]), queries, refs,
            variant="prototype_large", seed=43, fold=0, gallery="57x1",
            scorer_mode="prototype", temperature=0.1)
        self.assertEqual([row["correct"] for row in rows], [1, 0])
        self.assertEqual([row["true_class_rank"] for row in rows], [1, 2])
        self.assertEqual([row["same_genus_error"] for row in rows], [0, 1])
        self.assertEqual([row["same_scale_reference_count"] for row in rows], [1, 1])
        self.assertAlmostEqual(rows[0]["margin_cosine"], 0.1)
        with self.assertRaisesRegex(ValueError, "source-scan leakage"):
            self.diagnostics.query_diagnostics(
                self.np.asarray([[1., 0.]]), self.np.asarray(["acer_alpha", "acer_beta"]),
                [refs[0]], refs, variant="prototype_large", seed=43, fold=0,
                gallery="57x1", scorer_mode="prototype", temperature=0.1)

    def test_seed_comparison_requires_paired_queries(self):
        rows = []
        for seed, correct in ((42, 0), (43, 1), (44, 0)):
            rows.append({"variant": "prototype_large", "seed": seed,
                         "scorer_mode": "prototype", "gallery": "57x5", "fold": 0,
                         "query_path": "/scale_256/patch_acer_alpha_from_Tw003.jpg",
                         "true_label": "acer_alpha", "query_scale": 256,
                         "correct": correct, "margin_cosine": 0.1 * correct,
                         "true_class_rank": 2 - correct, "top64_oracle": 1})
        frame = self.pd.DataFrame(rows)
        paired = self.diagnostics.seed43_comparison(frame)
        self.assertEqual(len(paired), 1)
        self.assertEqual(paired.iloc[0]["seed43_correct_delta"], 1)
        boot = self.diagnostics.seed43_species_bootstrap(paired, n_boot=100)
        self.assertEqual(boot.iloc[0]["mean_species_delta"], 1)
        self.assertEqual(boot.iloc[0]["n_species_better"], 1)
        frame.loc[frame["seed"] == 44, "query_path"] = "different.jpg"
        with self.assertRaisesRegex(ValueError, "alignment failed"):
            self.diagnostics.seed43_comparison(frame)


if __name__ == "__main__":
    unittest.main()
