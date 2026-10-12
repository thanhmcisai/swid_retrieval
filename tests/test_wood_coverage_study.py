"""Coverage-study protocol and paired-evidence tests."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


@unittest.skipUnless(all(importlib.util.find_spec(name) is not None for name in
                         ("torch", "numpy", "pandas", "cv2", "albumentations")),
                     "Image-study dependencies unavailable")
class CoverageStudyTest(unittest.TestCase):
    def test_real_scan_disjoint_plan_reports_training_exposure(self):
        from swid_retrieval import wood_coverage_study as coverage

        items = [(f"/x/species_{species}/scale_{scale}/"
                  f"patch_{scale}_{patch}_from_Tw{species * 2 + scan}.jpg",
                  f"species_{species}")
                 for species in range(130) for scan, scale in
                 ((0, 256), (1, 512)) for patch in range(5)]
        _, valid = coverage.image_train._layout(items)
        multi = {k: coverage.ml._eligible(items, k)
                 for _, k in coverage.ml.SCHEDULE}
        report = coverage._plan_coverage(items, valid, multi, 43, 6, 1)
        self.assertEqual(report["total_episodes"], 6)
        self.assertGreater(report["unique_source_scans"], 64)
        self.assertGreater(report["unique_query_images"], 100)

    def test_planned_coverage_counts_unique_images_and_queries(self):
        from swid_retrieval import wood_coverage_study as coverage

        items = [(f"/x/species_a/scale_256/patch_{i}_from_Tw{i}.jpg", "species_a")
                 for i in range(5)]
        layouts = [(1, [], [], [], [], [])] * 2
        with patch.object(coverage.ml, "_plan", side_effect=[
                ([[0, 1], [0, 2]], layouts),
                ([[0, 3], [0, 4]], layouts)]):
            result = coverage._plan_coverage(items, {}, {}, 43, 2, 2)
        self.assertEqual(result["total_query_draws"], 4)
        self.assertEqual(result["unique_images"], 5)
        self.assertEqual(result["unique_query_images"], 4)
        self.assertEqual(result["unique_species"], 1)
        self.assertEqual(result["fraction_meta_train_images"], 1.0)
        self.assertEqual([x["cumulative_unique_images"] for x in
                          result["per_epoch"]], [3, 5])

    def test_baseline_requires_same_settings_and_more_episodes(self):
        from swid_retrieval import wood_coverage_study as coverage
        from swid_retrieval import wood_hard_negative as hard
        from swid_retrieval import gallery_experiment as study

        cfg = hard._settings()
        cfg["episodes"] = 600
        cfg["arms"] = [coverage.ARM]
        prior = dict(cfg, episodes=120, arms=list(hard.ARMS))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "train/random16").mkdir(parents=True)
            (out / "preflight.json").write_text(json.dumps({"settings": prior}))
            provenance = {"provenance": {
                "runner_sha256": study._hash_file(hard.__file__),
                "source_sha256": "source", "settings": {"episodes": 120}}}
            (out / "train/random16/epoch_01_provenance.json").write_text(
                json.dumps(provenance))
            with patch.object(hard, "_locked", return_value={}):
                previous, digest = coverage._baseline(out, cfg)
                self.assertEqual(previous["episodes"], 120)
                self.assertEqual(digest, "source")
                with self.assertRaisesRegex(ValueError, "matched setting: seed"):
                    coverage._baseline(out, dict(cfg, seed=44))
                with self.assertRaisesRegex(ValueError, "more episodes"):
                    coverage._baseline(out, dict(cfg, episodes=120))

    def test_paired_evidence_requires_exact_queries_and_checkpoint(self):
        import pandas as pd
        from swid_retrieval import wood_coverage_study as coverage
        from swid_retrieval import gallery_experiment as study

        with tempfile.TemporaryDirectory() as tmp:
            old, out = Path(tmp) / "old", Path(tmp) / "new"
            for folder, better in ((old, False), (out, True)):
                train = folder / "train/random16"
                train.mkdir(parents=True)
                checkpoint = train / "best.pt"
                checkpoint.write_bytes(b"old" if not better else b"new")
                val = folder / "validation/random16"
                val.mkdir(parents=True)
                (val / "protocol.json").write_text(json.dumps({
                    "checkpoint_sha256": study._hash_file(checkpoint),
                    "folds": [0, 1], "budgets": [10], "widths": [128]}))
                pd.DataFrame([{
                    "fold": fold, "budget": 10, "width": 128,
                    "method": "qkv_all_refs", "macro_r1": 0.4 if better else 0.2,
                    "candidate_macro_recall": 0.9, "ms_per_query": 4.0}
                    for fold in (0, 1)]).to_csv(val / "summary.csv", index=False)
                records = []
                for fold in (0, 1):
                    for species in range(57):
                        for query in range(5):
                            records.append({
                                "fold": fold, "query_path": f"/q/{fold}/{species}/{query}",
                                "true_label": f"species_{species}",
                                "budget": 10, "width": 128,
                                "method": "qkv_all_refs",
                                "correct": int(query < (2 if better else 1))})
                pd.DataFrame(records).to_csv(val / "per_query.csv", index=False)
            report = coverage._paired_evidence(old, out, samples=100)
            self.assertAlmostEqual(report["paired_delta_macro_r1"], 0.2)
            self.assertEqual(report["new_correct_old_wrong"], 114)
            self.assertEqual(report["old_correct_new_wrong"], 0)
            self.assertEqual(report["n_queries"], 570)
            with (out / "train/random16/best.pt").open("wb") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "checkpoint mismatch"):
                coverage._paired_evidence(old, out, samples=100)


if __name__ == "__main__":
    unittest.main()
