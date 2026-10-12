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

    def test_latest_validation_reuses_only_matching_checkpoint(self):
        import torch
        from swid_retrieval import wood_coverage_study as coverage
        from swid_retrieval import gallery_experiment as study

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            latest = root / "latest.pt"
            latest.write_bytes(b"epoch3")
            folder = root / "checkpoint_audit" / "latest"
            folder.mkdir(parents=True)
            protocol = {
                "checkpoint_sha256": study._hash_file(latest),
                "folds": [0, 1], "budgets": [10], "widths": [128],
                "reference_policy": "scan_scale", "shortlist": "raw_prototype",
                "meta_val_only": True}
            (folder / "protocol.json").write_text(json.dumps(protocol))
            (folder / "summary.csv").write_text("summary")
            (folder / "per_query.csv").write_text("queries")
            with patch.object(coverage.study, "validation_items",
                              side_effect=AssertionError("inference reran")):
                self.assertEqual(coverage._validate_latest(
                    root, root, {}, None, None, {}, {}, latest,
                    torch.device("cpu")), folder)
            latest.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "protocol changed"):
                coverage._validate_latest(root, root, {}, None, None,
                                          {}, {}, latest, torch.device("cpu"))

    def test_latest_validation_writes_matched_query_schema(self):
        import pandas as pd
        import torch
        from swid_retrieval import wood_coverage_study as coverage

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            latest = root / "latest.pt"
            latest.write_bytes(b"epoch3")
            refs = [(f"/refs/{i}.jpg", f"species_{i:03d}") for i in range(637)]
            queries = [(f"/queries/{fold}_{i}.jpg", f"species_{i // 5:03d}")
                       for fold in (0, 1) for i in range(285)]
            embeddings = torch.eye(637)
            def reference_data(_encoder, items, current, *_args):
                labels = [int(label.split("_")[1]) for _, label in current]
                return (embeddings, embeddings[:, :1, None],
                        embeddings[labels], embeddings[labels, :1, None])
            with patch.object(coverage.study, "validation_items",
                              side_effect=[(None, queries[:285]),
                                           (None, queries[285:])]), \
                 patch.object(coverage.budget, "_reference_pool", return_value=refs), \
                 patch.object(coverage.budget, "select_indices",
                              return_value=list(range(637))), \
                 patch.object(coverage.hard, "_reference_data",
                              side_effect=reference_data), \
                 patch.object(coverage.corr, "_transformed", side_effect=lambda _m, x: x), \
                 patch.object(coverage.ml, "rerank_all_references",
                              side_effect=lambda _m, _q, _qt, _r, _rt, _l,
                              _s, _c, adapted: adapted), \
                 patch.object(coverage.study, "source_scan_id", side_effect=lambda p: p), \
                 patch.object(coverage.study, "image_scale", return_value=256):
                folder = coverage._validate_latest(
                    root, root, {}, None, None, {},
                    {"image_batch": 16, "workers": 0}, latest,
                    torch.device("cpu"))
            summary = pd.read_csv(folder / "summary.csv")
            per_query = pd.read_csv(folder / "per_query.csv")
            self.assertEqual(len(summary), 6)
            self.assertEqual(len(per_query), 1710)
            self.assertEqual(set(summary.method),
                             {"raw_prototype", "adapted_prototype", "qkv_all_refs"})
            self.assertTrue((summary.macro_r1 == 1.0).all())
            self.assertEqual(set(per_query.fold), {0, 1})


if __name__ == "__main__":
    unittest.main()
