"""Focused checks for all-class wood correspondence scoring."""

import importlib.util
import io
import os
import runpy
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch unavailable")
class WoodCorrespondenceTest(unittest.TestCase):
    def setUp(self):
        import torch
        from swid_retrieval.wood_correspondence_method import (
            WoodCorrespondence, class_prototypes, scan_evidence)
        self.torch = torch
        self.model = WoodCorrespondence
        self.prototypes = class_prototypes
        self.pool = scan_evidence

    def _inputs(self, k=2):
        torch = self.torch
        torch.manual_seed(13)
        queries = torch.nn.functional.normalize(torch.randn(4, 16), dim=-1)
        references = torch.nn.functional.normalize(torch.randn(4 * k, 16), dim=-1)
        query_tokens = torch.nn.functional.normalize(torch.randn(4, 3, 16), dim=-1)
        reference_tokens = torch.nn.functional.normalize(torch.randn(4 * k, 3, 16), dim=-1)
        labels = torch.arange(4).repeat_interleave(k)
        scans = [f"scan_{i}_{j}" for i in range(4) for j in range(k)]
        return queries, query_tokens, references, reference_tokens, labels, scans

    def test_global_matches_normalized_prototypes_and_class_order(self):
        torch = self.torch
        q, qt, r, rt, labels, scans = self._inputs()
        labels = torch.tensor([2, 2, 0, 0, 3, 3, 1, 1])
        model = self.model(dimension=16, token_dim=8)
        score, classes = model(q, qt, r, rt, labels, scans, mode="global")
        prototypes, expected_classes, _ = self.prototypes(r, labels)
        self.assertTrue(torch.equal(classes, expected_classes))
        self.assertTrue(torch.allclose(score, q @ prototypes.T, atol=1e-6))

    def test_scan_consensus_uses_distinct_scans(self):
        torch = self.torch
        pair = torch.tensor([[0.9, 0.8, 0.2, 0.5]])
        labels = torch.tensor([0, 0, 0, 1])
        scans = ["a", "a", "b", "c"]
        consensus = self.pool(pair, labels, scans, 2, True)
        nearest = self.pool(pair, labels, scans, 2, False)
        self.assertAlmostEqual(consensus[0, 0].item(), 0.55, places=6)
        self.assertAlmostEqual(nearest[0, 0].item(), 0.9, places=6)
        self.assertAlmostEqual(consensus[0, 1].item(), 0.5, places=6)

    def test_all_class_and_full_shortlist_agree(self):
        torch = self.torch
        model = self.model(dimension=16, token_dim=8)
        for k in (1, 2):
            inputs = self._inputs(k)
            full, classes = model(*inputs, mode="qkv")
            equivalent, second_classes = model(*inputs, mode="qkv", top_classes=4)
            self.assertTrue(torch.equal(classes, second_classes))
            self.assertTrue(torch.allclose(full, equivalent, atol=1e-6))
            self.assertTrue(torch.isfinite(full).all())

    def test_shortlist_leaves_non_candidates_at_global_score(self):
        torch = self.torch
        model = self.model(dimension=16, token_dim=8)
        inputs = self._inputs()
        global_score, _ = model(*inputs, mode="global")
        shortcut, _ = model(*inputs, mode="qkv", top_classes=2)
        candidates = global_score.topk(2, dim=1).indices
        mask = torch.zeros_like(global_score, dtype=torch.bool).scatter_(1, candidates, True)
        self.assertTrue(torch.allclose(shortcut[~mask], global_score[~mask]))
        self.assertTrue(torch.isfinite(shortcut).all())

    def test_qkv_training_updates_attention_parameters(self):
        torch = self.torch
        model = self.model(dimension=16, token_dim=8)
        inputs = self._inputs()
        before = model.query_key.weight.detach().clone()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        score, classes = model(*inputs, mode="qkv", query_chunk=2, reference_chunk=3)
        self.assertTrue(torch.equal(classes, torch.arange(4)))
        loss = torch.nn.functional.cross_entropy(score / 0.07, torch.arange(4))
        loss.backward()
        self.assertTrue(torch.isfinite(model.query_key.weight.grad).all())
        optimizer.step()
        self.assertFalse(torch.equal(before, model.query_key.weight.detach()))


@unittest.skipUnless(all(importlib.util.find_spec(name) is not None for name in
                         ("torch", "numpy", "pandas", "cv2", "albumentations")),
                     "Gallery experiment dependencies unavailable")
class WoodCorrespondenceExperimentTest(unittest.TestCase):
    def test_runpy_dispatches_correspondence_only(self):
        from swid_retrieval import wood_correspondence_experiment as experiment
        flags = {name: "0" for name in (
            "RUN_WOOD_EVIDENCE_STUDY", "RUN_GALLERY_STUDY",
            "RUN_GALLERY_DIAGNOSTICS", "RUN_REPAIR_PUBLIC_ROWS",
            "RUN_FINAL_SCURD_RETRAIN", "RUN_FINAL_COLAB_AUDIT")}
        flags["RUN_WOOD_CORRESPONDENCE_STUDY"] = "1"
        with mock.patch.dict(os.environ, flags, clear=False):
            with mock.patch.object(experiment, "run") as pilot:
                with redirect_stdout(io.StringIO()):
                    runpy.run_module("swid_retrieval.run_overnight", run_name="__main__")
                pilot.assert_called_once_with()

    def test_recipe_and_meta_test_gate(self):
        from swid_retrieval import wood_correspondence_experiment as experiment
        self.assertEqual(experiment.recipe("global_only", "head")["score_mode"], "global")
        self.assertFalse(experiment.recipe("no_consensus", "joint")["consensus"])
        with self.assertRaises(ValueError):
            experiment.recipe("unknown", "head")

    def test_scan_disjoint_episode(self):
        import numpy as np
        from swid_retrieval import wood_correspondence_experiment as experiment
        items = [(f"/scale_256/patch_from_Tw{i * 3 + s:04d}.jpg", f"species_{i:03d}")
                 for i in range(70) for s in range(3)]
        by_class = experiment.evidence._episode_layout(items)
        hard = {label: [] for label in by_class}
        refs, queries, labels, scans = experiment._episode(
            by_class, items, np.random.default_rng(42), 16, 2, hard, True)
        self.assertEqual(len(refs), 32)
        self.assertEqual(len(queries), 16)
        for class_index, query_index in enumerate(queries):
            query_scan = experiment.study.source_scan_id(items[query_index][0])
            self.assertTrue(all(scan != query_scan for scan, label in zip(scans, labels)
                                if label == class_index))

    def test_joint_step_reaches_backbone(self):
        import torch
        from swid_retrieval import wood_correspondence_experiment as experiment

        class Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 8, 1)

            def forward_features(self, images):
                patches = self.conv(images).flatten(2).transpose(1, 2)
                return {"x_norm_clstoken": patches.mean(dim=1),
                        "x_norm_patchtokens": patches}

        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = Backbone()
                self.projection = torch.nn.Linear(8, 512)

            def project(self, features):
                return torch.nn.functional.normalize(self.projection(features), dim=-1)

        torch.manual_seed(5)
        encoder = Encoder()
        model = experiment.WoodCorrespondence()
        optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(model.parameters()),
                                      lr=0.001)
        before = encoder.backbone.conv.weight.detach().clone()
        images = torch.randn(6, 3, 4, 4)
        config = experiment.recipe("qkv", "joint")
        config["image_microbatch"] = 2
        loss, accuracy = experiment._joint_step(
            model, encoder, images, 4, [0, 0, 1, 1], ["a", "b", "c", "d"],
            config, optimizer, torch.device("cpu"))
        self.assertTrue(0 <= accuracy <= 1)
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertFalse(torch.equal(before, encoder.backbone.conv.weight.detach()))

    def test_query_reports_are_paired_and_stratified(self):
        import numpy as np
        import torch
        from swid_retrieval import wood_correspondence_experiment as experiment
        refs = [(f"/scale_256/patch_from_Tw{i:04d}.jpg", f"species_{i:03d}")
                for i in range(4)]
        queries = [(f"/scale_256/patch_from_Tw{i + 10:04d}.jpg", f"species_{i:03d}")
                   for i in range(4)]
        rng = np.random.default_rng(42)
        rg = rng.normal(size=(4, 512)).astype("float32")
        qg = rg.copy()
        rt = rng.normal(size=(4, 16, 512)).astype("float32")
        qt = rt.copy()
        features = (refs, queries, {"4x1": (np.arange(4), np.arange(4))},
                    (rg, rt), (qg, qt))
        model = experiment.WoodCorrespondence()
        config = experiment.recipe("qkv", "head")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with mock.patch.object(experiment, "_features_for_split", return_value=features):
                frame, summary, timing = experiment._evaluate(
                    out, out, {}, mock.Mock(), {}, "basehash", model,
                    config, torch.device("cpu"), folds=(0,),
                    modes=("global", "fixed_local", "global_adapt", "qkv_all"))
            self.assertEqual(len(frame), 16)
            self.assertEqual(set(summary["mode"]),
                             {"global", "fixed_local", "global_adapt", "qkv_all"})
            self.assertTrue(frame["same_genus_error"].isin([0, 1]).all())
            experiment._save_report(out, "synthetic", frame, summary, timing, {"test": True})
            for suffix in ("queries", "summary", "folds", "paired", "scale", "ref_scans", "timing"):
                self.assertTrue((out / f"synthetic_{suffix}.csv").is_file())
            paired = experiment.pd.read_csv(out / "synthetic_paired.csv")
            self.assertIn("fixed_local", set(paired["baseline"]))


if __name__ == "__main__":
    unittest.main()
