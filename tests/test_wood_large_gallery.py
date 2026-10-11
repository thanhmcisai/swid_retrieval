"""Tests for candidate identity, scan-balanced references and locked selection."""

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec("torch") is not None,
                     "PyTorch unavailable")
class CandidateMethodsTest(unittest.TestCase):
    def test_scores_and_union(self):
        import torch
        from swid_retrieval.wood_large_gallery_methods import (
            candidate_recall, candidate_union, class_scores)

        query = torch.eye(3)
        refs = torch.eye(3).repeat_interleave(2, dim=0)
        labels = torch.arange(3).repeat_interleave(2)
        proto, nearest = class_scores(query, refs, labels, chunk=2)
        self.assertEqual(tuple(proto.shape), (3, 3))
        self.assertTrue(torch.equal(proto.argmax(1), torch.arange(3)))
        self.assertTrue(torch.equal(nearest.argmax(1), torch.arange(3)))
        union, _ = candidate_union((proto, nearest), 2)
        self.assertEqual(tuple(union.shape), (3, 2))
        self.assertEqual(candidate_recall(union, torch.arange(3)), 1.0)

    def test_reranker_excludes_non_candidates(self):
        import torch
        from swid_retrieval.wood_correspondence_method import WoodCorrespondence
        from swid_retrieval.wood_large_gallery_methods import rerank_candidates

        model = WoodCorrespondence(dimension=4, token_dim=2)
        queries = torch.eye(4)[:2]
        refs = torch.eye(4)
        tokens = torch.ones(4, 16, 4)
        selected = torch.tensor([[1, 2], [0, 3]])
        result = rerank_candidates(model, queries, tokens[:2], refs, tokens,
                                   torch.arange(4), ["a", "b", "c", "d"],
                                   selected, queries @ refs.T, mode="global")
        self.assertTrue(torch.isneginf(result[0, [0, 3]]).all())
        self.assertTrue(torch.isneginf(result[1, [1, 2]]).all())
        self.assertTrue(torch.isfinite(result[0, selected[0]]).all())


@unittest.skipUnless(all(importlib.util.find_spec(name) is not None for name in
                         ("torch", "numpy", "pandas", "cv2", "albumentations")),
                     "Image-study dependencies unavailable")
class LargeGalleryProtocolTest(unittest.TestCase):
    def test_large_gallery_image_read_retries_without_skipping(self):
        from swid_retrieval import wood_large_gallery as large

        class TransientLoader:
            def __init__(self):
                self.calls = 0

            def load(self, path):
                self.calls += 1
                if self.calls < 3:
                    raise RuntimeError(f"Failed to read image: {path}")
                return path

        original = TransientLoader()
        with patch.object(large.time, "sleep") as sleep:
            self.assertEqual(large._RetryImageLoader(original).load("image.jpg"),
                             "image.jpg")
        self.assertEqual(original.calls, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_large_gallery_image_read_failure_preserves_cohort(self):
        from swid_retrieval import wood_large_gallery as large

        class MissingLoader:
            def load(self, path):
                raise RuntimeError(f"Failed to read image: {path}")

            def _cache_path(self, path):
                return Path(path + ".cache")

        with tempfile.TemporaryDirectory() as folder:
            missing = str(Path(folder) / "missing.jpg")
            with patch.object(large.time, "sleep"):
                with self.assertRaisesRegex(RuntimeError, "source_exists=False"):
                    large._RetryImageLoader(MissingLoader()).load(missing)

    def test_settings_survive_json_round_trip(self):
        from swid_retrieval import wood_large_gallery as large
        settings = large._settings()
        self.assertEqual(json.loads(json.dumps(settings)), settings)

    def test_upstream_seed_is_explicit(self):
        from swid_retrieval import wood_large_gallery as large
        with patch.dict(os.environ, {"WOOD_LARGE_SOURCE_SEED": "43"}):
            self.assertEqual(large._upstream_seed(42), 43)
        with patch.dict(os.environ, {"WOOD_LARGE_SOURCE_SEED": "match"}):
            self.assertEqual(large._upstream_seed(42), 42)

    def test_mrr_excludes_missing_shortlist_targets(self):
        import torch
        from swid_retrieval import wood_large_gallery as large
        scores = torch.tensor([[0.9, 0.8, -torch.inf], [0.7, 0.8, 0.6]])
        targets = torch.tensor([2, 0])
        self.assertAlmostEqual(large._macro_mrr(scores, targets), 0.25)

    def test_bank_loss_backpropagates_from_inference_bank(self):
        import torch
        from swid_retrieval import wood_large_gallery as large
        from swid_retrieval.wood_correspondence_method import WoodCorrespondence

        model = WoodCorrespondence(dimension=8, token_dim=4)
        query = torch.randn(4, 8, requires_grad=True)
        references = torch.randn(8, 8, requires_grad=True)
        labels = torch.arange(4).repeat_interleave(2).tolist()
        species = [f"species_{i}" for i in range(4)]
        bank_labels = species + [f"negative_{i}" for i in range(12)]
        with torch.inference_mode():
            bank = torch.randn(len(bank_labels), 8)
        for arm in ("hard_bank", "random_bank"):
            loss = large._bank_loss(model, query, references, labels, species,
                                    bank_labels, bank, 8, arm)
            loss.backward(retain_graph=True)
            self.assertTrue(torch.isfinite(query.grad).all())
            self.assertGreater(float(query.grad.norm()), 0)
            query.grad.zero_()

    def test_episode_and_candidate_smoke(self):
        import torch
        from swid_retrieval import wood_large_gallery as large
        large._smoke(torch.device("cpu"))

    def test_balanced_gallery_caps_each_species(self):
        from swid_retrieval import wood_large_gallery as large
        items = [(f"/scale_256/patch_{j}_from_Tw{i*10+s:05d}.jpg", f"species_{i}")
                 for i in range(3) for s in range(3) for j in range(4)]
        balanced = large._balanced_gallery(items, 5)
        self.assertEqual(len(balanced), 15)
        for i in range(3):
            paths = [path for path, label in balanced if label == f"species_{i}"]
            self.assertEqual(len(paths), 5)
            self.assertEqual(len({large.study.source_scan_id(p) for p in paths}), 3)

    def test_selection_lock_is_immutable(self):
        import pandas as pd
        from swid_retrieval import wood_large_gallery as large
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            pd.DataFrame([
                {"arm": "qkv_base", "seed": 43, "selection_score": 0.3,
                 "checkpoint_sha256": "q"},
                {"arm": "hard_bank", "seed": 43, "selection_score": 0.4,
                 "checkpoint_sha256": "h"},
            ]).to_csv(out / "selection_candidates.csv", index=False)
            selected = large._select(out, ["qkv_base", "hard_bank"], [43])
            self.assertEqual(selected["selected_arm"], "hard_bank")
            pd.DataFrame([
                {"arm": "qkv_base", "seed": 43, "selection_score": 0.5,
                 "checkpoint_sha256": "q"},
                {"arm": "hard_bank", "seed": 43, "selection_score": 0.4,
                 "checkpoint_sha256": "h"},
            ]).to_csv(out / "selection_candidates.csv", index=False)
            with self.assertRaises(ValueError):
                large._select(out, ["qkv_base", "hard_bank"], [43])
