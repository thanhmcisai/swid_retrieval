"""Tests for candidate identity, scan-balanced references and locked selection."""

import importlib.util
import json
import os
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
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
    def test_feature_cache_flushes_chunks_and_resumes_exact_rows(self):
        import numpy as np
        import torch
        from swid_retrieval import wood_large_gallery as large

        items = [(f"image_{i}.jpg", "species") for i in range(3201)]
        features = torch.arange(3201 * 512, dtype=torch.float32).reshape(3201, 512)

        class Batches:
            def __init__(self, offset):
                self.dataset = SimpleNamespace(loader=None)
                self.offset = offset

            def __iter__(self):
                if self.offset == 0:
                    yield features[:3200], None
                    raise RuntimeError("Interrupted after the first durable chunk")
                yield features[self.offset:], None

        def loader(remaining, cfg):
            return Batches(len(items) - len(remaining))

        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder) / "features.npy"
            with patch.object(large.study, "_loader", side_effect=loader), \
                 patch.object(large.study, "_autocast", return_value=nullcontext()):
                with self.assertRaisesRegex(RuntimeError, "Interrupted"):
                    large._cache_features(torch.nn.Identity(), items,
                                          {"image_batch": 64, "workers": 0},
                                          torch.device("cpu"), cache, "checkpoint", False)
                cursor = json.loads(cache.with_suffix(".progress.json").read_text())
                self.assertEqual(cursor["offset"], 3200)
                output, _ = large._cache_features(
                    torch.nn.Identity(), items, {"image_batch": 128, "workers": 4},
                    torch.device("cpu"), cache, "checkpoint", False)
            np.testing.assert_array_equal(output, features.numpy())

    def test_feature_cache_buffers_reference_tokens(self):
        import numpy as np
        import torch
        from swid_retrieval import wood_large_gallery as large

        items = [(f"image_{i}.jpg", "species") for i in range(3)]
        features = torch.arange(3 * 512, dtype=torch.float32).reshape(3, 512)

        class Batches:
            dataset = SimpleNamespace(loader=None)

            def __iter__(self):
                yield features[:2], None
                yield features[2:], None

        def encode(encoder, images):
            return images, images[:, None, :].expand(-1, 16, -1)

        with tempfile.TemporaryDirectory() as folder:
            with patch.object(large.study, "_loader", return_value=Batches()), \
                 patch.object(large.study, "_autocast", return_value=nullcontext()), \
                 patch.object(large.corr, "_encode_images", side_effect=encode):
                global_emb, tokens = large._cache_features(
                    torch.nn.Identity(), items, {"image_batch": 2, "workers": 0},
                    torch.device("cpu"), Path(folder) / "refs.npy", "checkpoint", True)
            np.testing.assert_array_equal(global_emb, features.numpy())
            np.testing.assert_array_equal(tokens,
                                          features.numpy()[:, None, :].repeat(16, axis=1)
                                          .astype(np.float16))

    def test_completed_gallery_skips_only_verified_artifacts(self):
        from swid_retrieval import wood_large_gallery as large

        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            self.assertFalse(large._completed_gallery(out, "checkpoint", "manifest",
                                                     "id", 0, 6189))
            (out / "protocol.json").write_text(json.dumps({
                "checkpoint_sha256": "checkpoint", "manifest_sha256": "manifest",
                "id_csv_sha256": "id", "query_limit_per_species": 0,
                "n_queries": 6189, "n_stage1_images": 176123, "n_species": 954}))
            for name in ("full954_summary.csv", "full954_queries.csv",
                         "cardinality_curve.csv", "per_source.csv"):
                (out / name).write_text("header\nvalue\n")
            self.assertTrue(large._completed_gallery(out, "checkpoint", "manifest",
                                                    "id", 0, 6189))
            with self.assertRaises(ValueError):
                large._completed_gallery(out, "changed", "manifest", "id", 0, 6189)
            (out / "per_source.csv").unlink()
            with self.assertRaises(RuntimeError):
                large._completed_gallery(out, "checkpoint", "manifest", "id", 0, 6189)

    def test_full_gallery_io_overrides_do_not_change_locked_settings(self):
        from swid_retrieval import wood_large_gallery as large

        settings = {"image_batch": 32, "workers": 2}
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("WOOD_LARGE_EVAL_IMAGE_BATCH", None)
            os.environ.pop("WOOD_LARGE_EVAL_WORKERS", None)
            self.assertEqual(large._evaluation_io(settings), (32, 2))
        with patch.dict(os.environ, {"WOOD_LARGE_EVAL_IMAGE_BATCH": "64",
                                     "WOOD_LARGE_EVAL_WORKERS": "8"}):
            self.assertEqual(large._evaluation_io(settings), (64, 8))
        self.assertEqual(settings, {"image_batch": 32, "workers": 2})
        with patch.dict(os.environ, {"WOOD_LARGE_EVAL_IMAGE_BATCH": "0"}):
            with self.assertRaises(ValueError):
                large._evaluation_io(settings)

    def test_locked_inference_accepts_only_exact_pre_retry_runner(self):
        from swid_retrieval import wood_large_gallery as large

        expected = {"runner_sha256": "current", "source_checkpoint_sha256": "source",
                    "manifest_sha256": "manifest", "method_sha256": "method",
                    "arm": "hard_bank", "seed": 43, "settings": {"episodes": 300}}
        previous = dict(expected, runner_sha256=large._PRE_RETRY_RUNNER_SHA256)
        self.assertTrue(large._inference_provenance_matches(expected, expected))
        self.assertTrue(large._inference_provenance_matches(previous, expected))
        self.assertFalse(large._inference_provenance_matches(
            dict(previous, runner_sha256="other"), expected))
        for field, value in (("source_checkpoint_sha256", "different"),
                             ("manifest_sha256", "different"),
                             ("method_sha256", "different"),
                             ("arm", "random_bank"), ("seed", 42),
                             ("settings", {"episodes": 60})):
            self.assertFalse(large._inference_provenance_matches(
                dict(previous, **{field: value}), expected), field)

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
