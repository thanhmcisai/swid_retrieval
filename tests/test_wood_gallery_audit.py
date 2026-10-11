"""Small invariants for the locked full-gallery follow-up audit."""

import importlib.util
import hashlib
import json
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(all(importlib.util.find_spec(name) is not None
                         for name in ("torch", "numpy")),
                     "Score dependencies unavailable")
class GalleryAuditTest(unittest.TestCase):
    def test_residual_preserves_full_scores_outside_shortlist(self):
        import torch
        from swid_retrieval.wood_gallery_audit import _blend_local

        full = torch.tensor([[0.8, 0.6, 0.4, 0.2]])
        capped = torch.tensor([[0.7, 0.3, 0.1, 0.2]])
        local = torch.tensor([[-torch.inf, 0.5, -torch.inf, 0.1]])
        shortlist = torch.tensor([[1, 3]])
        result = _blend_local(full, capped, local, shortlist, 0.25)
        torch.testing.assert_close(result, torch.tensor([[0.8, 0.65, 0.4, 0.175]]))
        torch.testing.assert_close(_blend_local(full, capped, local, shortlist, 0), full)
        with self.assertRaises(ValueError):
            _blend_local(full, capped, local, shortlist, 1.1)

    def test_scan_cap_matches_first_distinct_gallery_references(self):
        from swid_retrieval.wood_gallery_audit import (
            _first_distinct_scans, _reference_rows)

        items = [("a0", "a"), ("a1", "a"), ("a2", "a"),
                 ("a3", "a"), ("b0", "b"), ("b1", "b")]
        scan = {"a0": "s1", "a1": "s1", "a2": "s2", "a3": "s3",
                "b0": "s4", "b1": "s4"}
        self.assertEqual(_first_distinct_scans(
            items, [0, 0, 0, 0, 1, 1], scan.__getitem__, 2), [0, 2, 4])
        chosen = _reference_rows(items, [items[i] for i in (0, 2, 3, 4, 5)],
                                 [0, 1, 3], lambda path: 256 if path.endswith("0")
                                 else 512, scan.__getitem__, {"a"})
        self.assertEqual(chosen[0]["full_images"], 4)
        self.assertEqual(chosen[0]["capped3_scans"], 2)
        self.assertEqual(chosen[0]["is_public_id_species"], 1)
        self.assertEqual(chosen[1]["capped3_images"], 1)

    def test_incomplete_or_wrong_feature_cache_is_rejected(self):
        import numpy as np
        from swid_retrieval.wood_gallery_audit import _verified_cache

        items = [("path.jpg", "species")]
        checkpoint = "checkpoint"
        hash_bytes = lambda value: hashlib.sha256(value).hexdigest()
        signature = hash_bytes(json.dumps({
            "checkpoint": checkpoint, "items": items, "tokens": False,
            "image_size": 224}, separators=(",", ":")).encode())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "features.npy"
            np.save(path, np.zeros((1, 512), dtype=np.float32))
            cursor = path.with_suffix(".progress.json")
            cursor.write_text(json.dumps({"signature": signature, "offset": 0}))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                _verified_cache(path, checkpoint, items, False, hash_bytes)
            cursor.write_text(json.dumps({"signature": signature, "offset": 1}))
            result, tokens = _verified_cache(path, checkpoint, items, False,
                                             hash_bytes)
            self.assertEqual(result.shape, (1, 512))
            self.assertIsNone(tokens)
            with self.assertRaisesRegex(ValueError, "Unverified"):
                _verified_cache(path, "changed", items, False, hash_bytes)

    def test_shortlist_metrics_and_restricted_score(self):
        import torch
        from swid_retrieval.wood_gallery_audit import _metrics, _restricted

        scores = torch.tensor([[0.9, 0.8, 0.1, 0.0, -0.1],
                               [0.9, 0.0, 0.7, 0.3, 0.2]])
        truth = torch.tensor([1, 2])
        candidates = torch.tensor([[1, 2], [0, 2]])
        restricted = _restricted(scores, candidates)
        self.assertTrue(torch.isneginf(restricted[0, [0, 3, 4]]).all())
        metrics, correct = _metrics(restricted, truth, candidates)
        self.assertEqual(metrics["candidate_macro_recall"], 1.0)
        self.assertEqual(metrics["macro_r1"], 0.5)
        self.assertEqual(correct.tolist(), [True, False])
