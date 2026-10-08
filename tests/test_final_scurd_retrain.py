import io
import json
import os
import runpy
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

from swid_retrieval.final_scurd_retrain import _meta_path, _nearest_fingerprint, validate_meta


class FinalScurdRetrainTest(unittest.TestCase):
    def test_complete_support_logits_match_legacy_values_and_gradients(self):
        import importlib
        import torch

        engine = importlib.import_module(
            "swid_retrieval._engines.variance_retrieval_evidence_colab")
        generator = torch.Generator().manual_seed(17)
        query = torch.randn(7, 5, generator=generator)
        support = torch.randn(9, 5, generator=generator)
        targets = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
        old_query = query.clone().requires_grad_()
        new_query = query.clone().requires_grad_()
        old_support = support.clone().requires_grad_()
        new_support = support.clone().requires_grad_()

        old_logits = engine.urd_logits(old_query, old_support, targets, 3, 0.07)
        new_logits = engine.urd_logits(
            new_query, new_support, targets, 3, 0.07, support_complete=True)
        torch.testing.assert_close(old_logits, new_logits, rtol=0, atol=0)
        old_logits.sum().backward()
        new_logits.sum().backward()
        torch.testing.assert_close(old_query.grad, new_query.grad, rtol=0, atol=0)
        torch.testing.assert_close(old_support.grad, new_support.grad, rtol=0, atol=0)

    def test_epoch_metric_transfer_preserves_legacy_means(self):
        import torch

        values = np.random.RandomState(23).randn(25, 4).astype(np.float32)
        metrics = [torch.tensor(row) for row in values]
        batched = torch.stack(metrics).cpu().numpy().astype(np.float64)
        for column in range(4):
            legacy = float(np.mean([float(row[column].item()) for row in metrics]))
            self.assertEqual(float(np.mean(batched[:, column])), legacy)

    def test_cached_episode_pool_preserves_legacy_rng_sequence(self):
        import importlib

        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"ROOT_PATH": tmp, "RESULTS_DIR": tmp,
                                      "OUT_DIR": str(Path(tmp) / "logs")}, clear=False):
                engine = importlib.import_module(
                    "swid_retrieval._engines.variance_retrieval_evidence_colab")
        labels = np.repeat(np.array(["a", "b", "c", "d", "e"]), [7, 8, 6, 3, 9])

        def legacy_episode(rng, n_way, k_support, q_query):
            by_sp = {sp: np.where(labels == sp)[0] for sp in np.unique(labels)}
            valid = [sp for sp, idx in by_sp.items() if len(idx) >= k_support + q_query]
            if len(valid) < n_way:
                valid = [sp for sp, idx in by_sp.items() if len(idx) >= 2]
            chosen = rng.choice(valid, size=min(n_way, len(valid)), replace=False)
            support, query, ys, yq = [], [], [], []
            for ci, sp in enumerate(chosen):
                idx = by_sp[sp].copy()
                rng.shuffle(idx)
                n_s = min(k_support, max(1, len(idx) // 2))
                n_q = min(q_query, max(1, len(idx) - n_s))
                support.extend(idx[:n_s].tolist())
                query.extend(idx[n_s:n_s + n_q].tolist())
                ys.extend([ci] * n_s)
                yq.extend([ci] * n_q)
            return tuple(np.asarray(x, dtype=np.int64) for x in (support, query, ys, yq))

        for n_way, k_support, q_query in ((3, 2, 2), (5, 4, 3)):
            with self.subTest(n_way=n_way):
                old_rng = np.random.RandomState(42)
                fast_rng = np.random.RandomState(42)
                pool = engine.scurd_episode_pool(labels, k_support, q_query, n_way)
                for _ in range(30):
                    expected = legacy_episode(old_rng, n_way, k_support, q_query)
                    actual = engine.scurd_episode_indices(
                        labels, n_way, k_support, q_query, fast_rng, pool=pool)
                    for left, right in zip(expected, actual):
                        np.testing.assert_array_equal(left, right)

    def test_nearest_fingerprint_detects_row_permutation(self):
        embs = np.array([[1, 0], [0, 1], [-1, 0]], dtype=np.float32)
        match = _nearest_fingerprint(np.array([0, 1], dtype=np.float32),
                                     embs, ["a", "b", "c"])
        self.assertEqual(match["index"], 1)
        self.assertEqual(match["label"], "b")
        self.assertAlmostEqual(match["cosine"], 1.0)

    def _fixture(self, root):
        manifest = {
            "meta-train": [["a.jpg", "A species"], ["b.jpg", "B species"]],
            "meta-val": [["c.jpg", "C species"]],
            "meta-test": [["d.jpg", "D species"]],
        }
        manifest_path = root / "swi_manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        emb = np.zeros((4, 768), dtype=np.float32)
        emb[0, 0] = emb[1, 1] = emb[2, 2] = emb[3, 3] = 1.0
        cache_path = root / "cache.npz"
        np.savez_compressed(cache_path, embs_swi_dinov2=emb,
                            labels_swi_dinov2=np.array(["a_species", "b_species",
                                                       "c_species", "d_species"]))
        meta_path = root / "meta.npz"
        np.savez_compressed(meta_path, train_weak=emb[:2], train_strong=emb[:2],
                            train_labels=np.array(["a_species", "b_species"]),
                            val_weak=emb[2:3], val_labels=np.array(["c_species"]))
        return manifest_path, cache_path, meta_path

    def test_meta_cache_alignment_and_feature_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest, cache, meta = self._fixture(Path(tmp))
            report = validate_meta(meta, manifest, cache)
            self.assertEqual(report["train_weak_vs_v5_max_abs_diff"], 0.0)
            self.assertEqual(report["meta_train_images"], 2)
            np.savez_compressed(meta, train_weak=np.zeros((2, 768), np.float32),
                                train_strong=np.zeros((2, 768), np.float32),
                                train_labels=np.array(["a_species", "b_species"]),
                                val_weak=np.zeros((1, 768), np.float32),
                                val_labels=np.array(["c_species"]))
            with self.assertRaisesRegex(ValueError, "differ from v5"):
                validate_meta(meta, manifest, cache)

    def test_meta_missing_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.dict(os.environ, {"FINAL_SCURD_META_CACHE": str(root / "missing.npz")}):
                with self.assertRaisesRegex(FileNotFoundError, "FINAL_SCURD_META_CACHE"):
                    _meta_path(root, root / "results" / "run")

    def test_overnight_runpy_dispatches_head_only(self):
        from swid_retrieval import final_scurd_retrain
        with patch.dict(os.environ, {"RUN_FINAL_SCURD_RETRAIN": "1",
                                  "RUN_FINAL_COLAB_AUDIT": "0",
                                  "ROOT_PATH": "/tmp",
                                  "FORCE_REBUILD_FULL954": "1"}, clear=False):
            with patch.object(final_scurd_retrain, "run") as target:
                with redirect_stdout(io.StringIO()):
                    runpy.run_module("swid_retrieval.run_overnight", run_name="__main__")
                target.assert_called_once_with()

    def test_seed_training_records_meta_hash_and_rejects_stale_resume(self):
        import importlib
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.dict(os.environ, {"ROOT_PATH": tmp, "RESULTS_DIR": tmp,
                                      "OUT_DIR": str(root / "logs")}, clear=False):
                engine = importlib.import_module("swid_retrieval._engines.variance_retrieval_evidence_colab")
            weak = np.random.RandomState(7).randn(4, 768).astype(np.float32)
            strong = weak.copy()
            labels = np.array(["a", "a", "b", "b"])
            meta = {"train_weak": weak, "train_strong": strong, "train_labels": labels}
            changes = {
                "SCURD_CKPT_DIR": root / "checkpoints", "OUT_DIR": root / "logs",
                "SCURD_META_CACHE": root / "meta.npz", "SCURD_CACHE_VERSION": "vtest",
                "SCURD_TRAIN_EPOCHS": 1, "SCURD_TRAIN_EPISODES": 1,
                "SCURD_N_WAY": 2, "SCURD_K_SUPPORT": 1, "SCURD_Q_QUERY": 1,
                "SCURD_FORCE_RETRAIN_SEEDS": False,
            }
            (root / "logs").mkdir(exist_ok=True)
            with patch.dict(engine.__dict__, changes), patch.object(engine, "load_scurd_meta_cache", return_value=meta):
                with patch.dict(os.environ, {"SCURD_META_CACHE_SHA256": "fixture-hash"}):
                    path = engine.train_one_scurd_seed(42, "cpu")
                    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                    self.assertEqual(checkpoint["meta_cache_sha256"], "fixture-hash")
                    self.assertEqual(engine.train_one_scurd_seed(42, "cpu"), path)
                with patch.dict(os.environ, {"SCURD_META_CACHE_SHA256": "changed-hash"}):
                    with self.assertRaisesRegex(ValueError, "different provenance"):
                        engine.train_one_scurd_seed(42, "cpu")


if __name__ == "__main__":
    unittest.main()
