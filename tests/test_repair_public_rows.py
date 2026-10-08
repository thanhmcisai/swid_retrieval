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
import pandas as pd

from swid_retrieval.embeddings.repair_public_rows import (
    _extract_mapping, diagnose_duplicate_rows, rebuild_arrays, validate_correction_cohort,
    resolve_feature_equivalent_duplicates, validate_index_map,
)
from swid_retrieval.audit_support import sha256


class PublicRowRepairTest(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "file_path": ["a.jpg", "b.jpg", "c.jpg"],
            "label": ["A species", "B species", "C species"],
            "source_dataset": ["BD11", "FSDM41", "BD11"],
        })
        self.old_labels = np.array(["c_species", "old_fsdm_name", "a_species", "unused"])
        self.indices = np.array([2, 1, 0])
        self.scores = np.array([1.0, 0.99999, 1.0])
        self.runner_up = np.array([0.8, 0.7, 0.6])

    def test_correction_count_is_not_total_fsdm_count(self):
        columns = ["file_path", "label", "source_dataset", "source_original_name",
                   "corrected_name", "label_correction"]
        dfs = {
            "id": pd.DataFrame([["/dataset/ID/a.jpg", "a", "ID", "a", "", ""]],
                               columns=columns),
            "ood": pd.DataFrame([
                ["/dataset/FSDM41/a.jpg", "b", "FSDM41", "old-b", "new-b",
                 "FSDM41_PERMUTED_LABEL"],
                ["/dataset/FSDM41/c.jpg", "c", "FSDM41", "unchanged", "", ""],
            ], columns=columns),
        }
        self.assertEqual(validate_correction_cohort(dfs, {"old-b": "new-b"}, 1), 1)
        with self.assertRaisesRegex(ValueError, "Expected 2"):
            validate_correction_cohort(dfs, {"old-b": "new-b"}, 2)

    def test_verified_one_to_one_mapping_allows_only_fsdm_relabel(self):
        report = validate_index_map(self.df, self.old_labels, self.indices,
                                    self.scores, self.runner_up, "ood")
        self.assertEqual(report["matched_v3_rows"], 3)
        self.assertEqual(report["fsdm41_old_label_differences"], 1)

    def test_rejects_weak_ambiguous_reused_and_cross_label(self):
        cases = [
            (self.indices, [0.9, 1, 1], self.runner_up, "weak"),
            (self.indices, self.scores, [0.999999, 0.7, 0.6], "ambiguous"),
            ([2, 1, 2], self.scores, self.runner_up, "reused"),
            ([0, 1, 2], self.scores, self.runner_up, "labels disagree"),
        ]
        for indices, scores, runner_up, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                validate_index_map(self.df, self.old_labels, indices, scores,
                                   runner_up, "ood")

    def test_exact_image_copies_use_distinct_feature_equivalent_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first, second = root / "a.jpg", root / "a - Copy.jpg"
            first.write_bytes(b"same-image")
            second.write_bytes(b"same-image")
            df = pd.DataFrame({"file_path": [str(first), str(second)],
                               "label": ["A species", "A species"],
                               "source_dataset": ["VN26", "VN26"]})
            cache_path = root / "v3.npz"
            np.savez(cache_path,
                     embs_id_dinov2=np.array([[1.0, 0.0], [1.0, 0.0]]),
                     embs_id_arc=np.array([[0.5, 0.5], [0.5, 0.5]]),
                     logits_id_ce_narrow=np.array([[2.0], [2.0]]),
                     labels_id_dinov2=np.array(["a_species", "a_species"]))
            scores = np.array([1.0, 1.0])
            runner_up = np.array([1.0, 1.0])
            with np.load(cache_path, allow_pickle=False) as cache:
                indices, equivalent, groups = resolve_feature_equivalent_duplicates(
                    cache, df, "id", np.array([0, 0]), scores, runner_up)
            self.assertEqual(indices.tolist(), [0, 1])
            self.assertEqual(equivalent.tolist(), [True, True])
            self.assertEqual(len(groups), 1)
            report = validate_index_map(df, ["a_species", "a_species"],
                                        indices, scores, runner_up, "id",
                                        resolved_duplicates=equivalent)
            self.assertEqual(report["resolved_ambiguous_rows"], 2)

            second.write_bytes(b"different-image")
            with np.load(cache_path, allow_pickle=False) as cache:
                with self.assertRaisesRegex(ValueError, "different file bytes"):
                    resolve_feature_equivalent_duplicates(cache, df, "id",
                                                          np.array([0, 0]), scores, runner_up)
            second.write_bytes(b"same-image")
            np.savez(cache_path,
                     embs_id_dinov2=np.array([[1.0, 0.0], [1.0, 0.0]]),
                     embs_id_arc=np.array([[0.5, 0.5], [0.1, 0.9]]),
                     labels_id_dinov2=np.array(["a_species", "a_species"]))
            with np.load(cache_path, allow_pickle=False) as cache:
                with self.assertRaisesRegex(ValueError, "cannot assign one-to-one"):
                    resolve_feature_equivalent_duplicates(
                        cache, df, "id", np.array([0, 0]), scores, runner_up)

    def test_positional_tie_requires_all_other_id_rows_to_anchor_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [root / "a.jpg", root / "a - Copy.jpg", root / "b.jpg"]
            for path in paths:
                path.write_bytes(path.name.encode())
            df = pd.DataFrame({"file_path": [str(path) for path in paths],
                               "label": ["A species", "A species", "B species"],
                               "source_dataset": ["VN26"] * 3})
            cache_path = root / "v3.npz"
            np.savez(cache_path,
                     embs_id_dinov2=np.array([[1., 0.], [1., 0.], [0., 1.]]),
                     embs_id_arc=np.array([[0., 1.], [1., 0.], [0., 1.]]),
                     labels_id_dinov2=np.array(["a_species", "a_species", "b_species"]),
                     embs_ood_dinov2=np.array([[1., 0.], [1., 0.], [0., 1.]]),
                     embs_ood_arc=np.array([[0., 1.], [1., 0.], [0., 1.]]),
                     labels_ood_dinov2=np.array(["a_species", "a_species", "b_species"]))
            with np.load(cache_path, allow_pickle=False) as cache:
                indices, allowed, groups = resolve_feature_equivalent_duplicates(
                    cache, df, "id", np.array([1, 1, 2]),
                    np.ones(3), np.array([1., 1., 0.5]))
                self.assertEqual(indices.tolist(), [0, 1, 2])
                self.assertEqual(groups[0]["resolution"], "anchored_id_position")
                self.assertGreater(groups[0]["feature_different_candidates"][0][2], 0.1)
                with self.assertRaisesRegex(ValueError, "cannot assign one-to-one"):
                    resolve_feature_equivalent_duplicates(
                        cache, df, "id", np.array([1, 1, 0]),
                        np.ones(3), np.array([1., 1., 0.5]))
                with self.assertRaisesRegex(ValueError, "cannot assign one-to-one"):
                    resolve_feature_equivalent_duplicates(
                        cache, df, "ood", np.array([1, 1, 2]),
                        np.ones(3), np.array([1., 1., 0.5]))
                paths[1].write_bytes(paths[0].read_bytes())
                with self.assertRaisesRegex(ValueError, "identical image bytes but v3 features disagree"):
                    resolve_feature_equivalent_duplicates(
                        cache, df, "id", np.array([1, 1, 2]),
                        np.ones(3), np.array([1., 1., 0.5]))

    def test_ood_tie_uses_two_sided_local_offset_not_global_position(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            n, offset = 24, 3
            paths = []
            for row in range(n):
                path = root / f"image_{row}.jpg"
                path.write_bytes(str(row).encode())
                paths.append(str(path))
            df = pd.DataFrame({"file_path": paths, "label": ["A species"] * n,
                               "source_dataset": ["WRD25"] * n})
            old_dino = np.eye(n + offset, dtype=np.float32)
            old_dino[11] = old_dino[17]
            old_indices = np.arange(n) + offset
            old_indices[8] = 17
            scores = np.ones(n, dtype=np.float32)
            runner_up = np.zeros(n, dtype=np.float32)
            runner_up[[8, 14]] = 1
            cache_path = root / "v3.npz"
            np.savez(cache_path, embs_ood_dinov2=old_dino,
                     embs_ood_arc=np.arange(n + offset)[:, None].astype(np.float32),
                     labels_ood_dinov2=np.array(["a_species"] * (n + offset)))
            with np.load(cache_path, allow_pickle=False) as cache:
                indices, allowed, groups = resolve_feature_equivalent_duplicates(
                    cache, df, "ood", old_indices, scores, runner_up)
                self.assertEqual(indices[[8, 14]].tolist(), [11, 17])
                self.assertEqual(int(allowed.sum()), 2)
                self.assertEqual(groups[0]["resolution"], "anchored_ood_offset")
                self.assertEqual(groups[0]["anchor_evidence"]["offset"], offset)
                self.assertEqual(len(groups[0]["anchor_evidence"]["anchor_rows"]), 20)

                conflicting = old_indices.copy()
                conflicting[5] = 20
                with self.assertRaisesRegex(ValueError, "cannot assign one-to-one"):
                    resolve_feature_equivalent_duplicates(
                        cache, df, "ood", conflicting, scores, runner_up)

                Path(paths[14]).write_bytes(Path(paths[8]).read_bytes())
                with self.assertRaisesRegex(ValueError, "identical image bytes but v3 features disagree"):
                    resolve_feature_equivalent_duplicates(
                        cache, df, "ood", old_indices, scores, runner_up)

    def test_rebuild_preserves_ce_and_swi_reindexes_other_public_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            v3_path = root / "v3.npz"
            v5_path = root / "v5.npz"
            np.savez_compressed(v3_path,
                                labels_id_dinov2=np.array(["a", "b", "c"]),
                                labels_ood_dinov2=np.array(["c", "b", "a"]),
                                embs_id_dinov2=np.array([[1], [2], [3]]),
                                embs_ood_dinov2=np.array([[10], [20], [30]]),
                                embs_id_arc=np.array([[4], [5], [6]]),
                                embs_ood_arc=np.array([[40], [50], [60]]))
            np.savez_compressed(v5_path,
                                embs_swi_dinov2=np.array([[99]]),
                                labels_id_dinov2=np.array(["a", "b", "c"]),
                                labels_ood_dinov2=np.array(["a", "b", "c"]),
                                embs_id_dinov2=np.array([[1], [2], [3]]),
                                embs_ood_dinov2=np.array([[30], [20], [10]]),
                                embs_id_arc=np.array([[4], [5], [6]]),
                                embs_ood_arc=np.array([[60], [50], [40]]),
                                embs_id_ce_full_norm=np.array([[70], [80], [90]]),
                                embs_ood_ce_full_norm=np.array([[7], [8], [9]]),
                                logits_ood_ce_full=np.array([[11], [12], [13]]))
            dfs = {"id": self.df, "ood": self.df}
            with np.load(v3_path, allow_pickle=False) as v3, np.load(v5_path, allow_pickle=False) as v5:
                out = rebuild_arrays(v3, v5, dfs, {"id": np.arange(3), "ood": self.indices})
            self.assertEqual(out["embs_swi_dinov2"].tolist(), [[99]])
            self.assertEqual(out["embs_ood_arc"].ravel().tolist(), [60, 50, 40])
            self.assertEqual(out["embs_ood_ce_full_norm"].ravel().tolist(), [7, 8, 9])
            self.assertEqual(out["logits_ood_ce_full"].ravel().tolist(), [11, 12, 13])
            self.assertEqual(out["labels_ood_dinov2"].tolist(),
                             ["A species", "B species", "C species"])

    def test_diagnose_duplicate_rows_reads_saved_mapping_without_reextracting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            n, offset = 24, 3
            paths = []
            for row in range(n):
                path = root / f"image_{row}.jpg"
                path.write_bytes(str(row).encode())
                paths.append(str(path))
            Path(paths[14]).write_bytes(Path(paths[8]).read_bytes())
            old_dino = np.eye(n + offset, dtype=np.float32)
            old_dino[11] = old_dino[17]
            np.savez(root / "v3.npz", embs_ood_dinov2=old_dino,
                     embs_ood_arc=np.arange(n + offset)[:, None].astype(np.float32),
                     labels_ood_dinov2=np.array(["a_species"] * (n + offset)))
            old_indices = np.arange(n) + offset
            old_indices[8] = 17
            runner_up = np.zeros(n, dtype=np.float32)
            runner_up[[8, 14]] = 1
            pd.DataFrame({"file_path": paths, "label": ["A species"] * n,
                          "source_dataset": ["WRD25"] * n,
                          "nearest_v3_row": old_indices,
                          "cosine": np.ones(n),
                          "runner_up_cosine": runner_up}).to_csv(
                              root / "ood_row_identity.csv", index=False)
            with patch.dict(os.environ, {"ROOT_PATH": str(root),
                                      "PUBLIC_REPAIR_AUDIT_DIR": str(root),
                                      "PUBLIC_REPAIR_SOURCE_CACHE_NAME": "v3.npz"}):
                out = diagnose_duplicate_rows()
            report = json.loads(out.read_text())
            self.assertEqual(report["ambiguous_rows"], 2)
            self.assertEqual(report["pairs"], 1)
            self.assertEqual(report["anchored_pairs"], 1)
            self.assertEqual(report["identical_byte_pairs"], 1)
            self.assertEqual(report["groups"][0]["candidate_v3_rows"], [11, 17])
            arc = next(item for item in report["groups"][0]["features"]
                       if item["key"] == "embs_ood_arc")
            self.assertEqual(arc["max_abs_difference"], 6.0)
            self.assertFalse((root / "embedding_cache_full954_v6_public_row_verified.npz").exists())

    def test_opt_in_arc_anomaly_requires_matching_bytes_and_other_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            n, offset = 24, 3
            paths = []
            for row in range(n):
                path = root / f"image_{row}.jpg"
                path.write_bytes(str(row).encode())
                paths.append(str(path))
            Path(paths[14]).write_bytes(Path(paths[8]).read_bytes())
            df = pd.DataFrame({"file_path": paths, "label": ["A species"] * n,
                               "source_dataset": ["WRD25"] * n})
            old_dino = np.eye(n + offset, dtype=np.float32)
            old_dino[11] = old_dino[17]
            old_indices = np.arange(n) + offset
            old_indices[8] = 17
            scores = np.ones(n, dtype=np.float32)
            runner_up = np.zeros(n, dtype=np.float32)
            runner_up[[8, 14]] = 1
            arc = np.zeros((n + offset, 2), dtype=np.float32)
            arc[11] = [1., 0.]
            arc[17] = [1., 0.01]
            proto = np.zeros((n + offset, 2), dtype=np.float32)
            cache_path = root / "v3.npz"

            def resolve():
                with np.load(cache_path, allow_pickle=False) as cache:
                    return resolve_feature_equivalent_duplicates(
                        cache, df, "ood", old_indices, scores, runner_up)

            np.savez(cache_path, embs_ood_dinov2=old_dino, embs_ood_arc=arc,
                     embs_ood_proto=proto,
                     labels_ood_dinov2=np.array(["a_species"] * (n + offset)))
            with self.assertRaisesRegex(ValueError, "identical image bytes"):
                resolve()
            with patch.dict(os.environ, {"PUBLIC_REPAIR_ACCEPT_ARC_SHA256": "0" * 64}):
                with self.assertRaisesRegex(ValueError, "identical image bytes"):
                    resolve()
            with patch.dict(os.environ, {"PUBLIC_REPAIR_ACCEPT_ARC_SHA256": sha256(paths[8])}):
                indices, allowed, groups = resolve()
                self.assertEqual(indices[[8, 14]].tolist(), [11, 17])
                self.assertEqual(int(allowed.sum()), 2)
                self.assertEqual(groups[0]["resolution"], "anchored_ood_arc_anomaly")
                self.assertEqual(groups[0]["feature_exception"]["key"], "embs_ood_arc")
                self.assertAlmostEqual(groups[0]["feature_exception"]["max_abs_difference"], 0.01)

                proto[17, 0] = 0.001
                np.savez(cache_path, embs_ood_dinov2=old_dino, embs_ood_arc=arc,
                         embs_ood_proto=proto,
                         labels_ood_dinov2=np.array(["a_species"] * (n + offset)))
                with self.assertRaisesRegex(ValueError, "identical image bytes"):
                    resolve()

    def test_mapping_extracts_in_csv_order_and_resumes(self):
        import torch
        from PIL import Image
        from torchvision import transforms

        class Identity(torch.nn.Module):
            def forward(self, image):
                return image.flatten(1)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
            paths = []
            for i, color in enumerate(colors):
                path = root / f"{i}.png"
                Image.new("RGB", (1, 1), color).save(path)
                paths.append(str(path))
            df = pd.DataFrame({"file_path": paths})
            old = np.eye(3, dtype=np.float32)[[2, 0, 1]]
            partial = root / "map.partial.npz"
            transform = transforms.ToTensor()
            with patch.dict(os.environ, {"PUBLIC_REPAIR_BATCH_SIZE": "2",
                                      "PUBLIC_REPAIR_WORKERS": "0",
                                      "PUBLIC_REPAIR_SAVE_EVERY": "1"}):
                first = _extract_mapping(df, old, "id", partial, "sig", Identity(),
                                         transform, "cpu")
                second = _extract_mapping(df, old, "id", partial, "sig", Identity(),
                                          transform, "cpu")
            self.assertEqual(first[0].tolist(), [1, 2, 0])
            self.assertEqual(second[0].tolist(), first[0].tolist())
            self.assertTrue(np.all(first[1] > 0.999))
            with np.load(partial, allow_pickle=False) as saved:
                self.assertEqual(saved["signature"].item(), "sig")

    def test_overnight_runpy_dispatches_repair_only(self):
        from swid_retrieval.embeddings import repair_public_rows
        with patch.dict(os.environ, {"RUN_REPAIR_PUBLIC_ROWS": "1",
                                  "PUBLIC_REPAIR_DIAGNOSE_ONLY": "0",
                                  "RUN_FINAL_SCURD_RETRAIN": "0",
                                  "RUN_FINAL_COLAB_AUDIT": "0",
                                  "ROOT_PATH": "/tmp"}, clear=False):
            with patch.object(repair_public_rows, "run") as target:
                with redirect_stdout(io.StringIO()):
                    runpy.run_module("swid_retrieval.run_overnight", run_name="__main__")
                target.assert_called_once_with()

    def test_overnight_runpy_dispatches_diagnostic_only(self):
        from swid_retrieval.embeddings import repair_public_rows
        with patch.dict(os.environ, {"RUN_REPAIR_PUBLIC_ROWS": "1",
                                  "PUBLIC_REPAIR_DIAGNOSE_ONLY": "1",
                                  "RUN_FINAL_SCURD_RETRAIN": "0",
                                  "RUN_FINAL_COLAB_AUDIT": "0",
                                  "ROOT_PATH": "/tmp"}, clear=False):
            with patch.object(repair_public_rows, "diagnose_duplicate_rows") as diagnostic:
                with patch.object(repair_public_rows, "run") as repair:
                    with redirect_stdout(io.StringIO()):
                        runpy.run_module("swid_retrieval.run_overnight", run_name="__main__")
                    diagnostic.assert_called_once_with()
                    repair.assert_not_called()


if __name__ == "__main__":
    unittest.main()
