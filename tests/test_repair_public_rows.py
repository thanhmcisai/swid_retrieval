import io
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
    _extract_mapping, rebuild_arrays, validate_correction_cohort,
    validate_index_map,
)


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
                                  "RUN_FINAL_SCURD_RETRAIN": "0",
                                  "RUN_FINAL_COLAB_AUDIT": "0",
                                  "ROOT_PATH": "/tmp"}, clear=False):
            with patch.object(repair_public_rows, "run") as target:
                with redirect_stdout(io.StringIO()):
                    runpy.run_module("swid_retrieval.run_overnight", run_name="__main__")
                target.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
