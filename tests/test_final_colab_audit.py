import tempfile
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from swid_retrieval.audit_support import check_labels, compare_recipes, top_matches
from swid_retrieval.final_colab_audit import _ce_features, _validate_public_rows, _vn26_items, ce_vn26


class FinalAuditTest(unittest.TestCase):
    def test_batched_top_matches_equal_full_matrix(self):
        rng = np.random.RandomState(17)
        q = rng.randn(13, 7).astype(np.float32)
        g = rng.randn(23, 7).astype(np.float32)
        idx, vals = top_matches(q, g, k=5, batch=3)
        full = q @ g.T
        expected = np.argsort(-full, axis=1)[:, :5]
        np.testing.assert_array_equal(idx, expected)
        np.testing.assert_allclose(vals, np.take_along_axis(full, expected, axis=1))

    def test_reordered_labels_rejected(self):
        with self.assertRaisesRegex(ValueError, "Ordered label mismatch"):
            check_labels(["A a", "B b"], ["b_b", "a_a"], "fixture")

    def test_vn26_replays_folder_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for mag in ("x10", "x20", "x50"):
                folder = root / "datasets" / "VN26" / "species" / mag
                folder.mkdir(parents=True)
                (folder / "one.jpg").touch()
            row = {
                "dataset": repr(["VN26"] * 3),
                "folder_path": repr([f"/Users/admin/Downloads/Experimentals/datasets/VN26/species/{m}"
                                     for m in ("x10", "x20", "x50")]),
                "magnification": repr(["x10", "x20", "x50"]),
                "canonical_binomial": "Test species",
            }
            pd.DataFrame([row]).to_csv(root / "ID_species_public.csv", index=False)
            pd.DataFrame(columns=row).to_csv(root / "OOD_species_public.csv", index=False)
            mapping = _vn26_items(root)
            for mag in ("x10", "x20", "x50"):
                self.assertEqual(mapping[mag], [(str(root / "datasets" / "VN26" / "species" / mag / "one.jpg"),
                                                  "test_species")])

    def test_fsdm_correction_and_woodauth_exclusion(self):
        rows = pd.DataFrame([
            {"file_path": "/datasets/FSDM41/old/a.jpg", "label": "new_species",
             "source_dataset": "FSDM41", "source_original_name": "Old species",
             "corrected_name": "New species", "label_correction": "FSDM41_PERMUTED_LABEL"},
            {"file_path": "/datasets/VN26/other.jpg", "label": "other_species",
             "source_dataset": "VN26", "source_original_name": "Other species",
             "corrected_name": "", "label_correction": ""},
        ])
        self.assertEqual(_validate_public_rows(rows, ["new_species", "other_species"],
                                               {"Old species": "New species"}, "ood"), 1)
        bad = rows.copy()
        bad.at[0, "corrected_name"] = "Wrong species"
        with self.assertRaisesRegex(ValueError, "FSDM41 mapping"):
            _validate_public_rows(bad, rows["label"], {"Old species": "New species"}, "ood")
        bad = rows.copy()
        bad.at[1, "source_dataset"] = "WOODAUTH"
        with self.assertRaisesRegex(ValueError, "WoodAuth"):
            _validate_public_rows(bad, rows["label"], {"Old species": "New species"}, "ood")

    def test_missing_recipe_metadata_is_unknown(self):
        result = compare_recipes({"selected": {"epochs": 20}, "seed42": {"epochs": 20}})
        self.assertFalse(result["metadata_recipe_match"])
        self.assertIn("beta", result["missing"])

    def test_ce_exp4_requires_feature_embedding_not_classifier_output(self):
        calls = []
        fake = types.ModuleType("swid_retrieval.embeddings.extract")

        def extract(_model, _loader, _device, return_logits=False):
            calls.append(return_logits)
            return np.zeros((2, 512), np.float32), np.array([0, 1]), np.zeros((2, 954))

        fake.extract_embeddings = extract
        with patch.dict(sys.modules, {"swid_retrieval.embeddings.extract": fake}):
            features, logits, indices = _ce_features(None, None, "cpu", 2)
            self.assertEqual(features.shape, (2, 512))
            self.assertEqual(logits.shape, (2, 954))
            np.testing.assert_array_equal(indices, [0, 1])
            fake.extract_embeddings = lambda *args, **kwargs: (
                np.zeros((2, 954)), np.array([0, 1]), np.zeros((2, 954)))
            with self.assertRaisesRegex(ValueError, "CE extraction returned"):
                _ce_features(None, None, "cpu", 2)
            fake.extract_embeddings = lambda *args, **kwargs: (
                np.zeros((2, 512)), np.array([0, 1]), np.zeros((2, 512)))
            with self.assertRaisesRegex(ValueError, "CE logits returned"):
                _ce_features(None, None, "cpu", 2)
        self.assertEqual(calls, [True])

    def test_ce_vn26_keeps_legacy_logits_and_features_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "ce.pt"
            checkpoint.write_bytes(b"fixture")
            arrays = {}
            for source, parts in (("swi", (256, 512, 768)),
                                  ("vn26", ("x10", "x20", "x50"))):
                for part in parts:
                    key = f"{source}__CE_Full__{part}"
                    arrays[key] = np.ones((2, 954), dtype=np.float32)
                    arrays[key + "_feature512"] = np.zeros((2, 512), dtype=np.float32)
                    arrays[key + "_lbl"] = np.asarray(["test_species", "test_species"])
            np.savez_compressed(root / "ce_exp4_fresh.npz", **arrays)
            fake = types.ModuleType("swid_retrieval.experiments.rq4_vn26")

            def run(methods, out_dir):
                name, data = next(iter(methods.items()))
                dimension = data["swi_scales"][256][0].shape[1]
                out_dir.mkdir(parents=True)
                (out_dir / "rq4_generalization.json").write_text("{}")
                return {"cross_domain": {name: {"SWI_pool": {"VN26_all": {"mean": dimension}}}},
                        "cross_magnification": {name: {"x10": {"x20": {"mean": dimension}}}}}

            fake.run = run
            with patch.dict(sys.modules, {"swid_retrieval.experiments.rq4_vn26": fake}):
                ce_vn26({"out": root, "ce": checkpoint})
            result = json.loads((root / "ce_vn26_fresh.json").read_text())
            self.assertEqual(result["representations"]["legacy_logits_954"]["cross_domain"]["SWI_pool/VN26_all"]["mean"], 954)
            self.assertEqual(result["representations"]["features_512"]["cross_domain"]["SWI_pool/VN26_all"]["mean"], 512)


if __name__ == "__main__":
    unittest.main()
