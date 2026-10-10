"""Matched-control and paired-query checks for correspondence evaluation."""

import io
import os
import runpy
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import pandas as pd

from swid_retrieval import wood_correspondence_compare as compare


def _queries():
    rows = []
    for gallery in sorted(compare.GALLERIES):
        for fold in (0, 1):
            for species in ("a", "b"):
                rows.append({"split": "meta-val", "fold": fold, "gallery": gallery,
                             "query_path": f"/image/{fold}_{species}.jpg",
                             "true_label": species, "mode": "trained",
                             "rank": 1 if species == "a" else 2,
                             "correct": int(species == "a"),
                             "query_scale": 256, "query_scan": f"scan_{fold}_{species}"})
    return pd.DataFrame(rows)


class CorrespondenceCompareTest(unittest.TestCase):
    def test_paired_reports_use_identical_queries_and_species_bootstrap(self):
        global_only = _queries()
        qkv = global_only.copy()
        rescued = (qkv["gallery"] == "637x1") & (qkv["true_label"] == "b")
        qkv.loc[rescued, "correct"] = 1
        qkv.loc[rescued, "rank"] = 1
        paired = compare._paired(qkv, global_only)
        summary = compare._summary(paired, bootstrap=200)
        row = summary[(summary["gallery"] == "637x1") &
                      (summary["fold"] == "both")].iloc[0]
        self.assertEqual(row["n_queries"], 4)
        self.assertAlmostEqual(row["qkv_minus_global_only"], 0.5)
        self.assertEqual(row["rescued"], 2)
        self.assertEqual(row["harmed"], 0)
        changed = qkv.iloc[:-1]
        with self.assertRaises(ValueError):
            compare._paired(changed, global_only)

    def test_provenance_requires_matched_recipe_and_sampling(self):
        config = {"variant": "qkv", "score_mode": "qkv", "epochs": 2,
                  "episodes": 8, "seed": 43}
        shared = {field: field for field in (
            "base_checkpoint_sha256", "manifest_sha256", "runner_sha256",
            "correspondence_sha256", "method_sha256")}
        qkv = {"checkpoint_sha256": "qkv", "provenance":
               {**shared, "seed": 43, "config": config}}
        control = {"checkpoint_sha256": "global", "provenance":
                   {**shared, "seed": 43, "config":
                    {**config, "variant": "global_only", "score_mode": "global"}}}
        with TemporaryDirectory() as temp:
            qfolder = Path(temp) / "qkv"
            gfolder = Path(temp) / "control"
            qfolder.mkdir()
            gfolder.mkdir()
            epoch = {"epoch_coverage": {"image_draws": 10, "unique_images": 8},
                     "cumulative_unique_images": 8, "cumulative_unique_queries": 4}
            import json
            for index in (1, 2):
                for folder in (qfolder, gfolder):
                    (folder / f"epoch_{index:02d}.json").write_text(json.dumps(epoch))
            self.assertEqual(compare._check_matched_runs(
                qfolder, qkv, gfolder, control), 2)
            control["provenance"]["config"]["episodes"] = 7
            with self.assertRaises(ValueError):
                compare._check_matched_runs(qfolder, qkv, gfolder, control)

    def test_runpy_comparison_dispatch_isolated(self):
        flags = {name: "0" for name in (
            "RUN_WOOD_CORRESPONDENCE_IMAGE_TRAIN", "RUN_WOOD_CORRESPONDENCE_STUDY",
            "RUN_WOOD_EVIDENCE_STUDY", "RUN_GALLERY_DIAGNOSTICS",
            "RUN_GALLERY_STUDY", "RUN_REPAIR_PUBLIC_ROWS",
            "RUN_FINAL_SCURD_RETRAIN", "RUN_FINAL_COLAB_AUDIT")}
        flags["RUN_WOOD_CORRESPONDENCE_COMPARE"] = "1"
        with mock.patch.dict(os.environ, flags, clear=False):
            with mock.patch.object(compare, "run") as runner:
                with redirect_stdout(io.StringIO()):
                    runpy.run_module("swid_retrieval.run_overnight", run_name="__main__")
                runner.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
