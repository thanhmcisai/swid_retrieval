"""Protocol checks for image-based correspondence training."""

import importlib.util
import io
import os
import runpy
import sys
import unittest
from collections import defaultdict
from pathlib import Path
from types import ModuleType
from contextlib import redirect_stdout
from unittest import mock


class SamplerWithoutImageDependenciesTest(unittest.TestCase):
    def test_scan_disjoint_full_pool_and_query_coverage(self):
        import numpy as np
        import swid_retrieval as package

        fake = {}
        for name in ("data", "gallery_experiment", "wood_correspondence_experiment",
                     "wood_evidence_experiment"):
            fake[name] = ModuleType(f"swid_retrieval.{name}")
        fake["gallery_experiment"].canonical = lambda label: label.lower()
        fake["gallery_experiment"].source_scan_id = (
            lambda path: Path(path).stem.split("_from_")[1].lower())
        qualified = {module.__name__: module for module in fake.values()}
        path = Path(__file__).resolve().parents[1] / "wood_correspondence_image_train.py"
        spec = importlib.util.spec_from_file_location(
            "swid_retrieval._isolated_image_sampler_test", path)
        module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, qualified):
            with mock.patch.multiple(package, create=True, **fake):
                spec.loader.exec_module(module)
                items = [(f"/scale_256/patch_{j}_from_Tw{i * 4 + scan:05d}.jpg",
                          f"species_{i:03d}")
                         for i in range(70) for scan in range(2) for j in range(6)]
                items += [(f"/scale_256/patch_{j}_from_Tw99999.jpg", "single_scan")
                          for j in range(12)]
                by_class, valid = module._layout(items)
                audit = module._pool_audit(items, by_class, valid)
                self.assertEqual(audit["by_k"]["5"]["eligible_species"], 70)
                self.assertEqual(audit["by_k"]["5"]["eligible_images"], len(items) - 12)
                for n_way, k in module.SCHEDULE:
                    refs, queries, labels, scans = module._episode(
                        items, by_class, valid, np.random.default_rng(11),
                        n_way, k, defaultdict(list))
                    self.assertEqual(len(refs), n_way * k)
                    self.assertEqual(len(set(refs + queries)), len(refs + queries))
                    for class_id, query in enumerate(queries):
                        query_scan = fake["gallery_experiment"].source_scan_id(items[query][0])
                        self.assertTrue(all(scan != query_scan
                                            for scan, label in zip(scans, labels)
                                            if label == class_id))
                _, batches, coverage = module._epoch_plan(
                    items, by_class, valid, {"episodes": 16}, 43, 1)
                self.assertEqual(len(batches), 16)
                self.assertGreater(coverage["unique_query_images"], 0)
                self.assertLessEqual(coverage["unique_images"], len(items) - 12)


@unittest.skipUnless(all(importlib.util.find_spec(name) is not None for name in
                         ("torch", "numpy", "pandas", "cv2", "albumentations")),
                     "Image-training dependencies unavailable")
class ImageTrainTest(unittest.TestCase):
    def setUp(self):
        from swid_retrieval import wood_correspondence_image_train as image_train
        self.train = image_train
        self.items = [
            (f"/scale_256/patch_{j:03d}_from_Tw{i * 4 + scan:05d}.jpg", f"species_{i:03d}")
            for i in range(70) for scan in range(2) for j in range(6)
        ]

    def test_full_pool_scan_disjoint_episodes_and_coverage(self):
        import numpy as np
        by_class, valid = self.train._layout(self.items)
        self.assertEqual(len(valid[5]), 70)
        audit = self.train._pool_audit(self.items, by_class, valid)
        self.assertEqual(audit["by_k"]["5"]["eligible_images"], len(self.items))
        for n_way, k in self.train.SCHEDULE:
            refs, queries, labels, scans = self.train._episode(
                self.items, by_class, valid, np.random.default_rng(11),
                n_way, k, defaultdict(list))
            self.assertEqual(len(refs), n_way * k)
            self.assertEqual(len(queries), n_way)
            self.assertEqual(len(set(refs + queries)), len(refs + queries))
            for class_id, query in enumerate(queries):
                query_scan = self.train.study.source_scan_id(self.items[query][0])
                self.assertTrue(all(scan != query_scan for scan, label in zip(scans, labels)
                                    if label == class_id))
        config = {"episodes": 16}
        layouts, batches, coverage = self.train._epoch_plan(
            self.items, by_class, valid, config, 43, 1)
        self.assertEqual(len(batches), 16)
        self.assertEqual(len(layouts), 16)
        self.assertGreater(coverage["unique_query_images"], 0)
        self.assertLessEqual(coverage["unique_images"], len(self.items))

    def test_single_scan_species_excluded_not_claimed_as_trained(self):
        items = self.items + [(f"/scale_256/patch_{j}_from_Tw99999.jpg", "only_one_scan")
                              for j in range(12)]
        by_class, valid = self.train._layout(items)
        audit = self.train._pool_audit(items, by_class, valid)
        self.assertEqual(audit["meta_train_species"], 71)
        self.assertEqual(audit["by_k"]["5"]["eligible_species"], 70)
        self.assertEqual(audit["by_k"]["5"]["eligible_images"], len(self.items))

    def test_runpy_dispatch_isolated(self):
        flags = {name: "0" for name in (
            "RUN_WOOD_CORRESPONDENCE_STUDY", "RUN_WOOD_EVIDENCE_STUDY",
            "RUN_GALLERY_DIAGNOSTICS", "RUN_GALLERY_STUDY", "RUN_REPAIR_PUBLIC_ROWS",
            "RUN_FINAL_SCURD_RETRAIN", "RUN_FINAL_COLAB_AUDIT")}
        flags["RUN_WOOD_CORRESPONDENCE_IMAGE_TRAIN"] = "1"
        with mock.patch.dict(os.environ, flags, clear=False):
            with mock.patch.object(self.train, "run") as runner:
                with redirect_stdout(io.StringIO()):
                    runpy.run_module("swid_retrieval.run_overnight", run_name="__main__")
                runner.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
