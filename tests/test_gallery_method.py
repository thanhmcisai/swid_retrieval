"""Small synthetic checks for the isolated gallery study (run on Colab)."""

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


DEPS = all(importlib.util.find_spec(name) is not None for name in
           ("torch", "numpy", "pandas", "cv2", "albumentations"))


@unittest.skipUnless(DEPS, "Colab ML dependencies are unavailable locally")
class GalleryMethodTest(unittest.TestCase):
    def setUp(self):
        import numpy as np
        import torch
        from swid_retrieval import gallery_experiment as experiment
        from swid_retrieval import gallery_method as method
        self.np, self.torch, self.experiment, self.method = np, torch, experiment, method

    def test_score_contains_every_gallery_class(self):
        torch = self.torch
        scorer = self.method.GalleryScorer(top_m=1, mode="learned")
        refs = torch.nn.functional.normalize(torch.tensor([
            [1., 0.], [0.9, 0.1], [0., 1.], [-1., 0.]]), dim=1)
        labels = torch.tensor([2, 2, 5, 9])
        query = refs[:1]
        scores, classes = scorer(query, refs, labels)
        self.assertEqual(scores.shape, (1, 3))
        self.assertEqual(classes.tolist(), [2, 5, 9])
        self.assertTrue(torch.isfinite(scores).all())

    def test_image_ranking_metrics_are_separate_from_class_r1(self):
        np, torch = self.np, self.torch
        scorer = self.method.GalleryScorer(mode="prototype")
        gallery = np.asarray([[1., 0.], [0.9, 0.1], [0., 1.], [0.1, 0.9]],
                             dtype=np.float32)
        gallery = gallery / np.linalg.norm(gallery, axis=1, keepdims=True)
        result = self.experiment._predict(
            scorer, np.asarray([[1., 0.], [0., 1.]], dtype=np.float32),
            np.asarray(["a", "b"]), gallery, np.asarray(["a", "a", "b", "b"]),
            torch.device("cpu"))
        self.assertEqual(result["species_r1"], 1.0)
        self.assertEqual(result["image_map_at_100"], 1.0)
        self.assertEqual(result["image_mrr_at_100"], 1.0)

    def test_primary_variant_uses_exact_nearest_retrieval(self):
        np, torch = self.np, self.torch
        cfg = self.experiment.variant_config("metric_retrieval", pilot=True)
        self.assertEqual(cfg["scorer_mode"], "nearest")
        self.assertFalse(cfg["variable_gallery"])
        self.assertEqual(cfg["stability_weight"], 0.0)
        self.assertEqual(cfg["pseudo_ood_weight"], 0.0)
        scorer = self.method.GalleryScorer(mode="nearest", top_m=1)
        gallery = np.asarray([[1., 0.], [0., 1.], [-1., 0.]], dtype=np.float32)
        queries = np.asarray([[1., 0.], [0., 1.]], dtype=np.float32)
        result = self.experiment._predict(
            scorer, queries, np.asarray(["a", "b"]), gallery,
            np.asarray(["a", "b", "c"]), torch.device("cpu"))
        self.assertEqual(result["species_r1"], result["nearest_image_r1"])
        self.assertEqual(len(scorer.state_dict()), 0)
        no_memory = self.experiment.variant_config("metric_no_memory", pilot=True)
        self.assertEqual(no_memory["scorer_mode"], "nearest")
        self.assertEqual(no_memory["memory_size"], 0)

    def test_head_learning_rate_override_is_recorded_in_config(self):
        with patch.dict(os.environ, {"GALLERY_STUDY_HEAD_LR": "1e-5"}):
            cfg = self.experiment.variant_config("metric_no_memory", pilot=True)
        self.assertEqual(cfg["head_lr"], 1e-5)
        with patch.dict(os.environ, {"GALLERY_STUDY_HEAD_LR": "0"}):
            with self.assertRaisesRegex(ValueError, "Invalid microbatch or workers"):
                self.experiment.variant_config("metric_no_memory", pilot=True)

    def test_nonfinite_gallery_score_is_rejected(self):
        scorer = self.method.GalleryScorer(mode="nearest")
        with self.assertRaisesRegex(RuntimeError, "Non-finite gallery"):
            self.experiment.score_queries(
                scorer, self.np.asarray([[float("nan"), 0.]], dtype=self.np.float32),
                self.np.asarray([[1., 0.]], dtype=self.np.float32),
                self.np.asarray(["a"]), self.torch.device("cpu"))

    def test_variable_gallery_loss_reaches_encoder_features(self):
        torch = self.torch
        embeddings = torch.randn(12, 16, requires_grad=True)
        embeddings = torch.nn.functional.normalize(embeddings, dim=1)
        scorer = self.method.GalleryScorer(top_m=4, mode="learned")
        loss, parts = self.method.episode_objective(embeddings, scorer, 4, 2, 1)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("expansion", parts)
        self.assertIsNotNone(scorer.evidence_gate[-1].weight.grad)

    def test_normalized_evidence_is_invariant_to_identical_reference_copy(self):
        torch = self.torch
        query = torch.tensor([[1., 0.]])
        first = torch.tensor([[1., 0.], [0., 1.]])
        doubled = torch.tensor([[1., 0.], [1., 0.], [0., 1.]])
        corrected = self.method.GalleryScorer(top_m=4, mode="fixed", normalize_evidence=True)
        uncorrected = self.method.GalleryScorer(top_m=4, mode="fixed", normalize_evidence=False)
        a, _ = corrected(query, first, torch.tensor([0, 1]))
        b, _ = corrected(query, doubled, torch.tensor([0, 0, 1]))
        c, _ = uncorrected(query, first, torch.tensor([0, 1]))
        d, _ = uncorrected(query, doubled, torch.tensor([0, 0, 1]))
        self.assertTrue(torch.allclose(a, b, atol=1e-6))
        self.assertGreater(float(d[0, 0]), float(c[0, 0]))

    def test_sampler_is_reproducible_and_disjoint(self):
        labels = [f"genus_{i}" for i in range(8) for _ in range(6)]
        sampler = self.experiment.EpisodeSampler(labels, (4, 8), 2, 1, 5, 42)
        first = list(iter(sampler))
        second = list(iter(sampler))
        self.assertEqual(first, second)
        for indices in first:
            self.assertEqual(len(indices), len(set(indices)))

    def test_scan_disjoint_episode_and_validation(self):
        self.assertEqual(self.experiment.source_scan_id(
            "patch_256_1_2_from_Tw5023_2.jpg"), "tw5023")
        items = [(f"patch_256_{patch}_0_from_Tw{species * 2 + scan + 1000}.jpg",
                  f"genus_{species}")
                 for species in range(24) for scan in range(2) for patch in range(6)]
        labels = [label for _, label in items]
        groups = [self.experiment.source_scan_id(path) for path, _ in items]
        sampler = self.experiment.EpisodeSampler(
            labels, (4, 8), 2, 1, 5, 42, groups=groups)
        for indices in sampler:
            ways = len(indices) // 3
            for class_index in range(ways):
                support = indices[class_index * 2:class_index * 2 + 2]
                query = indices[ways * 2 + class_index]
                self.assertTrue(all(groups[index] != groups[query] for index in support))
        refs, probes = self.experiment.validation_items(
            {"meta-val": items}, group_mode="scan_disjoint")
        self.assertEqual(len(refs), 24 * 5)
        self.assertEqual(len(probes), 24 * 5)
        for species in set(labels):
            ref_scans = {self.experiment.source_scan_id(path)
                         for path, label in refs if label == species}
            query_scans = {self.experiment.source_scan_id(path)
                           for path, label in probes if label == species}
            self.assertFalse(ref_scans & query_scans)

    def test_meta_val_never_uses_train_or_test(self):
        manifest = {
            "meta-train": [["train.jpg", "train_species"]],
            "meta-val": [[f"val_{species}_{image}.jpg", f"genus_{species}"]
                         for species in range(24) for image in range(10)],
            "meta-test": [["test.jpg", "test_species"]],
        }
        refs, queries = self.experiment.validation_items(manifest)
        self.assertEqual(len(refs), 120)
        self.assertEqual(len(queries), 120)
        self.assertTrue(all(path.startswith("val_") for path, _ in refs + queries))
        self.assertFalse(set(path for path, _ in refs) & set(path for path, _ in queries))

    def test_synthetic_full_backbone_step(self):
        self.experiment.smoke_test(self.torch.device("cpu"))

    def test_supervised_warmup_updates_encoder_and_checks_source(self):
        torch = self.torch

        class CPUScaler:
            def scale(self, loss):
                return loss

            def unscale_(self, optimizer):
                pass

            def step(self, optimizer):
                optimizer.step()

            def update(self):
                pass

            def get_scale(self):
                return 1.0

            def is_enabled(self):
                return False

        encoder = self.method.GalleryEncoder(
            torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, 8)),
            feature_dim=8, embedding_dim=4)
        classifier = torch.nn.Linear(4, 3)
        optimizer = torch.optim.AdamW(
            list(encoder.parameters()) + list(classifier.parameters()), lr=1e-2)
        before = next(encoder.backbone.parameters()).detach().clone()
        result = self.experiment._supervised_step(
            torch.randn(6, 3, 2, 2), torch.tensor([0, 1, 2, 0, 1, 2]),
            encoder, classifier, optimizer, CPUScaler(), torch.device("cpu"))
        self.assertTrue(self.np.isfinite(result["loss"]))
        self.assertFalse(result["amp_skipped"])
        self.assertFalse(torch.equal(before, next(encoder.backbone.parameters())))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "swi_manifest.json").write_text("{}")
            checkpoint = root / "warmup.pt"
            torch.save({"checkpoint_kind": "supervised_warmup", "seed": 42, "epoch": 1,
                        "manifest_sha256": self.experiment._hash_file(root / "swi_manifest.json"),
                        "config": {"backbone": "woodpattern_tiny", "embedding_dim": 4},
                        "encoder": encoder.state_dict()}, checkpoint)
            cfg = {"variant": "metric_no_memory", "backbone": "woodpattern_tiny",
                   "embedding_dim": 4, "init_checkpoint": str(checkpoint),
                   "init_checkpoint_sha256": self.experiment._hash_file(checkpoint)}
            restored = self.method.GalleryEncoder(
                torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, 8)),
                feature_dim=8, embedding_dim=4)
            self.experiment._load_warmup_encoder(root, cfg, 42, restored)
            self.assertTrue(torch.equal(next(encoder.parameters()), next(restored.parameters())))
            with self.assertRaisesRegex(ValueError, "not aligned"):
                self.experiment._load_warmup_encoder(root, cfg, 43, restored)

    def test_supervised_warmup_config_is_isolated(self):
        with patch.dict(os.environ, {"GALLERY_STUDY_BACKBONE": "woodpattern_tiny",
                                  "GALLERY_STUDY_INIT_CHECKPOINT": ""}):
            cfg = self.experiment.variant_config("supervised_warmup")
        self.assertEqual(cfg["memory_size"], 0)
        self.assertEqual(cfg["scorer_mode"], "nearest")
        self.assertEqual(cfg["objective"], "supervised_warmup")

    def test_supervised_warmup_saves_meta_val_selected_checkpoint(self):
        torch = self.torch

        class TinyDataset(torch.utils.data.Dataset):
            def __init__(self, items, transform=None):
                self.samples = list(items)
                self.class_to_idx = {name: i for i, name in enumerate(sorted({y for _, y in items}))}

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, index):
                _, label = self.samples[index]
                return torch.randn(3, 2, 2), self.class_to_idx[label]

            def get_labels(self):
                return [self.class_to_idx[label] for _, label in self.samples]

        class CPUScaler:
            def __init__(self, *args, **kwargs):
                pass

            def scale(self, loss):
                return loss

            def unscale_(self, optimizer):
                pass

            def step(self, optimizer):
                optimizer.step()

            def update(self):
                pass

            def get_scale(self):
                return 1.0

            def is_enabled(self):
                return False

            def state_dict(self):
                return {}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "swi_manifest.json").write_text("{}")
            manifest = {"meta-train": [(f"{i}.jpg", "a" if i < 3 else "b")
                                       for i in range(6)]}
            with patch.dict(os.environ, {"GALLERY_STUDY_BACKBONE": "woodpattern_tiny",
                                      "GALLERY_STUDY_INIT_CHECKPOINT": "",
                                      "GALLERY_STUDY_PRELOAD": "0"}):
                cfg = self.experiment.variant_config("supervised_warmup")
            cfg.update({"warmup_epochs": 2, "warmup_steps": 2, "warmup_batch": 4,
                        "image_size": 2, "embedding_dim": 4, "workers": 0})
            encoder = self.method.GalleryEncoder(
                torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, 8)),
                feature_dim=8, embedding_dim=4)
            scorer = self.method.GalleryScorer(mode="nearest")
            scores = [0.1, 0.2]
            validations = [{"selection_score": score, "meta_val_r1_24": score,
                            "meta_val_r1_all": score, "validation_species": 57,
                            "reference_embedding_spread": 0.5} for score in scores]
            with patch.object(self.experiment.data, "ManifestDataset", TinyDataset), \
                    patch.object(self.experiment.data, "get_transforms", return_value=None), \
                    patch.object(self.experiment, "_model", return_value=(encoder, scorer)), \
                    patch.object(self.experiment, "_validation", side_effect=validations), \
                    patch.object(torch.amp, "GradScaler", CPUScaler, create=True):
                best = self.experiment.train_supervised_warmup(
                    root, root / "results", manifest, cfg, 42, torch.device("cpu"))
            saved = torch.load(best, map_location="cpu", weights_only=False)
            self.assertEqual(saved["epoch"], 2)
            self.assertEqual(saved["checkpoint_kind"], "supervised_warmup")
            self.assertEqual(saved["validation"]["selection_score"], 0.2)
            self.assertTrue((best.parent / "progress.json").exists())
            self.assertTrue((best.parent / "best_validation.json").exists())

    def test_metric_finetune_keeps_warmup_if_training_degrades(self):
        torch = self.torch

        class TinyDataset(torch.utils.data.Dataset):
            def __init__(self, items, transform=None):
                self.samples = list(items)
                self.class_to_idx = {name: i for i, name in enumerate(sorted({y for _, y in items}))}

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, index):
                _, label = self.samples[index]
                return torch.randn(3, 2, 2), self.class_to_idx[label]

        class CPUScaler:
            def __init__(self, *args, **kwargs):
                pass

            def scale(self, loss):
                return loss

            def unscale_(self, optimizer):
                pass

            def step(self, optimizer):
                optimizer.step()

            def update(self):
                pass

            def get_scale(self):
                return 1.0

            def is_enabled(self):
                return False

            def state_dict(self):
                return {}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "swi_manifest.json").write_text("{}")
            items = [(f"patch_256_{j}_{species}_from_Tw{1000 + 2 * species + (j == 2)}.jpg",
                      f"genus_{species}") for species in range(16) for j in range(3)]
            manifest = {"meta-train": items}
            with patch.dict(os.environ, {"GALLERY_STUDY_BACKBONE": "woodpattern_tiny",
                                      "GALLERY_STUDY_INIT_CHECKPOINT": "",
                                      "GALLERY_STUDY_PRELOAD": "0"}):
                cfg = self.experiment.variant_config("metric_no_memory")
            cfg.update({"init_checkpoint": "warmup.pt", "init_checkpoint_sha256": "test",
                        "epochs": 1, "episodes_per_epoch": 1, "ways": (4,), "workers": 0,
                        "microbatch": 12, "image_size": 2})
            encoder = self.method.GalleryEncoder(
                torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, 8)),
                feature_dim=8, embedding_dim=cfg["embedding_dim"])
            scorer = self.method.GalleryScorer(mode="nearest")
            validations = [{"selection_score": score, "meta_val_r1_24": score,
                            "meta_val_r1_all": score, "validation_species": 57,
                            "reference_embedding_spread": 0.5} for score in (0.5, 0.1)]
            with patch.object(self.experiment.data, "ManifestDataset", TinyDataset), \
                    patch.object(self.experiment.data, "get_transforms", return_value=None), \
                    patch.object(self.experiment, "_model", return_value=(encoder, scorer)), \
                    patch.object(self.experiment, "_load_warmup_encoder"), \
                    patch.object(self.experiment, "_validation", side_effect=validations), \
                    patch.object(torch.amp, "GradScaler", CPUScaler, create=True):
                best = self.experiment.train_variant(
                    root, root / "results", manifest, cfg, 42, torch.device("cpu"))
            selected = torch.load(best, map_location="cpu", weights_only=False)
            self.assertEqual(selected["epoch"], 0)
            self.assertEqual(selected["validation"]["selection_score"], 0.5)
            self.assertEqual(self.experiment.json.loads(
                (best.parent / "best_validation.json").read_text())["epoch"], 0)

    def test_smoke_retries_after_amp_skips(self):
        torch = self.torch

        class SkippingScaler:
            def __init__(self, *args, **kwargs):
                self.remaining = 2
                self.scale_value = 1024.0
                self.skipped = False

            def scale(self, loss):
                return loss

            def unscale_(self, optimizer):
                pass

            def step(self, optimizer):
                self.skipped = self.remaining > 0
                if self.skipped:
                    self.remaining -= 1
                else:
                    optimizer.step()

            def update(self):
                if self.skipped:
                    self.scale_value /= 2

            def get_scale(self):
                return self.scale_value

            def is_enabled(self):
                return True

        with patch.object(torch.amp, "GradScaler", SkippingScaler, create=True):
            loss = self.experiment.smoke_test(torch.device("cpu"))
        self.assertTrue(self.np.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
