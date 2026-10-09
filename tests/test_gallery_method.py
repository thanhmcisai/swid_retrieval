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
        if not hasattr(self.torch.amp, "GradScaler"):
            self.skipTest("Local PyTorch lacks torch.amp.GradScaler; Colab has it")
        self.experiment.smoke_test(self.torch.device("cpu"))

    def test_cross_scale_sampler_keeps_scan_disjoint_positives(self):
        items = [(f"scale_{scale}/patch_{scale}_{i}_from_Tw{100 + 2 * species + scan}.jpg",
                  f"genus_{species}")
                 for species in range(4) for scan, scale in enumerate((256, 512))
                 for i in range(3)]
        labels = [label for _, label in items]
        groups = [self.experiment.source_scan_id(path) for path, _ in items]
        scales = [self.experiment.image_scale(path) for path, _ in items]
        sampler = self.experiment.EpisodeSampler(labels, (4,), 2, 1, 8, 42,
                                                 groups=groups, scales=scales,
                                                 cross_scale=True)
        for indices in sampler:
            support, queries = indices[:8], indices[8:]
            for species in range(4):
                refs = support[species * 2:species * 2 + 2]
                query = queries[species]
                self.assertTrue(all(groups[ref] != groups[query] for ref in refs))
                self.assertTrue(all(scales[ref] != scales[query] for ref in refs))

    def test_frontier_recipes_are_isolated_and_dinov3_requires_weights(self):
        with patch.dict(os.environ, {"GALLERY_STUDY_BACKBONE": "woodpattern_tiny",
                                  "GALLERY_STUDY_INIT_CHECKPOINT": "",
                                  "GALLERY_STUDY_VALIDATION_TRAIN_DISTRACTORS": "1"}):
            large = self.experiment.variant_config("metric_large")
            local = self.experiment.variant_config("local_evidence")
            mixed = self.experiment.variant_config("local_evidence_ce_scale")
        self.assertEqual(large["ways"], (16, 32, 64))
        self.assertEqual(large["memory_size"], 0)
        self.assertEqual(local["local_weight"], 0.35)
        self.assertTrue(mixed["cross_scale"])
        self.assertEqual(local["validation_train_distractors"], 1)
        with patch.dict(os.environ, {"GALLERY_STUDY_BACKBONE": "dinov3_vits16",
                                  "GALLERY_STUDY_DINOV3_WEIGHTS": "",
                                  "GALLERY_STUDY_INIT_CHECKPOINT": ""}):
            with self.assertRaisesRegex(FileNotFoundError, "licensed local"):
                self.experiment.variant_config("pretrained_control")

    def test_dinov2_patch_recipes_isolate_backbone_and_score_ablations(self):
        with patch.dict(os.environ, {"GALLERY_STUDY_BACKBONE": "dinov2_vits14",
                                  "GALLERY_STUDY_INIT_CHECKPOINT": ""}):
            full = self.experiment.variant_config("prototype_large")
            frozen = self.experiment.variant_config("prototype_large_frozen")
            patch_model = self.experiment.variant_config("patch_evidence")
            patch_frozen = self.experiment.variant_config("patch_evidence_frozen")
            scale = self.experiment.variant_config("patch_evidence_scale")
        self.assertEqual(full["ways"], frozen["ways"])
        self.assertTrue(full["train_backbone"])
        self.assertFalse(frozen["train_backbone"])
        self.assertTrue(patch_model["train_backbone"])
        self.assertFalse(patch_frozen["train_backbone"])
        self.assertEqual(patch_model["local_feature_dim"], 384)
        self.assertEqual(patch_model["local_token_count"], 16)
        self.assertEqual(patch_model["scorer_mode"], "prototype")
        self.assertTrue(scale["cross_scale"])
        with patch.dict(os.environ, {"GALLERY_STUDY_BACKBONE": "woodpattern_tiny",
                                  "GALLERY_STUDY_INIT_CHECKPOINT": ""}):
            with self.assertRaisesRegex(ValueError, "requires a DINOv2"):
                self.experiment.variant_config("patch_evidence")

    def test_dense_patch_backbone_keeps_cls_and_trains_local_projection(self):
        torch = self.torch

        class TinyDino(torch.nn.Module):
            num_features = 8

            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 8, 1)

            def forward_features(self, x):
                patches = torch.nn.functional.adaptive_avg_pool2d(self.conv(x), (4, 4))
                patches = patches.flatten(2).transpose(1, 2)
                return {"x_norm_clstoken": patches.mean(1),
                        "x_norm_patchtokens": patches}

            def forward(self, x):
                return self.forward_features(x)["x_norm_clstoken"]

        base = TinyDino()
        dense = self.method.DensePatchBackbone(base)
        encoder = self.method.GalleryEncoder(dense, 8, 4, local_dim=4,
                                             local_feature_dim=8)
        x = torch.randn(2, 3, 8, 8)
        cls, patches = dense.forward_with_tokens(x)
        self.assertTrue(torch.allclose(cls, base(x)))
        self.assertEqual(patches.shape, (2, 16, 8))
        global_embedding, local_embedding = encoder.forward_with_tokens(x)
        self.assertEqual(global_embedding.shape, (2, 4))
        self.assertEqual(local_embedding.shape, (2, 16, 4))
        (global_embedding.square().mean() + local_embedding[:, 0].sum()).backward()
        self.assertIsNotNone(base.conv.weight.grad)
        self.assertIsNotNone(encoder.local_projection[1].weight.grad)

    def test_patch_episode_freeze_and_gradient_replay(self):
        torch = self.torch

        class TinyDino(torch.nn.Module):
            num_features = 8

            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 8, 1)

            def forward_features(self, images):
                patches = torch.nn.functional.adaptive_avg_pool2d(self.conv(images), (4, 4))
                patches = patches.flatten(2).transpose(1, 2)
                return {"x_norm_clstoken": patches.mean(1),
                        "x_norm_patchtokens": patches}

            def forward(self, images):
                return self.forward_features(images)["x_norm_clstoken"]

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

        images = torch.randn(6, 3, 8, 8)
        labels = torch.tensor([0, 0, 1, 1, 0, 1])
        cfg = {"support": 2, "queries": 1, "microbatch": 3,
               "objective": "episode", "variable_gallery": False,
               "local_weight": 0.25, "temperature": 0.07,
               "positive_weight": 0, "aux_ce_weight": 0}
        for train_backbone in (False, True):
            base = TinyDino()
            encoder = self.method.GalleryEncoder(
                self.method.DensePatchBackbone(base), 8, 4, local_dim=4,
                local_feature_dim=8)
            encoder.backbone.requires_grad_(train_backbone)
            scorer = self.method.GalleryScorer(mode="prototype")
            optimizer = torch.optim.SGD(
                (parameter for parameter in encoder.parameters() if parameter.requires_grad),
                lr=0.1)
            before_backbone = base.conv.weight.detach().clone()
            before_local = encoder.local_projection[1].weight.detach().clone()
            loss, details = self.experiment._episode_step(
                images, labels, {**cfg, "train_backbone": train_backbone},
                encoder, scorer, optimizer, CPUScaler(), torch.device("cpu"))
            self.assertTrue(self.np.isfinite(loss))
            self.assertEqual(details["gallery_images"], 4)
            self.assertEqual(torch.equal(before_backbone, base.conv.weight),
                             not train_backbone)
            self.assertFalse(torch.equal(before_local, encoder.local_projection[1].weight))

    def test_local_shortlist_reranks_and_backpropagates(self):
        torch, np = self.torch, self.np
        scorer = self.method.GalleryScorer(mode="nearest")
        vectors = np.asarray([[1., 0.], [1., 0.]], dtype=np.float32)
        qt = np.asarray([[[1., 0.], [1., 0.]]], dtype=np.float32)
        rt = np.asarray([[[0., 1.], [0., 1.]], [[1., 0.], [1., 0.]]], dtype=np.float16)
        cfg = {"local_candidates": 2, "local_refs_per_species": 1,
               "temperature": 0.07, "local_weight": 0.5}
        scores, classes = self.experiment.score_queries_evidence(
            scorer, vectors[:1], vectors, ["a", "b"], qt, rt, cfg, torch.device("cpu"))
        self.assertEqual(classes[scores.argmax(1)].tolist(), ["b"])

        class TinyBackbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.global_linear = torch.nn.Linear(12, 8)
                self.local_linear = torch.nn.Linear(12, 256)

            def forward_with_tokens(self, images):
                flat = images.flatten(1)
                return self.global_linear(flat), self.local_linear(flat).reshape(-1, 2, 128)

            def forward(self, images):
                return self.global_linear(images.flatten(1))

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

        encoder = self.method.GalleryEncoder(TinyBackbone(), 8, 4, local_dim=8)
        classifier = torch.nn.Linear(4, 4)
        optimizer = torch.optim.AdamW(list(encoder.parameters()) +
                                      list(classifier.parameters()), lr=1e-2)
        cfg.update({"support": 2, "queries": 1, "microbatch": 4,
                    "train_backbone": True, "objective": "episode", "variable_gallery": False,
                    "stability_weight": 0.0, "pseudo_ood_weight": 0.0,
                    "positive_weight": 0.1, "aux_ce_weight": 0.2})
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 0, 1, 2, 3])
        before = encoder.backbone.local_linear.weight.detach().clone()
        loss, details = self.experiment._episode_step(
            torch.randn(12, 3, 2, 2), labels, cfg, encoder, scorer, optimizer,
            CPUScaler(), torch.device("cpu"), classifier=classifier)
        self.assertTrue(np.isfinite(loss))
        self.assertIn("aux_ce", details)
        self.assertFalse(torch.equal(before, encoder.backbone.local_linear.weight))

    def test_meta_val_stress_uses_balanced_references_without_public_images(self):
        np, torch = self.np, self.torch
        manifest = {"meta-train": [(f"patch_256_0_from_Tw{100 + i}.jpg", f"train_{i}")
                                   for i in range(4)],
                    "meta-val": [(f"patch_256_{j}_from_Tw{1000 + 2 * i + scan}.jpg",
                                  f"val_{i}") for i in range(24) for scan in range(2)
                                 for j in range(5)]}
        label_order = sorted({label for split in manifest.values() for _, label in split})
        lookup = {label: i for i, label in enumerate(label_order)}

        def fake_encode(_encoder, items, _cfg, _device):
            matrix = np.zeros((len(items), len(lookup)), np.float32)
            for row, (_, label) in enumerate(items):
                matrix[row, lookup[label]] = 1
            return matrix

        cfg = {"group_mode": "scan_disjoint", "validation_folds": 2,
               "validation_train_distractors": 1, "local_weight": 0}
        with patch.object(self.experiment, "encode_items", side_effect=fake_encode):
            result = self.experiment._validation(None, self.method.GalleryScorer(mode="nearest"),
                                                 manifest, cfg, torch.device("cpu"))
        self.assertEqual(result["stress_gallery_species"], 28)
        self.assertEqual(result["stress_gallery_images"], 28)
        self.assertEqual(result["validation_folds"], 2)
        self.assertEqual(result["meta_val_r1_stress"], 1.0)
        self.assertEqual(result["meta_val_gallery_curve"]["28"], 1.0)
        self.assertEqual(result["meta_val_global_r1_24"], result["meta_val_r1_24"])
        self.assertEqual(result["meta_val_global_r1_stress"], result["meta_val_r1_stress"])
        self.assertEqual(result["selection_score"], 1.0)
        cfg.update({"local_weight": 0.35, "local_candidates": 8,
                    "local_refs_per_species": 2, "temperature": 0.07})

        def fake_local(encoder, items, recipe, device):
            global_emb = fake_encode(encoder, items, recipe, device)
            return global_emb, np.repeat(global_emb[:, None, :], 12, axis=1)

        with patch.object(self.experiment, "encode_items_with_tokens", side_effect=fake_local):
            local = self.experiment._validation_once(
                None, self.method.GalleryScorer(mode="nearest"), manifest, cfg,
                torch.device("cpu"), fold=0)
        self.assertEqual(local["stress_gallery_species"], 28)
        self.assertEqual(local["meta_val_global_r1_stress"], 1.0)
        self.assertEqual(local["selection_score"], 1.0)

    def test_meta_val_gallery_curve_uses_fixed_target_classes(self):
        np, torch = self.np, self.torch
        manifest = {"meta-train": [(f"train_{i}.jpg", f"train_{i}") for i in range(260)],
                    "meta-val": [(f"patch_256_{j}_from_Tw{1000 + 2 * i + scan}.jpg",
                                  f"val_{i}") for i in range(24) for scan in range(2)
                                 for j in range(5)]}
        labels = sorted({label for rows in manifest.values() for _, label in rows})
        lookup = {label: i for i, label in enumerate(labels)}

        def fake_encode(_encoder, items, _cfg, _device):
            matrix = np.zeros((len(items), len(labels)), dtype=np.float32)
            for row, (_, label) in enumerate(items):
                matrix[row, lookup[label]] = 1
            return matrix

        cfg = {"group_mode": "scan_disjoint", "validation_folds": 2,
               "validation_train_distractors": 1, "local_weight": 0}
        with patch.object(self.experiment, "encode_items", side_effect=fake_encode):
            result = self.experiment._validation(
                None, self.method.GalleryScorer(mode="prototype"), manifest,
                cfg, torch.device("cpu"))
        self.assertEqual(set(result["meta_val_gallery_curve"]), {"128", "256", "284"})
        self.assertTrue(all(score == 1 for score in result["meta_val_gallery_curve"].values()))
        self.assertEqual(result["meta_val_global_gallery_curve"], result["meta_val_gallery_curve"])
        self.assertEqual(len(result["meta_val_species_r1_all"]), 24)
        self.assertEqual(len(result["meta_val_gallery_curve_by_species"]["284"]), 24)
        self.assertEqual(result["meta_val_gallery_curve_by_species"]["284"]["val_0"], 1)
        self.assertEqual(result["stress_gallery_species"], 284)

    def test_local_embedding_cache_resumes_with_aligned_tokens(self):
        torch, np = self.torch, self.np

        class TinyEncoder(torch.nn.Module):
            def forward_with_tokens(self, images):
                return torch.nn.functional.normalize(images[:, :2], dim=1), images[:, None, :8].repeat(1, 4, 1)

        def fake_loader(items, _cfg):
            yield torch.arange(len(items) * 8, dtype=torch.float32).reshape(len(items), 8) + 1, None

        cfg = {"image_size": 224, "embedding_dim": 2, "backbone": "woodpattern_single_scale",
               "local_dim": 8, "local_weight": 0.35}
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "embeddings.npy"
            items = [("a.jpg", "a"), ("b.jpg", "b")]
            with patch.object(self.experiment, "_loader", side_effect=fake_loader):
                first = self.experiment._embedding_cache(
                    TinyEncoder(), items, cfg, torch.device("cpu"), target, "hash")
            second = self.experiment._embedding_cache(
                TinyEncoder(), items, cfg, torch.device("cpu"), target, "hash")
            tokens = np.load(target.with_name("embeddings_tokens.npy"))
            self.assertEqual(first.shape, (2, 2))
            self.assertTrue(np.array_equal(first, second))
            self.assertEqual(tokens.shape, (2, 4, 8))
            self.assertEqual(tokens.dtype, np.float16)

    def test_dinov2_patch_cache_uses_sixteen_tokens(self):
        torch, np = self.torch, self.np

        class TinyEncoder(torch.nn.Module):
            def forward_with_tokens(self, images):
                return torch.nn.functional.normalize(images[:, :2], dim=1), images[:, None, :8].repeat(1, 16, 1)

        def fake_loader(items, _cfg):
            yield torch.ones(len(items), 8), None

        cfg = {"image_size": 224, "embedding_dim": 2, "backbone": "dinov2_vits14",
               "local_dim": 8, "local_weight": 0.25, "local_token_count": 16}
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "embeddings.npy"
            with patch.object(self.experiment, "_loader", side_effect=fake_loader):
                self.experiment._embedding_cache(TinyEncoder(), [("a.jpg", "a")], cfg,
                                                 torch.device("cpu"), target, "hash")
            tokens = np.load(target.with_name("embeddings_tokens.npy"))
            self.assertEqual(tokens.shape, (1, 16, 8))

    def test_local_encoder_accepts_only_local_head_missing_from_warmup(self):
        torch = self.torch
        source = self.method.GalleryEncoder(torch.nn.Linear(4, 8), 8, 4)
        target = self.method.GalleryEncoder(torch.nn.Linear(4, 8), 8, 4, local_dim=8)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "swi_manifest.json").write_text("{}")
            checkpoint = root / "warmup.pt"
            torch.save({"checkpoint_kind": "supervised_warmup", "seed": 42, "epoch": 1,
                        "manifest_sha256": self.experiment._hash_file(root / "swi_manifest.json"),
                        "config": {"backbone": "woodpattern_tiny", "embedding_dim": 4},
                        "encoder": source.state_dict()}, checkpoint)
            cfg = {"variant": "local_evidence", "backbone": "woodpattern_tiny",
                   "embedding_dim": 4, "local_weight": 0.35,
                   "init_checkpoint": str(checkpoint),
                   "init_checkpoint_sha256": self.experiment._hash_file(checkpoint)}
            self.experiment._load_warmup_encoder(root, cfg, 42, target)
            self.assertTrue(torch.equal(source.backbone.weight, target.backbone.weight))

    def test_woodpattern_local_episode_replays_backbone_gradients(self):
        torch = self.torch
        from swid_retrieval.wood_encoder import WoodPatternNet

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

        backbone = WoodPatternNet()
        encoder = self.method.GalleryEncoder(backbone, backbone.num_features, 32, local_dim=16)
        scorer = self.method.GalleryScorer(mode="nearest")
        optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-3)
        before = backbone.stem[0].weight.detach().clone()
        cfg = {"support": 2, "queries": 1, "microbatch": 3, "train_backbone": True,
               "objective": "episode", "variable_gallery": False,
               "local_weight": 0.35, "temperature": 0.07,
               "positive_weight": 0, "aux_ce_weight": 0}
        loss, details = self.experiment._episode_step(
            torch.randn(6, 3, 64, 64), torch.tensor([0, 0, 1, 1, 0, 1]),
            cfg, encoder, scorer, optimizer, CPUScaler(), torch.device("cpu"))
        self.assertTrue(self.np.isfinite(loss))
        self.assertEqual(details["gallery_images"], 4)
        self.assertFalse(torch.equal(before, backbone.stem[0].weight))

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
