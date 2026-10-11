"""Focused protocol tests for the cross-scale end-to-end study."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


@unittest.skipUnless(all(importlib.util.find_spec(name) is not None for name in
                         ("torch", "numpy", "pandas", "cv2", "albumentations")),
                     "Image-study dependencies unavailable")
class MultilevelStudyTest(unittest.TestCase):
    def test_cross_scale_episodes_are_scan_disjoint(self):
        from swid_retrieval import wood_multilevel as ml
        from swid_retrieval import gallery_experiment as study
        from swid_retrieval import wood_correspondence_image_train as image_train

        items = []
        for species in range(128):
            for scan in range(2):
                scale = 256 if scan == 0 else 512
                for patch in range(7):
                    items.append((f"/train/sp_{species}/scale_{scale}/"
                                  f"patch_{scale}_{patch}_from_Tw{species*2+scan:05d}.jpg",
                                  f"sp_{species}"))
        _, valid = image_train._layout(items)
        multi = {k: ml._eligible(items, k) for k in (1, 2, 5)}
        self.assertEqual(len(multi[5]), 128)
        cfg = {"episodes": 6, "seed": 43}
        batches, layouts = ml._plan(items, valid, True, cfg, 1, multi)
        _, control_layouts = ml._plan(items, valid, False, cfg, 1, multi)
        self.assertEqual(len(batches), 6)
        for batch, layout, control in zip(batches, layouts, control_layouts):
            n_refs, labels, _, _, ref_scales, query_scales = layout
            self.assertEqual(layout[3], control[3])
            for class_id, query in enumerate(batch[n_refs:]):
                own = [batch[i] for i, label in enumerate(labels) if label == class_id]
                self.assertTrue(all(study.source_scan_id(items[i][0]) !=
                                    study.source_scan_id(items[query][0]) for i in own))
                self.assertTrue(any(study.image_scale(items[i][0]) !=
                                    study.image_scale(items[query][0]) for i in own))
                self.assertTrue(any(ref_scales[i] != query_scales[class_id]
                                    for i, label in enumerate(labels) if label == class_id))

    def test_cross_scale_loss_backpropagates_and_requires_positive(self):
        import torch
        from swid_retrieval import wood_multilevel as ml

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.global_adapter = torch.nn.Linear(3, 3, bias=False)

        model = Model()
        q = torch.randn(2, 3, requires_grad=True)
        r = torch.randn(4, 3, requires_grad=True)
        loss = ml.cross_scale_loss(model, q, r, [0, 0, 1, 1],
                                   [256, 512, 256, 512], [256, 256])
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(q.grad.abs().sum().item(), 0)
        self.assertGreater(model.global_adapter.weight.grad.abs().sum().item(), 0)
        with self.assertRaisesRegex(ValueError, "positive is missing"):
            ml.cross_scale_loss(model, q, r, [0, 0, 1, 1],
                                [256, 256, 256, 256], [256, 256])

    def test_rerank_uses_all_selected_references(self):
        import torch
        from swid_retrieval import wood_multilevel as ml

        class Model:
            def __init__(self):
                self.counts = []

            def __call__(self, query, query_tokens, references, reference_tokens,
                         reference_labels, scans, **kwargs):
                self.counts.append(len(references))
                classes = reference_labels.unique(sorted=True)
                return torch.arange(len(classes), dtype=torch.float32)[None], classes

        model = Model()
        q = torch.randn(1, 4)
        refs = torch.randn(6, 4)
        tokens = torch.randn(6, 2, 4)
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        selected = torch.tensor([[0, 2]])
        scores = ml.rerank_all_references(
            model, q, tokens[:1], refs, tokens, labels,
            ["scan0", "scan0", "scan1", "scan1", "scan2", "scan2"],
            selected, torch.zeros(1, 3))
        self.assertEqual(model.counts, [4])
        self.assertTrue(torch.isneginf(scores[0, 1]))
        self.assertTrue(torch.isfinite(scores[0, selected[0]]).all())

    def test_r5_does_not_count_masked_non_candidates(self):
        import torch
        from swid_retrieval import wood_multilevel as ml

        scores = torch.full((1, 6), -torch.inf)
        scores[0, 4] = 1.0
        result = ml._metrics(scores, torch.tensor([2]))
        self.assertEqual(result["macro_r1"], 0.0)
        self.assertEqual(result["macro_r5"], 0.0)

    def test_joint_step_updates_encoder_and_correspondence(self):
        import numpy as np
        import torch
        from torch.nn import functional as F
        from swid_retrieval import wood_multilevel as ml
        from swid_retrieval import wood_correspondence_experiment as corr
        from swid_retrieval.wood_correspondence_method import WoodCorrespondence

        class Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 512)

            def forward_features(self, images):
                global_ = self.linear(images.mean(dim=(2, 3)))
                return {"x_norm_clstoken": global_,
                        "x_norm_patchtokens": global_[:, None].repeat(1, 16, 1)}

        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = Backbone()

            def project(self, x):
                return F.normalize(x, dim=-1)

        encoder = Encoder()
        model = WoodCorrespondence(dimension=512, token_dim=8)
        optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(model.parameters()),
                                      lr=1e-3)
        before_encoder = encoder.backbone.linear.weight.detach().clone()
        before_head = model.query_key.weight.detach().clone()
        layout = (2, [0, 1], ["scan_a", "scan_b"],
                  ["species_0", "species_1"], [256, 256], [512, 512])
        bank = F.normalize(torch.randn(3, 512), dim=-1)
        config = corr.recipe("qkv", "joint")
        settings = {"microbatch": 2, "bank_negatives": 1, "bank_weight": 0.5,
                    "contrastive_weight": 0.2, "margin_weight": 0.2, "margin": 0.1}
        loss, accuracy = ml._step(model, encoder, torch.randn(4, 3, 4, 4), layout,
                                  ["species_0", "species_1", "species_2"], bank,
                                  "cross_scale_contrastive_margin", config, settings,
                                  optimizer, torch.device("cpu"))
        self.assertTrue(np.isfinite(loss))
        self.assertGreaterEqual(accuracy, 0)
        self.assertFalse(torch.equal(before_encoder, encoder.backbone.linear.weight))
        self.assertFalse(torch.equal(before_head, model.query_key.weight))

    def test_validation_lock_is_immutable(self):
        import pandas as pd
        from swid_retrieval import wood_multilevel as ml

        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            for arm, score in (("source", 0.2), ("cross_scale", 0.3)):
                folder = out / "validation" / arm
                folder.mkdir(parents=True)
                pd.DataFrame([{"arm": arm, "fold": fold, "budget": 20,
                               "width": 128, "method": "qkv_all_refs",
                               "macro_r1": score} for fold in (0, 1)]).to_csv(
                    folder / "summary.csv", index=False)
                (folder / "protocol.json").write_text('{"checkpoint_sha256":"test"}')
            cfg = {"arms": ["source", "cross_scale"], "folds": [0, 1]}
            ml._select(out, cfg)
            self.assertEqual(ml._locked_selection(out)["arm"], "cross_scale")
            ml._select(out, cfg)
            (out / "validation" / "cross_scale" / "protocol.json").write_text(
                '{"checkpoint_sha256":"changed"}')
            with self.assertRaisesRegex(ValueError, "provenance changed"):
                ml._locked_selection(out)

    def test_validation_reports_candidate_and_final_ranking(self):
        import numpy as np
        import pandas as pd
        import torch
        from swid_retrieval import wood_multilevel as ml

        refs = []
        queries = []
        for species in range(637):
            label = f"species_{species}"
            for scan, scale in ((0, 256), (1, 512)):
                refs.append((f"/train/{label}/scale_{scale}/"
                             f"patch_{scale}_from_Tw{species*3+scan:05d}.jpg", label))
            if species < 57:
                queries.extend((f"/val/{label}/scale_1024/"
                                f"patch_1024_{j}_from_Tw{species*3+2:05d}.jpg", label)
                               for j in range(5))

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.global_adapter = torch.nn.Linear(512, 512, bias=False)
                torch.nn.init.zeros_(self.global_adapter.weight)

            def forward(self, query, query_tokens, reference, reference_tokens,
                        reference_labels, scans, **kwargs):
                classes = reference_labels.unique(sorted=True)
                return torch.zeros(len(query), len(classes), device=query.device), classes

        def features(encoder, items, cfg, device, path, checkpoint_hash, tokens):
            emb = np.zeros((len(items), 512), dtype=np.float32)
            for i, (_, label) in enumerate(items):
                emb[i, int(label.rsplit("_", 1)[1]) % 512] = 1
            return emb, np.repeat(emb[:, None], 16, axis=1)

        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            checkpoint = out / "source.pt"
            checkpoint.write_text("synthetic")
            cfg = {"seed": 43, "source_seed": 43, "folds": [0],
                   "budgets": [1], "widths": [5], "workers": 0,
                   "image_batch": 32}
            with patch.object(ml, "_load", return_value=(
                    checkpoint, object(), Model(), {})), \
                 patch.object(ml.study, "validation_items", return_value=([], queries)), \
                 patch.object(ml.large, "_cache_features", side_effect=features) as cache:
                summary = ml._validate_arm(
                    out, out, {"meta-train": refs, "meta-val": []}, out, out,
                    cfg, "source", torch.device("cpu"))
                self.assertEqual(cache.call_count, 2)
                repeated = ml._validate_arm(
                    out, out, {"meta-train": refs, "meta-val": []}, out, out,
                    cfg, "source", torch.device("cpu"))
                self.assertEqual(cache.call_count, 2)
                self.assertEqual(len(summary), len(repeated))
                changed = dict(cfg, budgets=[1, 3])
                with self.assertRaisesRegex(ValueError, "protocol changed"):
                    ml._validate_arm(
                        out, out, {"meta-train": refs, "meta-val": []}, out, out,
                        changed, "source", torch.device("cpu"))
            self.assertEqual(set(summary.method), {"raw_prototype",
                                                   "adapted_prototype", "qkv_all_refs"})
            self.assertEqual(len(pd.read_csv(out / "validation" / "source" /
                                             "per_query.csv")), 285 * 3)
            self.assertTrue((summary.loc[summary.method == "qkv_all_refs",
                                         "candidate_macro_recall"] <= 1).all())


if __name__ == "__main__":
    unittest.main()
