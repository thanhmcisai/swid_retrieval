"""Protocol and gradient tests for shortlist-mined wood retrieval."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


@unittest.skipUnless(all(importlib.util.find_spec(name) is not None for name in
                         ("torch", "numpy", "pandas", "cv2", "albumentations")),
                     "Image-study dependencies unavailable")
class HardShortlistTest(unittest.TestCase):
    def test_negative_mining_is_within_raw_top128(self):
        import numpy as np
        import torch
        from swid_retrieval import wood_hard_negative as hard

        prototypes = torch.eye(130)
        query = torch.arange(130, dtype=torch.float32)
        query /= query.norm()
        raw = query @ prototypes.T
        raw[5] = -torch.inf
        top128 = set(raw.topk(128).indices.tolist())
        nearest = hard.mine_negative_classes(query, prototypes, 5, 16,
                                              False, np.random.default_rng(2))
        random = hard.mine_negative_classes(query, prototypes, 5, 16,
                                             True, np.random.default_rng(2))
        self.assertEqual(nearest.tolist(), raw.topk(15).indices.tolist())
        self.assertEqual(len(set(random.tolist())), 15)
        self.assertTrue(set(random.tolist()).issubset(top128))
        self.assertNotIn(5, nearest.tolist())
        self.assertNotIn(5, random.tolist())

    def test_query_scale_gallery_does_not_take_true_label(self):
        from swid_retrieval import wood_hard_negative as hard

        items = [("/x/a/scale_256/patch_256_0_from_Tw1.jpg", "a"),
                 ("/x/a/scale_512/patch_512_0_from_Tw2.jpg", "a"),
                 ("/x/b/scale_512/patch_512_0_from_Tw3.jpg", "b")]
        same, different, eligible = hard.query_scale_gallery(items, ["a", "b"], 256)
        self.assertEqual(same, [0, 2])
        self.assertEqual(different, [1, 2])
        self.assertEqual(eligible, [True, False])

    def test_random_control_preserves_query_rng_schedule(self):
        import numpy as np
        import torch
        from torch.nn import functional as F
        from swid_retrieval import wood_hard_negative as hard

        class Model:
            def __call__(self, q, qt, r, rt, labels, scans, **kwargs):
                classes = labels.unique(sorted=True)
                return torch.zeros(1, len(classes)), classes

        bank_g = F.normalize(torch.randn(130, 512), dim=-1)
        bank_t = torch.randn(130, 16, 512)
        names = [f"species_{i}" for i in range(130)]
        args = (Model(), torch.randn(2, 512), torch.randn(2, 16, 512),
                torch.randn(2, 512), torch.randn(2, 16, 512),
                [0, 1], ["scan_a", "scan_b"], names[:2], names,
                torch.arange(130), bank_g, bank_t,
                [f"scan_{i}" for i in range(130)], bank_g,
                16)
        rng_random, rng_hard = np.random.default_rng(7), np.random.default_rng(7)
        hard.shortlist_loss(*args, True, 2, rng_random,
                            {"gate": True, "consensus": True,
                             "reference_chunk": 64})
        hard.shortlist_loss(*args, False, 2, rng_hard,
                            {"gate": True, "consensus": True,
                             "reference_chunk": 64})
        self.assertEqual(rng_random.integers(100000), rng_hard.integers(100000))

    def test_shortlist_loss_reaches_encoder_and_qkv(self):
        import numpy as np
        import torch
        from torch.nn import functional as F
        from swid_retrieval import wood_hard_negative as hard
        from swid_retrieval import wood_correspondence_experiment as corr
        from swid_retrieval.wood_correspondence_method import WoodCorrespondence

        torch.set_num_threads(1)

        class Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 512)

            def forward_features(self, images):
                x = self.linear(images.mean(dim=(2, 3)))
                return {"x_norm_clstoken": x,
                        "x_norm_patchtokens": x[:, None].repeat(1, 16, 1)}

        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = Backbone()

            def project(self, x):
                return F.normalize(x, dim=-1)

        encoder = Encoder()
        model = WoodCorrespondence(dimension=512, token_dim=8)
        optimizer = torch.optim.AdamW(
            list(encoder.parameters()) + list(model.parameters()), lr=1e-3)
        before = encoder.backbone.linear.weight.detach().clone()
        before_head = model.query_key.weight.detach().clone()
        bank_labels = [f"species_{i}" for i in range(130)]
        bank_g = F.normalize(torch.randn(130, 512), dim=-1)
        bank_t = torch.randn(130, 16, 512)
        bank_proto = F.normalize(bank_g.clone(), dim=-1)
        bank = (bank_labels, torch.arange(130), bank_g, bank_t,
                bank_proto, [f"scan_{i}" for i in range(130)])
        cfg = {"microbatch": 2, "hard_queries": 2, "hard_weight": 0.5,
               "bank_weight": 0.5}
        loss, _, hard_acc = hard._step(
            model, encoder, torch.randn(4, 3, 4, 4),
            (2, [0, 1], ["scan_a", "scan_b"],
             bank_labels[:2], [256, 256], [512, 512]),
            bank, "hard16_cross_scale", corr.recipe("qkv", "joint"), cfg,
            optimizer, torch.device("cpu"), np.random.default_rng(43))
        self.assertTrue(np.isfinite(loss))
        self.assertGreaterEqual(hard_acc, 0)
        self.assertFalse(torch.equal(before, encoder.backbone.linear.weight))
        self.assertFalse(torch.equal(before_head, model.query_key.weight))

    def test_selection_lock_rejects_changed_evidence(self):
        import pandas as pd
        from swid_retrieval import wood_hard_negative as hard

        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            for arm, score in (("source", 0.2), ("cross_scale", 0.25),
                               ("hard16", 0.3)):
                folder = out / "validation" / arm
                folder.mkdir(parents=True)
                pd.DataFrame([{"arm": arm, "fold": fold, "budget": 10,
                               "width": 128, "method": "qkv_all_refs",
                               "macro_r1": score} for fold in (0, 1)]).to_csv(
                    folder / "summary.csv", index=False)
                folder.joinpath("protocol.json").write_text(json.dumps({
                    "folds": [0, 1], "budgets": [10], "widths": [128]}))
            cfg = {"arms": ["hard16"], "folds": [0, 1],
                   "budgets": [10], "widths": [128]}
            hard._select(out, cfg)
            self.assertEqual(hard._locked(out)["arm"], "hard16")
            hard._select(out, cfg)
            (out / "validation" / "hard16" / "summary.csv").write_text("changed")
            with self.assertRaisesRegex(ValueError, "evidence changed"):
                hard._locked(out)

    def test_validation_grid_and_scan_disjoint_queries(self):
        import numpy as np
        import pandas as pd
        import torch
        from swid_retrieval import wood_hard_negative as hard

        refs = []
        queries = []
        for species in range(637):
            label = f"species_{species}"
            refs.append((f"/train/{label}/scale_256/"
                         f"patch_256_from_Tw{species*2:05d}.jpg", label))
            if species < 57:
                queries.extend((f"/val/{label}/scale_512/"
                                f"patch_512_{j}_from_Tw{species*2+1:05d}.jpg",
                                label) for j in range(5))

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.global_adapter = torch.nn.Linear(512, 512, bias=False)
                torch.nn.init.zeros_(self.global_adapter.weight)

        def features(encoder, items, probes, base_cfg, device, cache, digest):
            def encoded(entries):
                arr = np.zeros((len(entries), 512), dtype=np.float32)
                for i, (_, label) in enumerate(entries):
                    arr[i, int(label.rsplit("_", 1)[1]) % 512] = 1
                return torch.from_numpy(arr)
            rg, qg = encoded(items), encoded(probes)
            return rg, rg[:, None].repeat(1, 16, 1), qg, qg[:, None].repeat(1, 16, 1)

        def rerank(model, qg, qt, rg, rt, ref_labels, scans, candidates, base):
            result = torch.full_like(base, -torch.inf)
            result.scatter_(1, candidates, base.gather(1, candidates))
            return result

        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            checkpoint = out / "best.pt"
            checkpoint.write_text("synthetic")
            cfg = {"folds": [0, 1], "budgets": [1], "widths": [5],
                   "image_batch": 16, "workers": 0}
            manifest = {"meta-train": refs, "meta-val": []}
            with patch.object(hard, "_load", return_value=(
                    checkpoint, object(), Model(), {})), \
                 patch.object(hard.study, "validation_items",
                              return_value=([], queries)), \
                 patch.object(hard, "_reference_data", side_effect=features), \
                 patch.object(hard.ml, "rerank_all_references", side_effect=rerank):
                hard._validate_arm(out, out, out, manifest, out, out,
                                   cfg, "hard16", torch.device("cpu"))
                hard._validate_arm(out, out, out, manifest, out, out,
                                   cfg, "hard16", torch.device("cpu"))
            summary = pd.read_csv(out / "validation/hard16/summary.csv")
            detail = pd.read_csv(out / "validation/hard16/per_query.csv")
            self.assertEqual(len(summary), 6)
            self.assertEqual(len(detail), 2 * 3 * 285)
            self.assertEqual(set(summary.fold), {0, 1})
            self.assertTrue(detail.candidate_hit.dropna().eq(True).all())

    def test_scale_stress_uses_paired_query_policy(self):
        import numpy as np
        import pandas as pd
        import torch
        from swid_retrieval import wood_hard_negative as hard

        refs, queries = [], []
        for species in range(637):
            label = f"species_{species}"
            for scan, scale in ((0, 256), (1, 512)):
                refs.append((f"/train/{label}/scale_{scale}/"
                             f"patch_{scale}_from_Tw{species*3+scan:05d}.jpg", label))
            if species < 57:
                queries.extend((f"/val/{label}/scale_{256 if j % 2 else 512}/"
                                f"patch_{j}_from_Tw{species*3+2:05d}.jpg", label)
                               for j in range(5))

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.global_adapter = torch.nn.Linear(512, 512, bias=False)
                torch.nn.init.zeros_(self.global_adapter.weight)

        def features(encoder, items, probes, base_cfg, device, cache, digest):
            def encoded(entries):
                arr = np.zeros((len(entries), 512), dtype=np.float32)
                for i, (_, label) in enumerate(entries):
                    arr[i, int(label.rsplit("_", 1)[1]) % 512] = 1
                return torch.from_numpy(arr)
            rg, qg = encoded(items), encoded(probes)
            return rg, rg[:, None].repeat(1, 16, 1), qg, qg[:, None].repeat(1, 16, 1)

        def rerank(model, qg, qt, rg, rt, ref_labels, scans, candidates, base):
            result = torch.full_like(base, -torch.inf)
            result.scatter_(1, candidates, base.gather(1, candidates))
            return result

        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            checkpoint = out / "best.pt"
            checkpoint.write_text("synthetic")
            (out / "selection_lock.json").write_text("locked")
            cfg = {"folds": [0, 1], "budgets": [20],
                   "image_batch": 16, "workers": 0}
            manifest = {"meta-train": refs, "meta-val": []}
            with patch.object(hard, "_load", return_value=(
                    checkpoint, object(), Model(), {})), \
                 patch.object(hard.study, "validation_items",
                              return_value=([], queries)), \
                 patch.object(hard, "_reference_data", side_effect=features), \
                 patch.object(hard.ml, "rerank_all_references", side_effect=rerank):
                hard._stress_arm(out, out, out, manifest, out, out,
                                 cfg, "hard16", torch.device("cpu"))
            result = pd.read_csv(out / "scale_stress/hard16/per_query.csv")
            self.assertEqual(len(result), 2 * 2 * 2 * 285)
            self.assertEqual(result.paired_eligible.sum(), len(result))
            self.assertTrue(result[result.policy == "same_query_scale"]
                            .eval("query_scale == reference_scale").all())
            self.assertTrue(result[result.policy == "different_query_scale"]
                            .eval("query_scale != reference_scale").all())
            self.assertEqual(result.groupby(["fold", "method", "query_path"])
                             .policy.nunique().min(), 2)

    def test_prior_comparator_can_be_filtered_for_pilot(self):
        import pandas as pd
        from swid_retrieval import wood_hard_negative as hard

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            old = base / "old" / "validation" / "source"
            old.mkdir(parents=True)
            checkpoint = base / "source.pt"
            checkpoint.write_text("synthetic")
            digest = hard.study._hash_file(checkpoint)
            rows, details = [], []
            for fold in (0, 1):
                for k in (1, 10):
                    for width, method in ((0, "raw_prototype"),
                                          (0, "adapted_prototype"),
                                          (32, "qkv_all_refs"),
                                          (128, "qkv_all_refs")):
                        rows.append({"arm": "source", "fold": fold, "budget": k,
                                     "width": width, "method": method,
                                     "macro_r1": 0.2})
                        details.extend({"arm": "source", "fold": fold,
                                        "budget": k, "width": width,
                                        "method": method, "query_path": f"q{i}"}
                                       for i in range(285))
            pd.DataFrame(rows).to_csv(old / "summary.csv", index=False)
            pd.DataFrame(details).to_csv(old / "per_query.csv", index=False)
            old.joinpath("protocol.json").write_text(json.dumps({
                "checkpoint_sha256": digest, "folds": [0, 1],
                "budgets": [1, 10], "widths": [32, 128]}))
            cfg = {"folds": [0, 1], "budgets": [10], "widths": [128]}
            with patch.object(hard, "_load", return_value=(
                    checkpoint, object(), object(), {})):
                hard._validate_arm(base, base / "new", base / "old", {},
                                   base, base, cfg, "source", None)
            filtered = pd.read_csv(base / "new/validation/source/summary.csv")
            per_query = pd.read_csv(base / "new/validation/source/per_query.csv")
            self.assertEqual(len(filtered), 6)
            self.assertEqual(len(per_query), 6 * 285)
            self.assertEqual(set(filtered.budget), {10})


if __name__ == "__main__":
    unittest.main()
