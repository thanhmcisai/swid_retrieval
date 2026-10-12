"""Focused protocol tests for the multi-scale/rank study."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from swid_retrieval import wood_scale_rank_method as method
from swid_retrieval import wood_scale_rank_study as runner
from swid_retrieval.wood_correspondence_method import WoodCorrespondence


def _items(n_species=18, n_scans=4):
    return [
        (f"/tmp/train/species_{species}/scale_{scale}/"
         f"patch_{scale}_0_0_from_Tw{species * n_scans + scan}.jpg",
         f"species_{species}")
        for species in range(n_species)
        for scan in range(n_scans)
        for scale in (256, 512, 1024, 2048)
    ]


def test_balanced_episodes_are_scan_disjoint_and_multiscale():
    items = _items()
    catalog = method.episode_catalog(items, (5, 10))
    batches, layouts = method.plan_episodes(
        items, catalog, [[16, 5], [16, 10]], 4, 43, 1, True)
    for batch, layout in zip(batches, layouts):
        n_refs, labels, scans, classes, ref_scales, query_scales = layout
        k = n_refs // 16
        assert k in (5, 10)
        assert len(classes) == 16
        assert len(batch) == n_refs + 16
        for row in range(16):
            refs = slice(row * k, (row + 1) * k)
            assert labels[refs] == [row] * k
            assert set(ref_scales[refs]) == {256, 512, 1024, 2048}
            assert set(ref_scales[refs]) - {query_scales[row]}
            query_scan = method.study.source_scan_id(items[batch[n_refs + row]][0])
            assert query_scan not in scans[refs]


def test_random_and_balanced_have_identical_queries_and_classes():
    items = _items()
    catalog = method.episode_catalog(items, (5, 10))
    a_batches, a_layouts = method.plan_episodes(
        items, catalog, [[16, 5], [16, 10]], 6, 43, 2, False)
    b_batches, b_layouts = method.plan_episodes(
        items, catalog, [[16, 5], [16, 10]], 6, 43, 2, True)
    for a, b, x, y in zip(a_batches, b_batches, a_layouts, b_layouts):
        assert x[3] == y[3]
        assert a[x[0]:] == b[y[0]:]


def test_position_loss_penalizes_lower_rank_and_has_gradient():
    losses = []
    for positive in (0.5, 0.1, -0.3):
        scores = torch.tensor([[positive, 0.2, 0.0]], requires_grad=True)
        loss, rank = method.position_loss(scores, 0)
        loss.backward()
        assert torch.isfinite(scores.grad).all()
        assert scores.grad[0, 0] < 0
        losses.append((float(loss), rank))
    assert [rank for _, rank in losses] == [1, 2, 3]
    assert losses[0][0] < losses[1][0] < losses[2][0]
    scores = torch.tensor([[-0.3, 0.2, 0.0]])
    plain, _ = method.position_loss(scores, 0)
    weighted, _ = method.position_loss(scores, 0, weight_by_rank=True)
    assert weighted > plain


def test_rank_loss_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        method.position_loss(torch.zeros(2, 3), 0)
    with pytest.raises(ValueError):
        method.position_loss(torch.zeros(1, 3), 3)


def test_smoke():
    runner._smoke(torch.device("cpu"))


def test_rank_aux_allrefs_uses_more_negatives_and_backpropagates():
    class SimpleRanker(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))
            self.seen = []

        def forward(self, query, query_tokens, refs, ref_tokens, labels,
                    scans, **kwargs):
            self.seen.append(len(refs))
            classes = labels.unique(sorted=True)
            pair = query @ refs.T * self.scale
            scores = torch.stack([pair[:, labels == c].mean(1)
                                  for c in classes.tolist()], 1)
            return scores, classes

    torch.manual_seed(7)
    names = [f"species_{i}" for i in range(130)]
    bank_labels = torch.arange(130).repeat_interleave(2)
    bank_g = torch.nn.functional.normalize(torch.randn(260, 8), dim=-1)
    bank_t = torch.randn(260, 2, 8)
    prototypes = torch.nn.functional.normalize(
        bank_g.reshape(130, 2, 8).mean(1), dim=-1)
    scans = [f"tw_{i}" for i in range(260)]
    first = torch.arange(130) * 2
    bank = (names, bank_labels, bank_g, bank_t, prototypes, scans, first)
    model = SimpleRanker()
    q = torch.randn(1, 8, requires_grad=True)
    r = torch.randn(5, 8, requires_grad=True)
    for allrefs in (False, True):
        loss, rank = runner._rank_aux(
            model, q, torch.randn(1, 2, 8), r, torch.randn(5, 2, 8),
            [0] * 5, [f"positive_{i}" for i in range(5)],
            ["species_0"], bank, 128, False, allrefs, 1,
            np.random.default_rng(43),
            {"gate": None, "consensus": None, "reference_chunk": 64})
        assert np.isfinite(rank)
        loss.backward(retain_graph=True)
    assert model.seen == [132, 259]
    assert q.grad.abs().sum() > 0
    assert r.grad.abs().sum() > 0
    assert model.scale.grad.abs().sum() > 0

    actual = WoodCorrespondence(dimension=8, token_dim=4)
    q2 = q.detach().requires_grad_(True)
    r2 = r.detach().requires_grad_(True)
    loss, _ = runner._rank_aux(
        actual, q2, torch.randn(1, 2, 8), r2, torch.randn(5, 2, 8),
        [0] * 5, [f"positive_{i}" for i in range(5)], ["species_0"],
        bank, 128, True, True, 1, np.random.default_rng(43),
        {"gate": True, "consensus": True, "reference_chunk": 64})
    loss.backward()
    assert torch.isfinite(loss)
    assert q2.grad.abs().sum() > 0
    assert r2.grad.abs().sum() > 0


def test_training_step_updates_encoder_and_reranker():
    class TinyBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(3, 8)

        def forward_features(self, images):
            pixels = images.flatten(2).transpose(1, 2)
            tokens = self.layer(pixels)
            return {"x_norm_clstoken": tokens.mean(1),
                    "x_norm_patchtokens": tokens}

    class TinyEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = TinyBackbone()
            self.projection = torch.nn.Linear(8, 8)

        def project(self, x):
            return torch.nn.functional.normalize(self.projection(x), dim=-1)

    torch.manual_seed(9)
    encoder = TinyEncoder()
    model = WoodCorrespondence(dimension=8, token_dim=4)
    names = [f"species_{i}" for i in range(130)]
    labels = torch.arange(130).repeat_interleave(2)
    bank_g = torch.nn.functional.normalize(torch.randn(260, 8), dim=-1)
    bank_t = torch.randn(260, 16, 8)
    prototypes = torch.nn.functional.normalize(
        bank_g.reshape(130, 2, 8).mean(1), dim=-1)
    bank = (names, labels, bank_g, bank_t, prototypes,
            [f"tw_{i}" for i in range(260)], torch.arange(130) * 2)
    images = torch.randn(96, 3, 4, 4)
    layout = (80, [i for i in range(16) for _ in range(5)],
              [f"ref_{i}" for i in range(80)], names[:16],
              [256, 512, 1024, 2048, 4096] * 16, [256] * 16)
    recipe = {"score_mode": "qkv", "gate": True, "consensus": True,
              "query_chunk": 4, "reference_chunk": 64,
              "global_aux": 0.0, "supcon": 0.0}
    cfg = {"microbatch": 16, "bank_weight": 0.5, "hard_weight": 0.5,
           "rank_weight": 0.2, "rank_queries": 1,
           "contrastive_weight": 0.2}
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(model.parameters()), lr=1e-3)
    old_encoder = encoder.backbone.layer.weight.detach().clone()
    old_ranker = model.query_key.weight.detach().clone()
    loss, accuracy, rank = runner._step(
        model, encoder, images, layout, bank, "balanced_rank128_allrefs",
        recipe, cfg, optimizer, torch.device("cpu"), np.random.default_rng(43))
    assert np.isfinite(loss)
    assert 0 <= accuracy <= 1
    assert rank is not None
    assert not torch.equal(old_encoder, encoder.backbone.layer.weight)
    assert not torch.equal(old_ranker, model.query_key.weight)


def test_selection_uses_fold0_only_and_rejects_changed_evidence(tmp_path,
                                                                  monkeypatch):
    monkeypatch.setattr(runner.study, "_hash_file", lambda p: str(Path(p).stat().st_size))
    cfg = {"arms": ["random", "balanced"], "seeds": [43], "epochs": 2,
           "selection_budget": 10, "selection_width": 128}
    scores = {("random", 1): 0.30, ("random", 2): 0.31,
              ("balanced", 1): 0.32, ("balanced", 2): 0.29}
    for (arm, epoch), score in scores.items():
        checkpoint = (tmp_path / "train" / arm / "seed_43" /
                      f"epoch_{epoch:02d}.pt")
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"weights")
        folder = runner._validation_folder(tmp_path, arm, 43, epoch)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "protocol.json").write_text(json.dumps({
            "checkpoint_sha256": "7", "arm": arm, "folds": [0],
            "budgets": [10], "widths": [128]}))
        pd.DataFrame([{"fold": 0, "budget": 10, "width": 128,
                       "method": "qkv_all_refs", "macro_r1": score,
                       "candidate_macro_recall": 0.98}]).to_csv(
                           folder / "summary.csv", index=False)
        (folder / "per_query.csv").write_text("query,correct\nx,1\n")
    runner._select(tmp_path, cfg)
    lock = runner._locked(tmp_path, cfg)
    assert lock["selected"]["arm"] == "balanced"
    assert lock["selected"]["epoch"] == 1
    assert lock["fold1_not_used_for_this_selection"]
    folder = runner._validation_folder(tmp_path, "balanced", 43, 1)
    (folder / "per_query.csv").write_text("query,correct\nx,0\ny,0\n")
    with pytest.raises(ValueError, match="changed"):
        runner._locked(tmp_path, cfg)
