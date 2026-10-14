import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from swid_retrieval import wood_gallery_listwise as pilot
from swid_retrieval.gallery_method import ArcFaceClassifier
from swid_retrieval.wood_correspondence_method import WoodCorrespondence


def items():
    return [
        (f"/x/{label}/scale_{512 if i % 2 else 1024}/"
         f"patch_{i}_from_Tw{scan}.jpg", label)
        for label, scans in (("a", (1, 2, 3, 4, 5)),
                             ("b", (6, 7, 8, 9, 10)))
        for i, scan in enumerate(scans)]


def make_bank():
    rows = items()
    labels, plan = pilot.bank_plan(rows, 43, 5)
    torch.manual_seed(3)
    features = F.normalize(torch.randn(10, 512), dim=-1)
    return rows, plan, pilot.ReferenceBank(rows, labels, plan, features,
                                          torch.device("cpu"))


def test_bank_plan_is_deterministic_scan_diverse_and_training_only():
    rows = items()
    labels, plan = pilot.bank_plan(rows, 43, 5)
    assert (labels, plan) == pilot.bank_plan(rows, 43, 5)
    assert labels == ["a", "b"]
    assert len(plan) == 2 and all(len(group) == 5 for group in plan)
    for group in plan:
        assert len(set(group)) == len(group)
        assert len({rows[index][0].split("Tw")[-1] for index in group}) == 5
    audit = pilot.bank_plan_audit(rows, plan)
    assert audit["species"] == 2
    assert audit["reference_images"] == 10
    assert audit["species_with_multiple_scans"] == 2
    assert audit["species_with_multiple_scales"] == 2


@pytest.mark.parametrize("k", [1, 3, 5])
def test_gallery_loss_backpropagates_through_query_and_live_reference(k):
    rows, _, bank = make_bank()
    query = torch.randn(2, 512, requires_grad=True)
    reference = torch.randn(2, 512, requires_grad=True)
    labels = torch.tensor([0, 1])
    indices = torch.tensor([0, 5])
    logits = bank.scores(query, reference, labels, indices, k, 0.07)
    loss, details = pilot.gallery_loss(logits, labels, 1, 0.05 / 0.07, 0.2)
    assert logits.shape == (2, 2) and torch.isfinite(loss)
    assert 0 <= details["gallery_r1"] <= 1
    loss.backward()
    assert query.grad.abs().sum() > 0
    assert reference.grad.abs().sum() > 0
    assert bank.features.grad is None


def test_bank_replaces_self_match_and_updates_only_selected_references():
    rows = items()
    labels, plan = pilot.bank_plan(rows, 43, 5)
    features = torch.zeros(10, 512)
    features[0, 0] = 1
    features[1, 1] = 1
    bank = pilot.ReferenceBank(rows, labels, plan, features, torch.device("cpu"))
    self_index = plan[0][0]
    query = torch.zeros(1, 512)
    query[0, 0] = 1
    reference = torch.zeros(1, 512)
    reference[0, 1] = 1
    logits = bank.scores(query, reference, torch.tensor([0]),
                         torch.tensor([self_index]), 1, 1)
    assert logits[0, 0].item() == pytest.approx(0)
    before = bank.features.clone()
    new = torch.zeros(1, 512)
    new[0, 2] = 1
    assert bank.update(torch.tensor([self_index]), new, 0) == 1
    assert bank.features[0, 2].item() == pytest.approx(1)
    assert torch.equal(bank.features[1:], before[1:])


def test_bank_multiscale_budget_has_real_effect():
    _, plan, bank = make_bank()
    query = torch.randn(2, 512)
    reference = torch.randn(2, 512)
    labels = torch.tensor([0, 1])
    indices = torch.tensor([plan[0][0], plan[1][0]])
    one = bank.scores(query, reference, labels, indices, 1, 0.07)
    five = bank.scores(query, reference, labels, indices, 5, 0.07)
    assert not torch.allclose(one, five)


def test_inference_mode_source_features_are_safe_for_backward():
    rows = items()
    labels, plan = pilot.bank_plan(rows, 43, 5)
    with torch.inference_mode():
        extracted = torch.randn(10, 512)
    bank = pilot.ReferenceBank(rows, labels, plan, extracted,
                               torch.device("cpu"))
    assert not bank.features.is_inference()
    query = torch.randn(2, 512, requires_grad=True)
    reference = torch.randn(2, 512, requires_grad=True)
    labels_tensor = torch.tensor([0, 1])
    logits = bank.scores(query, reference, labels_tensor,
                         torch.tensor([0, 5]), 3, 0.07)
    F.cross_entropy(logits, labels_tensor).backward()
    assert query.grad.abs().sum() > 0


def test_validation_cache_fingerprint_rejects_missing_and_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(pilot.data, "CACHE_DIR", tmp_path)
    path = "/x/test.jpg"
    with pytest.raises(FileNotFoundError, match="not cached"):
        pilot._cache_fingerprint([path])
    import hashlib
    cached = tmp_path / (hashlib.md5(path.encode()).hexdigest() + ".jpg")
    cached.write_bytes(b"first")
    first = pilot._cache_fingerprint([path])
    cached.write_bytes(b"second")
    assert pilot._cache_fingerprint([path]) != first


def test_selection_preserves_small_gallery_retention(tmp_path):
    cfg = {"retention_tolerance": 0.03}
    source = {"meta_val": {"24x5": 0.66, "637x1": 0.15}}
    reports = [
        {"arm": "gallery_full", "best_meta_val": {"24x5": 0.60,
                                                    "637x1": 0.30},
         "best_sha256": "bad"},
        {"arm": "gallery_hard_scale", "best_meta_val": {"24x5": 0.64,
                                                          "637x1": 0.20},
         "best_sha256": "good"},
    ]
    lock = pilot._selection(tmp_path, reports, source,
                            {"source_sha256": "source"}, cfg)
    assert lock["selected_arm"] == "gallery_hard_scale"
    assert not lock["meta_test_evaluated"] and not lock["public_id_evaluated"]


@pytest.mark.parametrize("device_name", ["cpu", "cuda"] if torch.cuda.is_available()
                         else ["cpu"])
def test_one_gallery_training_step_updates_encoder(device_name):
    class TinyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(12, 512)

        def forward_features(self, images):
            x = self.linear(images.flatten(1))
            return {"x_norm_clstoken": x,
                    "x_norm_patchtokens": x[:, None].expand(-1, 16, -1)}

    class TinyEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = TinyBackbone()
            self.projection = nn.Linear(512, 512)

        def project(self, features):
            return F.normalize(self.projection(features), dim=-1)

    device = torch.device(device_name)
    rows, plan, bank = make_bank()
    bank = pilot.ReferenceBank(rows, ["a", "b"], plan,
                               bank.features.detach().clone(), device)
    encoder = TinyEncoder().to(device)
    model = WoodCorrespondence().to(device)
    classifier = ArcFaceClassifier(512, 2).to(device)
    optimizer = torch.optim.AdamW(list(encoder.parameters()) +
                                  list(classifier.parameters()), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler() if device_name == "cuda" else None
    initial = encoder.backbone.linear.weight.detach().clone()
    batch = (torch.randn(4, 3, 2, 2), torch.randn(4, 3, 2, 2),
             torch.tensor([0, 0, 1, 1]), torch.tensor([0, 1, 5, 6]))
    refs = torch.tensor([1, 2, 3, 4, 6, 7, 8, 9, 0, 5], device=device)
    cfg = {"base": {"microbatch": 4, "contrastive_weight": 0.2,
                    "contrastive_temperature": 0.07, "arcface_weight": 1.0},
           "temperature": 0.07, "hard_classes": 1,
           "hard_margin": 0.05, "hard_weight": 0.2,
           "gallery_weight": 0.25, "bank_momentum": 0.5}
    loss, details = pilot._step_with_bank(
        encoder, model, classifier, bank, batch, refs, cfg, optimizer, scaler,
        device, 1, "gallery_hard_scale")
    assert np.isfinite(loss)
    assert details["gallery_k"] == 1
    assert not torch.equal(initial, encoder.backbone.linear.weight)
