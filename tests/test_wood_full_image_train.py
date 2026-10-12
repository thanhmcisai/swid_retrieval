import numpy as np
import torch

from swid_retrieval import wood_full_image_train as full
from swid_retrieval.wood_correspondence_method import WoodCorrespondence
from swid_retrieval.gallery_method import ArcFaceClassifier


def _items():
    return [
        ("/x/a/scale_512/patch_512_1_from_Tw1.jpg", "a"),
        ("/x/a/scale_1024/patch_1024_1_from_Tw2.jpg", "a"),
        ("/x/a/scale_512/patch_512_2_from_Tw2.jpg", "a"),
        ("/x/b/scale_512/patch_512_1_from_Tw3.jpg", "b"),
        ("/x/b/scale_1024/patch_1024_1_from_Tw3.jpg", "b"),
    ]


def test_reference_plan_covers_every_query_and_prefers_disjoint_scan():
    items = _items()
    refs, audit = full.reference_plan(items, 43, 1)
    assert len(refs) == len(items)
    assert all(i != j and items[i][1] == items[j][1] for i, j in enumerate(refs))
    assert audit["cross_scan_pairs"] == 3
    assert audit["same_scan_fallback_pairs"] == 2
    assert np.all((refs >= 0) & (refs < len(items)))


def test_class_balanced_batches_cover_full_dataset_once():
    items = _items() * 5
    batches = full.class_balanced_batches(items, 3, 4)
    assert sorted(index for batch in batches for index in batch) == list(range(len(items)))
    assert all(1 <= len(batch) <= 3 for batch in batches)


def test_arcface_and_qkv_losses_backpropagate_to_embeddings():
    torch.manual_seed(3)
    embeddings = torch.nn.functional.normalize(torch.randn(8, 512), dim=1).requires_grad_()
    tokens = torch.randn(8, 16, 512, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 3])
    model = WoodCorrespondence()
    classifier = ArcFaceClassifier(512, 4)
    loss, metrics = full._loss(model, classifier, embeddings, tokens, labels,
                               "arcface_qkv", 0.5)
    assert torch.isfinite(loss)
    assert metrics["qkv"] is not None
    loss.backward()
    assert embeddings.grad is not None and embeddings.grad.abs().sum() > 0
    assert tokens.grad is not None and tokens.grad.abs().sum() > 0
    assert model.query_key.weight.grad is not None


def test_single_class_query_uses_extra_negative_reference():
    torch.manual_seed(7)
    embeddings = torch.nn.functional.normalize(torch.randn(5, 512), dim=1).requires_grad_()
    tokens = torch.randn(5, 16, 512, requires_grad=True)
    labels = torch.tensor([0, 0])
    model = WoodCorrespondence()
    classifier = ArcFaceClassifier(512, 2)
    loss, metrics = full._loss(model, classifier, embeddings, tokens, labels,
                               "arcface_qkv", 0.5, extra_label=1)
    assert torch.isfinite(loss)
    assert metrics["candidate_classes"] == 2
    loss.backward()
    assert tokens.grad is not None and tokens.grad.abs().sum() > 0


def test_full_step_updates_encoder_and_qkv(monkeypatch):
    class TinyEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.global_layer = torch.nn.Linear(12, 512)
            self.token_layer = torch.nn.Linear(12, 512)

        def forward(self, images):
            flat = images.flatten(1)
            g = torch.nn.functional.normalize(self.global_layer(flat), dim=1)
            t = self.token_layer(flat)[:, None, :].expand(-1, 16, -1)
            return g, t

    monkeypatch.setattr(full.corr, "_encode_images", lambda encoder, images: encoder(images))
    torch.manual_seed(5)
    encoder = TinyEncoder()
    model = WoodCorrespondence()
    classifier = ArcFaceClassifier(512, 2)
    params = list(encoder.parameters()) + list(model.parameters()) + list(classifier.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    before = encoder.global_layer.weight.detach().clone()
    batch = (torch.randn(2, 3, 2, 2), torch.randn(2, 3, 2, 2),
             torch.tensor([0, 1]), torch.tensor([0, 1]))
    loss, metrics = full._step(encoder, model, classifier, batch, "arcface_qkv",
                               {"microbatch": 2, "qkv_weight": 0.5}, optimizer,
                               torch.device("cpu"))
    assert np.isfinite(loss)
    assert metrics["candidate_classes"] == 2
    assert not torch.equal(before, encoder.global_layer.weight)
