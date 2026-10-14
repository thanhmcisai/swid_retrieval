import json
import zipfile

import pytest
import torch

from swid_retrieval import wood_full_image_early_stop as early


def test_milestone_steps_and_cosine_schedule():
    assert early.milestone_steps(1947) == {487: 0.25, 974: 0.5, 1947: 1.0}
    assert early.lr_multiplier("constant", 500, 1947) == 1.0
    assert early.lr_multiplier("cosine", 1, 1947) == pytest.approx(1.0)
    assert early.lr_multiplier("cosine", 1947, 1947) == pytest.approx(0.1)
    assert early.lr_multiplier("cosine", 974, 1947) == pytest.approx(0.55)
    with pytest.raises(ValueError):
        early.milestone_steps(3)
    with pytest.raises(ValueError):
        early.lr_multiplier("unknown", 1, 4)


def test_selection_score_uses_three_locked_galleries():
    assert early.selection_score({"57x5": 0.5, "57x1": 0.4,
                                  "637x1": 0.1, "24x5": 0.9}) == pytest.approx(1 / 3)


def test_train_one_records_fractional_coverage_and_skips_completed(tmp_path, monkeypatch):
    class TinyEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(2, 2)
            self.projection = torch.nn.Linear(2, 2)

    class TinyPairs(torch.utils.data.Dataset):
        def __init__(self, items, refs):
            self.items = items
            self.lookup = {"a": 0, "b": 1}

        def __len__(self):
            return len(self.items)

        def __getitem__(self, index):
            label = self.lookup[self.items[index][1]]
            return torch.ones(3, 2, 2), torch.ones(3, 2, 2), label, index

    items = [(f"/x/{label}/scale_512/patch_512_{i}_from_Tw{i}.jpg", label)
             for label in ("a", "b") for i in range(4)]
    monkeypatch.setattr(early.evidence, "_encoder_from_state",
                        lambda *args: TinyEncoder())
    monkeypatch.setattr(early.full, "PairDataset", TinyPairs)
    monkeypatch.setattr(early.full, "_step",
                        lambda *args, **kwargs: (1.0, {}))
    values = iter((0.4, 0.5, 0.45))
    monkeypatch.setattr(early.full, "_validate", lambda *args, **kwargs: {
        "57x5": (value := next(values)), "57x1": value, "637x1": value})
    cfg = {"seed": 43, "base": {"batch": 2, "workers": 0,
           "backbone_lr": 2e-6, "head_lr": 1e-4,
           "print_every": 100, "step_mode": "direct"}}
    report = early._train_one(tmp_path, {}, items, {}, {}, "sourcehash",
                              {"test": True}, cfg, "arcface", "cosine",
                              torch.device("cpu"), 0.3)
    assert [row["unique_queries_seen"] for row in report["records"]] == [2, 4, 8]
    assert report["best_fraction"] == 0.5
    assert report["best_score"] == pytest.approx(0.5)
    assert (tmp_path / "arcface/cosine/seed_43/best.pt").is_file()
    assert json.loads((tmp_path / "arcface/cosine/seed_43/complete.json").read_text())[
        "best_fraction"] == 0.5
    assert early._train_one(tmp_path, {}, items, {}, {}, "sourcehash",
                            {"test": True}, cfg, "arcface", "cosine",
                            torch.device("cpu"), 0.3) == report


def test_export_excludes_checkpoints(tmp_path):
    (tmp_path / "summary.json").write_text('{"complete": true}')
    (tmp_path / "best.pt").write_bytes(b"weights")
    with zipfile.ZipFile(early._export(tmp_path)) as archive:
        assert archive.namelist() == ["summary.json"]
