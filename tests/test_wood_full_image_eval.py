import pytest
import numpy as np
import torch

from swid_retrieval import wood_full_image_eval as evaluation
from swid_retrieval.wood_correspondence_method import WoodCorrespondence


def _image(scan, index):
    return f"/x/a/scale_512/patch_512_{index}_from_Tw{scan}.jpg"


def test_reference_pool_excludes_query_scan_and_requires_ten_images():
    manifest = {"meta-val": [(_image(1, i), "a") for i in range(5)] +
                [(_image(2, i), "a") for i in range(10)]}
    queries = [(_image(1, i), "a") for i in range(5)]
    pool = evaluation._reference_pool(manifest, queries)
    assert len(pool) == 10
    assert all("Tw2" in path for path, _ in pool)
    with pytest.raises(ValueError, match="Fewer than 10"):
        evaluation._reference_pool({"meta-val": manifest["meta-val"][:-1]}, queries)


def test_unique_items_checks_path_label_identity():
    items, lookup = evaluation._unique_items(
        [(_image(1, 1), "a")], [(_image(1, 1), "a"), (_image(2, 1), "a")])
    assert len(items) == 2
    assert lookup[_image(1, 1)] == 0
    with pytest.raises(ValueError, match="Conflicting labels"):
        evaluation._unique_items([(_image(1, 1), "a")], [(_image(1, 1), "b")])


def test_summary_reports_conditional_retrieval_without_leaking_query_count():
    rows = [{"gallery": "637x1", "policy": "standard", "mode": "global",
             "query_label": label, "correct": int(index == 0), "r5": 1,
             "candidate_r32": int(index == 0), "candidate_r128": 1,
             "n_references": 637}
            for label in ("a", "b") for index in range(2)]
    summary = evaluation._summaries(rows)[0]
    assert summary["n_queries"] == 4
    assert summary["n_species"] == 2
    assert summary["r1"] == 0.5
    assert summary["candidate_r128"] == 1.0
    assert summary["r1_given_candidate128"] == 0.5


def test_conditional_r1_excludes_correct_predictions_outside_shortlist():
    rows = [{"gallery": "637x1", "policy": "standard", "mode": "qkv",
             "query_label": "a", "correct": correct, "r5": correct,
             "candidate_r32": candidate, "candidate_r128": candidate,
             "n_references": 637}
            for correct, candidate in ((1, 0), (0, 1))]
    summary = evaluation._summaries(rows)[0]
    assert summary["r1"] == 0.5
    assert summary["r1_given_candidate128"] == 0.0
    assert summary["correct_outside_candidate128"] == 1


def test_score_gallery_produces_paired_global_and_qkv_query_rows():
    refs = [(_image(1, 1), "a"), (_image(2, 1), "b")]
    queries = [(_image(3, 1), "a"), (_image(4, 1), "b")]
    items, lookup = evaluation._unique_items(refs, queries)
    rng = np.random.default_rng(7)
    embeddings = rng.normal(size=(len(items), 512)).astype(np.float32)
    tokens = rng.normal(size=(len(items), 16, 512)).astype(np.float32)
    model = WoodCorrespondence().eval()
    rows = evaluation._score_gallery(
        model, "arcface_qkv", 0, "2x1", "standard", refs, queries,
        lookup, embeddings, tokens, torch.device("cpu"))
    assert len(rows) == 4
    assert {row["mode"] for row in rows} == {"global", "qkv"}
    assert {row["query_path"] for row in rows} == {path for path, _ in queries}


def test_validate_keeps_identical_queries_across_reference_budgets(monkeypatch, tmp_path):
    refs = [(f"/x/{label}/scale_512/patch_512_{i}_from_Tw{scan}.jpg", label)
            for label, scan in (("a", 1), ("b", 2)) for i in range(10)]
    queries = [(f"/x/{label}/scale_512/patch_512_{i}_from_Tw{scan}.jpg", label)
               for label, scan in (("a", 3), ("b", 4)) for i in range(5)]
    manifest = {"meta-val": refs + queries}
    monkeypatch.setattr(evaluation.evidence, "_validation_sets",
                        lambda *args: (refs, queries, []))
    monkeypatch.setattr(evaluation.evidence, "_gallery_plan",
                        lambda *args: {"2x1": (np.asarray([0, 10]), np.arange(10))})
    monkeypatch.setattr(evaluation.evidence, "_extract",
                        lambda encoder, items, cfg, device: (
                            np.eye(len(items), 512, dtype=np.float32),
                            np.zeros((len(items), 16, 512), dtype=np.float32)))
    evaluation.evaluate(manifest, None, WoodCorrespondence().eval(),
                        {"image_size": 224}, "arcface", torch.device("cpu"),
                        tmp_path, "checkpoint-hash", "manifest-hash")
    with (tmp_path / "per_query.csv").open() as stream:
        import csv
        rows = list(csv.DictReader(stream))
    for fold in ("0", "1"):
        expected = {row["query_path"] for row in rows
                    if row["fold"] == fold and row["gallery"] == "2x1" and
                    row["policy"] == "standard"}
        for policy in evaluation.POLICIES:
            for k in evaluation.BUDGETS:
                observed = {row["query_path"] for row in rows
                            if row["fold"] == fold and row["gallery"] == f"2x{k}" and
                            row["policy"] == policy}
                assert observed == expected
