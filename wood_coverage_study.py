"""Isolated training-exposure study for the selected random-shortlist model.

The source, sampler, losses and validation protocol remain identical to the
previous hard-shortlist run. Only the number of episodes per epoch changes.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import data, gallery_experiment as study
from . import wood_correspondence_image_train as image_train
from . import wood_hard_negative as hard
from . import wood_multilevel as ml


ARM = "random16"
ENDPOINT = {"budget": 10, "width": 128, "method": "qkv_all_refs"}
MATCHED_SETTINGS = ("seed", "epochs", "hard_queries", "backbone_lr",
                    "head_lr", "hard_weight", "bank_weight", "microbatch",
                    "workers", "image_batch", "folds", "budgets", "widths")


def _paths(root):
    old = Path(os.environ.get(
        "WOOD_COVERAGE_BASELINE_OUT",
        root / "results" / "wood_hard_shortlist_study_v1")).resolve()
    out = Path(os.environ.get(
        "WOOD_COVERAGE_OUT",
        root / "results" / "wood_coverage_random16_v1")).resolve()
    ml_out = Path(os.environ.get(
        "WOOD_ML_OUT", root / "results" / "wood_multilevel_study_v1")).resolve()
    base_out = Path(os.environ.get(
        "WOOD_LARGE_BASE_OUT", root / "results" /
        "dinov2s_retrieval_matched_v1")).resolve()
    image_out = Path(os.environ.get(
        "WOOD_LARGE_QKV_OUT", root / "results" /
        "wood_correspondence_image_full_v1")).resolve()
    if old == out:
        raise ValueError("Coverage output must not overwrite the earlier study")
    return old, out, ml_out, base_out, image_out


def _config():
    cfg = hard._settings()
    cfg["arms"] = [ARM]
    cfg["episodes"] = int(os.environ.get("WOOD_COVERAGE_EPISODES", "600"))
    if cfg["episodes"] < 1:
        raise ValueError("WOOD_COVERAGE_EPISODES must be positive")
    return cfg


def _baseline(old, cfg):
    prior = json.loads((old / "preflight.json").read_text())["settings"]
    if prior["arms"] != list(hard.ARMS):
        raise ValueError("Unexpected prior shortlist arm set")
    for key in MATCHED_SETTINGS:
        if cfg[key] != prior[key]:
            raise ValueError(f"Coverage study changed matched setting: {key}")
    if cfg["episodes"] <= prior["episodes"]:
        raise ValueError("Coverage run must use more episodes than baseline")
    hard._locked(old)
    train_meta = json.loads((old / "train" / ARM /
                             "epoch_01_provenance.json").read_text())
    if train_meta["provenance"]["runner_sha256"] != study._hash_file(hard.__file__):
        raise ValueError("Original training implementation changed")
    if train_meta["provenance"]["settings"]["episodes"] != prior["episodes"]:
        raise ValueError("Baseline training and preflight disagree")
    return prior, train_meta["provenance"]["source_sha256"]


def _plan_coverage(items, valid, multi, seed, episodes, epochs):
    image_ids, query_ids, species, scans = set(), set(), set(), set()
    per_epoch = []
    query_draws = 0
    for epoch in range(1, epochs + 1):
        batches, layouts = ml._plan(items, valid, False,
                                    {"seed": seed, "episodes": episodes},
                                    epoch, multi=multi)
        seen, queried = set(), set()
        for batch, layout in zip(batches, layouts):
            n_refs = layout[0]
            seen.update(batch)
            queried.update(batch[n_refs:])
            query_draws += len(batch) - n_refs
        image_ids.update(seen)
        query_ids.update(queried)
        species.update(study.canonical(items[i][1]) for i in seen)
        scans.update(study.source_scan_id(items[i][0]) for i in seen)
        per_epoch.append({"epoch": epoch, "episodes": episodes,
                          "unique_images": len(seen),
                          "unique_query_images": len(queried),
                          "cumulative_unique_images": len(image_ids),
                          "cumulative_unique_queries": len(query_ids)})
    return {"episodes_per_epoch": episodes,
            "total_episodes": episodes * epochs,
            "total_query_draws": query_draws,
            "unique_images": len(image_ids),
            "unique_query_images": len(query_ids),
            "unique_species": len(species),
            "unique_source_scans": len(scans),
            "fraction_meta_train_images": len(image_ids) / len(items),
            "per_epoch": per_epoch}


def _preflight(root, old, out, image_out, manifest, cfg):
    prior, source_hash = _baseline(old, cfg)
    manifest_hash = study._hash_file(root / "swi_manifest.json")
    original_training = json.loads((old / "train" / ARM /
                                    "epoch_01_provenance.json").read_text())
    if original_training["provenance"]["manifest_sha256"] != manifest_hash:
        raise ValueError("Meta-train manifest differs from the baseline")
    source_path = image_out / "qkv" / f"seed_{cfg['seed']}" / "best.pt"
    if study._hash_file(source_path) != source_hash:
        raise ValueError("Source checkpoint differs from the baseline")
    items = manifest["meta-train"]
    _, valid = image_train._layout(items)
    multi = {k: ml._eligible(items, k) for _, k in ml.SCHEDULE}
    low = _plan_coverage(items, valid, multi, cfg["seed"],
                         prior["episodes"], cfg["epochs"])
    high = _plan_coverage(items, valid, multi, cfg["seed"],
                          cfg["episodes"], cfg["epochs"])
    if high["unique_images"] <= low["unique_images"]:
        raise ValueError("More episodes did not increase unique image coverage")
    audit = {"hypothesis": "more training exposure with the same learning rule",
             "meta_train_images": len(items),
             "manifest_sha256": manifest_hash,
             "baseline_source_sha256": source_hash,
             "baseline_lock_sha256": study._hash_file(
                 old / "selection_lock.json"),
             "baseline": low, "increased_coverage": high,
             "settings": cfg, "test_evaluated": False}
    path = out / "preflight.json"
    if path.is_file() and json.loads(path.read_text()) != audit:
        raise ValueError("Coverage preflight changed; use a new output folder")
    out.mkdir(parents=True, exist_ok=True)
    study._json(path, audit)
    print("[wood-coverage] planned images: "
          f"{low['unique_images']}/{len(items)} -> "
          f"{high['unique_images']}/{len(items)}; "
          f"query images: {low['unique_query_images']} -> "
          f"{high['unique_query_images']}", flush=True)
    return audit


def _assert_preflight(root, old, out, image_out, cfg):
    audit = json.loads((out / "preflight.json").read_text())
    _, source_hash = _baseline(old, cfg)
    source_path = image_out / "qkv" / f"seed_{cfg['seed']}" / "best.pt"
    if (audit["settings"] != cfg or
            audit["baseline_source_sha256"] != source_hash or
            audit["manifest_sha256"] != study._hash_file(
                root / "swi_manifest.json") or
            study._hash_file(source_path) != source_hash or
            audit["baseline_lock_sha256"] != study._hash_file(
                old / "selection_lock.json") or
            audit["meta_train_images"] != len(data.load_swi_manifest(
                root / "swi_manifest.json")["meta-train"])):
        raise ValueError("Coverage preflight is stale")


def _endpoint(frame):
    rows = frame[(frame.budget == ENDPOINT["budget"]) &
                 (frame.width == ENDPOINT["width"]) &
                 (frame.method == ENDPOINT["method"])]
    if len(rows) != 2 or set(rows.fold) != {0, 1}:
        raise ValueError("Matched K=10/top-128 endpoint is incomplete")
    return rows


def _paired_evidence(old, out, samples=10000):
    old_dir = old / "validation" / ARM
    new_dir = out / "validation" / ARM
    for folder, checkpoint in ((old_dir, old / "train" / ARM / "best.pt"),
                               (new_dir, out / "train" / ARM / "best.pt")):
        protocol = json.loads((folder / "protocol.json").read_text())
        if (protocol["checkpoint_sha256"] != study._hash_file(checkpoint) or
                protocol["folds"] != [0, 1] or
                ENDPOINT["budget"] not in protocol["budgets"] or
                ENDPOINT["width"] not in protocol["widths"]):
            raise ValueError(f"Validation protocol/checkpoint mismatch: {folder}")
    old_summary = _endpoint(pd.read_csv(old_dir / "summary.csv"))
    new_summary = _endpoint(pd.read_csv(new_dir / "summary.csv"))
    def selected_queries(folder):
        frame = pd.read_csv(folder / "per_query.csv")
        frame = frame[(frame.budget == ENDPOINT["budget"]) &
                      (frame.width == ENDPOINT["width"]) &
                      (frame.method == ENDPOINT["method"])]
        if len(frame) != 570 or frame.duplicated(["fold", "query_path"]).any():
            raise ValueError("Expected 285 unique queries in each of two folds")
        return frame.set_index(["fold", "query_path"]).sort_index()
    old_queries = selected_queries(old_dir)
    new_queries = selected_queries(new_dir).reindex(old_queries.index)
    if (new_queries.true_label.isna().any() or
            not np.array_equal(old_queries.true_label.to_numpy(),
                               new_queries.true_label.to_numpy())):
        raise ValueError("The two validation cohorts are not identical")
    labels = old_queries.true_label.to_numpy()
    delta = new_queries.correct.to_numpy() - old_queries.correct.to_numpy()
    classes = sorted(set(labels))
    by_species = np.array([delta[labels == name].mean() for name in classes])
    rng = np.random.default_rng(43001)
    draws = rng.integers(0, len(classes), size=(samples, len(classes)))
    lower, upper = np.quantile(by_species[draws].mean(axis=1), [0.025, 0.975])
    return {"endpoint": ENDPOINT, "selection_split": "meta-val",
            "test_evaluated": False, "n_queries": 570,
            "n_species": len(classes),
            "baseline_macro_r1": float(old_summary.macro_r1.mean()),
            "increased_coverage_macro_r1": float(new_summary.macro_r1.mean()),
            "baseline_candidate_recall": float(
                old_summary.candidate_macro_recall.mean()),
            "increased_coverage_candidate_recall": float(
                new_summary.candidate_macro_recall.mean()),
            "baseline_ms_per_query": float(old_summary.ms_per_query.mean()),
            "increased_coverage_ms_per_query": float(
                new_summary.ms_per_query.mean()),
            "paired_delta_macro_r1": float(by_species.mean()),
            "paired_species_bootstrap_95_ci": [float(lower), float(upper)],
            "bootstrap_samples": samples,
            "new_correct_old_wrong": int((delta == 1).sum()),
            "old_correct_new_wrong": int((delta == -1).sum()),
            "both_same_outcome": int((delta == 0).sum()),
            "baseline_validation_sha256": study._hash_file(
                old_dir / "summary.csv"),
            "increased_coverage_validation_sha256": study._hash_file(
                new_dir / "summary.csv")}


def run():
    root = Path(os.environ.get("ROOT_PATH", "/content/drive/MyDrive/NCS")).resolve()
    old, out, ml_out, base_out, image_out = _paths(root)
    cfg = _config()
    mode = os.environ.get("WOOD_COVERAGE_MODE", "preflight")
    if mode not in {"preflight", "train", "validate", "compare"}:
        raise ValueError("WOOD_COVERAGE_MODE must be preflight/train/validate/compare")
    manifest = data.load_swi_manifest(root / "swi_manifest.json")
    if mode == "preflight":
        _preflight(root, old, out, image_out, manifest, cfg)
        return
    _assert_preflight(root, old, out, image_out, cfg)
    if mode == "train":
        if not torch.cuda.is_available():
            raise RuntimeError("Coverage training requires CUDA")
        latest = out / "train" / ARM / "latest.pt"
        if latest.is_file():
            saved = torch.load(latest, map_location="cpu", weights_only=False)
            if saved["epoch"] >= cfg["epochs"]:
                expected = saved["provenance"]["settings"]
                if (saved["epoch"] != cfg["epochs"] or
                        any(cfg[key] != value for key, value in expected.items()) or
                        saved["provenance"]["source_sha256"] !=
                        json.loads((out / "preflight.json").read_text())[
                            "baseline_source_sha256"] or
                        not (out / "train" / ARM / "best.pt").is_file()):
                    raise ValueError("Completed training checkpoint differs from study")
                print("[wood-coverage] training already complete", flush=True)
                return
        hard._preload_training(manifest, cfg)
        hard._train(root, out, manifest, base_out, image_out,
                    cfg, ARM, torch.device("cuda"))
    elif mode == "validate":
        if not torch.cuda.is_available():
            raise RuntimeError("Coverage validation requires CUDA")
        hard._validate_arm(root, out, ml_out, manifest, base_out, image_out,
                           cfg, ARM, torch.device("cuda"))
    else:
        report = _paired_evidence(old, out)
        path = out / "coverage_comparison.json"
        if path.is_file() and json.loads(path.read_text()) != report:
            raise ValueError("Coverage evidence changed; use a new output folder")
        study._json(path, report)
        print(f"[wood-coverage] paired meta-val: {report}", flush=True)


if __name__ == "__main__":
    run()
