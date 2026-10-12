"""Isolated training-exposure study for the selected random-shortlist model.

The source, sampler, losses and validation protocol remain identical to the
previous hard-shortlist run. Only the number of episodes per epoch changes.
"""

import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import data, gallery_experiment as study
from . import wood_correspondence_experiment as corr
from . import wood_correspondence_image_train as image_train
from . import wood_hard_negative as hard
from . import wood_multilevel as ml
from . import wood_reference_budget as budget
from .wood_large_gallery_methods import class_scores


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


def _paired_report(old_dir, old_checkpoint, new_dir, new_checkpoint,
                   samples=10000):
    for folder, checkpoint in ((old_dir, old_checkpoint),
                               (new_dir, new_checkpoint)):
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


def _paired_evidence(old, out, samples=10000):
    return _paired_report(old / "validation" / ARM,
                          old / "train" / ARM / "best.pt",
                          out / "validation" / ARM,
                          out / "train" / ARM / "best.pt", samples)


def _validate_latest(root, out, manifest, encoder, model, base_cfg, cfg,
                     latest_path, device):
    folder = out / "checkpoint_audit" / "latest"
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint_hash = study._hash_file(latest_path)
    protocol = {"checkpoint_sha256": checkpoint_hash, "folds": [0, 1],
                "budgets": [10], "widths": [128],
                "reference_policy": "scan_scale", "shortlist": "raw_prototype",
                "meta_val_only": True}
    protocol_path = folder / "protocol.json"
    if protocol_path.is_file():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("Latest validation protocol changed")
        if all((folder / name).is_file() for name in ("summary.csv", "per_query.csv")):
            print("[wood-coverage] reusing latest-checkpoint validation", flush=True)
            return folder
    rows, query_rows = [], []
    for fold in protocol["folds"]:
        _, queries = study.validation_items(
            manifest, group_mode="scan_disjoint", fold=fold)
        refs = budget._reference_pool(manifest, queries)
        class_names = sorted({label for _, label in refs})
        if len(class_names) != 637 or len(queries) != 285:
            raise ValueError("Meta-val cohort changed")
        class_ids = {label: i for i, label in enumerate(class_names)}
        selected = budget.select_indices(refs, 10, "scan_scale")
        items = [refs[i] for i in selected]
        ref_labels = torch.tensor([class_ids[label] for _, label in items],
                                  device=device)
        scans = [study.source_scan_id(path) for path, _ in items]
        target_scans = defaultdict(set)
        for path, label in items:
            target_scans[label].add(study.source_scan_id(path))
        if any(study.source_scan_id(path) in target_scans[study.canonical(label)]
               for path, label in queries):
            raise ValueError("Meta-val source-scan leakage")
        rg, rt, qg, qt = hard._reference_data(
            encoder, items, queries,
            dict(base_cfg, image_batch=cfg["image_batch"], workers=cfg["workers"]),
            device, folder / "features" / f"fold_{fold}", checkpoint_hash)
        truth = torch.tensor([class_ids[study.canonical(label)]
                              for _, label in queries], device=device)
        with torch.inference_mode():
            raw, _ = class_scores(qg, rg, ref_labels)
            adapted, _ = class_scores(corr._transformed(model, qg),
                                      corr._transformed(model, rg), ref_labels)
        for method, width, scores, candidates, elapsed in (
                ("raw_prototype", 0, raw, None, 0.0),
                ("adapted_prototype", 0, adapted, None, 0.0)):
            rows.append(hard._summary_row(ARM, fold, 10, width, method,
                                          scores, truth, candidates, elapsed))
            predictions = scores.argmax(dim=1).tolist()
            for i, (path, label) in enumerate(queries):
                query_rows.append({
                    "arm": ARM, "fold": fold, "budget": 10, "width": width,
                    "method": method, "query_path": path,
                    "true_label": study.canonical(label),
                    "predicted_label": class_names[predictions[i]],
                    "correct": int(predictions[i] == int(truth[i])),
                    "candidate_hit": None, "query_scale": study.image_scale(path)})
        candidates = raw.topk(128, dim=1).indices
        started = time.perf_counter()
        with torch.inference_mode():
            scores = ml.rerank_all_references(
                model, qg, qt, rg, rt, ref_labels, scans, candidates, adapted)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = 1000 * (time.perf_counter() - started) / len(qg)
        rows.append(hard._summary_row(ARM, fold, 10, 128, "qkv_all_refs",
                                      scores, truth, candidates, elapsed))
        predictions = scores.argmax(dim=1).tolist()
        hits = (candidates == truth[:, None]).any(dim=1).tolist()
        for i, (path, label) in enumerate(queries):
            query_rows.append({
                "arm": ARM, "fold": fold, "budget": 10, "width": 128,
                "method": "qkv_all_refs", "query_path": path,
                "true_label": study.canonical(label),
                "predicted_label": class_names[predictions[i]],
                "correct": int(predictions[i] == int(truth[i])),
                "candidate_hit": hits[i], "query_scale": study.image_scale(path)})
        print(f"[wood-coverage] latest fold={fold} K=10 top128 "
              f"R@1={rows[-1]['macro_r1']:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(folder / "summary.csv", index=False)
    pd.DataFrame(query_rows).to_csv(folder / "per_query.csv", index=False)
    study._json(protocol_path, protocol)
    return folder


def _audit_latest(root, old, out, base_out, image_out, manifest, cfg):
    if not torch.cuda.is_available():
        raise RuntimeError("Latest-checkpoint audit requires CUDA")
    train = out / "train" / ARM
    best_path, latest_path = train / "best.pt", train / "latest.pt"
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    latest = torch.load(latest_path, map_location="cpu", weights_only=False)
    source, encoder, model, base_cfg, _ = hard._source(
        root, base_out, image_out, cfg["seed"], torch.device("cuda"))
    expected = hard._provenance(root, source, cfg, ARM)
    if (best["provenance"] != expected or latest["provenance"] != expected or
            latest["epoch"] != cfg["epochs"] or
            best["epoch"] >= latest["epoch"]):
        raise ValueError("Expected distinct best and final checkpoints from the completed run")
    best_meta = json.loads((train / f"epoch_{best['epoch']:02d}_provenance.json").read_text())
    latest_meta = json.loads((train / f"epoch_{latest['epoch']:02d}_provenance.json").read_text())
    if (best_meta["provenance"] != expected or latest_meta["provenance"] != expected or
            best["selection_score"] != best_meta["score"] or
            latest["best_score"] != best["selection_score"]):
        raise ValueError("Checkpoint and per-epoch validation evidence disagree")
    best_dir = out / "validation" / ARM
    if not all((best_dir / name).is_file() for name in
               ("protocol.json", "summary.csv", "per_query.csv")):
        raise ValueError("Evaluate best.pt with coverage validate mode first")
    encoder.load_state_dict(latest["encoder"], strict=True)
    model.load_state_dict(latest["model"], strict=True)
    audit_dir = _validate_latest(root, out, manifest, encoder.eval(), model.eval(),
                                 base_cfg, cfg, latest_path, torch.device("cuda"))
    baseline_dir = old / "validation" / ARM
    baseline_path = old / "train" / ARM / "best.pt"
    reports = {
        "baseline_vs_best": _paired_report(baseline_dir, baseline_path,
                                          best_dir, best_path),
        "baseline_vs_latest": _paired_report(baseline_dir, baseline_path,
                                            audit_dir, latest_path),
        "best_vs_latest": _paired_report(best_dir, best_path,
                                        audit_dir, latest_path),
    }
    result = {
        "selection_split": "meta-val", "test_evaluated": False,
        "endpoint": ENDPOINT, "note": "Diagnostic only; latest was not selected before this audit",
        "best_epoch": int(best["epoch"]), "latest_epoch": int(latest["epoch"]),
        "best_selection_score": float(best_meta["score"]),
        "latest_selection_score": float(latest_meta["score"]),
        "best_epoch_unique_train_images": int(best_meta["unique_train_images"]),
        "latest_epoch_unique_train_images": int(latest_meta["unique_train_images"]),
        "best_checkpoint_sha256": study._hash_file(best_path),
        "latest_checkpoint_sha256": study._hash_file(latest_path),
        "latest_validation_protocol_sha256": study._hash_file(audit_dir / "protocol.json"),
        "comparisons": reports,
    }
    path = out / "checkpoint_audit" / "comparison.json"
    if path.is_file() and json.loads(path.read_text()) != result:
        raise ValueError("Latest-checkpoint audit changed; use a new output folder")
    study._json(path, result)
    print(f"[wood-coverage] checkpoint audit: {path}", flush=True)
    return result


def run():
    root = Path(os.environ.get("ROOT_PATH", "/content/drive/MyDrive/NCS")).resolve()
    old, out, ml_out, base_out, image_out = _paths(root)
    cfg = _config()
    mode = os.environ.get("WOOD_COVERAGE_MODE", "preflight")
    if mode not in {"preflight", "train", "validate", "compare", "audit_latest"}:
        raise ValueError("WOOD_COVERAGE_MODE must be preflight/train/validate/compare/audit_latest")
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
    elif mode == "audit_latest":
        _audit_latest(root, old, out, base_out, image_out, manifest, cfg)
    else:
        report = _paired_evidence(old, out)
        path = out / "coverage_comparison.json"
        if path.is_file() and json.loads(path.read_text()) != report:
            raise ValueError("Coverage evidence changed; use a new output folder")
        study._json(path, report)
        print(f"[wood-coverage] paired meta-val: {report}", flush=True)


if __name__ == "__main__":
    run()
