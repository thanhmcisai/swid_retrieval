"""Inference-only diagnosis of selected SWI meta-validation retrieval checkpoints."""

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import data, gallery_experiment as study
from .gallery_method import GalleryScorer


def _extra_references(manifest, target_labels):
    by_class = defaultdict(list)
    for split in ("meta-train", "meta-val"):
        for path, label in manifest[split]:
            label = study.canonical(label)
            if label not in target_labels:
                by_class[label].append((path, label))
    return [min(rows, key=lambda item: study._hash_bytes(item[0].encode()))
            for _, rows in sorted(by_class.items())]


def gallery_plan(reference_items, probe_items, extra_items):
    """Use the same deterministic references and distractors as training validation."""
    ref_labels = np.asarray([study.canonical(label) for _, label in reference_items])
    query_labels = np.asarray([study.canonical(label) for _, label in probe_items])
    classes = set(ref_labels)
    if len(classes) != 57 or set(query_labels) != classes or len(extra_items) != 580:
        raise ValueError("Expected 57 scan-disjoint targets and 580 stress distractors")
    first = {}
    for index, label in enumerate(ref_labels):
        first.setdefault(label, index)
    balanced = np.asarray(list(first.values()), dtype=np.int64)
    subset = set(sorted(classes, key=lambda label: study._hash_bytes(label.encode()))[:24])
    extra_order = sorted(range(len(extra_items)),
                         key=lambda index: study._hash_bytes(extra_items[index][1].encode()))
    plan = {
        "24x5": (np.flatnonzero(np.isin(ref_labels, list(subset))),
                 np.flatnonzero(np.isin(query_labels, list(subset)))),
        "57x5": (np.arange(len(reference_items)), np.arange(len(probe_items))),
        "57x1": (balanced, np.arange(len(probe_items))),
    }
    for size in (128, 256, 637):
        chosen = np.asarray(extra_order[:size - len(balanced)], dtype=np.int64)
        plan[f"{size}x1"] = (np.r_[balanced, len(reference_items) + chosen],
                            np.arange(len(probe_items)))
    if len(plan["637x1"][0]) != 637:
        raise ValueError("Full stress gallery is incomplete")
    return plan


def query_diagnostics(scores, classes, query_items, gallery_items, *,
                      variant, seed, fold, gallery, scorer_mode, temperature):
    """Return per-query ranking, oracle-shortlist, and scan/scale diagnostics."""
    scores = np.asarray(scores)
    classes = np.asarray(classes)
    if scores.shape != (len(query_items), len(classes)) or len(classes) < 2:
        raise ValueError("Score shape or gallery cardinality changed")
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite retrieval score")
    class_index = {label: index for index, label in enumerate(classes)}
    refs = defaultdict(list)
    for path, label in gallery_items:
        refs[study.canonical(label)].append(path)
    order = np.argsort(-scores, axis=1, kind="stable")
    ranks = np.empty_like(order)
    ranks[np.arange(len(scores))[:, None], order] = np.arange(1, len(classes) + 1)
    rows = []
    for index, (path, raw_label) in enumerate(query_items):
        label = study.canonical(raw_label)
        if label not in class_index:
            raise ValueError(f"Target species absent from gallery: {label}")
        true_index = class_index[label]
        best_index = int(order[index, 0])
        wrong = scores[index].copy()
        wrong[true_index] = -np.inf
        nearest_wrong_index = int(wrong.argmax())
        predicted = str(classes[best_index])
        query_scan = study.source_scan_id(path)
        ref_paths = refs[label]
        if any(study.source_scan_id(ref) == query_scan for ref in ref_paths):
            raise ValueError("Reference/query source-scan leakage")
        query_scale = study.image_scale(path)
        ref_scales = [study.image_scale(ref) for ref in ref_paths]
        rows.append({
            "variant": variant, "seed": seed, "fold": fold, "gallery": gallery,
            "scorer_mode": scorer_mode, "is_native_scorer": int(
                (variant == "metric_large" and scorer_mode == "nearest") or
                (variant != "metric_large" and scorer_mode == "prototype")),
            "query_path": str(path), "query_scan": query_scan,
            "query_scale": query_scale, "true_label": label,
            "true_genus": label.split("_", 1)[0], "predicted_label": predicted,
            "predicted_genus": predicted.split("_", 1)[0],
            "nearest_wrong_label": str(classes[nearest_wrong_index]),
            "correct": int(predicted == label),
            "same_genus_error": int(predicted != label and
                                    predicted.split("_", 1)[0] == label.split("_", 1)[0]),
            "true_class_rank": int(ranks[index, true_index]),
            "top64_oracle": int(ranks[index, true_index] <= 64),
            "margin_cosine": float(temperature *
                                   (scores[index, true_index] - wrong[nearest_wrong_index])),
            "true_score_cosine": float(temperature * scores[index, true_index]),
            "wrong_score_cosine": float(temperature * wrong[nearest_wrong_index]),
            "reference_images": len(ref_paths),
            "same_scale_reference_count": sum(scale == query_scale for scale in ref_scales),
            "true_reference_scales": ",".join(str(scale) for scale in ref_scales),
        })
    return rows


def _summarize(frame, keys):
    rows = []
    for key, part in frame.groupby(keys, sort=True, dropna=False):
        key = (key,) if not isinstance(key, tuple) else key
        by_species = part.groupby("true_label", sort=True)
        errors = part[part["correct"] == 0]
        rows.append({**dict(zip(keys, key)),
                     "n_queries": len(part), "n_species": part["true_label"].nunique(),
                     "macro_r1": float(by_species["correct"].mean().mean()),
                     "macro_top64_oracle": float(by_species["top64_oracle"].mean().mean()),
                     "mean_true_rank": float(part["true_class_rank"].mean()),
                     "mean_margin_cosine": float(part["margin_cosine"].mean()),
                     "same_genus_fraction_of_errors": (
                         float(errors["same_genus_error"].mean()) if len(errors) else 0.0)})
    return pd.DataFrame(rows)


def seed43_comparison(frame):
    """Pair exactly the same validation queries across prototype training seeds."""
    focus = frame[(frame["variant"] == "prototype_large") &
                  (frame["scorer_mode"] == "prototype")]
    keys = ["gallery", "fold", "query_path"]
    columns = keys + ["true_label", "query_scale", "correct", "margin_cosine",
                      "true_class_rank", "top64_oracle"]
    parts = {}
    for seed in (42, 43, 44):
        part = focus[focus["seed"] == seed][columns].copy()
        if part.duplicated(keys).any():
            raise ValueError(f"Repeated validation query for seed {seed}")
        parts[seed] = part.rename(columns={
            name: f"{name}_{seed}" for name in columns if name not in keys})
    merged = parts[43].merge(parts[42], on=keys, validate="one_to_one")
    merged = merged.merge(parts[44], on=keys, validate="one_to_one")
    if len(merged) != len(parts[43]) or any(
            not (merged[f"true_label_{seed}"] == merged["true_label_43"]).all()
            for seed in (42, 44)) or any(
            not (merged[f"query_scale_{seed}"] == merged["query_scale_43"]).all()
            for seed in (42, 44)):
        raise ValueError("Seed comparison query alignment failed")
    merged["other_seed_mean_correct"] = (merged["correct_42"] + merged["correct_44"]) / 2
    merged["seed43_correct_delta"] = merged["correct_43"] - merged["other_seed_mean_correct"]
    merged["seed43_margin_delta"] = (merged["margin_cosine_43"] -
                                     (merged["margin_cosine_42"] +
                                      merged["margin_cosine_44"]) / 2)
    return merged


def seed43_species_bootstrap(paired, n_boot=5000, seed=2026):
    """Exploratory species-cluster intervals; selected checkpoints share these folds."""
    rng = np.random.default_rng(seed)
    rows = []
    for gallery, part in paired.groupby("gallery", sort=True):
        species = part.groupby("true_label_43", sort=True)["seed43_correct_delta"].mean()
        deltas = species.to_numpy(dtype=np.float64)
        draws = rng.integers(0, len(deltas), size=(n_boot, len(deltas)))
        boot = deltas[draws].mean(axis=1)
        rows.append({"gallery": gallery, "n_species": len(deltas),
                     "mean_species_delta": float(deltas.mean()),
                     "ci95_low": float(np.quantile(boot, 0.025)),
                     "ci95_high": float(np.quantile(boot, 0.975)),
                     "n_species_better": int((deltas > 0).sum()),
                     "n_species_worse": int((deltas < 0).sum()),
                     "n_species_tied": int((deltas == 0).sum()),
                     "interpretation": "descriptive; checkpoints selected on these meta-val folds"})
    return pd.DataFrame(rows)


def _checkpoint_rows(root, study_out, manifest, variant, seed, device):
    path = study_out / "dinov2_vits14" / variant / f"seed_{seed}" / "best.pt"
    state = torch.load(path, map_location="cpu", weights_only=False)
    cfg = state["config"]
    if (cfg["variant"] != variant or cfg["backbone"] != "dinov2_vits14" or
            cfg["group_mode"] != "scan_disjoint" or cfg["validation_folds"] != 2 or
            cfg["validation_train_distractors"] != 1 or cfg["image_size"] != 224 or
            state["seed"] != seed or
            state["signature"] != study._run_signature(root, cfg, seed)):
        raise ValueError(f"Checkpoint recipe/provenance mismatch: {path}")
    if cfg.get("local_weight", 0) or cfg["scorer_mode"] not in {"nearest", "prototype"}:
        raise ValueError(f"Unsupported diagnostic scorer or local branch: {path}")
    encoder, _ = study._model(cfg, device, pretrained=True)
    if study._state_sha256(encoder.backbone) != state["initial_backbone_sha256"]:
        raise ValueError(f"Pretrained DINOv2-S weights changed: {path}")
    encoder.load_state_dict(state["encoder"], strict=True)
    encoder.eval()
    rows = []
    for fold in range(2):
        reference_items, probe_items = study.validation_items(
            manifest, group_mode="scan_disjoint", fold=fold)
        target_labels = {study.canonical(label) for _, label in reference_items}
        extra_items = _extra_references(manifest, target_labels)
        plan = gallery_plan(reference_items, probe_items, extra_items)
        all_reference_items = reference_items + extra_items
        refs = study.encode_items(encoder, reference_items, cfg, device)
        queries = study.encode_items(encoder, probe_items, cfg, device)
        extras = study.encode_items(encoder, extra_items, cfg, device)
        all_refs = np.concatenate((refs, extras))
        for gallery, (ref_index, query_index) in plan.items():
            gallery_items = [all_reference_items[index] for index in ref_index]
            selected_queries = [probe_items[index] for index in query_index]
            for mode in ("nearest", "prototype"):
                scorer = GalleryScorer(cfg["top_m"], cfg["temperature"], mode,
                                       cfg["normalize_evidence"]).to(device)
                scores, classes, *_ = study.score_queries(
                    scorer, queries[query_index], all_refs[ref_index],
                    [label for _, label in gallery_items], device)
                rows.extend(query_diagnostics(
                    scores, classes, selected_queries, gallery_items,
                    variant=variant, seed=seed, fold=fold, gallery=gallery,
                    scorer_mode=mode, temperature=cfg["temperature"]))
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows, state["validation"], path, int(state["epoch"])


def run():
    root = Path(os.environ.get("ROOT_PATH", "/content/drive/MyDrive/NCS")).resolve()
    study_out = Path(os.environ.get(
        "GALLERY_DIAG_STUDY_OUT", root / "results" / "dinov2s_retrieval_matched_v1")).resolve()
    out = Path(os.environ.get(
        "GALLERY_DIAG_OUT", root / "results" / "dinov2s_seed43_diagnostics_v1")).resolve()
    variants = [item.strip() for item in os.environ.get(
        "GALLERY_DIAG_VARIANTS",
        "prototype_large,metric_large,prototype_large_frozen").split(",")]
    seeds = [int(item) for item in os.environ.get("GALLERY_DIAG_SEEDS", "42,43,44").split(",")]
    if (len(set(variants)) != len(variants) or len(set(seeds)) != len(seeds) or
            not variants or not seeds or
            any(item not in {"prototype_large", "metric_large", "prototype_large_frozen"}
                for item in variants)):
        raise ValueError("Invalid diagnostic variants or seeds")
    if "prototype_large" not in variants or not {42, 43, 44}.issubset(seeds):
        raise ValueError("Seed-43 comparison requires prototype_large seeds 42/43/44")
    device = torch.device("cuda" if os.environ.get("DEVICE", "cuda") == "cuda" and
                          torch.cuda.is_available() else "cpu")
    manifest = data.load_swi_manifest(root / "swi_manifest.json")
    screen = pd.read_csv(study_out / "meta_val_screen.csv")
    if os.environ.get("GALLERY_DIAG_PRELOAD", "1") == "1":
        paths = []
        for fold in range(2):
            refs, queries = study.validation_items(manifest, group_mode="scan_disjoint", fold=fold)
            paths.extend(path for path, _ in refs + queries +
                         _extra_references(manifest, {study.canonical(label) for _, label in refs}))
        stats = data.preload_image_cache(
            sorted(set(paths)), max_workers=int(os.environ.get("GALLERY_DIAG_PRELOAD_WORKERS", "16")),
            desc="Gallery diagnostic SWI images")
        if stats["bad"]:
            raise RuntimeError(f"Failed to preload {stats['bad']} diagnostic images")
    out.mkdir(parents=True, exist_ok=True)
    all_rows, checkpoints = [], []
    for variant in variants:
        for seed in seeds:
            matching = screen[(screen["backbone"] == "dinov2_vits14") &
                              (screen["variant"] == variant) & (screen["seed"] == seed)]
            if len(matching) != 1:
                raise ValueError(f"Expected exactly one screen row for {variant} seed {seed}")
            path = study_out / "dinov2_vits14" / variant / f"seed_{seed}" / "best.pt"
            if study._hash_file(path) != matching.iloc[0]["checkpoint_sha256"]:
                raise ValueError(f"Screen/checkpoint hash mismatch: {path}")
            if any(matching.iloc[0][column] != study._hash_file(filename) for column, filename in (
                    ("runner_sha256", study.__file__),
                    ("model_sha256", Path(study.__file__).with_name("gallery_method.py")),
                    ("wood_encoder_sha256", Path(study.__file__).with_name("wood_encoder.py")))):
                raise ValueError(f"Screen/code provenance mismatch: {path}")
            print(f"[diagnostics] {variant} seed={seed}", flush=True)
            rows, selected, path, epoch = _checkpoint_rows(root, study_out, manifest,
                                                           variant, seed, device)
            if epoch != int(matching.iloc[0]["epoch"]):
                raise ValueError(f"Screen/checkpoint epoch mismatch: {path}")
            frame = pd.DataFrame(rows)
            native = "nearest" if variant == "metric_large" else "prototype"
            for gallery, expected in (
                    ("24x5", selected["meta_val_r1_24"]),
                    ("57x5", selected["meta_val_r1_all"]),
                    ("128x1", selected["meta_val_gallery_curve"]["128"]),
                    ("256x1", selected["meta_val_gallery_curve"]["256"]),
                    ("637x1", selected["meta_val_r1_stress"])):
                observed = _summarize(frame[(frame["scorer_mode"] == native) &
                                            (frame["gallery"] == gallery)], ["gallery"])
                actual = float(observed.iloc[0]["macro_r1"])
                if abs(actual - expected) > 1e-6:
                    raise ValueError(f"{variant} seed={seed} {gallery}: "
                                     f"diagnostic R@1={actual} vs checkpoint={expected}")
            all_rows.extend(rows)
            checkpoints.append({"variant": variant, "seed": seed,
                                "checkpoint": str(path), "sha256": matching.iloc[0]["checkpoint_sha256"],
                                "selected_epoch": epoch})
    frame = pd.DataFrame(all_rows)
    frame.to_csv(out / "query_diagnostics.csv", index=False)
    _summarize(frame, ["variant", "seed", "scorer_mode", "gallery"]).to_csv(
        out / "summary.csv", index=False)
    _summarize(frame, ["variant", "seed", "scorer_mode", "gallery", "fold"]).to_csv(
        out / "fold_summary.csv", index=False)
    _summarize(frame, ["variant", "seed", "scorer_mode", "gallery", "true_label"]).to_csv(
        out / "species_summary.csv", index=False)
    _summarize(frame, ["variant", "seed", "scorer_mode", "gallery", "query_scale"]).to_csv(
        out / "scale_summary.csv", index=False)
    _summarize(frame, ["variant", "seed", "scorer_mode", "gallery", "query_scale",
                       "same_scale_reference_count"]).to_csv(
        out / "scale_match_summary.csv", index=False)
    paired = seed43_comparison(frame)
    paired.to_csv(out / "seed43_query_comparison.csv", index=False)
    seed43_species_bootstrap(paired).to_csv(out / "seed43_species_bootstrap.csv", index=False)
    _summarize(frame[(frame["variant"] == "prototype_large") &
                     (frame["scorer_mode"] == "prototype")],
               ["variant", "seed", "scorer_mode", "gallery"]).to_csv(
                   out / "seed43_context.csv", index=False)
    paired.groupby(["gallery", "true_label_43"], sort=True).agg(
        n_queries=("seed43_correct_delta", "size"),
        seed43_r1=("correct_43", "mean"),
        other_seed_r1=("other_seed_mean_correct", "mean"),
        r1_delta=("seed43_correct_delta", "mean"),
        margin_delta=("seed43_margin_delta", "mean")).reset_index().to_csv(
            out / "seed43_species_comparison.csv", index=False)
    study._json(out / "diagnostic_manifest.json", {
        "status": "exploratory_meta_validation_only", "public_test_used": False,
        "study_out": str(study_out), "runner_sha256": study._hash_file(study.__file__),
        "model_sha256": study._hash_file(Path(study.__file__).with_name("gallery_method.py")),
        "checkpoint_rows": checkpoints,
        "gallery_protocol": "24/57 species x 5 references; 57/128/256/637 x 1 reference",
        "top64_oracle": "true species is in top 64 global class scores",
        "selection_warning": "These same folds selected each best checkpoint; diagnosis is not an independent test."})
    print(f"[diagnostics] saved {len(frame)} query rows to {out}", flush=True)
    return out


if __name__ == "__main__":
    run()
