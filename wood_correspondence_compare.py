"""Paired meta-val comparison of image-trained QKV and global-only controls.

This module reads completed reports and checkpoints. It never trains or opens
meta-test, and it does not modify either experiment's provenance-bearing code.
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


GALLERIES = {"24x5", "57x5", "57x1", "128x1", "256x1", "637x1"}
KEYS = ["split", "fold", "gallery", "query_path", "true_label"]


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def _load_run(root, variant, seed):
    folder = root / variant / f"seed_{seed}"
    report = folder / "selected_meta_val"
    checkpoint = folder / "best.pt"
    provenance_file = report / "selected_provenance.json"
    queries_file = report / "selected_queries.csv"
    for path in (checkpoint, provenance_file, queries_file):
        if not path.is_file():
            raise FileNotFoundError(path)
    provenance = _json(provenance_file)
    if (provenance.get("meta_test_evaluated") is not False or
            provenance.get("selection_fold") != 0 or
            provenance.get("descriptive_fold") != 1 or
            provenance.get("provenance", {}).get("config", {}).get("variant") != variant or
            provenance["provenance"].get("seed") != seed or
            _sha256(checkpoint) != provenance.get("checkpoint_sha256")):
        raise ValueError(f"Checkpoint/report provenance mismatch: {folder}")
    frame = pd.read_csv(queries_file)
    required = set(KEYS + ["mode", "rank", "correct", "query_scale", "query_scan"])
    if not required.issubset(frame.columns):
        raise ValueError(f"Incomplete query report: {queries_file}")
    frame = frame[frame["mode"] == "trained"].copy()
    if (frame.empty or frame.duplicated(KEYS).any() or
            set(frame["split"]) != {"meta-val"} or
            set(frame["fold"]) != {0, 1} or
            set(frame["gallery"]) != GALLERIES or
            not frame["correct"].isin([0, 1]).all()):
        raise ValueError(f"Incomplete or repeated matched evaluation: {queries_file}")
    return folder, frame, provenance


def _check_matched_runs(qkv_folder, qkv_provenance, global_folder, global_provenance):
    qa = qkv_provenance["provenance"]
    gb = global_provenance["provenance"]
    for field in ("base_checkpoint_sha256", "manifest_sha256", "runner_sha256",
                  "correspondence_sha256", "method_sha256", "seed"):
        if qa.get(field) != gb.get(field):
            raise ValueError(f"QKV/global-only {field} mismatch")
    qc, gc = dict(qa["config"]), dict(gb["config"])
    if (qc.pop("variant") != "qkv" or qc.pop("score_mode") != "qkv" or
            gc.pop("variant") != "global_only" or gc.pop("score_mode") != "global" or
            qc != gc):
        raise ValueError("QKV/global-only training configurations are not matched")
    if qkv_provenance["checkpoint_sha256"] == global_provenance["checkpoint_sha256"]:
        raise ValueError("Both methods point to the same checkpoint")
    epochs = int(qc["epochs"])
    for epoch in range(1, epochs + 1):
        qa_epoch = _json(qkv_folder / f"epoch_{epoch:02d}.json")
        gb_epoch = _json(global_folder / f"epoch_{epoch:02d}.json")
        if (qa_epoch["epoch_coverage"] != gb_epoch["epoch_coverage"] or
                qa_epoch["cumulative_unique_images"] !=
                gb_epoch["cumulative_unique_images"] or
                qa_epoch["cumulative_unique_queries"] !=
                gb_epoch["cumulative_unique_queries"]):
            raise ValueError(f"Image sampling coverage differs at epoch {epoch}")
    return epochs


def _paired(qkv, global_only):
    left = global_only[KEYS + ["rank", "correct", "query_scale", "query_scan"]].rename(
        columns={"rank": "global_only_rank", "correct": "global_only_correct"})
    right = qkv[KEYS + ["rank", "correct", "query_scale", "query_scan"]].rename(
        columns={"rank": "qkv_rank", "correct": "qkv_correct"})
    paired = left.merge(right, on=KEYS, how="outer", validate="one_to_one",
                        suffixes=("_global_only", "_qkv"), indicator=True)
    if (len(paired) != len(left) or len(paired) != len(right) or
            not paired["_merge"].eq("both").all() or
            not paired["query_scale_global_only"].eq(paired["query_scale_qkv"]).all() or
            not paired["query_scan_global_only"].eq(paired["query_scan_qkv"]).all()):
        raise ValueError("QKV/global-only reports have different query identity")
    paired = paired.drop(columns=["_merge", "query_scale_qkv", "query_scan_qkv"])
    paired = paired.rename(columns={"query_scale_global_only": "query_scale",
                                    "query_scan_global_only": "query_scan"})
    paired["delta"] = paired["qkv_correct"] - paired["global_only_correct"]
    return paired.sort_values(KEYS).reset_index(drop=True)


def _summary(paired, bootstrap=5000, seed=2026):
    if bootstrap < 100:
        raise ValueError("At least 100 species-bootstrap replicates are required")
    rows = []
    for gallery in sorted(GALLERIES):
        for fold in (None, 0, 1):
            part = paired[paired["gallery"] == gallery]
            if fold is not None:
                part = part[part["fold"] == fold]
            species_delta = part.groupby("true_label")["delta"].mean().to_numpy()
            if not len(species_delta):
                raise ValueError(f"Empty paired gallery {gallery}, fold={fold}")
            rng = np.random.default_rng(seed + (fold or 0) + len(part))
            sampled = species_delta[rng.integers(
                len(species_delta), size=(bootstrap, len(species_delta)))].mean(axis=1)
            rows.append({"gallery": gallery, "fold": "both" if fold is None else fold,
                         "n_queries": len(part), "n_species": len(species_delta),
                         "global_only_macro_r1": float(part.groupby("true_label")[
                             "global_only_correct"].mean().mean()),
                         "qkv_macro_r1": float(part.groupby("true_label")[
                             "qkv_correct"].mean().mean()),
                         "qkv_minus_global_only": float(species_delta.mean()),
                         "ci95_low": float(np.quantile(sampled, 0.025)),
                         "ci95_high": float(np.quantile(sampled, 0.975)),
                         "rescued": int(((part["global_only_correct"] == 0) &
                                         (part["qkv_correct"] == 1)).sum()),
                         "harmed": int(((part["global_only_correct"] == 1) &
                                        (part["qkv_correct"] == 0)).sum())})
    return pd.DataFrame(rows)


def run():
    root = Path(os.environ.get("ROOT_PATH", "/content/drive/MyDrive/NCS")).resolve()
    qkv_root = Path(os.environ.get("WOOD_COMPARE_QKV_OUT", root / "results" /
                                   "wood_correspondence_image_full_v1")).resolve()
    global_root = Path(os.environ.get("WOOD_COMPARE_GLOBAL_OUT", root / "results" /
                                      "wood_correspondence_global_only_v1")).resolve()
    out = Path(os.environ.get("WOOD_COMPARE_OUT", root / "results" /
                              "wood_correspondence_matched_comparison_v1")).resolve()
    seed = int(os.environ.get("WOOD_COMPARE_SEED", "43"))
    qkv_folder, qkv, qp = _load_run(qkv_root, "qkv", seed)
    global_folder, global_only, gp = _load_run(global_root, "global_only", seed)
    epochs = _check_matched_runs(qkv_folder, qp, global_folder, gp)
    paired = _paired(qkv, global_only)
    summary = _summary(paired)
    out.mkdir(parents=True, exist_ok=True)
    paired.to_csv(out / "paired_queries.csv", index=False)
    summary.to_csv(out / "paired_summary.csv", index=False)
    manifest = {"qkv_checkpoint_sha256": qp["checkpoint_sha256"],
                "global_only_checkpoint_sha256": gp["checkpoint_sha256"],
                "qkv_selected_epoch": qp["selected_epoch"],
                "global_only_selected_epoch": gp["selected_epoch"],
                "base_checkpoint_sha256": qp["provenance"]["base_checkpoint_sha256"],
                "manifest_sha256": qp["provenance"]["manifest_sha256"],
                "same_sampling_coverage_all_epochs": True,
                "matched_training_epochs": epochs, "seed": seed,
                "meta_test_evaluated": False,
                "inference": "Paired species bootstrap on model-selected meta-val; "
                             "does not account for seed or model-selection variance. "
                             "The pretrained base checkpoint was selected using "
                             "both meta-val folds, so fold 1 is not an independent "
                             "holdout for the complete pipeline."}
    with (out / "comparison_manifest.json").open("w") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
    print(summary[summary["fold"] == "both"].to_string(index=False), flush=True)
    print(f"[wood-compare] matched report: {out}", flush=True)
    return out
