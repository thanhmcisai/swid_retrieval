"""Inference-only audit for the corrected-public, retrained-CE submission run.

The source caches and prior results are read-only. This command records which
checkpoint, scoring mode, class map and input rows produced every new number.
"""
import argparse
import ast
import json
import os
import runpy
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .audit_support import (canonical, check_labels, compare_recipes, completed,
                            finish, sha256, top_matches, write_json)


SCALES = (256, 512, 768)
MAGS = ("x10", "x20", "x50")
SEEDS = (42, 43, 44)


def paths(args):
    root = args.root.resolve()
    run = args.run_root.resolve()
    research = args.research_dir.resolve() if args.research_dir else run / "deployment" / "research_directions"
    return {
        "root": root, "run": run, "research": research,
        "cache": (args.cache or root / "embedding_cache_full954_v5_retrained_ce_corrected_public.npz").resolve(),
        "exp4": (args.exp4 or root / "exp4_embedding_cache_v3.npz").resolve(),
        "ce": (args.ce_checkpoint or root / "checkpoints" / "ce_954sp_convnext_base.pt").resolve(),
        "id": root / "ID_images_expanded.csv", "ood": root / "OOD_images_expanded.csv",
        "correction": root / "dataset_label_corrections.json",
        "manifest": root / "swi_manifest.json",
        "selected": run / "hyperparameters" / "scurd_hyperparameter_selection.json",
        "out": (args.out or run / "final_colab_audit").resolve(),
    }


def _e4key(prefix, method, split):
    return f"{prefix}__{method}__{split}"


def _vn26_items(root):
    """Replay the original VN26 folder/magnification mapping, retaining paths."""
    local_prefix = "/Users/admin/Downloads/Experimentals/datasets"
    items = {mag: [] for mag in MAGS}
    for name in ("ID_species_public.csv", "OOD_species_public.csv"):
        csv = root / name
        if not csv.is_file():
            raise FileNotFoundError(f"VN26 source mapping requires {csv}")
        df = pd.read_csv(csv)
        for column in ("dataset", "folder_path", "magnification", "canonical_binomial"):
            if column not in df:
                raise ValueError(f"{csv} lacks {column}")
        for _, row in df.iterrows():
            datasets = ast.literal_eval(row["dataset"])
            folders = ast.literal_eval(row["folder_path"])
            mags = ast.literal_eval(row["magnification"])
            if not (len(datasets) == len(folders) == len(mags)):
                raise ValueError(f"Misaligned VN26 source lists in {csv}")
            for ds, folder, mag in zip(datasets, folders, mags):
                if ds != "VN26" or mag not in items:
                    continue
                folder = Path(str(folder).replace(local_prefix, str(root / "datasets")))
                if not folder.is_dir():
                    raise FileNotFoundError(f"VN26 folder missing: {folder}")
                label = canonical([row["canonical_binomial"]])[0]
                items[mag].extend((str(img), label) for img in sorted(folder.iterdir())
                                  if img.suffix.lower() in {".jpg", ".jpeg", ".png"})
    for mag, rows in items.items():
        if not rows or len({p for p, _ in rows}) != len(rows):
            raise ValueError(f"Empty or duplicate VN26 paths for {mag}")
    return items


def _checkpoint_metadata(p):
    import torch
    checkpoint = torch.load(p, map_location="cpu", weights_only=False)
    return {k: checkpoint.get(k) for k in ("in_dim", "out_dim", "head_type", "epochs",
            "episodes_per_epoch", "n_way", "k_support", "q_query", "lambda_cons",
            "lr", "weight_decay", "tau", "beta", "learnable_beta",
            "selection_metric", "seed", "meta_cache", "cache_version",
            "research_cache_version", "method", "training_complete")}, checkpoint


def _validate_public_rows(df, labels, overrides, side):
    required = {"file_path", "label", "source_dataset", "source_original_name",
                "corrected_name", "label_correction"}
    if not required.issubset(df.columns):
        raise ValueError(f"{side} CSV lacks correction audit columns: {sorted(required - set(df.columns))}")
    if df["file_path"].duplicated().any():
        raise ValueError(f"Duplicate paths in corrected {side} CSV")
    check_labels(df["label"], labels, side)
    if df["file_path"].astype(str).str.contains("woodauth|wood-auth|wood_auth", case=False).any():
        raise ValueError(f"WoodAuth rows remain in corrected {side} CSV")
    if df["source_dataset"].astype(str).str.upper().eq("WOODAUTH").any():
        raise ValueError(f"WoodAuth source remains in corrected {side} CSV")
    fsdm = df["file_path"].astype(str).str.contains("/FSDM41/", case=False, regex=False)
    fsdm |= df["source_dataset"].astype(str).str.upper().eq("FSDM41")
    corrected = 0
    for _, row in df[fsdm].iterrows():
        expected = overrides.get(str(row["source_original_name"]))
        marker = str(row["label_correction"])
        if expected:
            if marker != "FSDM41_PERMUTED_LABEL" or str(row["corrected_name"]) != expected:
                raise ValueError(f"FSDM41 mapping differs from correction JSON: {row['file_path']}")
            corrected += 1
        elif marker == "FSDM41_PERMUTED_LABEL":
            raise ValueError(f"Unexpected FSDM41 correction: {row['file_path']}")
    return corrected


def preflight(p):
    if p["cache"].parent != p["root"] or p["exp4"].parent != p["root"]:
        raise ValueError("Cache and exp4 files must be directly under --root for the validated gallery engine")
    if p["research"] != p["run"] / "deployment" / "research_directions":
        raise ValueError("Research checkpoints must be in the deployment research directory")
    required = ("cache", "exp4", "ce", "id", "ood", "correction", "manifest", "selected")
    for key in required:
        if not p[key].is_file():
            raise FileNotFoundError(f"Missing {key}: {p[key]}")
    selected = json.loads(p["selected"].read_text())["selected"]
    if selected.get("mode") != "raw":
        raise ValueError("Meta-val selected scoring mode is not raw; inspect the selection artifact")
    if not {"checkpoint", "tau", "top_m"}.issubset(selected):
        raise ValueError("Incomplete meta-val selection artifact")
    overrides = json.loads(p["correction"].read_text())["FSDM41"]["overrides"]
    ckpts = {"selected": p["research"] / selected["checkpoint"]}
    ckpts.update({f"seed{seed}": p["research"] /
                  f"sc_urd_checkpoint_scurd_r01_e20_seed{seed}_v2.pt" for seed in SEEDS})
    for name, ckpt in ckpts.items():
        if not ckpt.is_file():
            raise FileNotFoundError(f"Missing {name} checkpoint: {ckpt}")

    fsdm_corrected = 0
    with np.load(p["cache"], allow_pickle=False) as cache:
        id_labels = canonical(cache["labels_id_dinov2"])
        ood_labels = canonical(cache["labels_ood_dinov2"])
        swi_labels = canonical(cache["labels_swi_dinov2"])
        observed = (len(id_labels), len(ood_labels), len(swi_labels),
                    len(set(id_labels)), len(set(ood_labels)), len(set(swi_labels)))
        expected_counts = (6189, 30868, 176123, 24, 153, 954)
        if observed != expected_counts:
            raise ValueError(f"Not the corrected v5 cohort: observed {observed}, expected {expected_counts}")
        overlap = set(id_labels) & set(ood_labels)
        if overlap:
            raise ValueError(f"Corrected public ID/OOD species overlap: {sorted(overlap)[:5]}")
        for side, labels in (("id", id_labels), ("ood", ood_labels)):
            df = pd.read_csv(p[side])
            fsdm_corrected += _validate_public_rows(df, labels, overrides, side)
            for method in ("dinov2", "ce_full_norm"):
                if len(cache[f"embs_{side}_{method}"]) != len(df):
                    raise ValueError(f"{side}/{method} feature count differs from CSV")
        for method in ("dinov2", "ce_full_norm"):
            if len(cache[f"embs_swi_{method}"]) != len(swi_labels):
                raise ValueError(f"SWI/{method} feature count differs from labels")
        if len(cache["logits_id_ce_full"]) != len(id_labels):
            raise ValueError("CE logits count differs from ID rows")
        with p["manifest"].open() as f:
            manifest = json.load(f)
        swi_items = sum((manifest[split] for split in ("meta-train", "meta-val", "meta-test")), [])
        check_labels([label for _, label in swi_items], swi_labels, "SWI manifest/cache")
    if fsdm_corrected != 2901:
        raise ValueError(f"Expected 2901 FSDM41 relabelled images, got {fsdm_corrected}")

    with np.load(p["exp4"], allow_pickle=False) as exp4:
        vn_items = _vn26_items(p["root"])
        for scale in SCALES:
            key = _e4key("swi", "DINOv2", scale)
            if key not in exp4 or f"{key}_lbl" not in exp4:
                raise KeyError(f"Missing {key} in exp4 cache")
            expected = [(path, label) for path, label in manifest["meta-test"]
                        if Path(path).parent.name == f"scale_{scale}"]
            check_labels([label for _, label in expected], exp4[f"{key}_lbl"], f"exp4/{key}")
            if len(exp4[key]) != len(expected):
                raise ValueError(f"exp4/{key} row count mismatch")
            ce_key = _e4key("swi", "CE_Full", scale)
            if ce_key not in exp4 or f"{ce_key}_lbl" not in exp4:
                raise KeyError(f"Missing {ce_key} in exp4 cache")
            if exp4[ce_key].ndim != 2 or exp4[ce_key].shape[1] != 954:
                raise ValueError(f"Legacy exp4/{ce_key} is not 954-class logits")
            check_labels([label for _, label in expected], exp4[f"{ce_key}_lbl"], f"exp4/{ce_key}")
        for mag in MAGS:
            key = _e4key("vn26", "DINOv2", mag)
            if key not in exp4 or f"{key}_lbl" not in exp4:
                raise KeyError(f"Missing {key} in exp4 cache")
            ce_key = _e4key("vn26", "CE_Full", mag)
            if ce_key not in exp4 or f"{ce_key}_lbl" not in exp4:
                raise KeyError(f"Missing {ce_key} in exp4 cache")
            if exp4[ce_key].ndim != 2 or exp4[ce_key].shape[1] != 954:
                raise ValueError(f"Legacy exp4/{ce_key} is not 954-class logits")
            for method_key in (key, ce_key):
                if Counter(canonical(label for _, label in vn_items[mag])) != Counter(canonical(exp4[f"{method_key}_lbl"])):
                    raise ValueError(f"VN26 {mag} image labels differ from exp4/{method_key}")

    ce_meta, ce_ckpt = _checkpoint_metadata(p["ce"])
    classes = ce_ckpt.get("ce_species_list")
    if classes is None or len(classes) != 954:
        raise ValueError("CE checkpoint lacks its ordered 954-class map")
    if ce_ckpt.get("metrics", {}).get("n_classes") != 954:
        raise ValueError("CE checkpoint class count does not match 954")
    check_labels(sorted(set(swi_labels)), classes, "CE class map/SWI manifest")
    if not set(id_labels).issubset(set(canonical(classes))):
        raise ValueError("Public ID labels absent from CE class map")
    recipes = {name: _checkpoint_metadata(path)[0] for name, path in ckpts.items()}
    recipe_comparison = compare_recipes(recipes)
    meta_caches = {name: row.get("meta_cache") for name, row in recipes.items()}
    report = {
        "selected": selected, "checkpoint_paths": {name: str(path) for name, path in ckpts.items()},
        "recipe_comparison": recipe_comparison, "ce_epoch": ce_ckpt.get("epoch"),
        "checkpoint_meta_caches": meta_caches,
        "counts": {"id_images": len(id_labels), "ood_images": len(ood_labels),
                   "swi_images": len(swi_labels), "id_species": len(set(id_labels)),
                   "ood_species": len(set(ood_labels)), "swi_species": len(set(swi_labels)),
                   "fsdm41_relabelled": fsdm_corrected},
        "known_limitations": [
            "Legacy exp4 stores labels/features but not ordered image paths; original VN26 image identity cannot be proven from that cache.",
            "Matching checkpoint metadata does not prove matching training data or preprocessing.",
        ],
    }
    return report, ckpts, manifest


def _input_hashes(p, ckpts, meta_caches):
    files = {key: p[key] for key in ("cache", "exp4", "ce", "id", "ood", "correction", "manifest", "selected")}
    files.update({f"checkpoint_{key}": path for key, path in ckpts.items()})
    files.update({name: p["root"] / name for name in ("ID_species_public.csv", "OOD_species_public.csv")})
    package = Path(__file__).parent
    for name in ("final_colab_audit.py", "run_overnight.py", "audit_support.py", "config.py",
                 "data.py", "models.py", "gallery.py", "scurd.py",
                 "embeddings/extract.py", "experiments/audit_native_ce.py",
                 "experiments/registry.py", "experiments/rq1_native.py",
                 "experiments/rq4_vn26.py",
                 "_engines/variance_retrieval_evidence_colab.py"):
        files[f"code_{name}"] = package / name
    for name, value in meta_caches.items():
        if value is not None and Path(value).is_file():
            files[f"training_meta_{name}"] = Path(value)
    hashes = {}
    result = {}
    for key, path in files.items():
        resolved = path.resolve()
        if resolved not in hashes:
            hashes[resolved] = sha256(resolved)
        result[key] = {"path": str(resolved), "sha256": hashes[resolved]}
    return result


def _ce_features(model, loader, device, count):
    from .embeddings.extract import extract_embeddings
    features, indices, logits = extract_embeddings(model, loader, device, return_logits=True)
    if features.ndim != 2 or features.shape != (count, 512):
        raise ValueError(f"CE extraction returned {features.shape}, expected {(count, 512)}")
    if logits.ndim != 2 or logits.shape != (count, 954):
        raise ValueError(f"CE logits returned {logits.shape}, expected {(count, 954)}")
    return features, logits, indices


def _extract_ce(model, items, args):
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from .data import get_transforms
    transform = get_transforms(224, augment=False)
    class DirectItemsDataset(Dataset):
        def __init__(self, rows):
            self.rows = rows
            self.idx_to_class = {i: label for i, (_, label) in enumerate(rows)}

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            with Image.open(self.rows[i][0]) as image:
                pixels = np.array(image.convert("RGB"))
            return transform(image=pixels)["image"], i

    ds = DirectItemsDataset(items)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=args.device == "cuda")
    features, logits, indices = _ce_features(model, loader, args.device, len(items))
    labels = np.asarray([ds.idx_to_class[int(i)] for i in indices])
    check_labels([label for _, label in items], labels, "CE extraction order")
    return features, logits, labels


def ce_exp4(p, args, manifest):
    import torch
    from .models import CEClassifier
    ckpt = torch.load(p["ce"], map_location="cpu", weights_only=False)
    model = CEClassifier("convnext_base", n_classes=954, embedding_dim=512, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(args.device).eval()
    vn = _vn26_items(p["root"])
    outputs = {}
    with np.load(p["exp4"], allow_pickle=False) as old:
        for scale in SCALES:
            items = [(path, label) for path, label in manifest["meta-test"]
                     if Path(path).parent.name == f"scale_{scale}"]
            key = _e4key("swi", "CE_Full", scale)
            features, logits, labels = _extract_ce(model, items, args)
            check_labels(labels, old[f"{key}_lbl"], key)
            outputs[key], outputs[f"{key}_feature512"] = logits, features
            outputs[f"{key}_lbl"] = labels
            outputs[f"{key}_paths"] = np.asarray([path for path, _ in items])
        for mag in MAGS:
            key = _e4key("vn26", "CE_Full", mag)
            items = vn[mag]
            features, logits, labels = _extract_ce(model, items, args)
            if Counter(canonical(labels)) != Counter(canonical(old[f"{key}_lbl"])):
                raise ValueError(f"VN26 {mag} label multiset differs from prior exp4 cache")
            outputs[key], outputs[f"{key}_feature512"] = logits, features
            outputs[f"{key}_lbl"] = labels
            outputs[f"{key}_paths"] = np.asarray([path for path, _ in items])
    out = p["out"] / "ce_exp4_fresh.npz"
    np.savez_compressed(out, **outputs)
    write_json(p["out"] / "ce_exp4_provenance.json", {
        "checkpoint_sha256": sha256(p["ce"]),
        "input_mapping": [str(p["root"] / name) for name in
                          ("ID_species_public.csv", "OOD_species_public.csv", "swi_manifest.json")],
        "transform": "get_transforms(224, augment=False)",
        "image_reading": "PIL RGB directly for both SWI and VN26, as in original exp4",
        "representations": {"legacy_protocol": "954-class CE logits",
                            "sensitivity": "512-dimensional normalized penultimate features"},
        "comparison": "SWI ordered labels and VN26 label multisets match prior exp4; prior image paths unavailable",
        "n_images": {k: len(v) for k, v in outputs.items() if k.endswith("_lbl")},
    })
    return [out, p["out"] / "ce_exp4_provenance.json"]


def _scores(q, g):
    from .experiments.registry import _norm
    _, best = top_matches(_norm(q), _norm(g), k=1)
    return 1.0 - best[:, 0]


def _scurd_metrics(projected, labels, mode, tau, top_m, mask, splits, vn26, ood_metrics):
    from .experiments.registry import scurd_retrieval_eval
    q, g, o = projected["id"], projected["swi"][mask], projected["ood"]
    li, lg, lo = labels["id"], labels["swi"][mask], labels["ood"]
    scorer = lambda a, al, b, bl: scurd_retrieval_eval(a, al, b, bl,
                                                       mode=mode, tau=tau, top_m=top_m)
    baseline = scorer(q, li, g, lg)
    pool, queries = [], []
    for sp in splits:
        pool.extend(splits[sp]["pool_indices"][:10])
        queries.extend(splits[sp]["query_indices"])
    expanded_g = np.concatenate((g, o[pool]))
    expanded_l = np.concatenate((lg, lo[pool]))
    old = scorer(q, li, expanded_g, expanded_l)
    new = scorer(o[queries], lo[queries], expanded_g, expanded_l)
    result = {
        "public_id_macro_R1": baseline["mean"],
        "ood_auroc": ood_metrics["auroc"], "ood_fpr95": ood_metrics["fpr95"],
        "old_after_plus50_K10": old["mean"], "new50_after_K10": new["mean"],
        "n_ood_test_images": ood_metrics["n_test"],
        "n_new_gallery_images": len(pool), "n_new_query_images": len(queries),
        "per_species": {"public_id": baseline["per_species"],
                        "old_after": old["per_species"], "new50": new["per_species"]},
    }
    if vn26:
        swi_e, swi_l, vn_e, vn_l = vn26
        common = set(swi_l) & set(vn_l)
        gm = np.isin(swi_l, list(common))
        qm = np.isin(vn_l, list(common))
        ev = scorer(vn_e[qm], vn_l[qm], swi_e[gm], swi_l[gm])
        result["vn26_swi_pool_all"] = ev["mean"]
        result["vn26_n_common"] = len(common)
        result["per_species"]["vn26"] = ev["per_species"]
    return result


def _ood_metrics(projected, labels, gallery_mask):
    from sklearn.metrics import roc_auc_score, roc_curve
    g = projected["swi"][gallery_mask]
    sid = _scores(projected["id"], g)
    sod = _scores(projected["ood"][labels["ood_test_mask"]], g)
    y = np.r_[np.zeros(len(sid)), np.ones(len(sod))]
    scores = np.r_[sid, sod]
    fpr, tpr, _ = roc_curve(y, scores)
    return {"auroc": float(roc_auc_score(y, scores)),
            "fpr95": float(fpr[min(np.searchsorted(tpr, .95), len(fpr) - 1)]),
            "n_test": len(sod)}


def _triage_rows(projected, labels, gallery_mask, mode, tau, top_m):
    from .experiments.registry import _norm, scurd_class_scores
    query = projected["id"]
    gallery = projected["swi"][gallery_mask]
    gallery_labels = labels["swi"][gallery_mask]
    scores, classes = scurd_class_scores(query, gallery, gallery_labels,
                                         mode=mode, tau=tau, top_m=top_m)
    correct = classes[np.argmax(scores, axis=1)] == labels["id"]
    if mode == "centered":
        mean = gallery.mean(axis=0, keepdims=True)
        query, gallery = query - mean, gallery - mean
    _, nearest = top_matches(_norm(query), _norm(gallery), k=1)
    evidence = nearest[:, 0]
    thresholds = sorted(set([float(evidence.min())] +
                            [float(np.quantile(evidence, q)) for q in (.25, .50, .75, .90)]))
    return [{"mode": mode, "gallery_scope": "id_only", "similarity_threshold": th,
             "auto_decided_coverage_pct": float(100 * np.mean(evidence >= th)),
             "top1_accuracy_auto_decided": float(np.mean(correct[evidence >= th])),
             "flagged_for_review_pct": float(100 * np.mean(evidence < th)),
             "n_auto_decided": int(np.sum(evidence >= th)), "n_total": len(evidence)}
            for th in thresholds]


def _gallery_resampling(projected, labels, root, mode, tau, top_m, n_repeats):
    from .experiments.registry import scurd_retrieval_eval
    df = pd.read_csv(root / "ID_images_expanded.csv")
    check_labels(df["label"], labels["id"], "gallery resampling ID rows")
    if df["file_path"].duplicated().any():
        raise ValueError("Duplicate ID paths in gallery resampling")
    rng = np.random.RandomState(42)
    split = {}
    for sp in sorted(set(labels["id"])):
        idx = np.where(labels["id"] == sp)[0]
        rng.shuffle(idx)
        n_query = min(10, len(idx) // 2)
        split[sp] = {"query": idx[:n_query], "pool": idx[n_query:]}
    query_idx = np.concatenate([split[sp]["query"] for sp in sorted(split)])
    q, q_labels = projected["id"][query_idx], labels["id"][query_idx]
    mask = np.isin(labels["swi"], np.unique(labels["id"]))
    swi_g, swi_l = projected["swi"][mask], labels["swi"][mask]
    if len(q) != 240:
        raise ValueError(f"Expected 240 fixed public-ID gallery queries, got {len(q)}")
    rows = []
    for strategy in ("A_swi_only", "B_public_only", "C_mixed"):
        for k in (("base",) if strategy == "A_swi_only" else (1, 3, 5, 10, "all")):
            vals = []
            repeats = 1 if k in ("base", "all") else n_repeats
            for rep in range(repeats):
                draw = np.random.RandomState(42 + rep)
                selected = []
                for sp in sorted(split):
                    pool = split[sp]["pool"]
                    if k == "base":
                        continue
                    if k == "all":
                        selected.extend(pool)
                    else:
                        chosen = draw.choice(len(pool), min(int(k), len(pool)), replace=False)
                        selected.extend(pool[chosen])
                if strategy == "A_swi_only":
                    gallery, gallery_labels = swi_g, swi_l
                elif strategy == "B_public_only":
                    gallery = projected["id"][selected]
                    gallery_labels = labels["id"][selected]
                else:
                    gallery = np.concatenate((swi_g, projected["id"][selected]))
                    gallery_labels = np.concatenate((swi_l, labels["id"][selected]))
                ev = scurd_retrieval_eval(q, q_labels, gallery, gallery_labels,
                                          mode=mode, tau=tau, top_m=top_m)
                vals.append(ev["mean"])
            vals = np.asarray(vals)
            rows.append({"method": "SC-URD", "mode": mode, "strategy": strategy,
                         "K": str(k), "mean": float(vals.mean()),
                         "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                         "min": float(vals.min()), "max": float(vals.max()),
                         "n_repeats": len(vals), "n_queries": len(q)})
            print(f"gallery SC-URD/{mode} {strategy} K={k}: {vals.mean():.4f}", flush=True)
    return rows


def _ood_kshot(projected, labels, splits, mode, tau, top_m, n_repeats):
    from .experiments.registry import scurd_retrieval_eval
    values = []
    for rep in range(n_repeats):
        rng = np.random.RandomState(42 + rep)
        queries, gallery = [], []
        for sp in splits:
            pool = splits[sp]["pool_indices"]
            chosen = rng.choice(len(pool), min(10, len(pool)), replace=False)
            gallery.extend(pool[i] for i in chosen)
            queries.extend(splits[sp]["query_indices"])
        ev = scurd_retrieval_eval(projected["ood"][queries], labels["ood"][queries],
                                  projected["ood"][gallery], labels["ood"][gallery],
                                  mode=mode, tau=tau, top_m=top_m)
        values.append(ev["mean"])
    vals = np.asarray(values)
    return {"mean": float(vals.mean()), "std": float(vals.std(ddof=0)),
            "n_repeats": len(vals), "n_queries": len(queries)}


def scurd_audit(p, args, preflight_report, ckpts):
    from .experiments import registry as R
    from .experiments.rq4_vn26 import run as run_rq4
    from .experiments.rq1_native import prototype_macro_top1
    from .scurd import load_scurd_model, project_np
    with np.load(p["cache"], allow_pickle=False) as cache, np.load(p["exp4"], allow_pickle=False) as exp4:
        labels = {side: canonical(cache[f"labels_{side}_dinov2"]) for side in ("id", "ood", "swi")}
        labels["ood_test_mask"] = R.ood_test_mask(labels["ood"])
        raw = {side: cache[f"embs_{side}_dinov2"] for side in ("id", "ood", "swi")}
        scale = {s: (exp4[_e4key("swi", "DINOv2", s)],
                     canonical(exp4[_e4key("swi", "DINOv2", s) + "_lbl"])) for s in SCALES}
        mags = {m: (exp4[_e4key("vn26", "DINOv2", m)],
                    canonical(exp4[_e4key("vn26", "DINOv2", m) + "_lbl"])) for m in MAGS}
    mask24 = np.isin(labels["swi"], np.unique(labels["id"]))
    if mask24.sum() != 3315:
        raise ValueError(f"Expected 3315 SWI references for deployment, got {mask24.sum()}")
    splits, top50 = R.ood_species_splits()
    if len(top50) != 50:
        raise ValueError("OOD top-50 selection failed")
    rows = []
    gallery_rows = []
    triage_rows = []
    rq4_files = []
    selected = preflight_report["selected"]
    for name, checkpoint in ckpts.items():
        model, _ = load_scurd_model(checkpoint, in_dim=raw["id"].shape[1], device=args.device)
        proj = {side: project_np(model, value, args.device) for side, value in raw.items()}
        sw = [project_np(model, scale[s][0], args.device) for s in SCALES]
        vn = [project_np(model, mags[m][0], args.device) for m in MAGS]
        vn26 = (np.concatenate(sw), np.concatenate([scale[s][1] for s in SCALES]),
                np.concatenate(vn), np.concatenate([mags[m][1] for m in MAGS]))
        ood_metrics = _ood_metrics(proj, labels, mask24)
        for mode in ("raw", "centered"):
            entry = _scurd_metrics(proj, labels, mode, float(selected["tau"]),
                                   int(selected["top_m"]), mask24, splits, vn26, ood_metrics)
            entry.update({"checkpoint_role": name, "checkpoint": checkpoint.name,
                          "checkpoint_sha256": sha256(checkpoint), "mode": mode,
                          "gallery_scope": "id_only", "gallery_images": int(mask24.sum()),
                          "gallery_species": 24, "tau": float(selected["tau"]),
                          "top_m": int(selected["top_m"])})
            rows.append(entry)
            if name == "selected":
                triage_rows.extend(_triage_rows(proj, labels, mask24, mode,
                                                float(selected["tau"]), int(selected["top_m"])))
                swi_parts = {s: (sw[i], scale[s][1]) for i, s in enumerate(SCALES)}
                swi_parts["pool"] = (vn26[0], vn26[1])
                vn_parts = {m: (vn[i], mags[m][1]) for i, m in enumerate(MAGS)}
                rq4_dir = p["out"] / f"scurd_rq4_{mode}"
                rq4 = run_rq4({"SC-URD": {"swi_scales": swi_parts,
                                            "vn26_mags": vn_parts,
                                            "scurd_mode": mode,
                                            "scurd_tau": float(selected["tau"]),
                                            "scurd_top_m": int(selected["top_m"]) }}, rq4_dir)
                rq4_main = rq4["cross_domain"]["SC-URD"]["SWI_pool"]["VN26_all"]["mean"]
                if abs(rq4_main - entry["vn26_swi_pool_all"]) > 1e-6:
                    raise ValueError(f"SC-URD {mode} VN26 main cell disagrees with RQ4 evaluator")
                rq4_files.append(rq4_dir / "rq4_generalization.json")
                entry["ood_only_kshot_K10"] = _ood_kshot(
                    proj, labels, splits, mode, float(selected["tau"]),
                    int(selected["top_m"]), args.ood_kshot_repeats)
                gallery_rows.extend(_gallery_resampling(
                    proj, labels, p["root"], mode, float(selected["tau"]),
                    int(selected["top_m"]), args.gallery_repeats))
            print(f"{name}/{mode}: R@1={entry['public_id_macro_R1']:.4f} "
                  f"AUROC={entry['ood_auroc']:.4f} VN26={entry['vn26_swi_pool_all']:.4f}", flush=True)
        if name == "selected":
            from .experiments.registry import scurd_retrieval_eval
            for mode in ("raw", "centered"):
                ev = scurd_retrieval_eval(proj["id"], labels["id"], proj["swi"], labels["swi"],
                                          mode=mode, tau=float(selected["tau"]),
                                          top_m=int(selected["top_m"]))
                proto = prototype_macro_top1(proj["id"], labels["id"], proj["swi"], labels["swi"],
                                              centered=mode == "centered")
                rows.append({"checkpoint_role": name, "checkpoint": checkpoint.name,
                             "checkpoint_sha256": sha256(checkpoint), "mode": mode,
                             "gallery_scope": "full_swi", "gallery_images": len(labels["swi"]),
                             "gallery_species": len(set(labels["swi"])),
                             "public_id_macro_R1": ev["mean"], "prototype_macro_R1": proto["mean"],
                             "per_species": {"public_id": ev["per_species"],
                                             "prototype": proto["per_species"]}})
        del model, proj, sw, vn, vn26
    out = p["out"] / "scurd_raw_centered_seed_audit.json"
    write_json(out, {"selected_by_meta_val": selected,
                     "recipe_comparison": preflight_report["recipe_comparison"],
                     "rows": rows, "old_ood_statistics_reused": False,
                     "training_performed": False})
    summary = pd.DataFrame([{k: v for k, v in row.items()
                             if k not in {"per_species", "ood_only_kshot_K10"}} for row in rows])
    csv = p["out"] / "scurd_raw_centered_seed_audit.csv"
    summary.to_csv(csv, index=False)
    metrics = ("public_id_macro_R1", "ood_auroc", "ood_fpr95",
               "old_after_plus50_K10", "new50_after_K10", "vn26_swi_pool_all")
    seed_summary = []
    for mode in ("raw", "centered"):
        seed_rows = [row for row in rows if row["checkpoint_role"].startswith("seed") and
                     row["mode"] == mode and row["gallery_scope"] == "id_only"]
        if len(seed_rows) != len(SEEDS):
            raise ValueError(f"Incomplete {mode} seed evaluation")
        for metric in metrics:
            vals = np.asarray([row[metric] for row in seed_rows], dtype=float)
            seed_summary.append({"mode": mode, "metric": metric,
                                 "mean": float(vals.mean()), "sample_sd": float(vals.std(ddof=1)),
                                 "n_seeds": len(vals)})
    seed_csv = p["out"] / "scurd_seed_summary.csv"
    pd.DataFrame(seed_summary).to_csv(seed_csv, index=False)
    gallery_csv = p["out"] / "scurd_gallery_resampling_raw_centered.csv"
    pd.DataFrame(gallery_rows).to_csv(gallery_csv, index=False)
    triage_csv = p["out"] / "scurd_triage_raw_centered.csv"
    pd.DataFrame(triage_rows).to_csv(triage_csv, index=False)
    return [out, csv, seed_csv, gallery_csv, triage_csv, *rq4_files]


def ce_vn26(p):
    from .experiments.rq4_vn26 import run as run_rq4
    variants = (("legacy_logits_954", "", "CE-Full-logits954"),
                ("features_512", "_feature512", "CE-Full-features512"))
    results, outputs = {}, []
    with np.load(p["out"] / "ce_exp4_fresh.npz", allow_pickle=False) as cache:
        for variant, suffix, method in variants:
            swi = {s: (cache[_e4key("swi", "CE_Full", s) + suffix],
                       canonical(cache[_e4key("swi", "CE_Full", s) + "_lbl"])) for s in SCALES}
            vn = {m: (cache[_e4key("vn26", "CE_Full", m) + suffix],
                      canonical(cache[_e4key("vn26", "CE_Full", m) + "_lbl"])) for m in MAGS}
            swi["pool"] = (np.concatenate([swi[s][0] for s in SCALES]),
                           np.concatenate([swi[s][1] for s in SCALES]))
            rq4_dir = p["out"] / f"ce_rq4_{variant}"
            rq4 = run_rq4({method: {"swi_scales": swi, "vn26_mags": vn}}, rq4_dir)
            results[variant] = {
                "cross_domain": {f"{g}/{q}": cell for g, queries in rq4["cross_domain"][method].items()
                                 for q, cell in queries.items()},
                "cross_magnification": {f"{g}/{q}": cell for g, queries in rq4["cross_magnification"][method].items()
                                        for q, cell in queries.items()},
            }
            outputs.append(rq4_dir / "rq4_generalization.json")
    result = {
        "primary_protocol": "legacy_logits_954",
        "representations": results,
        "checkpoint_sha256": sha256(p["ce"]), "training_performed": False,
    }
    out = p["out"] / "ce_vn26_fresh.json"
    write_json(out, result)
    return [out, *outputs]


def paired_baselines(p):
    from scipy.stats import wilcoxon
    from .experiments.registry import _fuse, _norm
    result = json.loads((p["out"] / "scurd_raw_centered_seed_audit.json").read_text())
    with np.load(p["cache"], allow_pickle=False) as cache:
        labels_id = canonical(cache["labels_id_dinov2"])
        labels_swi = canonical(cache["labels_swi_dinov2"])
        features = {
            "DINOv2": (cache["embs_id_dinov2"], cache["embs_swi_dinov2"]),
            "ArcFace-557": (cache["embs_id_arc"], cache["embs_swi_arc"]),
            "CE-Full": (cache["embs_id_ce_full_norm"], cache["embs_swi_ce_full_norm"]),
            "SupCon": (cache["embs_id_Var_CvNxt_SupCon"], cache["embs_swi_Var_CvNxt_SupCon"]),
            "Fusion": (_fuse(cache["embs_id_arc"], cache["embs_id_dinov2"]),
                       _fuse(cache["embs_swi_arc"], cache["embs_swi_dinov2"])),
        }
    rows, tests = [], []
    for scope in ("id_only", "full_swi"):
        gallery_mask = np.isin(labels_swi, np.unique(labels_id)) if scope == "id_only" else np.ones(len(labels_swi), bool)
        selected = next(row for row in result["rows"] if row["checkpoint_role"] == "selected"
                        and row["mode"] == "raw" and row["gallery_scope"] == scope)
        primary = selected["per_species"]["public_id"]
        for name, (query, gallery) in features.items():
            idx, _ = top_matches(_norm(query), _norm(gallery[gallery_mask]), k=1)
            preds = labels_swi[gallery_mask][idx[:, 0]]
            per_species = {str(sp): float(np.mean(preds[labels_id == sp] == sp)) for sp in np.unique(labels_id)}
            rows.append({"scope": scope, "method": name,
                         "macro_R1": float(np.mean(list(per_species.values()))),
                         "per_species": per_species})
            species = sorted(set(primary) & set(per_species))
            a = np.asarray([primary[sp] for sp in species])
            b = np.asarray([per_species[sp] for sp in species])
            diff = a - b
            p_value = 1.0 if np.allclose(diff, 0) else float(wilcoxon(
                a, b, zero_method="wilcox", correction=False, alternative="two-sided", method="approx").pvalue)
            tests.append({"scope": scope, "comparison": f"SC-URD selected raw vs {name}",
                          "mean_difference": float(diff.mean()), "p_unadjusted": p_value,
                          "n_species": len(species)})
        scoped = sorted((item for item in tests if item["scope"] == scope), key=lambda x: x["p_unadjusted"])
        adjusted = 0.0
        for rank, item in enumerate(scoped):
            adjusted = max(adjusted, min(1.0, (len(scoped) - rank) * item["p_unadjusted"]))
            item["p_holm"] = adjusted
    out = p["out"] / "paired_raw_baselines.json"
    write_json(out, {"baseline_rows": rows, "paired_tests": tests,
                     "test": "Wilcoxon signed rank, normal approximation, two-sided, zero_method=wilcox, no continuity correction; Holm within each gallery scope",
                     "checkpoint_selection": "SC-URD raw selected on SWI meta-val only"})
    return [out]


def _run_with_environment(script, overrides):
    previous = {key: os.environ.get(key) for key in overrides}
    try:
        os.environ.update(overrides)
        runpy.run_path(str(script), run_name="__main__")
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def raw_gallery_matrix(p, args, selected):
    if p["cache"].parent != p["root"] or p["exp4"].parent != p["root"]:
        raise ValueError("The validated gallery engine requires cache and exp4 files directly under --root")
    if p["research"] != p["run"] / "deployment" / "research_directions":
        raise ValueError("The gallery engine requires the deployment research directory")
    engine = Path(__file__).parent / "_engines" / "variance_retrieval_evidence_colab.py"
    out = p["out"] / "raw_gallery_engine"
    overrides = {
        "ROOT_PATH": str(p["root"]), "RESULTS_DIR": str(p["run"] / "deployment"),
        "OUT_DIR": str(out), "EMB_CACHE_NAME": p["cache"].name,
        "EXP4_CACHE_NAME": p["exp4"].name, "GALLERY_SCOPE": "id_only",
        "SCURD_MAIN_MODE": "raw", "SCURD_TAU": str(selected["tau"]),
        "SCURD_TOP_M": str(selected["top_m"]),
        "SCURD_MAIN_CKPT_NAME": selected["checkpoint"],
        "SCURD_PROJ_CACHE_NAME": "final_audit_force_projection_missing.npz",
        "RUN_MAP": "0", "RUN_HEADLINE_RECOMPUTE": "0",
        "RUN_GALLERY_RESAMPLING": "1", "RUN_REVIEWER_GAP_FULL_GALLERY": "0",
        "RUN_RQ5_FULL_GALLERY": "0", "RUN_SCURD_SEED_SENSITIVITY": "0",
        "RUN_TRAIN_SCURD_SEEDS": "0", "SCURD_FORCE_RETRAIN_SEEDS": "0",
        "N_GALLERY_REPEATS": str(args.gallery_repeats), "DEVICE": args.device,
    }
    _run_with_environment(engine, overrides)
    matrix = out / "gallery_resampling_variance.csv"
    engine_df = pd.read_csv(matrix)
    expected_methods = {"DINOv2", "ArcFace-557", "CE-Full", "Fusion", "SupCon", "SC-URD"}
    if (set(engine_df["method"]) != expected_methods or len(engine_df) != 66
            or not engine_df.groupby("method").size().eq(11).all()):
        raise ValueError("Raw gallery engine did not produce the full 6-method x 11-setting matrix")
    own = pd.read_csv(p["out"] / "scurd_gallery_resampling_raw_centered.csv")
    own = own[own["mode"].eq("raw")]
    merged = engine_df[engine_df["method"].eq("SC-URD")].merge(
        own, on=["method", "strategy", "K"], suffixes=("_engine", "_independent"), validate="one_to_one")
    if (len(merged) != 11
            or not np.allclose(merged["mean_engine"], merged["mean_independent"], atol=1e-6)
            or not np.allclose(merged["std_engine"], merged["std_independent"], atol=1e-6)):
        raise ValueError("Independent SC-URD raw resampling disagrees with validated engine")
    write_json(out / "independent_comparison.json", {
        "n_settings": len(merged),
        "max_abs_mean_difference": float(np.max(np.abs(merged["mean_engine"] - merged["mean_independent"]))),
        "mode": "raw", "selected_checkpoint": selected["checkpoint"],
        "selected_checkpoint_sha256": sha256(p["research"] / selected["checkpoint"]),
    })
    return [matrix, out / "variance_evidence_manifest.json", out / "independent_comparison.json"]


def run(args):
    if args.batch_size < 1 or args.workers < 0 or args.gallery_repeats < 1 or args.ood_kshot_repeats < 1:
        raise ValueError("Batch size and repetitions must be positive; workers must be nonnegative")
    p = paths(args)
    os.environ["ROOT_PATH"] = str(p["root"])
    os.environ["FULL954_CACHE_NAME"] = p["cache"].name
    os.environ["EXP4_CACHE_NAME"] = p["exp4"].name
    os.environ["RESULTS_DIR"] = str(p["run"] / "deployment")
    os.environ["DEVICE"] = args.device
    report, ckpts, manifest = preflight(p)
    if args.preflight:
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
    out = p["out"]
    out.mkdir(parents=True, exist_ok=True)
    inputs = _input_hashes(p, ckpts, report["checkpoint_meta_caches"])
    input_manifest = out / "inputs.json"
    if input_manifest.exists() and json.loads(input_manifest.read_text()) != inputs:
        raise ValueError("Audit inputs changed; choose a new --out directory")
    write_json(input_manifest, inputs)
    write_json(out / "preflight.json", report)
    if not completed(out, "native_ce"):
        print("[final-audit] Native CE checkpoint/cache alignment", flush=True)
        native_out = out / "native_ce"
        if native_out.exists():
            raise ValueError(f"Unfinished native CE directory: {native_out}; inspect it before rerunning")
        from .experiments.audit_native_ce import main as native_ce_main
        native_ce_main(["--root", str(p["root"]), "--checkpoint", str(p["ce"]),
                        "--id-csv", str(p["id"]), "--cache", str(p["cache"]),
                        "--out", str(native_out),
                        "--device", args.device, "--batch-size", str(args.batch_size),
                        "--workers", str(args.workers)])
        finish(out, "native_ce", [native_out / "native_ce_audit.json", native_out / "native_ce_queries.npz"])
    if not completed(out, "ce_exp4"):
        print("[final-audit] CE-Full VN26 image extraction", flush=True)
        finish(out, "ce_exp4", ce_exp4(p, args, manifest))
    if not completed(out, "ce_vn26"):
        print("[final-audit] CE-Full VN26 scoring", flush=True)
        finish(out, "ce_vn26", ce_vn26(p))
    if not completed(out, "scurd"):
        print("[final-audit] SC-URD raw/centered checkpoints", flush=True)
        finish(out, "scurd", scurd_audit(p, args, report, ckpts))
    if not completed(out, "raw_gallery_matrix"):
        print("[final-audit] Full raw gallery matrix", flush=True)
        finish(out, "raw_gallery_matrix", raw_gallery_matrix(p, args, report["selected"]))
    if not completed(out, "paired_raw"):
        print("[final-audit] Paired raw comparisons", flush=True)
        finish(out, "paired_raw", paired_baselines(p))
    native = json.loads((out / "native_ce" / "native_ce_audit.json").read_text())
    scurd = json.loads((out / "scurd_raw_centered_seed_audit.json").read_text())
    ce = json.loads((out / "ce_vn26_fresh.json").read_text())
    summary = {
        "native_ce_macro": native["mean"],
        "ce_vn26_legacy_logits_swi_pool_all": ce["representations"]["legacy_logits_954"]["cross_domain"]["SWI_pool/VN26_all"]["mean"],
        "ce_vn26_features512_swi_pool_all": ce["representations"]["features_512"]["cross_domain"]["SWI_pool/VN26_all"]["mean"],
        "selected_raw_deployment": next(r for r in scurd["rows"] if r["checkpoint_role"] == "selected"
                                        and r["mode"] == "raw" and r["gallery_scope"] == "id_only"),
        "seed_recipe_metadata_match": report["recipe_comparison"]["metadata_recipe_match"],
        "training_meta_cache_hash_match": (
            len([key for key in inputs if key.startswith("training_meta_")]) == len(ckpts)
            and len({value["sha256"] for key, value in inputs.items()
                     if key.startswith("training_meta_")}) == 1),
        "training_data_identity_verified": False,
        "legacy_exp4_image_paths_verified": False,
        "paired_raw_baselines": str(out / "paired_raw_baselines.json"),
        "raw_gallery_matrix": str(out / "raw_gallery_engine" / "gallery_resampling_variance.csv"),
        "paper_numbers_updated": False,
    }
    write_json(out / "summary.json", summary)
    print(f"Audit finished: {out / 'summary.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/content/drive/MyDrive/NCS"))
    parser.add_argument("--run-root", type=Path, default=Path("/content/drive/MyDrive/NCS/results/paper_reframe_full954_retrained_ce_corrected_public"))
    parser.add_argument("--research-dir", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--exp4", type=Path)
    parser.add_argument("--ce-checkpoint", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gallery-repeats", type=int, default=100)
    parser.add_argument("--ood-kshot-repeats", type=int, default=50)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0 or args.gallery_repeats < 1 or args.ood_kshot_repeats < 1:
        parser.error("batch-size and repeats must be positive; workers must be nonnegative")
    run(args)


if __name__ == "__main__":
    main()
