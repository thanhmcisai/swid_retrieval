"""Validation-locked reference-budget study for large wood galleries.

All model/score selection uses scan-disjoint meta-val queries. The public-ID
pass only evaluates the locked recipe against the existing full-gallery cache.
"""

import json
import os
import re
import zipfile
from hashlib import sha256
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

BUDGETS = (1, 3, 5, 10, 20)
POLICIES = ("scan_hash", "scan_scale")
SCORERS = ("prototype", "scale_top2", "prototype_scale_half")

_SCAN_ID = re.compile(r"_from_(tw[0-9]+)(?:_|$)", re.IGNORECASE)


def _canonical(label):
    return str(label).replace(" ", "_").lower().strip()


def _hash_bytes(value):
    return sha256(value).hexdigest()


def _source_scan_id(path):
    match = _SCAN_ID.search(Path(path).stem)
    if match is None:
        raise ValueError(f"Cannot recover source scan from {path}")
    return match.group(1).lower()


def _image_scale(path):
    parent = Path(path).parent.name.lower()
    match = re.fullmatch(r"scale_(\d+)", parent)
    if match is None:
        match = re.match(r"patch_(\d+)_", Path(path).name.lower())
    return int(match.group(1)) if match else None


def select_indices(items, budget, policy):
    """Return nested, deterministic per-class reference prefixes."""
    if budget < 1 or policy not in POLICIES:
        raise ValueError("Invalid reference budget/policy")
    by_class = defaultdict(list)
    for index, (path, label) in enumerate(items):
        by_class[_canonical(label)].append(index)
    chosen = []
    for label in sorted(by_class):
        indices = by_class[label]
        if policy == "scan_hash":
            by_scan = defaultdict(list)
            for index in indices:
                by_scan[_source_scan_id(items[index][0])].append(index)
            prefix = [min(by_scan[scan], key=lambda i: _hash_bytes(
                str(items[i][0]).encode())) for scan in sorted(
                    by_scan, key=lambda x: _hash_bytes(x.encode()))]
            remaining = sorted(set(indices) - set(prefix), key=lambda i:
                               _hash_bytes(str(items[i][0]).encode()))
            ordered = prefix + remaining
        else:
            scans = {i: _source_scan_id(items[i][0]) for i in indices}
            scales = {i: _image_scale(items[i][0]) for i in indices}
            hashed = {i: _hash_bytes(str(items[i][0]).encode()) for i in indices}
            scan_count, scale_count, pair_count = Counter(), Counter(), Counter()
            remaining = set(indices)
            ordered = []
            for _ in range(min(budget, len(indices))):
                index = min(remaining, key=lambda i: (
                    scale_count[scales[i]], scan_count[scans[i]],
                    pair_count[(scans[i], scales[i])], hashed[i]))
                ordered.append(index)
                remaining.remove(index)
                scan_count[scans[index]] += 1
                scale_count[scales[index]] += 1
                pair_count[(scans[index], scales[index])] += 1
        chosen.extend(ordered[:budget])
    if len(chosen) < len(by_class):
        raise ValueError("Reference selection dropped a species")
    return chosen


def score_references(queries, references, labels, scales, n_classes):
    """Compare one global and two scale-aware all-class scorers."""
    if (len(references) != len(labels) or len(labels) != len(scales) or
            len(references) == 0 or queries.shape[1] != references.shape[1]):
        raise ValueError("Reference score inputs are inconsistent")
    if (labels.min() < 0 or labels.max() >= n_classes or
            (torch.bincount(labels, minlength=n_classes) == 0).any()):
        raise ValueError("At least one class has no reference")
    q = F.normalize(queries.float(), dim=-1)
    r = F.normalize(references.float(), dim=-1)
    groups = sorted(set(zip(labels.tolist(), scales)))
    group_id = {pair: i for i, pair in enumerate(groups)}
    inverse = torch.tensor([group_id[pair] for pair in zip(labels.tolist(), scales)],
                           device=r.device)
    group_labels = torch.tensor([pair[0] for pair in groups], device=r.device)
    group_sums = r.new_zeros((len(groups), r.shape[1]))
    group_sums.index_add_(0, inverse, r)
    group_counts = torch.bincount(inverse, minlength=len(groups))
    group_prototypes = F.normalize(group_sums / group_counts[:, None], dim=-1)
    class_sums = r.new_zeros((n_classes, r.shape[1]))
    class_sums.index_add_(0, labels, r)
    class_counts = torch.bincount(labels, minlength=n_classes)
    prototype = q @ F.normalize(class_sums / class_counts[:, None], dim=-1).T
    group_scores = q @ group_prototypes.T
    scale_top2 = torch.empty_like(prototype)
    for label in range(n_classes):
        values = group_scores[:, group_labels == label]
        scale_top2[:, label] = values.topk(min(2, values.shape[1]), dim=1).values.mean(1)
    return {"prototype": prototype, "scale_top2": scale_top2,
            "prototype_scale_half": (prototype + scale_top2) / 2}


def _metrics(scores, truth):
    prediction = scores.argmax(1)
    labels = torch.unique(truth)
    result = {"macro_r1": float(torch.stack([
        (prediction[truth == label] == label).float().mean()
        for label in labels]).mean()),
        "macro_r5": float(torch.stack([
            (scores[truth == label].topk(5, dim=1).indices == label).any(1)
            .float().mean() for label in labels]).mean())}
    for k in (32, 128):
        result[f"macro_recall{k}"] = float(torch.stack([
            (scores[truth == label].topk(min(k, scores.shape[1]), dim=1).indices
             == label).any(1).float().mean() for label in labels]).mean())
    return result, prediction


def _cohorts(manifest):
    from . import gallery_experiment as study

    folds = {}
    for fold in (0, 1):
        _, queries = study.validation_items(
            manifest, group_mode="scan_disjoint", fold=fold)
        refs = _reference_pool(manifest, queries)
        target_scans = {label for _, label in queries}
        if (len(queries) != 285 or len(target_scans) != 57 or
                len({label for _, label in refs}) != 637):
            raise ValueError("Meta-val reference/query cohort changed")
        folds[fold] = (refs, queries)
    return folds


def _reference_pool(manifest, queries):
    target_scans = defaultdict(set)
    for path, label in queries:
        target_scans[_canonical(label)].add(_source_scan_id(path))
    return [(path, _canonical(label)) for split in ("meta-train", "meta-val")
            for path, label in manifest[split]
            if _source_scan_id(path) not in target_scans[_canonical(label)]]


def _winning_recipe(frame):
    if (len(frame) != 2 * len(POLICIES) * len(BUDGETS) * len(SCORERS) or
            frame.groupby(["policy", "budget", "scorer"]).fold.nunique().min() != 2):
        raise ValueError("Validation matrix is incomplete")
    grouped = frame.groupby(["policy", "budget", "scorer"], as_index=False)[
        ["macro_r1", "macro_r5", "macro_recall128"]].mean()
    grouped = grouped.sort_values(
        ["macro_r1", "macro_r5", "budget", "policy", "scorer"],
        ascending=[False, False, True, True, True])
    return grouped


def _paths(root):
    large_out = Path(os.environ.get(
        "WOOD_LARGE_OUT", root / "results" / "wood_large_gallery_final_v1")).resolve()
    study_out = Path(os.environ.get(
        "WOOD_LARGE_BASE_OUT", root / "results" /
        "dinov2s_retrieval_matched_v1")).resolve()
    image_out = Path(os.environ.get(
        "WOOD_LARGE_QKV_OUT", root / "results" /
        "wood_correspondence_image_full_v1")).resolve()
    out = Path(os.environ.get(
        "WOOD_REF_OUT", root / "results" / "wood_reference_budget_v1")).resolve()
    if out == large_out or large_out in out.parents:
        raise ValueError("Reference study output must be separate from locked test run")
    return large_out, study_out, image_out, out


def _source(root, large_out, study_out, image_out):
    from . import (gallery_experiment as study, wood_gallery_audit as audit,
                   wood_large_gallery as large)

    lock = json.loads((large_out / "selection_lock.json").read_text())
    plan = json.loads((large_out / "study_plan.json").read_text())
    if lock["public_id_test_evaluated"] or lock["meta_test_evaluated"]:
        raise ValueError("The source checkpoint was selected after viewing test data")
    arm, seed = lock["selected_arm"], int(lock["report_seed"])
    if arm not in {"random_bank", "hard_bank", "qkv_base"}:
        raise ValueError("Selected checkpoint arm is unsupported")
    source_seed = plan["settings"]["source_seed"]
    configured_seed = os.environ.get("WOOD_LARGE_SOURCE_SEED")
    if configured_seed is not None and configured_seed != str(source_seed):
        raise ValueError("WOOD_LARGE_SOURCE_SEED differs from locked study plan")
    os.environ["WOOD_LARGE_SOURCE_SEED"] = str(source_seed)
    upstream = seed if source_seed == "match" else int(source_seed)
    checkpoint = audit._checkpoint_path(root, large_out, image_out, arm, seed,
                                        upstream)
    if study._hash_file(checkpoint) != lock["selected_checkpoint_sha256"]:
        raise ValueError("Selected source checkpoint changed")
    return large, lock, arm, seed, checkpoint


def _cohort_signature(folds, checkpoint_hash):
    from . import gallery_experiment as study

    return study._hash_bytes(json.dumps({
        "checkpoint": checkpoint_hash,
        "folds": {str(k): {"refs": refs, "queries": queries}
                  for k, (refs, queries) in folds.items()},
        "budgets": BUDGETS, "policies": POLICIES, "scorers": SCORERS,
    }, sort_keys=True).encode())


def _validation(root, large_out, study_out, image_out, out):
    from . import data, gallery_experiment as study

    large, source_lock, arm, seed, checkpoint = _source(
        root, large_out, study_out, image_out)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Validation feature extraction requires CUDA")
    manifest = data.load_swi_manifest(root / "swi_manifest.json")
    folds = _cohorts(manifest)
    checkpoint_hash = study._hash_file(checkpoint)
    signature = _cohort_signature(folds, checkpoint_hash)
    protocol_path = out / "validation_protocol.json"
    if protocol_path.is_file():
        previous = json.loads(protocol_path.read_text())
        if (previous["checkpoint_sha256"] != checkpoint_hash or
                previous["cohort_signature"] != signature or
                previous["manifest_sha256"] != study._hash_file(
                    root / "swi_manifest.json") or
                previous["source_selection_sha256"] != study._hash_file(
                    large_out / "selection_lock.json")):
            raise ValueError("Validation protocol changed; use a new output folder")
    _, encoder, _, base_cfg, _ = large._load_arm(
        root, study_out, image_out, large_out, arm, seed, device)
    batch = int(os.environ.get("WOOD_REF_IMAGE_BATCH", "64"))
    workers = int(os.environ.get("WOOD_REF_WORKERS", "2"))
    if batch < 1 or workers < 0:
        raise ValueError("Invalid image batch/workers")
    cfg = dict(base_cfg, image_batch=batch, workers=min(
        workers, len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity")
        else os.cpu_count() or 1))
    rows, outcomes = [], []
    for fold, (refs, queries) in folds.items():
        labels = sorted({label for _, label in refs})
        label_to_id = {label: i for i, label in enumerate(labels)}
        selections = {(policy, budget): select_indices(refs, budget, policy)
                      for policy in POLICIES for budget in BUDGETS}
        pool_indices = sorted(set().union(*(set(x) for x in selections.values())))
        pool_items = [refs[i] for i in pool_indices]
        position = {original: i for i, original in enumerate(pool_indices)}
        cache = out / "features" / f"fold_{fold}"
        print(f"[wood-ref] fold={fold}: {len(pool_items)} reference images, "
              f"{len(queries)} queries, {len(labels)} species", flush=True)
        ref_g, _ = large._cache_features(
            encoder, pool_items, cfg, device, cache / "refs.npy",
            checkpoint_hash, False)
        query_g, _ = large._cache_features(
            encoder, queries, cfg, device, cache / "queries.npy",
            checkpoint_hash, False)
        reference = torch.from_numpy(np.asarray(ref_g).copy()).to(device)
        query = torch.from_numpy(np.asarray(query_g).copy()).to(device)
        truth = torch.tensor([label_to_id[label] for _, label in queries], device=device)
        truth_ids = truth.cpu().tolist()
        for policy in POLICIES:
            for budget in BUDGETS:
                selected = selections[(policy, budget)]
                local = [position[i] for i in selected]
                chosen = [refs[i] for i in selected]
                ref_labels = torch.tensor([label_to_id[label] for _, label in chosen],
                                          device=device)
                scales = [_image_scale(path) for path, _ in chosen]
                with torch.inference_mode():
                    scores = score_references(query, reference[local], ref_labels,
                                              scales, len(labels))
                for scorer, matrix in scores.items():
                    metrics, predictions = _metrics(matrix, truth)
                    rows.append({"fold": fold, "policy": policy,
                                 "budget": budget, "scorer": scorer,
                                 "n_references": len(selected), **metrics})
                    pred = predictions.cpu().tolist()
                    outcomes.extend({"fold": fold, "policy": policy,
                                     "budget": budget, "scorer": scorer,
                                     "query_path": str(path), "true_label": label,
                                     "predicted_label": labels[pred[i]],
                                     "correct": int(pred[i] == truth_ids[i])}
                                    for i, (path, label) in enumerate(queries))
                    print(f"[wood-ref] fold={fold} {policy} K={budget} {scorer} "
                          f"R@1={metrics['macro_r1']:.4f} "
                          f"R@128={metrics['macro_recall128']:.4f}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "validation_summary.csv", index=False)
    pd.DataFrame(outcomes).to_csv(out / "validation_queries.csv", index=False)
    record = {"checkpoint_sha256": checkpoint_hash,
              "manifest_sha256": study._hash_file(root / "swi_manifest.json"),
              "source_selection_sha256": study._hash_file(
                  large_out / "selection_lock.json"),
              "cohort_signature": signature, "folds": [0, 1],
              "training_performed": False, "test_evaluated": False,
              "protocol": "scan-disjoint meta-val; 57 target + 580 distractor species"}
    path = protocol_path
    if path.is_file() and json.loads(path.read_text()) != record:
        raise ValueError("Validation protocol changed; use a new output folder")
    study._json(path, record)
    return rows


def _select(root, large_out, study_out, image_out, out):
    from . import gallery_experiment as study

    _, _, _, _, checkpoint = _source(root, large_out, study_out, image_out)
    protocol = json.loads((out / "validation_protocol.json").read_text())
    if (protocol["checkpoint_sha256"] != study._hash_file(checkpoint) or
            protocol["manifest_sha256"] != study._hash_file(root / "swi_manifest.json") or
            protocol["source_selection_sha256"] != study._hash_file(
                large_out / "selection_lock.json")):
        raise ValueError("Validation provenance changed")
    frame = pd.read_csv(out / "validation_summary.csv")
    grouped = _winning_recipe(frame)
    grouped.to_csv(out / "selection_candidates.csv", index=False)
    chosen = grouped.iloc[0]
    lock = {"policy": str(chosen.policy), "budget": int(chosen.budget),
            "scorer": str(chosen.scorer), "validation_macro_r1": float(chosen.macro_r1),
            "validation_macro_recall128": float(chosen.macro_recall128),
            "selection_metric": "mean macro R@1 over scan-disjoint meta-val folds 0 and 1",
            "validation_protocol_sha256": study._hash_file(
                out / "validation_protocol.json"),
            "selection_candidates_sha256": study._hash_file(
                out / "selection_candidates.csv"),
            "checkpoint_sha256": study._hash_file(checkpoint),
            "public_id_evaluated_at_lock": False}
    path = out / "selection_lock.json"
    if path.is_file() and json.loads(path.read_text()) != lock:
        raise ValueError("Existing selection differs; use a new output folder")
    study._json(path, lock)
    print(f"[wood-ref] LOCKED: {lock['policy']} K={lock['budget']} "
          f"{lock['scorer']} val R@1={lock['validation_macro_r1']:.4f}", flush=True)
    return lock


def _public_id(root, large_out, study_out, image_out, out):
    from . import (data, gallery_experiment as study,
                   wood_gallery_audit as audit)

    _, source_lock, arm, seed, checkpoint = _source(
        root, large_out, study_out, image_out)
    selected = json.loads((out / "selection_lock.json").read_text())
    protocol = json.loads((out / "validation_protocol.json").read_text())
    if (selected["checkpoint_sha256"] != study._hash_file(checkpoint) or
            selected["validation_protocol_sha256"] != study._hash_file(
                out / "validation_protocol.json") or
            selected["selection_candidates_sha256"] != study._hash_file(
                out / "selection_candidates.csv") or
            protocol["manifest_sha256"] != study._hash_file(root / "swi_manifest.json") or
            protocol["source_selection_sha256"] != study._hash_file(
                large_out / "selection_lock.json")):
        raise ValueError("Selection lock or source provenance changed")
    manifest = data.load_swi_manifest(root / "swi_manifest.json")
    full_items = [(path, study.canonical(label))
                  for path, label in data.full_swi_items(manifest)]
    frame = study._public_frames(root, manifest)["id"]
    queries = [(path, study.canonical(label))
               for path, label in zip(frame.file_path, frame.label)]
    if len(full_items) != 176123 or len(queries) != 6189:
        raise ValueError("Locked full-gallery/public-ID cohort changed")
    folder = large_out / "full954" / arm / f"seed_{seed}"
    old_protocol = json.loads((folder / "protocol.json").read_text())
    if (old_protocol["checkpoint_sha256"] != selected["checkpoint_sha256"] or
            old_protocol["id_csv_sha256"] != study._hash_file(
                root / "ID_images_expanded.csv") or
            old_protocol["query_limit_per_species"] != 0):
        raise ValueError("Existing public-ID feature cache has wrong provenance")
    cache = folder / "features"
    full_g, _ = audit._verified_cache(
        cache / "swi_full.npy", selected["checkpoint_sha256"], full_items,
        False, study._hash_bytes)
    query_g, _ = audit._verified_cache(
        cache / "public_id.npy", selected["checkpoint_sha256"], queries,
        True, study._hash_bytes)
    labels = sorted({label for _, label in full_items})
    label_id = {label: i for i, label in enumerate(labels)}
    picked = sorted(select_indices(full_items, selected["budget"],
                                   selected["policy"]))
    chosen = [full_items[i] for i in picked]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reference = torch.from_numpy(np.asarray(full_g[picked]).copy()).to(device)
    query = torch.from_numpy(np.asarray(query_g).copy()).to(device)
    ref_labels = torch.tensor([label_id[label] for _, label in chosen], device=device)
    truth = torch.tensor([label_id[label] for _, label in queries], device=device)
    scales = [_image_scale(path) for path, _ in chosen]
    with torch.inference_mode():
        scores = score_references(query, reference, ref_labels, scales,
                                  len(labels))[selected["scorer"]]
    metrics, predictions = _metrics(scores, truth)
    predictions = predictions.cpu().tolist()
    truth_ids = truth.cpu().tolist()
    out.mkdir(parents=True, exist_ok=True)
    rows = [{"query_path": str(path), "true_label": label,
             "source_dataset": str(frame.source_dataset.iloc[i]),
             "predicted_label": labels[predictions[i]],
             "correct": int(predictions[i] == truth_ids[i])}
            for i, (path, label) in enumerate(queries)]
    pd.DataFrame(rows).to_csv(out / "public_id_queries.csv", index=False)
    reference_groups = defaultdict(lambda: {"scans": set(), "scales": set(),
                                             "images": 0})
    for path, label in chosen:
        group = reference_groups[label]
        group["images"] += 1
        group["scans"].add(_source_scan_id(path))
        group["scales"].add(_image_scale(path))
    pd.DataFrame([{"species": label, "images": value["images"],
                   "scans": len(value["scans"]),
                   "scales": len(value["scales"])}
                  for label, value in sorted(reference_groups.items())]).to_csv(
                      out / "public_id_reference_counts.csv", index=False)
    grouped = pd.DataFrame(rows).groupby(
        ["source_dataset", "true_label"], as_index=False).agg(
            n_queries=("correct", "size"), r1=("correct", "mean"))
    grouped.to_csv(out / "public_id_per_species.csv", index=False)
    grouped.groupby("source_dataset", as_index=False).agg(
        n_species=("true_label", "size"), macro_r1=("r1", "mean")).to_csv(
            out / "public_id_per_source.csv", index=False)
    old = pd.read_csv(folder / "full954_summary.csv")
    baseline = old.loc[(old.method == "raw_prototype") & (old.k == 1)]
    if len(baseline) != 1:
        raise ValueError("Missing original full-prototype baseline")
    old_macro = float(baseline.macro_r1.iloc[0])
    old_queries = pd.read_csv(folder / "full954_queries.csv")
    old_queries = old_queries.loc[
        (old_queries.method == "raw_prototype") & (old_queries.width == 0),
        ["query_path", "true_label", "correct"]]
    paired = pd.DataFrame(rows).merge(
        old_queries, on=["query_path", "true_label"], how="outer",
        validate="one_to_one", suffixes=("_selected", "_full"), indicator=True)
    if len(paired) != len(queries) or not paired._merge.eq("both").all():
        raise ValueError("Original and selected public-ID query sets differ")
    rescued = int(((paired.correct_selected == 1) &
                   (paired.correct_full == 0)).sum())
    harmed = int(((paired.correct_selected == 0) &
                  (paired.correct_full == 1)).sum())
    report = {"selection_lock_sha256": study._hash_file(out / "selection_lock.json"),
              "checkpoint_sha256": selected["checkpoint_sha256"],
              "id_csv_sha256": old_protocol["id_csv_sha256"],
              "n_queries": len(queries), "n_gallery_species": len(labels),
              "n_references": len(chosen), "policy": selected["policy"],
              "budget": selected["budget"], "scorer": selected["scorer"],
              **metrics, "full_prototype_macro_r1": old_macro,
              "rescued_vs_full_prototype": rescued,
              "harmed_vs_full_prototype": harmed,
              "training_performed": False,
              "public_id_prior_exploration_disclosed": True}
    study._json(out / "public_id_report.json", report)
    print(f"[wood-ref] locked public-ID macro R@1={metrics['macro_r1']:.4f}; "
          f"R@128={metrics['macro_recall128']:.4f}", flush=True)
    return report


def _export(out):
    names = ("validation_summary.csv", "validation_queries.csv",
             "validation_protocol.json", "selection_candidates.csv",
             "selection_lock.json", "public_id_queries.csv", "public_id_report.json",
             "public_id_reference_counts.csv", "public_id_per_species.csv",
             "public_id_per_source.csv")
    missing = [name for name in names if not (out / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete reference-budget study: {missing}")
    target = out / "wood_reference_budget_review.zip"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            archive.write(out / name, name)
    print(f"[wood-ref] review ZIP: {target} ({target.stat().st_size/2**20:.2f} MiB)",
          flush=True)
    return target


def run():
    from . import data

    root = Path(os.environ.get("ROOT_PATH", "/content/drive/MyDrive/NCS")).resolve()
    large_out, study_out, image_out, out = _paths(root)
    mode = os.environ.get("WOOD_REF_MODE", "preflight").lower()
    if mode == "export":
        return _export(out)
    if mode not in {"preflight", "validate", "select", "public_id"}:
        raise ValueError("WOOD_REF_MODE must be preflight, validate, select, public_id, or export")
    _, _, arm, seed, checkpoint = _source(
        root, large_out, study_out, image_out)
    print(f"[wood-ref] {mode}: arm={arm} seed={seed} "
          f"checkpoint={checkpoint.name} out={out}", flush=True)
    if mode == "preflight":
        folds = _cohorts(data.load_swi_manifest(root / "swi_manifest.json"))
        print(f"[wood-ref] {len(folds)} scan-disjoint meta-val folds; "
              "637 gallery species per fold; no test read", flush=True)
        return None
    if mode == "validate":
        if (out / "selection_lock.json").is_file():
            raise ValueError("Selection already locked; use a new output folder")
        return _validation(root, large_out, study_out, image_out, out)
    if mode == "select":
        return _select(root, large_out, study_out, image_out, out)
    return _public_id(root, large_out, study_out, image_out, out)
