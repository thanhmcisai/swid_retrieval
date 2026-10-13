"""Locked meta-validation readout for full-image wood retrieval ablations."""

import csv
import os
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from . import gallery_experiment as study
from . import wood_evidence_experiment as evidence
from . import wood_reference_budget as budget


BUDGETS = (1, 3, 5, 10)
POLICIES = ("scan_hash", "scan_scale")
QKV_ARMS = {"arcface_qkv", "arcface_multipos_qkv"}


def _reference_pool(manifest, queries):
    query_scans = defaultdict(set)
    labels = set()
    for path, label in queries:
        name = study.canonical(label)
        labels.add(name)
        query_scans[name].add(study.source_scan_id(path))
    pool = [(path, study.canonical(label)) for path, label in manifest["meta-val"]
            if study.canonical(label) in labels and
            study.source_scan_id(path) not in query_scans[study.canonical(label)]]
    counts = defaultdict(int)
    for _, label in pool:
        counts[label] += 1
    missing = sorted(label for label in labels if counts[label] < max(BUDGETS))
    if missing:
        raise ValueError(f"Fewer than {max(BUDGETS)} scan-disjoint references: {missing[:5]}")
    return pool


def _unique_items(*groups):
    items, lookup = [], {}
    for group in groups:
        for path, label in group:
            path = str(path)
            name = study.canonical(label)
            if path in lookup:
                if items[lookup[path]][1] != name:
                    raise ValueError(f"Conflicting labels for {path}")
                continue
            lookup[path] = len(items)
            items.append((path, name))
    return items, lookup


def _score_gallery(model, arm, fold, gallery, policy, references, queries,
                   lookup, embeddings, tokens, device, evaluate_qkv=True):
    names = sorted({study.canonical(label) for _, label in references})
    class_lookup = {name: i for i, name in enumerate(names)}
    ref_idx = [lookup[str(path)] for path, _ in references]
    query_idx = [lookup[str(path)] for path, _ in queries]
    ref_labels = torch.tensor([class_lookup[study.canonical(label)]
                               for _, label in references], device=device)
    truth = torch.tensor([class_lookup[study.canonical(label)]
                          for _, label in queries], device=device)
    ref_scans = [study.source_scan_id(path) for path, _ in references]
    r = torch.from_numpy(embeddings[ref_idx]).to(device)
    rt = torch.from_numpy(tokens[ref_idx]).to(device)
    q = torch.from_numpy(embeddings[query_idx]).to(device)
    qt = torch.from_numpy(tokens[query_idx]).to(device)
    modes = ("global", "qkv") if arm in QKV_ARMS and evaluate_qkv else ("global",)
    rows = []
    with torch.inference_mode():
        for start in range(0, len(q), 8):
            end = min(start + 8, len(q))
            base, classes = model(q[start:end], qt[start:end], r, rt,
                                  ref_labels, ref_scans, mode="global")
            candidate32 = (classes[base.topk(min(32, len(classes)), dim=1).indices] ==
                           truth[start:end, None]).any(1)
            candidate128 = (classes[base.topk(min(128, len(classes)), dim=1).indices] ==
                            truth[start:end, None]).any(1)
            for mode in modes:
                if mode == "global":
                    scores = base
                else:
                    scores, scored_classes = model(
                        q[start:end], qt[start:end], r, rt, ref_labels, ref_scans,
                        mode="qkv", top_classes=128 if len(classes) > 128 else None,
                        query_chunk=8, reference_chunk=64)
                    if not torch.equal(scored_classes, classes):
                        raise RuntimeError("QKV changed the gallery class order")
                predicted = classes[scores.argmax(1)]
                top5 = (classes[scores.topk(min(5, len(classes)), dim=1).indices] ==
                        truth[start:end, None]).any(1)
                for offset in range(end - start):
                    path, label = queries[start + offset]
                    rows.append({
                        "fold": fold, "gallery": gallery, "policy": policy,
                        "mode": mode, "query_path": str(path),
                        "query_label": study.canonical(label),
                        "query_scan": study.source_scan_id(path),
                        "query_scale": study.image_scale(path),
                        "predicted_label": names[int(predicted[offset])],
                        "correct": int(predicted[offset] == truth[start + offset]),
                        "r5": int(top5[offset]),
                        "candidate_r32": int(candidate32[offset]),
                        "candidate_r128": int(candidate128[offset]),
                        "n_classes": len(classes), "n_references": len(references),
                    })
    return rows


def _summaries(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["gallery"], row["policy"], row["mode"])].append(row)
    summaries = []
    for (gallery, policy, mode), cohort in sorted(groups.items()):
        by_species = defaultdict(list)
        for row in cohort:
            by_species[row["query_label"]].append(row)
        counts = {len(part) for part in by_species.values()}
        if len(counts) != 1:
            raise ValueError(f"Unbalanced query counts for {gallery}/{policy}/{mode}")
        correct = sum(row["correct"] for row in cohort)
        candidate = sum(row["candidate_r128"] for row in cohort)
        correct_and_candidate = sum(row["correct"] and row["candidate_r128"]
                                    for row in cohort)
        summaries.append({
            "gallery": gallery, "policy": policy, "mode": mode,
            "n_queries": len(cohort), "n_species": len(by_species),
            "n_references": cohort[0]["n_references"],
            "r1": correct / len(cohort),
            "r5": sum(row["r5"] for row in cohort) / len(cohort),
            "candidate_r32": sum(row["candidate_r32"] for row in cohort) / len(cohort),
            "candidate_r128": candidate / len(cohort),
            "r1_given_candidate128": (correct_and_candidate / candidate
                                        if candidate else None),
            "correct_outside_candidate128": correct - correct_and_candidate,
        })
    return summaries


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(manifest, encoder, model, base_cfg, arm, device, output,
             checkpoint_hash, manifest_hash):
    output = Path(output)
    all_rows = []
    for fold in (0, 1):
        refs, queries, extras = evidence._validation_sets(manifest, "meta-val", fold)
        standard = evidence._gallery_plan(refs, queries, extras, "meta-val")
        pool = _reference_pool(manifest, queries)
        selected = {(policy, k): [pool[i] for i in budget.select_indices(pool, k, policy)]
                    for policy in POLICIES for k in BUDGETS}
        top_refs = [item for policy in POLICIES for item in selected[(policy, 10)]]
        items, lookup = _unique_items(refs, extras, top_refs, queries)
        cfg = dict(base_cfg, image_batch=int(os.environ.get("WOOD_FULL_EVAL_BATCH", "64")),
                   workers=int(os.environ.get("WOOD_FULL_EVAL_WORKERS", "2")))
        embeddings, tokens = evidence._extract(encoder, items, cfg, device)
        for gallery, (ri, qi) in standard.items():
            reference = [(refs + extras)[int(i)] for i in ri]
            query = [queries[int(i)] for i in qi]
            all_rows.extend(_score_gallery(model, arm, fold, gallery, "standard",
                                           reference, query, lookup, embeddings,
                                           tokens, device))
        for policy in POLICIES:
            for k in BUDGETS:
                all_rows.extend(_score_gallery(
                    model, arm, fold, f"{len(set(label for _, label in queries))}x{k}",
                    policy, selected[(policy, k)], queries, lookup, embeddings,
                    tokens, device))
    summary = _summaries(all_rows)
    _write_csv(output / "per_query.csv", all_rows)
    _write_csv(output / "summary.csv", summary)
    study._json(output / "protocol.json", {
        "split": "meta-val", "folds": [0, 1],
        "query_grouping": "scan_disjoint", "query_reuse_across_budgets": True,
        "budget_policies": list(POLICIES), "budgets": list(BUDGETS),
        "checkpoint_sha256": checkpoint_hash, "manifest_sha256": manifest_hash,
        "evaluator_sha256": study._hash_file(__file__),
        "qkv_top_classes": 128, "source_scan_is_not_independent_patch": True,
    })
    print(f"[wood-full] validation: {output / 'summary.csv'}", flush=True)


def export_review(out):
    out = Path(out)
    files = sorted(path for path in out.rglob("*") if path.is_file() and
                   (path.name.startswith("epoch_") and path.suffix == ".json" or
                    path.name in {"provenance.json", "summary.csv", "per_query.csv",
                                  "protocol.json"}))
    if not any(path.name == "summary.csv" for path in files):
        raise ValueError("No validation summaries exist; run validate before export")
    archive = out / "wood_full_image_ablation_review.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as stream:
        for path in files:
            stream.write(path, path.relative_to(out))
    if archive.stat().st_size < 1024:
        raise RuntimeError("Review archive is unexpectedly small")
    print(f"[wood-full] review archive: {archive} ({archive.stat().st_size:,} bytes)",
          flush=True)
    return archive
