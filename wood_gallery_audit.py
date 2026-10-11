"""Inference-only diagnosis of the locked full-gallery retrieval result.

This reads completed feature caches; it never extracts images, trains weights,
or selects a model using public-ID outcomes. The score blends are exploratory.
"""

import json
import os
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .wood_correspondence_method import WoodCorrespondence
from .wood_large_gallery_methods import (candidate_union, class_scores,
                                         rerank_candidates)


def _macro(values, truth):
    return float(torch.stack([values[truth == label].float().mean()
                              for label in torch.unique(truth)]).mean())


def _metrics(scores, truth, candidates=None):
    prediction = scores.argmax(1)
    correct = prediction == truth
    top5 = (scores.topk(5, dim=1).indices == truth[:, None]).any(1)
    result = {"macro_r1": _macro(correct, truth),
              "macro_r5": _macro(top5, truth),
              "micro_r1": float(correct.float().mean())}
    if candidates is not None:
        hit = (candidates == truth[:, None]).any(1)
        result.update(candidate_macro_recall=_macro(hit, truth),
                      candidate_micro_recall=float(hit.float().mean()),
                      correct_given_candidate=float(correct[hit].float().mean())
                      if bool(hit.any()) else 0.0)
    return result, correct


def _restricted(scores, candidates):
    result = torch.full_like(scores, -torch.inf)
    result.scatter_(1, candidates, scores.gather(1, candidates))
    return result


def _blend_local(full_scores, capped_global, local_scores, candidates, weight):
    if not 0 <= weight <= 1:
        raise ValueError("Local residual weight must be in [0, 1]")
    result = full_scores.clone()
    residual = local_scores.gather(1, candidates) - capped_global.gather(1, candidates)
    result.scatter_add_(1, candidates, weight * residual)
    return result


def _first_distinct_scans(items, class_ids, source_scan_id, max_scans=3):
    kept, seen = [], defaultdict(set)
    for index, ((path, _), class_id) in enumerate(zip(items, class_ids)):
        scan = source_scan_id(path)
        if scan not in seen[class_id] and len(seen[class_id]) < max_scans:
            kept.append(index)
            seen[class_id].add(scan)
    if len(seen) != len(set(class_ids)):
        raise ValueError("Capped reference gallery lost a species")
    return kept


def _reference_rows(full_items, balanced, three, image_scale, source_scan_id,
                    query_species):
    def summarize(items):
        result = defaultdict(lambda: {"scans": set(), "scales": Counter(), "images": 0})
        for path, label in items:
            result[label]["images"] += 1
            result[label]["scans"].add(source_scan_id(path))
            result[label]["scales"][str(image_scale(path))] += 1
        return result

    full = summarize(full_items)
    five = summarize(balanced)
    three_summary = summarize([balanced[i] for i in three])
    rows = []
    for label in sorted(full):
        row = {"species": label, "is_public_id_species": int(label in query_species)}
        for name, values in (("full", full[label]), ("capped5", five[label]),
                             ("capped3", three_summary[label])):
            row[f"{name}_images"] = values["images"]
            row[f"{name}_scans"] = len(values["scans"])
            row[f"{name}_scales"] = json.dumps(dict(sorted(values["scales"].items())))
        rows.append(row)
    return rows


def _verified_cache(path, checkpoint_hash, items, tokens, hash_bytes):
    cursor = path.with_suffix(".progress.json")
    token_path = path.with_name(path.stem + "_tokens.npy")
    if not path.is_file() or not cursor.is_file() or (tokens and not token_path.is_file()):
        raise FileNotFoundError(f"Completed feature cache missing: {path}")
    signature = hash_bytes(json.dumps({
        "checkpoint": checkpoint_hash, "items": items, "tokens": tokens,
        "image_size": 224}, separators=(",", ":")).encode())
    status = json.loads(cursor.read_text())
    if status != {"signature": signature, "offset": len(items)}:
        raise ValueError(f"Unverified or incomplete feature cache: {path}")
    globals_ = np.load(path, mmap_mode="r", allow_pickle=False)
    locals_ = np.load(token_path, mmap_mode="r", allow_pickle=False) if tokens else None
    if globals_.shape != (len(items), 512) or (tokens and
            locals_.shape != (len(items), 16, 512)):
        raise ValueError(f"Feature cache shape mismatch: {path}")
    return globals_, locals_


def _checkpoint_path(root, large_out, image_out, arm, seed, upstream_seed):
    if arm in {"hard_bank", "random_bank"}:
        return large_out / arm / f"seed_{seed}" / "best.pt"
    if arm == "qkv_base":
        return image_out / "qkv" / f"seed_{upstream_seed}" / "best.pt"
    if arm == "global_only_base":
        global_out = Path(os.environ.get(
            "WOOD_LARGE_GLOBAL_OUT", root / "results" /
            "wood_correspondence_global_only_v1"))
        return global_out / "global_only" / f"seed_{upstream_seed}" / "best.pt"
    raise ValueError(f"Unsupported audit arm: {arm}")


def _comparison_rows(truth, baseline_correct, results):
    rows = []
    for name, correct in results.items():
        rescued = int((correct & ~baseline_correct).sum())
        harmed = int((~correct & baseline_correct).sum())
        rows.append({"method": name, "rescued_vs_full_prototype": rescued,
                     "harmed_vs_full_prototype": harmed,
                     "net_correct_queries": rescued - harmed,
                     "n_queries": len(truth)})
    return rows


def _write_review(out):
    target = out / "wood_gallery_audit_review.zip"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ("summary.csv", "paired_changes.csv", "per_source.csv",
                     "per_species.csv", "reference_selection.csv",
                     "query_outcomes.csv", "audit.json"):
            archive.write(out / name, name)
    print(f"[wood-audit] review ZIP: {target} ({target.stat().st_size/2**20:.2f} MiB)",
          flush=True)
    return target


def run():
    # Import the image-data modules only for a real Colab run. Pure score tests do
    # not need the image preprocessing stack.
    from . import data, gallery_experiment as study, wood_large_gallery as large

    root = Path(os.environ.get("ROOT_PATH", "/content/drive/MyDrive/NCS")).resolve()
    large_out = Path(os.environ.get(
        "WOOD_LARGE_OUT", root / "results" / "wood_large_gallery_v1")).resolve()
    image_out = Path(os.environ.get(
        "WOOD_LARGE_QKV_OUT", root / "results" /
        "wood_correspondence_image_full_v1")).resolve()
    out = Path(os.environ.get(
        "WOOD_GALLERY_AUDIT_OUT", root / "results" /
        "wood_gallery_audit_v1")).resolve()
    if out == large_out or large_out in out.parents:
        raise ValueError("Audit output must be separate from the locked study")
    lock = json.loads((large_out / "selection_lock.json").read_text())
    plan = json.loads((large_out / "study_plan.json").read_text())
    if lock["public_id_test_evaluated"] or lock["meta_test_evaluated"]:
        raise ValueError("Selection lock was created after viewing test outcomes")
    arm, seed = lock["selected_arm"], int(lock["report_seed"])
    if arm not in {"hard_bank", "random_bank", "qkv_base"}:
        raise ValueError("QKV residual audit requires a QKV-selected arm")
    upstream_seed = seed if plan["settings"]["source_seed"] == "match" else int(
        plan["settings"]["source_seed"])
    checkpoint = _checkpoint_path(root, large_out, image_out, arm, seed,
                                  upstream_seed)
    if study._hash_file(checkpoint) != lock["selected_checkpoint_sha256"]:
        raise ValueError("Selected checkpoint hash changed")
    folder = large_out / "full954" / arm / f"seed_{seed}"
    protocol = json.loads((folder / "protocol.json").read_text())
    manifest_path = root / "swi_manifest.json"
    id_csv = root / "ID_images_expanded.csv"
    if (protocol["checkpoint_sha256"] != lock["selected_checkpoint_sha256"] or
            protocol["manifest_sha256"] != study._hash_file(manifest_path) or
            protocol["id_csv_sha256"] != study._hash_file(id_csv) or
            protocol["query_limit_per_species"] != 0):
        raise ValueError("Locked full-gallery protocol changed")

    manifest = data.load_swi_manifest(manifest_path)
    full_items = [(path, study.canonical(label))
                  for path, label in data.full_swi_items(manifest)]
    balanced = large._balanced_gallery(full_items)
    frame = study._public_frames(root, manifest)["id"]
    queries = list(zip(frame["file_path"], frame["label"]))
    if (len(full_items), len(balanced), len(queries)) != (176123, 4763, 6189):
        raise ValueError("Public/SWI cohort differs from locked evaluation")
    labels = sorted({label for _, label in full_items})
    label_id = {label: i for i, label in enumerate(labels)}
    if len(labels) != 954 or any(study.canonical(label) not in label_id
                                 for _, label in queries):
        raise ValueError("Gallery label identity mismatch")
    checkpoint_hash = lock["selected_checkpoint_sha256"]
    cache = folder / "features"
    full_g, _ = _verified_cache(cache / "swi_full.npy", checkpoint_hash,
                                full_items, False, study._hash_bytes)
    ref_g, ref_t = _verified_cache(cache / "swi_capped5.npy", checkpoint_hash,
                                   balanced, True, study._hash_bytes)
    query_g, query_t = _verified_cache(cache / "public_id.npy", checkpoint_hash,
                                       queries, True, study._hash_bytes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Full-gallery score audit requires a CUDA device")
    full = torch.from_numpy(np.array(full_g)).to(device)
    refs = torch.from_numpy(np.array(ref_g)).to(device)
    q = torch.from_numpy(np.array(query_g)).to(device)
    full_labels = torch.tensor([label_id[label] for _, label in full_items], device=device)
    ref_ids = [label_id[study.canonical(label)] for _, label in balanced]
    ref_labels = torch.tensor(ref_ids, device=device)
    truth = torch.tensor([label_id[study.canonical(label)] for _, label in queries],
                         device=device)
    scans = [study.source_scan_id(path) for path, _ in balanced]
    three = _first_distinct_scans(balanced, ref_ids, study.source_scan_id)
    reference_rows = _reference_rows(
        full_items, balanced, three, study.image_scale, study.source_scan_id,
        {study.canonical(label) for _, label in queries})
    with torch.inference_mode():
        full_proto, full_max = class_scores(q, full, full_labels)
        cap5_proto, cap5_max = class_scores(q, refs, ref_labels)
        cap3_proto, cap3_max = class_scores(q, refs[three], ref_labels[three])
    old = pd.read_csv(folder / "full954_summary.csv")
    old_value = float(old.loc[(old.method == "raw_prototype") & (old.k == 1),
                              "macro_r1"].iloc[0])
    current_value = _metrics(full_proto, truth)[0]["macro_r1"]
    if abs(old_value - current_value) > 1e-5:
        raise ValueError(f"Full-prototype replay mismatch: {old_value} vs "
                         f"{current_value}")

    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = WoodCorrespondence().to(device).eval()
    model.load_state_dict(saved["model"], strict=True)
    from . import wood_correspondence_experiment as corr
    with torch.inference_mode():
        adapted_full = corr._transformed(model, full)
        adapted_q = corr._transformed(model, q)
        adapt_proto, adapt_max = class_scores(adapted_q, adapted_full, full_labels)
        cap3_adapt, _ = class_scores(adapted_q, corr._transformed(model, refs[three]),
                                     ref_labels[three])
        shortlist, _ = candidate_union((adapt_proto, adapt_max, full_proto), 128)
        ref_tokens = torch.from_numpy(np.array(ref_t, dtype=np.float32)).to(device)
        query_tokens = torch.from_numpy(np.array(query_t, dtype=np.float32)).to(device)

    methods = {"full_prototype": full_proto, "full_image_max": full_max,
               "full_adapted_prototype": adapt_proto,
               "capped5_prototype": cap5_proto,
               "capped5_image_max": cap5_max,
               "capped3_prototype": cap3_proto,
               "capped3_image_max": cap3_max,
               "capped3_adapted_prototype": cap3_adapt}
    for width in (32, 128):
        selected = shortlist[:, :width]
        old_rerank = old.loc[(old.method == "qkv_union") & (old.k == width)]
        if len(old_rerank) != 1:
            raise ValueError(f"Missing locked reranker row for width={width}")
        shortlist_macro = _macro((selected == truth[:, None]).any(1), truth)
        if abs(shortlist_macro - float(old_rerank.candidate_macro_recall.iloc[0])) > 1e-5:
            raise ValueError(f"Shortlist replay mismatch at width={width}")
        methods[f"full_prototype_shortlist_{width}"] = _restricted(full_proto, selected)
        methods[f"capped3_prototype_shortlist_{width}"] = _restricted(cap3_proto, selected)
        methods[f"capped3_adapted_shortlist_{width}"] = _restricted(cap3_adapt, selected)
        with torch.inference_mode():
            local = rerank_candidates(model, q, query_tokens, refs, ref_tokens,
                                      ref_labels, scans, selected, adapt_proto,
                                      mode="qkv", max_references_per_class=3)
        local_r1 = _metrics(local, truth)[0]["macro_r1"]
        if abs(local_r1 - float(old_rerank.macro_r1.iloc[0])) > 1e-5:
            raise ValueError(f"QKV replay mismatch at width={width}: "
                             f"{local_r1} vs {old_rerank.macro_r1.iloc[0]}")
        methods[f"qkv_only_{width}"] = local
        for weight in (0.1, 0.25, 0.5):
            methods[f"full_plus_local_{width}_{weight}"] = _blend_local(
                adapt_proto, cap3_adapt, local, selected, weight)
        print(f"[wood-audit] scored shortlist width={width}", flush=True)

    summary, per_source, per_species, outcomes, correct_by_method = [], [], [], [], {}
    source = frame["source_dataset"].astype(str).tolist()
    query_labels = [study.canonical(label) for _, label in queries]
    query_paths = [str(path) for path, _ in queries]
    for name, scores in methods.items():
        width = int(name.split("_")[3]) if name.startswith("full_plus_local_") else (
            int(name.rsplit("_", 1)[1]) if name.endswith(("_32", "_128")) else 0)
        selected = shortlist[:, :width] if width else None
        metrics, correct = _metrics(scores, truth, selected)
        baseline = _metrics(full_proto, truth)[1]
        correct_by_method[name] = correct
        summary.append({"method": name, "width": width, **metrics,
                        "rescued_vs_full_prototype": int((correct & ~baseline).sum()),
                        "harmed_vs_full_prototype": int((~correct & baseline).sum())})
        predictions = scores.argmax(1).cpu().numpy()
        correct_np = correct.cpu().numpy()
        for i, path in enumerate(query_paths):
            outcomes.append({"method": name, "query_path": path,
                             "true_label": query_labels[i], "source_dataset": source[i],
                             "predicted_label": labels[int(predictions[i])],
                             "correct": int(correct_np[i])})
        grouped = defaultdict(list)
        for i, (label, origin) in enumerate(zip(query_labels, source)):
            grouped[(origin, label)].append(int(correct_np[i]))
        for (origin, label), values in grouped.items():
            per_species.append({"method": name, "source_dataset": origin,
                                "species": label, "n_queries": len(values),
                                "r1": sum(values) / len(values)})
        sources = defaultdict(list)
        for (origin, _), values in grouped.items():
            sources[origin].append(sum(values) / len(values))
        for origin, values in sources.items():
            per_source.append({"method": name, "source_dataset": origin,
                               "n_species": len(values),
                               "macro_r1": sum(values) / len(values)})
        print(f"[wood-audit] {name}: macro R@1={metrics['macro_r1']:.4f}", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
    pd.DataFrame(_comparison_rows(truth, correct_by_method["full_prototype"],
                                  correct_by_method)).to_csv(
                                      out / "paired_changes.csv", index=False)
    pd.DataFrame(per_source).to_csv(out / "per_source.csv", index=False)
    pd.DataFrame(per_species).to_csv(out / "per_species.csv", index=False)
    pd.DataFrame(reference_rows).to_csv(out / "reference_selection.csv", index=False)
    pd.DataFrame(outcomes).to_csv(out / "query_outcomes.csv", index=False)
    audit = {"selected_arm": arm, "report_seed": seed,
             "checkpoint_sha256": checkpoint_hash,
             "manifest_sha256": protocol["manifest_sha256"],
             "id_csv_sha256": protocol["id_csv_sha256"],
             "n_queries": len(queries), "n_gallery_species": len(labels),
             "n_full_references": len(full_items),
             "n_capped5_references": len(balanced),
             "n_capped3_references": len(three),
             "training_performed": False, "image_extraction_performed": False,
             "model_selection_performed": False,
             "public_id_results_are_exploratory": True,
             "scoring_formula": "full adapted prototype + lambda * "
                                "(QKV capped-three score - adapted capped-three prototype)",
             "weights_tested_not_selected": [0.1, 0.25, 0.5]}
    (out / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return _write_review(out)
