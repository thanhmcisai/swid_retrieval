"""End-to-end cross-scale retrieval ablations, selected on meta-validation only."""

import os
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from . import data, gallery_experiment as study
from . import wood_correspondence_experiment as corr
from . import wood_correspondence_image_train as image_train
from . import wood_large_gallery as large
from . import wood_reference_budget as budget
from .wood_large_gallery_methods import class_scores, rerank_candidates


ARMS = ("source", "scan_control", "cross_scale", "cross_scale_contrastive",
        "cross_scale_contrastive_margin")
SCHEDULE = ((32, 1), (64, 1), (128, 1), (32, 2), (64, 2), (32, 5))


def _settings():
    cfg = {
        "epochs": int(os.environ.get("WOOD_ML_EPOCHS", "3")),
        "episodes": int(os.environ.get("WOOD_ML_EPISODES", "60")),
        "workers": int(os.environ.get("WOOD_ML_WORKERS", "2")),
        "microbatch": int(os.environ.get("WOOD_ML_MICROBATCH", "4")),
        "image_batch": int(os.environ.get("WOOD_ML_IMAGE_BATCH", "64")),
        "backbone_lr": float(os.environ.get("WOOD_ML_BACKBONE_LR", "1e-6")),
        "head_lr": float(os.environ.get("WOOD_ML_HEAD_LR", "5e-5")),
        "bank_weight": float(os.environ.get("WOOD_ML_BANK_WEIGHT", "0.5")),
        "contrastive_weight": float(os.environ.get("WOOD_ML_CONTRASTIVE_WEIGHT", "0.2")),
        "margin_weight": float(os.environ.get("WOOD_ML_MARGIN_WEIGHT", "0.2")),
        "margin": float(os.environ.get("WOOD_ML_MARGIN", "0.1")),
        "bank_negatives": int(os.environ.get("WOOD_ML_BANK_NEGATIVES", "128")),
        "seed": int(os.environ.get("WOOD_ML_SEED", "43")),
        "source_seed": int(os.environ.get("WOOD_ML_SOURCE_SEED", "43")),
        "folds": [int(x) for x in os.environ.get("WOOD_ML_FOLDS", "0,1").split(",")],
        "budgets": [int(x) for x in os.environ.get("WOOD_ML_BUDGETS", "1,3,10,20").split(",")],
        "widths": [int(x) for x in os.environ.get("WOOD_ML_WIDTHS", "32,128").split(",")],
        "arms": [x.strip() for x in os.environ.get(
            "WOOD_ML_ARMS", ",".join(ARMS)).split(",")],
    }
    if (cfg["epochs"] < 1 or cfg["episodes"] < 1 or cfg["workers"] < 0 or
            cfg["microbatch"] < 1 or cfg["image_batch"] < 1 or
            cfg["backbone_lr"] <= 0 or cfg["head_lr"] <= 0 or
            cfg["bank_weight"] < 0 or cfg["contrastive_weight"] < 0 or
            cfg["margin_weight"] < 0 or cfg["margin"] < 0 or
            cfg["bank_negatives"] < 1 or not cfg["arms"] or
            any(x not in ARMS for x in cfg["arms"]) or
            len(cfg["arms"]) != len(set(cfg["arms"])) or
            len(cfg["folds"]) != len(set(cfg["folds"])) or
            len(cfg["budgets"]) != len(set(cfg["budgets"])) or
            len(cfg["widths"]) != len(set(cfg["widths"])) or
            any(x not in (0, 1) for x in cfg["folds"]) or
            any(x < 1 or x > 20 for x in cfg["budgets"]) or
            any(x < 1 or x > 637 for x in cfg["widths"])):
        raise ValueError("Invalid multilevel study configuration")
    return cfg


def _eligible(items, min_support=1):
    scales = [study.image_scale(path) for path, _ in items]
    if any(scale is None for scale in scales):
        raise ValueError("All meta-train patches must have a crop-width scale")
    groups = defaultdict(lambda: defaultdict(list))
    for index, (path, label) in enumerate(items):
        groups[study.canonical(label)][study.source_scan_id(path)].append(index)
    eligible = defaultdict(list)
    for label, scans in groups.items():
        scan_scales = {scan: {scales[i] for i in rows}
                       for scan, rows in scans.items()}
        total = sum(map(len, scans.values()))
        for query_scan, query_rows in scans.items():
            if total - len(query_rows) < min_support:
                continue
            other_scales = set().union(*(scales for scan, scales in scan_scales.items()
                                         if scan != query_scan))
            for query in query_rows:
                scale = scales[query]
                if other_scales - {scale}:
                    eligible[label].append(query)
    return eligible


def _plan(items, valid, cross_scale, cfg, epoch, multi=None):
    rng = np.random.default_rng(cfg["seed"] * 100003 + epoch)
    by_class, _ = image_train._layout(items)
    if cross_scale and multi is None:
        multi = {k: _eligible(items, k) for _, k in SCHEDULE}
    batches, layouts = [], []
    query_decks = defaultdict(list)
    for step in range(cfg["episodes"]):
        n_way, k = SCHEDULE[step % len(SCHEDULE)]
        choices = sorted(label for label in valid[k] if multi is None or
                         label in multi[k])
        if len(choices) < n_way:
            raise ValueError(f"Only {len(choices)} eligible species for {n_way}-way")
        class_rng = np.random.default_rng(cfg["seed"] * 100003 + epoch * 1009 + step)
        labels = class_rng.choice(choices, size=n_way, replace=False).tolist()
        refs, queries, ref_labels, scans = [], [], [], []
        for class_id, label in enumerate(labels):
            if cross_scale:
                candidates = multi[k][label]
                key = (k, label)
                if not query_decks[key]:
                    query_decks[key] = rng.permutation(candidates).tolist()
                query = int(query_decks[key].pop())
                query_scan = study.source_scan_id(items[query][0])
                alternatives = [index for scan, rows in by_class[label].items()
                                if scan != query_scan for index in rows
                                if study.image_scale(items[index][0]) !=
                                study.image_scale(items[query][0])]
                selected = [int(rng.choice(alternatives))]
            else:
                key = (k, label)
                if not query_decks[key]:
                    candidates = [i for scan in valid[k][label]
                                  for i in by_class[label][scan]]
                    query_decks[key] = rng.permutation(candidates).tolist()
                query = int(query_decks[key].pop())
                query_scan = study.source_scan_id(items[query][0])
                selected = []
            support = [i for scan, rows in by_class[label].items() if scan != query_scan
                       for i in rows if i not in selected]
            if len(support) < k - len(selected):
                raise ValueError("Insufficient scan-disjoint references")
            if k > len(selected):
                selected += rng.choice(support, size=k - len(selected),
                                       replace=False).astype(int).tolist()
            refs.extend(selected)
            queries.append(query)
            ref_labels.extend([class_id] * k)
            scans.extend(study.source_scan_id(items[i][0]) for i in selected)
        indices = refs + queries
        batches.append(indices)
        layouts.append((len(refs), ref_labels, scans, labels,
                        [study.image_scale(items[i][0]) for i in refs],
                        [study.image_scale(items[i][0]) for i in queries]))
    return batches, layouts


def cross_scale_loss(model, query, references, ref_labels, ref_scales, query_scales,
                     temperature=0.07):
    """Multi-positive InfoNCE over cross-scan references at different crop scales."""
    q = corr._transformed(model, query)
    r = corr._transformed(model, references)
    labels = torch.as_tensor(ref_labels, device=q.device)
    scale_r = torch.as_tensor(ref_scales, device=q.device)
    scale_q = torch.as_tensor(query_scales, device=q.device)
    same_species = labels[None, :] == torch.arange(len(q), device=q.device)[:, None]
    positive = same_species & (scale_r[None, :] != scale_q[:, None])
    if not positive.any(dim=1).all():
        raise ValueError("Cross-scale positive is missing from an episode")
    logits = q @ r.T / temperature
    denominator = logits.masked_fill(same_species & ~positive, -torch.inf)
    numerator = logits.masked_fill(~positive, -torch.inf)
    return (torch.logsumexp(denominator, dim=1) -
            torch.logsumexp(numerator, dim=1)).mean()


def _step(model, encoder, images, layout, bank_labels, bank, arm, config,
          cfg, optimizer, device):
    n_refs, labels, scans, species, ref_scales, query_scales = layout
    optimizer.zero_grad(set_to_none=True)
    globals_, tokens_ = [], []
    with torch.no_grad():
        for chunk in images.split(cfg["microbatch"]):
            g, t = corr._encode_images(encoder, chunk.to(device, non_blocking=True))
            globals_.append(g.float())
            tokens_.append(t.float())
    all_g = torch.cat(globals_).detach().requires_grad_(True)
    all_t = torch.cat(tokens_).detach().requires_grad_(True)
    r, q = all_g[:n_refs], all_g[n_refs:]
    loss, accuracy = corr._loss(model, q, all_t[n_refs:], r, all_t[:n_refs],
                                labels, scans, config)
    bank_ce = large._bank_loss(model, q, r, labels, species, bank_labels, bank,
                               cfg["bank_negatives"], "hard_bank")
    loss = loss + cfg["bank_weight"] * bank_ce
    if arm in {"cross_scale_contrastive", "cross_scale_contrastive_margin"}:
        loss = loss + cfg["contrastive_weight"] * cross_scale_loss(
            model, q, r, labels, ref_scales, query_scales)
    if arm == "cross_scale_contrastive_margin":
        scores, classes = model(q, all_t[n_refs:], r, all_t[:n_refs],
                                torch.as_tensor(labels, device=device), scans,
                                mode="qkv", query_chunk=config["query_chunk"],
                                reference_chunk=config["reference_chunk"])
        targets = torch.arange(len(q), device=device)
        if not torch.equal(classes, targets):
            raise ValueError("Episode class ordering changed")
        impostors = scores.masked_fill(F.one_hot(targets, len(q)).bool(), -torch.inf)
        loss = loss + cfg["margin_weight"] * F.softplus(
            cfg["margin"] + impostors.max(dim=1).values - scores.diag()).mean()
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite multilevel training loss")
    loss.backward()
    g_grad, t_grad = all_g.grad.detach(), all_t.grad.detach()
    if not torch.isfinite(g_grad).all() or not torch.isfinite(t_grad).all():
        raise RuntimeError("Non-finite feature gradient")
    offset = 0
    for chunk in images.split(cfg["microbatch"]):
        size = len(chunk)
        g, t = corr._encode_images(encoder, chunk.to(device, non_blocking=True))
        torch.autograd.backward((g.float(), t.float()),
                                (g_grad[offset:offset + size], t_grad[offset:offset + size]))
        offset += size
    torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(encoder.parameters()), 1.0)
    optimizer.step()
    return float(loss.detach()), accuracy


def _source(root, study_out, image_out, cfg, device):
    return large._source_checkpoint(root, study_out, image_out, cfg["source_seed"], device)


def _provenance(root, source, cfg, arm):
    train_keys = ("epochs", "episodes", "workers", "microbatch", "image_batch",
                  "backbone_lr", "head_lr", "bank_weight", "contrastive_weight",
                  "margin_weight", "margin", "bank_negatives", "seed", "source_seed")
    return {"source_sha256": study._hash_file(source),
            "manifest_sha256": study._hash_file(root / "swi_manifest.json"),
            "runner_sha256": study._hash_file(__file__),
            "method_sha256": study._hash_file(Path(__file__).with_name(
                "wood_correspondence_method.py")),
            "settings": {key: cfg[key] for key in train_keys},
            "arm": arm, "selection_split": "meta-val"}


def _train(root, out, manifest, study_out, image_out, cfg, arm, device):
    if arm == "source":
        return
    study._seed(cfg["seed"])
    source, encoder, model, base_cfg, config = _source(
        root, study_out, image_out, cfg, device)
    folder = out / "train" / arm
    folder.mkdir(parents=True, exist_ok=True)
    provenance = _provenance(root, source, cfg, arm)
    latest, best = folder / "latest.pt", folder / "best.pt"
    items = manifest["meta-train"]
    _, valid = image_train._layout(items)
    multi = {k: _eligible(items, k) for _, k in SCHEDULE}
    if len(multi[5]) < 128:
        raise ValueError("Too few cross-scan/cross-scale eligible species")
    print(f"[wood-ml] eligible cross-scale K=5 species={len(multi[5])} / "
          f"{len({study.canonical(label) for _, label in items})}", flush=True)
    optimizer = torch.optim.AdamW([
        {"params": model.parameters(), "lr": cfg["head_lr"]},
        {"params": encoder.parameters(), "lr": cfg["backbone_lr"]}], weight_decay=1e-3)
    start, best_score = 1, -float("inf")
    if latest.is_file():
        saved = torch.load(latest, map_location="cpu", weights_only=False)
        if saved["provenance"] != provenance:
            raise ValueError("Training provenance changed; choose a new output folder")
        encoder.load_state_dict(saved["encoder"], strict=True)
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        start, best_score = saved["epoch"] + 1, saved["best_score"]
        print(f"[wood-ml] resume {arm} epoch={start}", flush=True)
    dataset = data.ManifestDataset(items, transform=data.get_transforms(224, augment=True))
    dataset.loader = large._RetryImageLoader(dataset.loader)
    reps = large._bank_representatives(manifest)
    bank_cfg = dict(base_cfg, image_batch=cfg["image_batch"], workers=cfg["workers"])
    for epoch in range(start, cfg["epochs"] + 1):
        bank_labels, bank = large._bank(encoder, model, reps, bank_cfg, device)
        batches, layouts = _plan(items, valid, arm != "scan_control", cfg, epoch,
                                 multi=multi)
        unique_images = len({index for batch in batches for index in batch})
        cross_scale_pairs = sum(sum(ref_scale != query_scales[class_id]
                                    for class_id, ref_scale in zip(labels, ref_scales))
                                for _, labels, _, _, ref_scales, query_scales in layouts)
        torch.manual_seed(cfg["seed"] * 100003 + epoch)
        loader = DataLoader(dataset, batch_sampler=batches,
                            num_workers=min(cfg["workers"], len(os.sched_getaffinity(0))
                                            if hasattr(os, "sched_getaffinity") else
                                            os.cpu_count() or 1), pin_memory=True)
        encoder.backbone.eval()
        encoder.projection.train()
        model.train()
        losses, accuracies = [], []
        started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for step, ((images, _), layout) in enumerate(zip(loader, layouts), 1):
            loss, accuracy = _step(model, encoder, images, layout, bank_labels, bank,
                                   arm, config, cfg, optimizer, device)
            losses.append(loss)
            accuracies.append(accuracy)
            if step % 10 == 0 or step == len(layouts):
                print(f"[wood-ml] {arm} epoch={epoch} step={step}/{len(layouts)} "
                      f"loss={np.mean(losses):.4f} train_R1={np.mean(accuracies):.4f} "
                      f"elapsed_s={time.perf_counter()-started:.1f}", flush=True)
        train_seconds = time.perf_counter() - started
        train_peak = (torch.cuda.max_memory_allocated() / 2**30
                      if device.type == "cuda" else None)
        candidate = folder / "candidate.pt"
        payload = {"provenance": provenance, "epoch": epoch,
                   "encoder": encoder.state_dict(), "model": model.state_dict()}
        study._save_checkpoint(candidate, payload)
        frame, summary, timing = corr._evaluate(
            root, out / "early_stop" / arm, manifest, encoder, base_cfg,
            study._hash_file(candidate), model, config, device, folds=(0,),
            modes=("global", "trained"))
        score = corr._selected_score(summary, "trained")
        corr._save_report(folder, f"epoch_{epoch:02d}", frame, summary, timing,
                          {"provenance": provenance, "score": score,
                           "unique_train_images": unique_images,
                           "cross_scale_positive_pairs": cross_scale_pairs,
                           "training_images_total": len(items),
                           "train_seconds": train_seconds,
                           "peak_gpu_gb": train_peak})
        if score > best_score:
            best_score = score
            study._save_checkpoint(best, {**payload, "selection_score": score})
        study._save_checkpoint(latest, {**payload, "optimizer": optimizer.state_dict(),
                                        "best_score": best_score})
        print(f"[wood-ml] {arm} epoch={epoch} val={score:.4f} best={best_score:.4f} "
              f"unique_images={unique_images}/{len(items)} "
              f"cross_scale_pairs={cross_scale_pairs} "
              f"peak_gb={train_peak:.2f}",
              flush=True)


def _load(root, out, study_out, image_out, cfg, arm, device):
    source, encoder, model, base_cfg, config = _source(
        root, study_out, image_out, cfg, device)
    if arm != "source":
        path = out / "train" / arm / "best.pt"
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved["provenance"] != _provenance(root, source, cfg, arm):
            raise ValueError("Selected checkpoint provenance mismatch")
        encoder.load_state_dict(saved["encoder"], strict=True)
        model.load_state_dict(saved["model"], strict=True)
    else:
        path = source
    return path, encoder.eval(), model.eval(), base_cfg


def _metrics(scores, truth):
    labels = truth.unique()
    prediction = scores.argmax(dim=1)
    correct = prediction == truth
    top5 = scores.topk(5, dim=1).indices
    r5 = (top5 == truth[:, None]).any(dim=1) & torch.isfinite(
        scores.gather(1, truth[:, None]).squeeze(1))
    return {"macro_r1": float(torch.stack([correct[truth == c].float().mean()
                                            for c in labels]).mean()),
            "micro_r1": float(correct.float().mean()),
            "macro_r5": float(torch.stack([
                r5[truth == c].float().mean() for c in labels]).mean())}


def rerank_all_references(model, query, query_tokens, references, reference_tokens,
                          reference_labels, scans, candidates, base_scores):
    """Rerank the shortlist with every selected reference, including other scales."""
    if (len(query) != len(candidates) or len(references) != len(scans) or
            len(references) != len(reference_labels) or
            len(references) != len(reference_tokens)):
        raise ValueError("Invalid all-reference rerank inputs")
    by_class = {int(label): (reference_labels == label).nonzero(as_tuple=True)[0]
                for label in reference_labels.unique().tolist()}
    result = torch.full_like(base_scores, -torch.inf)
    for row in range(len(query)):
        chosen = candidates[row].unique(sorted=True)
        indices = torch.cat([by_class[int(label)] for label in chosen.tolist()])
        local_labels = torch.searchsorted(chosen, reference_labels[indices])
        scores, classes = model(query[row:row + 1], query_tokens[row:row + 1],
                                references[indices], reference_tokens[indices],
                                local_labels, [scans[i] for i in indices.tolist()],
                                mode="qkv", query_chunk=1)
        if not torch.equal(classes, torch.arange(len(chosen), device=query.device)):
            raise ValueError("Reranker class indexing changed")
        result[row, chosen] = scores[0]
    return result


def _validate_arm(root, out, manifest, study_out, image_out, cfg, arm, device):
    path, encoder, model, base_cfg = _load(root, out, study_out, image_out,
                                           cfg, arm, device)
    checkpoint_hash = study._hash_file(path)
    folder = out / "validation" / arm
    protocol_path = folder / "protocol.json"
    protocol = {"checkpoint_sha256": checkpoint_hash,
                "meta_val_only": True, "meta_test_evaluated": False,
                "public_id_evaluated": False, "folds": cfg["folds"],
                "budgets": cfg["budgets"], "widths": cfg["widths"],
                "selection_policy": "scan_scale", "candidate_rule": "raw_prototype"}
    if protocol_path.is_file():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("Existing validation protocol changed; use a new output folder")
        if all((folder / name).is_file() for name in
               ("summary.csv", "per_query.csv", "by_scale.csv")):
            print(f"[wood-ml] reusing completed validation arm={arm}", flush=True)
            return pd.read_csv(folder / "summary.csv")
    extract_cfg = dict(base_cfg, image_batch=cfg["image_batch"], workers=cfg["workers"])
    rows, query_rows = [], []
    for fold in cfg["folds"]:
        _, queries = study.validation_items(manifest, group_mode="scan_disjoint", fold=fold)
        refs = budget._reference_pool(manifest, queries)
        labels = sorted({label for _, label in refs})
        if len(labels) != 637 or len(queries) != 285:
            raise ValueError("Meta-val cohort changed")
        label_ids = {label: i for i, label in enumerate(labels)}
        selected = budget.select_indices(refs, max(cfg["budgets"]), "scan_scale")
        selected_items = [refs[i] for i in selected]
        cache = out / "features" / arm / f"fold_{fold}"
        rg, rt = large._cache_features(encoder, selected_items, extract_cfg, device,
                                       cache / "references.npy", checkpoint_hash, True)
        qg, qt = large._cache_features(encoder, queries, extract_cfg, device,
                                       cache / "queries.npy", checkpoint_hash, True)
        rg = torch.from_numpy(np.asarray(rg).copy()).to(device)
        rt = torch.from_numpy(np.asarray(rt).copy().astype(np.float32)).to(device)
        qg = torch.from_numpy(np.asarray(qg).copy()).to(device)
        qt = torch.from_numpy(np.asarray(qt).copy().astype(np.float32)).to(device)
        truth = torch.tensor([label_ids[study.canonical(label)] for _, label in queries],
                             device=device)
        truth_list = truth.tolist()
        for count in cfg["budgets"]:
            chosen = budget.select_indices(refs, count, "scan_scale")
            positions = {index: j for j, index in enumerate(selected)}
            ids = torch.tensor([positions[i] for i in chosen], device=device)
            chosen_items = [refs[i] for i in chosen]
            ref_labels = torch.tensor([label_ids[label] for _, label in chosen_items],
                                      device=device)
            scans = [study.source_scan_id(item[0]) for item in chosen_items]
            scales_by_class = defaultdict(set)
            scans_by_class = defaultdict(set)
            for path_r, label_r in chosen_items:
                scales_by_class[label_r].add(study.image_scale(path_r))
                scans_by_class[label_r].add(study.source_scan_id(path_r))
            for path_q, label_q in queries:
                scan_q = study.source_scan_id(path_q)
                if scan_q in scans_by_class[study.canonical(label_q)]:
                    raise ValueError("Validation scan leakage")
            with torch.inference_mode():
                raw, _ = class_scores(qg, rg[ids], ref_labels)
                adapted_r = corr._transformed(model, rg[ids])
                adapted_q = corr._transformed(model, qg)
                adapted, _ = class_scores(adapted_q, adapted_r, ref_labels)
            base = {"raw_prototype": raw, "adapted_prototype": adapted}
            def record(name, scores, width, hit=None):
                prediction = scores.argmax(dim=1).tolist()
                hit_list = hit.tolist() if hit is not None else [None] * len(queries)
                for i, (path_q, label_q) in enumerate(queries):
                    label_q = study.canonical(label_q)
                    query_rows.append({"arm": arm, "fold": fold, "budget": count,
                                       "width": width, "method": name,
                                       "query_path": path_q, "true_label": label_q,
                                       "predicted_label": labels[prediction[i]],
                                       "correct": int(prediction[i] == truth_list[i]),
                                       "candidate_hit": hit_list[i],
                                       "query_scale": study.image_scale(path_q),
                                       "same_scale_reference": int(
                                           study.image_scale(path_q) in scales_by_class[label_q]),
                                       "distinct_reference_scans": len(scans_by_class[label_q]),
                                       "distinct_reference_scales": len(scales_by_class[label_q])})
            for name, scores in base.items():
                recall32 = (scores.topk(32, dim=1).indices == truth[:, None]).any(dim=1)
                recall128 = (scores.topk(128, dim=1).indices == truth[:, None]).any(dim=1)
                rows.append({"arm": arm, "fold": fold, "budget": count,
                             "width": 0, "method": name, **_metrics(scores, truth),
                             "candidate_macro_recall": None,
                             "candidate_macro_recall32": float(torch.stack([
                                 recall32[truth == c].float().mean()
                                 for c in truth.unique()]).mean()),
                             "candidate_macro_recall128": float(torch.stack([
                                 recall128[truth == c].float().mean()
                                 for c in truth.unique()]).mean()),
                             "conditional_macro_r1": None, "ms_per_query": 0.0})
                record(name, scores, 0)
            for width in cfg["widths"]:
                shortlist = raw.topk(width, dim=1).indices
                hit = (shortlist == truth[:, None]).any(dim=1)
                candidate_macro = float(torch.stack([
                    hit[truth == c].float().mean() for c in truth.unique()]).mean())
                start = time.perf_counter()
                with torch.inference_mode():
                    reranked = rerank_all_references(
                        model, qg, qt, rg[ids], rt[ids], ref_labels, scans,
                        shortlist, adapted)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                methods = [("qkv_all_refs", reranked)]
                if (count >= 3 and count == max(cfg["budgets"]) and
                        width == max(cfg["widths"])):
                    scan_start = time.perf_counter()
                    with torch.inference_mode():
                        scan_three = rerank_candidates(
                            model, qg, qt, rg[ids], rt[ids], ref_labels, scans,
                            shortlist, adapted, max_references_per_class=3)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    methods.append(("qkv_scan3", scan_three))
                    scan_elapsed = time.perf_counter() - scan_start
                for name, scores in methods:
                    metrics = _metrics(scores, truth)
                    conditional_values = [
                        (scores.argmax(1)[(truth == c) & hit] == c).float().mean()
                        for c in truth.unique() if ((truth == c) & hit).any()]
                    conditional = (float(torch.stack(conditional_values).mean())
                                   if conditional_values else None)
                    rows.append({"arm": arm, "fold": fold, "budget": count,
                                 "width": width, "method": name, **metrics,
                                 "candidate_macro_recall": candidate_macro,
                                 "candidate_macro_recall32": None,
                                 "candidate_macro_recall128": None,
                                 "conditional_macro_r1": conditional,
                                 "ms_per_query": 1000 * (
                                     scan_elapsed if name == "qkv_scan3" else elapsed) / len(qg)})
                    record(name, scores, width, hit)
                    print(f"[wood-ml] {arm} fold={fold} budget={count} width={width} "
                          f"{name} candidate={candidate_macro:.4f} "
                          f"R1={metrics['macro_r1']:.4f}", flush=True)
    folder.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(folder / "summary.csv", index=False)
    pd.DataFrame(query_rows).to_csv(folder / "per_query.csv", index=False)
    pd.DataFrame(query_rows).groupby(
        ["fold", "budget", "width", "method", "query_scale",
         "same_scale_reference"], as_index=False).agg(
        n_queries=("correct", "size"),
        n_species=("true_label", "nunique"),
        micro_r1=("correct", "mean")).to_csv(folder / "by_scale.csv", index=False)
    study._json(protocol_path, protocol)
    return frame


def _select(out, cfg):
    frames = []
    for arm in cfg["arms"]:
        path = out / "validation" / arm / "summary.csv"
        frame = pd.read_csv(path)
        if set(frame.fold) != set(cfg["folds"]):
            raise ValueError("Incomplete validation folds")
        frames.append(frame)
    all_rows = pd.concat(frames, ignore_index=True)
    grouped = all_rows.groupby(["arm", "budget", "width", "method"],
                               dropna=False, as_index=False).agg(
        macro_r1=("macro_r1", "mean"), n_folds=("fold", "nunique"))
    if (grouped.n_folds != len(cfg["folds"])).any():
        raise ValueError("Some candidate methods are missing validation folds")
    selected = grouped.sort_values(["macro_r1", "budget", "width", "arm", "method"],
                                   ascending=[False, True, True, True, True]).iloc[0]
    lock = {
        "criterion": "highest mean meta-val macro R@1, tie: lower reference budget",
        "meta_test_evaluated": False, "public_id_evaluated": False,
        "source_run": "meta-train image fine-tuning from meta-val-selected source",
        "validation_sha256": {arm: study._hash_file(
            out / "validation" / arm / "summary.csv") for arm in cfg["arms"]},
        "validation_provenance_sha256": {arm: study._hash_file(
            out / "validation" / arm / "protocol.json") for arm in cfg["arms"]},
        "selected": {"arm": str(selected.arm), "budget": int(selected.budget),
                     "width": int(selected.width), "method": str(selected.method),
                     "meta_val_macro_r1": float(selected.macro_r1)},
    }
    lock_path = out / "selection_lock.json"
    if lock_path.is_file() and json.loads(lock_path.read_text()) != lock:
        raise ValueError("Selection lock already exists with different evidence")
    study._json(lock_path, lock)
    print(f"[wood-ml] validation lock: {out / 'selection_lock.json'}", flush=True)


def _locked_selection(out):
    lock_path = out / "selection_lock.json"
    lock = json.loads(lock_path.read_text())
    if lock["meta_test_evaluated"] or lock["public_id_evaluated"]:
        raise ValueError("Selection was not locked before test evaluation")
    for arm, expected in lock["validation_sha256"].items():
        if study._hash_file(out / "validation" / arm / "summary.csv") != expected:
            raise ValueError("Validation evidence changed after selection")
        if study._hash_file(out / "validation" / arm / "protocol.json") != lock[
                "validation_provenance_sha256"][arm]:
            raise ValueError("Validation checkpoint provenance changed after selection")
    return lock["selected"]


def _public_scores(root, out, manifest, study_out, image_out, cfg, arm,
                   selected, device):
    path, encoder, model, base_cfg = _load(root, out, study_out, image_out,
                                           cfg, arm, device)
    checkpoint_hash = study._hash_file(path)
    protocol = json.loads((out / "validation" / arm / "protocol.json").read_text())
    if protocol["checkpoint_sha256"] != checkpoint_hash:
        raise ValueError("Selected checkpoint differs from validated checkpoint")
    full_items = [(path, study.canonical(label)) for path, label in
                  data.full_swi_items(manifest)]
    frame = study._public_frames(root, manifest)["id"]
    queries = [(path, study.canonical(label)) for path, label in
               zip(frame.file_path, frame.label)]
    labels = sorted({label for _, label in full_items})
    if (len(full_items) != 176123 or len(labels) != 954 or len(queries) != 6189 or
            len({label for _, label in queries}) != 24 or
            frame.source_dataset.astype(str).str.casefold().eq("woodauth").any()):
        raise ValueError("Corrected full-gallery/public-ID cohort changed")
    budget_k, width, method = (int(selected["budget"]), int(selected["width"]),
                               selected["method"])
    ids = budget.select_indices(full_items, budget_k, "scan_scale")
    references = [full_items[i] for i in ids]
    folder = out / "public_id" / arm
    folder.mkdir(parents=True, exist_ok=True)
    signature = study._hash_bytes(json.dumps({
        "checkpoint": checkpoint_hash, "lock": study._hash_file(
            out / "selection_lock.json"), "id_csv": study._hash_file(
            root / "ID_images_expanded.csv"), "references": references,
        "queries": queries, "method": method, "width": width,
    }, separators=(",", ":")).encode())
    protocol_path = folder / "protocol.json"
    if protocol_path.is_file() and json.loads(protocol_path.read_text())["signature"] != signature:
        raise ValueError("Public-ID protocol changed; choose a new output folder")
    extract_cfg = dict(base_cfg, image_batch=cfg["image_batch"], workers=cfg["workers"])
    need_tokens = method in {"qkv_all_refs", "qkv_scan3"}
    rg, rt = large._cache_features(encoder, references, extract_cfg, device,
                                   folder / "references.npy", checkpoint_hash,
                                   need_tokens)
    qg, qt = large._cache_features(encoder, queries, extract_cfg, device,
                                   folder / "queries.npy", checkpoint_hash,
                                   need_tokens)
    rg = torch.from_numpy(np.asarray(rg).copy()).to(device)
    rt = (torch.from_numpy(np.asarray(rt).copy().astype(np.float32)).to(device)
          if need_tokens else None)
    qg = torch.from_numpy(np.asarray(qg).copy()).to(device)
    qt = (torch.from_numpy(np.asarray(qt).copy().astype(np.float32)).to(device)
          if need_tokens else None)
    label_ids = {label: i for i, label in enumerate(labels)}
    ref_labels = torch.tensor([label_ids[label] for _, label in references], device=device)
    truth = torch.tensor([label_ids[label] for _, label in queries], device=device)
    scans = [study.source_scan_id(path) for path, _ in references]
    with torch.inference_mode():
        raw, _ = class_scores(qg, rg, ref_labels)
        adapted, _ = class_scores(corr._transformed(model, qg),
                                  corr._transformed(model, rg), ref_labels)
    shortlist = raw.topk(width, dim=1).indices if width else None
    hit = ((shortlist == truth[:, None]).any(dim=1) if shortlist is not None else None)
    if method == "raw_prototype":
        scores = raw
    elif method == "adapted_prototype":
        scores = adapted
    elif method in {"qkv_all_refs", "qkv_scan3"}:
        scores_path = folder / "scores.npy"
        cursor_path = folder / "scores.progress.json"
        if cursor_path.is_file():
            cursor = json.loads(cursor_path.read_text())
            if cursor["signature"] != signature or not scores_path.is_file():
                raise ValueError("Stale public rerank cache")
            offset = int(cursor["offset"])
            saved = np.lib.format.open_memmap(scores_path, mode="r+")
        else:
            if scores_path.exists():
                raise ValueError("Unverified public rerank cache")
            saved = np.lib.format.open_memmap(
                scores_path, mode="w+", dtype=np.float32, shape=(len(queries), len(labels)))
            offset = 0
            study._json(cursor_path, {"signature": signature, "offset": offset})
        if saved.shape != (len(queries), len(labels)) or not 0 <= offset <= len(queries):
            raise ValueError("Public rerank cache shape/cursor mismatch")
        started = time.perf_counter()
        for first in range(offset, len(queries), 16):
            stop = min(first + 16, len(queries))
            with torch.inference_mode():
                if method == "qkv_scan3":
                    part = rerank_candidates(
                        model, qg[first:stop], qt[first:stop], rg, rt,
                        ref_labels, scans, shortlist[first:stop], adapted[first:stop],
                        max_references_per_class=3)
                else:
                    part = rerank_all_references(
                        model, qg[first:stop], qt[first:stop], rg, rt,
                        ref_labels, scans, shortlist[first:stop], adapted[first:stop])
            saved[first:stop] = part.cpu().numpy()
            saved.flush()
            study._json(cursor_path, {"signature": signature, "offset": stop})
            if stop % 160 == 0 or stop == len(queries):
                print(f"[wood-ml] public {arm} {method} {stop}/{len(queries)} "
                      f"elapsed_s={time.perf_counter()-started:.1f}", flush=True)
        scores = torch.from_numpy(np.asarray(saved).copy()).to(device)
    else:
        raise ValueError("Selected method is not supported by public-ID evaluator")
    if not torch.isfinite(scores.max(dim=1).values).all():
        raise ValueError("Public-ID scores contain an unscored query")
    result = _metrics(scores, truth)
    result.update({"arm": arm, "method": method, "budget": budget_k,
                   "width": width, "n_queries": len(queries),
                   "n_gallery_species": len(labels), "n_references": len(references),
                   "candidate_macro_recall": (float(torch.stack([
                       hit[truth == c].float().mean() for c in truth.unique()]).mean())
                       if hit is not None else None),
                   "conditional_macro_r1": (float(torch.stack([
                       (scores.argmax(1)[(truth == c) & hit] == c).float().mean()
                       for c in truth.unique() if ((truth == c) & hit).any()]).mean())
                       if hit is not None and hit.any() else None),
                   "public_id_prior_exploration_disclosed": True})
    prediction = scores.argmax(dim=1).tolist()
    truth_ids = truth.tolist()
    pd.DataFrame({"query_path": [path for path, _ in queries],
                  "true_label": [label for _, label in queries],
                  "source_dataset": frame.source_dataset.astype(str).tolist(),
                  "predicted_label": [labels[i] for i in prediction],
                  "correct": [int(a == b) for a, b in zip(prediction, truth_ids)],
                  "candidate_hit": (hit.int().tolist() if hit is not None else
                                    [None] * len(queries))}).to_csv(
        folder / "per_query.csv", index=False)
    study._json(folder / "summary.json", result)
    study._json(protocol_path, {"signature": signature,
               "checkpoint_sha256": checkpoint_hash,
               "selection_lock_sha256": study._hash_file(out / "selection_lock.json"),
               "n_gallery_images_pool": len(full_items),
               "n_gallery_references_scored": len(references),
               "public_id_prior_exploration_disclosed": True})
    print(f"[wood-ml] public {arm}: macro R@1={result['macro_r1']:.4f} "
          f"candidate={result['candidate_macro_recall']}", flush=True)
    return result


def _public_id(root, out, manifest, study_out, image_out, cfg, device):
    selected = _locked_selection(out)
    arms = list(dict.fromkeys(("source", selected["arm"])))
    rows = [_public_scores(root, out, manifest, study_out, image_out, cfg,
                           arm, selected, device) for arm in arms]
    pd.DataFrame(rows).to_csv(out / "public_id_comparison.csv", index=False)
    for arm in arms:
        frame = pd.read_csv(out / "public_id" / arm / "per_query.csv")
        by_species = frame.groupby(["source_dataset", "true_label"],
                                   as_index=False).correct.mean()
        by_species.groupby("source_dataset", as_index=False).agg(
            n_species=("true_label", "size"), macro_r1=("correct", "mean")
        ).to_csv(out / "public_id" / arm / "per_source.csv", index=False)
    if len(arms) == 2:
        source = pd.read_csv(out / "public_id" / arms[0] / "per_query.csv")
        selected_rows = pd.read_csv(out / "public_id" / arms[1] / "per_query.csv")
        paired = source.merge(selected_rows, on=["query_path", "true_label",
                                                "source_dataset"], how="outer",
                              validate="one_to_one", indicator=True,
                              suffixes=("_source", "_selected"))
        if len(paired) != 6189 or not paired._merge.eq("both").all():
            raise ValueError("Source and selected public-ID queries differ")
        paired["rescued"] = ((paired.correct_source == 0) &
                             (paired.correct_selected == 1)).astype(int)
        paired["harmed"] = ((paired.correct_source == 1) &
                            (paired.correct_selected == 0)).astype(int)
        paired.to_csv(out / "public_id_paired.csv", index=False)
        study._json(out / "public_id_paired_summary.json", {
            "source_arm": arms[0], "selected_arm": arms[1],
            "rescued": int(paired.rescued.sum()), "harmed": int(paired.harmed.sum()),
            "n_queries": len(paired),
            "public_id_prior_exploration_disclosed": True})


def _smoke(root, out, manifest, study_out, image_out, cfg, device):
    source, encoder, model, _, config = _source(
        root, study_out, image_out, cfg, device)
    items = manifest["meta-train"]
    multi = _eligible(items)
    by_class, _ = image_train._layout(items)
    labels = sorted(multi)[:4]
    if len(labels) != 4:
        raise ValueError("Smoke needs four cross-scale eligible species")
    refs, queries, scales_r, scales_q, scans = [], [], [], [], []
    for label in labels:
        query = multi[label][0]
        query_scan = study.source_scan_id(items[query][0])
        query_scale = study.image_scale(items[query][0])
        reference = next(index for scan, rows in by_class[label].items()
                         if scan != query_scan for index in rows
                         if study.image_scale(items[index][0]) != query_scale)
        refs.append(reference)
        queries.append(query)
        scales_r.append(study.image_scale(items[reference][0]))
        scales_q.append(query_scale)
        scans.append(study.source_scan_id(items[reference][0]))
    samples = [items[i] for i in refs + queries]
    dataset = data.ManifestDataset(samples, transform=data.get_transforms(224, augment=True))
    dataset.loader = large._RetryImageLoader(dataset.loader)
    images, _ = next(iter(DataLoader(dataset, batch_size=8, num_workers=0)))
    bank_labels = labels + [next(label for label in sorted(by_class) if label not in labels)]
    bank = F.normalize(torch.randn(5, 512, device=device), dim=-1)
    optimizer = torch.optim.AdamW([
        {"params": model.parameters(), "lr": cfg["head_lr"]},
        {"params": encoder.parameters(), "lr": cfg["backbone_lr"]}])
    encoder.backbone.eval()
    encoder.projection.train()
    model.train()
    local_cfg = dict(cfg, microbatch=min(2, cfg["microbatch"]), bank_negatives=1)
    loss, accuracy = _step(model, encoder, images,
                           (4, list(range(4)), scans, labels, scales_r, scales_q),
                           bank_labels, bank, "cross_scale_contrastive_margin",
                           config, local_cfg, optimizer, device)
    backbone_grad = encoder.backbone.blocks[-1].attn.qkv.weight.grad
    qkv_grad = model.query_key.weight.grad
    if (backbone_grad is None or qkv_grad is None or
            not torch.isfinite(backbone_grad).all() or
            not torch.isfinite(qkv_grad).all() or
            backbone_grad.norm() <= 0 or qkv_grad.norm() <= 0):
        raise RuntimeError("Real-image smoke did not backpropagate to encoder and QKV")
    out.mkdir(parents=True, exist_ok=True)
    report = {"source_checkpoint_sha256": study._hash_file(source),
              "loss": loss, "train_r1": accuracy,
              "encoder_attention_grad_norm": float(backbone_grad.norm()),
              "qkv_grad_norm": float(qkv_grad.norm()),
              "n_images": 8, "training_checkpoint_saved": False}
    study._json(out / "smoke.json", report)
    print(f"[wood-ml] real-image smoke passed: {report}", flush=True)


def run():
    root = Path(os.environ.get("ROOT_PATH", "/content/drive/MyDrive/NCS")).resolve()
    out = Path(os.environ.get("WOOD_ML_OUT", root / "results" /
                              "wood_multilevel_study_v1")).resolve()
    study_out = Path(os.environ.get("WOOD_LARGE_BASE_OUT", root / "results" /
                                  "dinov2s_retrieval_matched_v1")).resolve()
    image_out = Path(os.environ.get("WOOD_LARGE_QKV_OUT", root / "results" /
                                  "wood_correspondence_image_full_v1")).resolve()
    cfg = _settings()
    mode = os.environ.get("WOOD_ML_MODE", "preflight")
    if mode not in {"preflight", "smoke", "train", "validate", "select", "public_id"}:
        raise ValueError("Invalid WOOD_ML_MODE")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = data.load_swi_manifest(root / "swi_manifest.json")
    if mode == "preflight":
        items = manifest["meta-train"]
        eligible = _eligible(items)
        audit = {"meta_train_images": len(items), "meta_train_species": len({
            study.canonical(label) for _, label in items}),
                 "cross_scan_cross_scale_eligible_species": len(eligible),
                 "source_seed": cfg["source_seed"], "settings": cfg}
        out.mkdir(parents=True, exist_ok=True)
        study._json(out / "preflight.json", audit)
        print(f"[wood-ml] preflight: {audit}", flush=True)
        return
    if device.type != "cuda":
        raise RuntimeError("Image training and extraction require CUDA")
    if mode == "smoke":
        _smoke(root, out, manifest, study_out, image_out, cfg, device)
    elif mode == "train":
        for arm in cfg["arms"]:
            _train(root, out, manifest, study_out, image_out, cfg, arm, device)
    elif mode == "validate":
        for arm in cfg["arms"]:
            _validate_arm(root, out, manifest, study_out, image_out, cfg, arm, device)
    elif mode == "select":
        _select(out, cfg)
    else:
        _public_id(root, out, manifest, study_out, image_out, cfg, device)


if __name__ == "__main__":
    run()
