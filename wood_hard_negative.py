"""Shortlist-mined QKV training and query-scale reference controls.

All mining uses meta-train images. Model choice uses meta-val only. The
previously inspected public-ID cohort is deliberately absent from this runner.
"""

import json
import os
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
from . import wood_multilevel as ml
from . import wood_reference_budget as budget
from .wood_correspondence_method import class_prototypes
from .wood_large_gallery_methods import class_scores


ARMS = ("random16", "hard16", "hard16_cross_scale", "hard32_cross_scale")
COMPARATORS = ("source", "cross_scale")


def _settings():
    cfg = {
        "seed": int(os.environ.get("WOOD_HARD_SEED", "43")),
        "epochs": int(os.environ.get("WOOD_HARD_EPOCHS", "3")),
        "episodes": int(os.environ.get("WOOD_HARD_EPISODES", "120")),
        "workers": int(os.environ.get("WOOD_HARD_WORKERS", "2")),
        "preload_workers": int(os.environ.get("WOOD_HARD_PRELOAD_WORKERS", "8")),
        "image_batch": int(os.environ.get("WOOD_HARD_IMAGE_BATCH", "64")),
        "microbatch": int(os.environ.get("WOOD_HARD_MICROBATCH", "4")),
        "hard_queries": int(os.environ.get("WOOD_HARD_QUERIES", "4")),
        "backbone_lr": float(os.environ.get("WOOD_HARD_BACKBONE_LR", "1e-6")),
        "head_lr": float(os.environ.get("WOOD_HARD_HEAD_LR", "5e-5")),
        "hard_weight": float(os.environ.get("WOOD_HARD_WEIGHT", "0.5")),
        "bank_weight": float(os.environ.get("WOOD_HARD_BANK_WEIGHT", "0.5")),
        "folds": [int(x) for x in os.environ.get("WOOD_HARD_FOLDS", "0,1").split(",")],
        "budgets": [int(x) for x in os.environ.get("WOOD_HARD_BUDGETS", "1,3,10,20").split(",")],
        "widths": [int(x) for x in os.environ.get("WOOD_HARD_WIDTHS", "32,128").split(",")],
        "arms": [x.strip() for x in os.environ.get(
            "WOOD_HARD_ARMS", ",".join(ARMS)).split(",")],
    }
    if (cfg["seed"] < 0 or cfg["epochs"] < 1 or cfg["episodes"] < 1 or
            cfg["workers"] < 0 or cfg["preload_workers"] < 1 or
            cfg["image_batch"] < 1 or
            cfg["microbatch"] < 1 or cfg["hard_queries"] < 1 or
            cfg["backbone_lr"] <= 0 or cfg["head_lr"] <= 0 or
            cfg["hard_weight"] < 0 or cfg["bank_weight"] < 0 or
            not cfg["arms"] or len(set(cfg["arms"])) != len(cfg["arms"]) or
            any(arm not in ARMS for arm in cfg["arms"]) or
            set(cfg["folds"]) != {0, 1} or len(cfg["folds"]) != 2 or
            not cfg["budgets"] or len(set(cfg["budgets"])) != len(cfg["budgets"]) or
            any(k < 1 or k > 20 for k in cfg["budgets"]) or
            not cfg["widths"] or len(set(cfg["widths"])) != len(cfg["widths"]) or
            any(w < 1 or w > 128 for w in cfg["widths"])):
        raise ValueError("Invalid hard-negative study settings")
    return cfg


def _source(root, base_out, image_out, seed, device):
    return large._source_checkpoint(root, base_out, image_out, seed, device)


def _provenance(root, source, cfg, arm):
    keys = ("seed", "epochs", "episodes", "microbatch", "hard_queries",
            "backbone_lr", "head_lr", "hard_weight", "bank_weight")
    return {"source_sha256": study._hash_file(source),
            "manifest_sha256": study._hash_file(root / "swi_manifest.json"),
            "runner_sha256": study._hash_file(__file__),
            "method_sha256": study._hash_file(Path(corr.__file__).with_name(
                "wood_correspondence_method.py")),
            "settings": {key: cfg[key] for key in keys},
            "arm": arm, "mining_split": "meta-train",
            "selection_split": "meta-val"}


def mine_negative_classes(query, bank_prototypes, positive_class, width,
                          random_control, rng):
    """Choose negatives *within* the same raw top-128 candidate pool."""
    scores = F.normalize(query.detach().float(), dim=-1) @ bank_prototypes.T
    scores = scores.clone()
    scores[int(positive_class)] = -torch.inf
    shortlist = scores.topk(128).indices
    if random_control:
        chosen = rng.choice(128, size=width - 1, replace=False)
        return shortlist[torch.as_tensor(chosen, device=shortlist.device)]
    return shortlist[:width - 1]


def _bank_features(encoder, representatives, base_cfg, device, path, checkpoint_hash):
    extract_cfg = dict(base_cfg, image_batch=base_cfg.get("image_batch", 64),
                       workers=base_cfg.get("workers", 2))
    globals_, tokens = large._cache_features(
        encoder, representatives, extract_cfg, device, path, checkpoint_hash, True)
    labels = sorted({study.canonical(label) for _, label in representatives})
    label_ids = {label: i for i, label in enumerate(labels)}
    ref_labels = torch.tensor([label_ids[study.canonical(label)]
                               for _, label in representatives], device=device)
    bank_g = torch.from_numpy(np.asarray(globals_).copy()).to(device)
    bank_t = torch.from_numpy(np.asarray(tokens).copy().astype(np.float32)).to(device)
    raw_prototypes, classes, _ = class_prototypes(
        F.normalize(bank_g.float(), dim=-1), ref_labels)
    if len(labels) < 129 or not torch.equal(
            classes, torch.arange(len(labels), device=device)):
        raise ValueError("Incomplete meta-train bank")
    scans = [study.source_scan_id(path) for path, _ in representatives]
    return labels, ref_labels, bank_g, bank_t, raw_prototypes, scans


def shortlist_loss(model, q, qt, r, rt, episode_labels, episode_scans,
                   species, bank_labels, bank_ref_labels, bank_g, bank_t, bank_scans,
                   bank_prototypes, width, random_control, query_count, rng,
                   config):
    """Backpropagate QKV ranking through query and in-episode positive images."""
    lookup = {label: i for i, label in enumerate(bank_labels)}
    chosen_queries = rng.choice(len(q), size=min(query_count, len(q)),
                                replace=False)
    negative_rng = np.random.default_rng(rng.integers(0, 2**63))
    losses, accuracies = [], []
    for row in chosen_queries:
        positive_class = lookup[species[int(row)]]
        negatives = mine_negative_classes(q[row], bank_prototypes, positive_class,
                                          width, random_control, negative_rng)
        mask = torch.isin(bank_ref_labels, negatives)
        bank_indices = mask.nonzero(as_tuple=True)[0]
        positive_indices = (torch.as_tensor(episode_labels, device=q.device) == row
                            ).nonzero(as_tuple=True)[0]
        refs = torch.cat((r[positive_indices], bank_g[bank_indices]), dim=0)
        tokens = torch.cat((rt[positive_indices], bank_t[bank_indices]), dim=0)
        labels = torch.cat((bank_ref_labels.new_full((len(positive_indices),),
                                                 positive_class),
                            bank_ref_labels[bank_indices]))
        scans = ([episode_scans[int(i)] for i in positive_indices.tolist()] +
                 [bank_scans[int(i)] for i in bank_indices.tolist()])
        scores, classes = model(
            q[row:row + 1], qt[row:row + 1], refs, tokens, labels, scans,
            mode="qkv", gate=config["gate"], consensus=config["consensus"],
            query_chunk=1, reference_chunk=config["reference_chunk"])
        target = torch.searchsorted(classes, labels.new_tensor([positive_class]))
        if target.item() >= len(classes) or classes[target.item()] != positive_class:
            raise ValueError("Positive class missing from QKV shortlist")
        losses.append(F.cross_entropy(scores / 0.07, target))
        accuracies.append(float((scores.argmax(dim=1) == target).float().mean().detach()))
    return torch.stack(losses).mean(), float(np.mean(accuracies))


def _step(model, encoder, images, layout, bank, arm, config, cfg,
          optimizer, device, rng):
    n_refs, labels, scans, species, _, _ = layout
    bank_labels, bank_ref_labels, bank_g, bank_t, prototypes, bank_scans = bank
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
    old_loss, old_accuracy = corr._loss(
        model, q, all_t[n_refs:], r, all_t[:n_refs], labels, scans, config)
    with torch.no_grad():
        adapted_bank, classes, _ = class_prototypes(
            corr._transformed(model, bank_g), bank_ref_labels)
    if len(classes) != len(bank_labels):
        raise ValueError("Incomplete adapted bank")
    bank_loss = large._bank_loss(model, q, r, labels, species, bank_labels,
                                 adapted_bank, 128, "hard_bank")
    hard_loss, hard_accuracy = shortlist_loss(
        model, q, all_t[n_refs:], r, all_t[:n_refs], labels, scans,
        species, bank_labels, bank_ref_labels, bank_g, bank_t, bank_scans,
        prototypes, 32 if arm == "hard32_cross_scale" else 16,
        arm == "random16", cfg["hard_queries"], rng, config)
    loss = old_loss + cfg["bank_weight"] * bank_loss + cfg["hard_weight"] * hard_loss
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite shortlist training loss")
    loss.backward()
    g_grad, t_grad = all_g.grad.detach(), all_t.grad.detach()
    if not torch.isfinite(g_grad).all() or not torch.isfinite(t_grad).all():
        raise RuntimeError("Non-finite encoded feature gradient")
    offset = 0
    for chunk in images.split(cfg["microbatch"]):
        size = len(chunk)
        g, t = corr._encode_images(encoder, chunk.to(device, non_blocking=True))
        torch.autograd.backward((g.float(), t.float()),
                                (g_grad[offset:offset + size],
                                 t_grad[offset:offset + size]))
        offset += size
    torch.nn.utils.clip_grad_norm_(
        list(model.parameters()) + list(encoder.parameters()), 1.0)
    optimizer.step()
    return float(loss.detach()), old_accuracy, hard_accuracy


def _train(root, out, manifest, base_out, image_out, cfg, arm, device):
    study._seed(cfg["seed"])
    source, encoder, model, base_cfg, recipe = _source(
        root, base_out, image_out, cfg["seed"], device)
    folder = out / "train" / arm
    folder.mkdir(parents=True, exist_ok=True)
    provenance = _provenance(root, source, cfg, arm)
    latest, best = folder / "latest.pt", folder / "best.pt"
    optimizer = torch.optim.AdamW([
        {"params": model.parameters(), "lr": cfg["head_lr"]},
        {"params": encoder.parameters(), "lr": cfg["backbone_lr"]}],
        weight_decay=1e-3)
    start, best_score = 1, -float("inf")
    if latest.is_file():
        saved = torch.load(latest, map_location="cpu", weights_only=False)
        if saved["provenance"] != provenance:
            raise ValueError("Training provenance changed; use a new output folder")
        encoder.load_state_dict(saved["encoder"], strict=True)
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        start, best_score = saved["epoch"] + 1, saved["best_score"]
        print(f"[wood-hard] resume {arm} at epoch={start}", flush=True)
    items = manifest["meta-train"]
    _, valid = image_train._layout(items)
    multi = {k: ml._eligible(items, k) for _, k in ml.SCHEDULE}
    if len(multi[5]) < 128:
        raise ValueError("Too few scan/crop-scale eligible meta-train species")
    dataset = data.ManifestDataset(
        items, transform=data.get_transforms(224, augment=True))
    dataset.loader = large._RetryImageLoader(dataset.loader)
    representatives = large._bank_representatives(manifest)
    bank_cfg = dict(base_cfg, image_batch=cfg["image_batch"], workers=cfg["workers"])
    for epoch in range(start, cfg["epochs"] + 1):
        prior = latest if epoch > 1 else source
        bank = _bank_features(
            encoder, representatives, bank_cfg, device,
            (out / "bank_features" / "shared_source.npy" if epoch == 1 else
             out / "bank_features" / arm / f"epoch_{epoch:02d}.npy"),
            study._hash_file(prior))
        plan_cfg = {"episodes": cfg["episodes"], "seed": cfg["seed"]}
        cross_scale = arm.endswith("cross_scale")
        batches, layouts = ml._plan(items, valid, cross_scale, plan_cfg,
                                     epoch, multi=multi)
        unique_images = len({i for batch in batches for i in batch})
        loader = DataLoader(
            dataset, batch_sampler=batches,
            num_workers=min(cfg["workers"], len(os.sched_getaffinity(0))
                            if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1),
            pin_memory=True)
        encoder.backbone.eval()
        encoder.projection.train()
        model.train()
        rng = np.random.default_rng(cfg["seed"] * 101 + epoch)
        losses, old_acc, shortlist_acc = [], [], []
        started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for step, ((images, _), layout) in enumerate(zip(loader, layouts), 1):
            loss, old, hard = _step(model, encoder, images, layout, bank, arm,
                                    recipe, cfg, optimizer, device, rng)
            losses.append(loss)
            old_acc.append(old)
            shortlist_acc.append(hard)
            if step % 10 == 0 or step == len(layouts):
                print(f"[wood-hard] {arm} epoch={epoch} step={step}/{len(layouts)} "
                      f"loss={np.mean(losses):.4f} episode_R1={np.mean(old_acc):.4f} "
                      f"shortlist_R1={np.mean(shortlist_acc):.4f} "
                      f"elapsed_s={time.perf_counter()-started:.1f}", flush=True)
        train_seconds = time.perf_counter() - started
        payload = {"provenance": provenance, "epoch": epoch,
                   "encoder": encoder.state_dict(), "model": model.state_dict()}
        candidate = folder / "candidate.pt"
        study._save_checkpoint(candidate, payload)
        frame, summary, timing = corr._evaluate(
            root, out / "early_stop" / arm, manifest, encoder, base_cfg,
            study._hash_file(candidate), model, recipe, device, folds=(0,),
            modes=("global", "trained"))
        score = corr._selected_score(summary, "trained")
        corr._save_report(folder, f"epoch_{epoch:02d}", frame, summary,
                          timing, {"provenance": provenance, "score": score,
                                   "unique_train_images": unique_images,
                                   "training_images_total": len(items),
                                   "shortlist_training_queries": cfg["episodes"] *
                                   cfg["hard_queries"],
                                   "train_seconds": train_seconds,
                                   "peak_gpu_gb": (torch.cuda.max_memory_allocated() / 2**30
                                                   if device.type == "cuda" else None)})
        if score > best_score:
            best_score = score
            study._save_checkpoint(best, {**payload, "selection_score": score})
        study._save_checkpoint(latest, {**payload,
                                        "optimizer": optimizer.state_dict(),
                                        "best_score": best_score})
    print(f"[wood-hard] {arm} epoch={epoch} val={score:.4f} "
              f"best={best_score:.4f} unique_images={unique_images}/{len(items)}",
              flush=True)


def _preload_training(manifest, cfg):
    if os.environ.get("WOOD_HARD_PRELOAD", "1") != "1":
        return
    items = manifest["meta-train"]
    _, valid = image_train._layout(items)
    multi = {k: ml._eligible(items, k) for _, k in ml.SCHEDULE}
    needed = set()
    plan_cfg = {"episodes": cfg["episodes"], "seed": cfg["seed"]}
    for epoch in range(1, cfg["epochs"] + 1):
        for cross_scale in {arm.endswith("cross_scale") for arm in cfg["arms"]}:
            batches, _ = ml._plan(items, valid, cross_scale, plan_cfg,
                                   epoch, multi=multi)
            needed.update(i for batch in batches for i in batch)
    paths = [items[i][0] for i in sorted(needed)]
    paths.extend(path for path, _ in large._bank_representatives(manifest))
    print(f"[wood-hard] preloading {len(set(paths))} planned training/bank images",
          flush=True)
    data.preload_image_cache(paths, max_workers=cfg["preload_workers"],
                             desc="Hard-shortlist planned images")


def _load(root, out, ml_out, base_out, image_out, cfg, arm, device):
    if arm in COMPARATORS:
        prior_cfg = json.loads((ml_out / "preflight.json").read_text())["settings"]
        path, encoder, model, base_cfg = ml._load(
            root, ml_out, base_out, image_out, prior_cfg, arm, device)
        return path, encoder, model, base_cfg
    source, encoder, model, base_cfg, _ = _source(
        root, base_out, image_out, cfg["seed"], device)
    path = out / "train" / arm / "best.pt"
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved["provenance"] != _provenance(root, source, cfg, arm):
        raise ValueError("Hard-negative checkpoint provenance mismatch")
    encoder.load_state_dict(saved["encoder"], strict=True)
    model.load_state_dict(saved["model"], strict=True)
    return path, encoder.eval(), model.eval(), base_cfg


def _reference_data(encoder, items, queries, base_cfg, device, cache,
                    checkpoint_hash):
    extract_cfg = dict(base_cfg, image_batch=base_cfg.get("image_batch", 64),
                       workers=base_cfg.get("workers", 2))
    rg, rt = large._cache_features(encoder, items, extract_cfg, device,
                                   cache / "references.npy", checkpoint_hash, True)
    qg, qt = large._cache_features(encoder, queries, extract_cfg, device,
                                   cache / "queries.npy", checkpoint_hash, True)
    return (torch.from_numpy(np.asarray(rg).copy()).to(device),
            torch.from_numpy(np.asarray(rt).copy().astype(np.float32)).to(device),
            torch.from_numpy(np.asarray(qg).copy()).to(device),
            torch.from_numpy(np.asarray(qt).copy().astype(np.float32)).to(device))


def _summary_row(arm, fold, count, width, method, scores, truth,
                 candidate=None, elapsed=None):
    result = {"arm": arm, "fold": fold, "budget": count, "width": width,
              "method": method, **ml._metrics(scores, truth),
              "ms_per_query": elapsed, "candidate_macro_recall": None,
              "conditional_macro_r1": None}
    if candidate is not None:
        hit = (candidate == truth[:, None]).any(dim=1)
        result["candidate_macro_recall"] = float(torch.stack([
            hit[truth == c].float().mean() for c in truth.unique()]).mean())
        included = [
            (scores.argmax(1)[(truth == c) & hit] == c).float().mean()
            for c in truth.unique() if ((truth == c) & hit).any()]
        result["conditional_macro_r1"] = (
            float(torch.stack(included).mean()) if included else None)
    return result


def _validate_arm(root, out, ml_out, manifest, base_out, image_out,
                  cfg, arm, device):
    folder = out / "validation" / arm
    folder.mkdir(parents=True, exist_ok=True)
    path, encoder, model, base_cfg = _load(
        root, out, ml_out, base_out, image_out, cfg, arm, device)
    checkpoint_hash = study._hash_file(path)
    protocol = {"checkpoint_sha256": checkpoint_hash, "folds": cfg["folds"],
                "budgets": cfg["budgets"], "widths": cfg["widths"],
                "reference_policy": "scan_scale", "shortlist": "raw_prototype",
                "meta_val_only": True}
    protocol_path = folder / "protocol.json"
    if protocol_path.is_file():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("Validation protocol changed; use a new output folder")
        if all((folder / name).is_file() for name in ("summary.csv", "per_query.csv")):
            print(f"[wood-hard] reusing validation {arm}", flush=True)
            return
    if arm in COMPARATORS:
        previous = ml_out / "validation" / arm
        prior = json.loads((previous / "protocol.json").read_text())
        if (prior["checkpoint_sha256"] != checkpoint_hash or
                prior["folds"] != cfg["folds"] or
                not set(cfg["budgets"]).issubset(prior["budgets"]) or
                not set(cfg["widths"]).issubset(prior["widths"])):
            raise ValueError("Previous validation is not a matched comparator")
        summary = pd.read_csv(previous / "summary.csv")
        per_query = pd.read_csv(previous / "per_query.csv")
        summary = summary[summary.budget.isin(cfg["budgets"]) &
                          ((summary.width == 0) | summary.width.isin(cfg["widths"]))]
        per_query = per_query[per_query.budget.isin(cfg["budgets"]) &
                              ((per_query.width == 0) |
                               per_query.width.isin(cfg["widths"]))]
        if (set(summary.fold) != {0, 1} or
                len(per_query) != 570 * len(summary) // 2):
            raise ValueError("Previous validation evidence is incomplete")
        summary.to_csv(folder / "summary.csv", index=False)
        per_query.to_csv(folder / "per_query.csv", index=False)
        study._json(protocol_path, protocol)
        return
    rows, query_rows = [], []
    for fold in cfg["folds"]:
        _, queries = study.validation_items(
            manifest, group_mode="scan_disjoint", fold=fold)
        refs = budget._reference_pool(manifest, queries)
        class_names = sorted({label for _, label in refs})
        if len(class_names) != 637 or len(queries) != 285:
            raise ValueError("Meta-val cohort changed")
        class_ids = {label: i for i, label in enumerate(class_names)}
        selected = budget.select_indices(refs, max(cfg["budgets"]), "scan_scale")
        selected_items = [refs[i] for i in selected]
        rg, rt, qg, qt = _reference_data(
            encoder, selected_items, queries, dict(base_cfg,
                image_batch=cfg["image_batch"], workers=cfg["workers"]), device,
            out / "features" / arm / f"fold_{fold}", checkpoint_hash)
        truth = torch.tensor([class_ids[study.canonical(label)]
                              for _, label in queries], device=device)
        position = {index: i for i, index in enumerate(selected)}
        for count in cfg["budgets"]:
            chosen = budget.select_indices(refs, count, "scan_scale")
            ids = torch.tensor([position[i] for i in chosen], device=device)
            items = [refs[i] for i in chosen]
            ref_labels = torch.tensor([class_ids[label] for _, label in items],
                                      device=device)
            scans = [study.source_scan_id(path) for path, _ in items]
            target_scans = defaultdict(set)
            for (path, label) in items:
                target_scans[label].add(study.source_scan_id(path))
            if any(study.source_scan_id(path) in target_scans[
                    study.canonical(label)] for path, label in queries):
                raise ValueError("Meta-val source-scan leakage")
            with torch.inference_mode():
                raw, _ = class_scores(qg, rg[ids], ref_labels)
                adapted, _ = class_scores(corr._transformed(model, qg),
                                          corr._transformed(model, rg[ids]),
                                          ref_labels)
            def record(method, width, scores, candidates=None):
                predictions = scores.argmax(dim=1).tolist()
                hit = ((candidates == truth[:, None]).any(dim=1).tolist()
                       if candidates is not None else [None] * len(queries))
                for i, (path, label) in enumerate(queries):
                    query_rows.append({
                        "arm": arm, "fold": fold, "budget": count,
                        "width": width, "method": method, "query_path": path,
                        "true_label": study.canonical(label),
                        "predicted_label": class_names[predictions[i]],
                        "correct": int(predictions[i] == int(truth[i])),
                        "candidate_hit": hit[i],
                        "query_scale": study.image_scale(path)})
            for method, scores in (("raw_prototype", raw),
                                   ("adapted_prototype", adapted)):
                rows.append(_summary_row(arm, fold, count, 0, method,
                                         scores, truth, elapsed=0.0))
                record(method, 0, scores)
            for width in cfg["widths"]:
                candidates = raw.topk(width, dim=1).indices
                started = time.perf_counter()
                with torch.inference_mode():
                    scores = ml.rerank_all_references(
                        model, qg, qt, rg[ids], rt[ids], ref_labels,
                        scans, candidates, adapted)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = 1000 * (time.perf_counter() - started) / len(qg)
                rows.append(_summary_row(
                    arm, fold, count, width, "qkv_all_refs", scores, truth,
                    candidates, elapsed))
                record("qkv_all_refs", width, scores, candidates)
                print(f"[wood-hard] {arm} fold={fold} K={count} width={width} "
                      f"candidate={rows[-1]['candidate_macro_recall']:.4f} "
                      f"R1={rows[-1]['macro_r1']:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(folder / "summary.csv", index=False)
    pd.DataFrame(query_rows).to_csv(folder / "per_query.csv", index=False)
    study._json(protocol_path, protocol)


def _select(out, cfg):
    arms = list(COMPARATORS) + cfg["arms"]
    frames = []
    for arm in arms:
        folder = out / "validation" / arm
        protocol = json.loads((folder / "protocol.json").read_text())
        if (protocol["folds"] != cfg["folds"] or
                protocol["budgets"] != cfg["budgets"] or
                protocol["widths"] != cfg["widths"]):
            raise ValueError("Validation protocol differs across arms")
        frame = pd.read_csv(folder / "summary.csv")
        frame = frame[frame.method.isin(
            ("raw_prototype", "adapted_prototype", "qkv_all_refs"))]
        if set(frame.fold) != {0, 1}:
            raise ValueError("Incomplete validation folds")
        frames.append(frame)
    all_rows = pd.concat(frames, ignore_index=True)
    grouped = all_rows.groupby(["arm", "budget", "width", "method"],
                               as_index=False).agg(
        macro_r1=("macro_r1", "mean"), n_folds=("fold", "nunique"))
    if (grouped.n_folds != 2).any():
        raise ValueError("Missing matched validation result")
    best = grouped.sort_values(
        ["macro_r1", "budget", "width", "arm", "method"],
        ascending=[False, True, True, True, True]).iloc[0]
    lock = {"selection_split": "meta-val", "test_evaluated": False,
            "public_id_evaluated": False,
            "criterion": "highest two-fold macro R@1; lower K on ties",
            "validation_sha256": {arm: study._hash_file(
                out / "validation" / arm / "summary.csv") for arm in arms},
            "protocol_sha256": {arm: study._hash_file(
                out / "validation" / arm / "protocol.json") for arm in arms},
            "selected": {"arm": str(best.arm), "budget": int(best.budget),
                         "width": int(best.width), "method": str(best.method),
                         "meta_val_macro_r1": float(best.macro_r1)}}
    path = out / "selection_lock.json"
    if path.is_file() and json.loads(path.read_text()) != lock:
        raise ValueError("Selection lock already exists with different evidence")
    study._json(path, lock)
    print(f"[wood-hard] locked meta-val selection: {lock['selected']}", flush=True)


def _locked(out):
    lock = json.loads((out / "selection_lock.json").read_text())
    if lock["test_evaluated"] or lock["public_id_evaluated"]:
        raise ValueError("Selection was not locked before external evaluation")
    for arm, digest in lock["validation_sha256"].items():
        if study._hash_file(out / "validation" / arm / "summary.csv") != digest:
            raise ValueError("Validation evidence changed after selection")
        if study._hash_file(out / "validation" / arm / "protocol.json") != lock[
                "protocol_sha256"][arm]:
            raise ValueError("Validation checkpoint changed after selection")
    return lock["selected"]


def query_scale_gallery(items, class_names, query_scale):
    """Select one reference per class using only the observable query crop width."""
    by_class = defaultdict(list)
    for index, (path, label) in enumerate(items):
        by_class[label].append((index, study.image_scale(path)))
    same, different, eligible = [], [], []
    for label in class_names:
        choices = by_class[label]
        if not choices:
            raise ValueError("Missing gallery species")
        matched = [index for index, scale in choices if scale == query_scale]
        mismatched = [index for index, scale in choices if scale != query_scale]
        same.append(matched[0] if matched else choices[0][0])
        different.append(mismatched[0] if mismatched else choices[0][0])
        eligible.append(bool(matched and mismatched))
    return same, different, eligible


def _stress_arm(root, out, ml_out, manifest, base_out, image_out,
                cfg, arm, device):
    path, encoder, model, base_cfg = _load(
        root, out, ml_out, base_out, image_out, cfg, arm, device)
    checkpoint_hash = study._hash_file(path)
    folder = out / "scale_stress" / arm
    folder.mkdir(parents=True, exist_ok=True)
    protocol = {"checkpoint_sha256": checkpoint_hash,
                "selection_lock_sha256": study._hash_file(out / "selection_lock.json"),
                "folds": cfg["folds"], "reference_pool_budget": 20,
                "gallery_species": 637, "reference_per_species": 1,
                "candidate_width": 128,
                "policies": ["same_query_scale", "different_query_scale"],
                "gallery_policy_uses_true_label": False}
    if (folder / "protocol.json").is_file():
        if json.loads((folder / "protocol.json").read_text()) != protocol:
            raise ValueError("Scale-stress protocol changed")
        if (folder / "summary.csv").is_file() and (folder / "per_query.csv").is_file():
            print(f"[wood-hard] reusing scale stress {arm}", flush=True)
            return
    rows = []
    for fold in cfg["folds"]:
        _, queries = study.validation_items(
            manifest, group_mode="scan_disjoint", fold=fold)
        refs = budget._reference_pool(manifest, queries)
        class_names = sorted({label for _, label in refs})
        if len(class_names) != 637 or len(queries) != 285:
            raise ValueError("Meta-val scale-stress cohort changed")
        ids = budget.select_indices(refs, 20, "scan_scale")
        items = [refs[i] for i in ids]
        reference_scans = defaultdict(set)
        for path, label in items:
            reference_scans[label].add(study.source_scan_id(path))
        if any(study.source_scan_id(path) in reference_scans[
                study.canonical(label)] for path, label in queries):
            raise ValueError("Scale-stress source-scan leakage")
        feature_root = (
            ((ml_out if arm in COMPARATORS else out) / "features")
            if max(cfg["budgets"]) == 20 else out / "scale_stress_features"
        ) / arm / f"fold_{fold}"
        rg, rt, qg, qt = _reference_data(
            encoder, items, queries, dict(base_cfg,
                image_batch=cfg["image_batch"], workers=cfg["workers"]),
            device, feature_root, checkpoint_hash)
        label_ids = {label: i for i, label in enumerate(class_names)}
        for scale in sorted({study.image_scale(path) for path, _ in queries}):
            query_ids = [i for i, (path, _) in enumerate(queries)
                         if study.image_scale(path) == scale]
            same, different, eligible = query_scale_gallery(items, class_names, scale)
            for policy, ref_indices in (("same_query_scale", same),
                                        ("different_query_scale", different)):
                ref_ids = torch.tensor(ref_indices, device=device)
                reference_labels = torch.arange(len(class_names), device=device)
                gallery_scans = [study.source_scan_id(items[i][0]) for i in ref_indices]
                for first in range(0, len(query_ids), 16):
                    current = query_ids[first:first + 16]
                    q = qg[current]
                    qt_part = qt[current]
                    with torch.inference_mode():
                        raw, _ = class_scores(q, rg[ref_ids], reference_labels)
                        adapted, _ = class_scores(
                            corr._transformed(model, q),
                            corr._transformed(model, rg[ref_ids]),
                            reference_labels)
                        shortlist = raw.topk(128, dim=1).indices
                        reranked = ml.rerank_all_references(
                            model, q, qt_part, rg[ref_ids], rt[ref_ids],
                            reference_labels, gallery_scans, shortlist, adapted)
                    for method, scores in (("raw_prototype", raw),
                                           ("qkv_all_refs", reranked)):
                        predictions = scores.argmax(dim=1).tolist()
                        for local, original in enumerate(current):
                            path_q, label_q = queries[original]
                            true_id = label_ids[study.canonical(label_q)]
                            rows.append({
                                "arm": arm, "fold": fold, "policy": policy,
                                "method": method, "query_path": path_q,
                                "true_label": study.canonical(label_q),
                                "query_scale": scale,
                                "paired_eligible": int(eligible[true_id]),
                                "reference_scale": study.image_scale(
                                    items[ref_indices[true_id]][0]),
                                "predicted_label": class_names[predictions[local]],
                                "correct": int(predictions[local] == true_id),
                                "candidate_hit": int((shortlist[local] == true_id).any())})
    frame = pd.DataFrame(rows)
    if len(frame) != 2 * 2 * 2 * 285:
        raise ValueError("Scale-stress query grid is incomplete")
    frame.to_csv(folder / "per_query.csv", index=False)
    summary = []
    for (fold, policy, method), part in frame.groupby(["fold", "policy", "method"]):
        for cohort, selected in (("all", part),
                                 ("paired_eligible", part[part.paired_eligible == 1])):
            by_species = selected.groupby("true_label").agg(
                r1=("correct", "mean"), candidate=("candidate_hit", "mean"))
            summary.append({"arm": arm, "fold": fold, "policy": policy,
                            "method": method, "cohort": cohort,
                            "n_queries": len(selected),
                            "n_species": len(by_species),
                            "macro_r1": by_species.r1.mean(),
                            "candidate_macro_recall": by_species.candidate.mean()})
    pd.DataFrame(summary).to_csv(folder / "summary.csv", index=False)
    study._json(folder / "protocol.json", protocol)
    print(f"[wood-hard] scale-stress {arm}: {folder / 'summary.csv'}", flush=True)


def _smoke(root, out, manifest, base_out, image_out, cfg, device):
    source, encoder, model, base_cfg, recipe = _source(
        root, base_out, image_out, cfg["seed"], device)
    items = manifest["meta-train"]
    eligible = ml._eligible(items)
    by_class, _ = image_train._layout(items)
    labels = sorted(eligible)[:4]
    if len(labels) != 4:
        raise ValueError("Smoke requires four cross-scale eligible classes")
    refs, queries, scans, ref_scales, query_scales = [], [], [], [], []
    for label in labels:
        query = eligible[label][0]
        query_scan = study.source_scan_id(items[query][0])
        query_scale = study.image_scale(items[query][0])
        reference = next(
            index for scan, rows in by_class[label].items() if scan != query_scan
            for index in rows if study.image_scale(items[index][0]) != query_scale)
        refs.append(reference)
        queries.append(query)
        scans.append(study.source_scan_id(items[reference][0]))
        ref_scales.append(study.image_scale(items[reference][0]))
        query_scales.append(query_scale)
    smoke_items = [items[i] for i in refs + queries]
    dataset = data.ManifestDataset(
        smoke_items, transform=data.get_transforms(224, augment=True))
    dataset.loader = large._RetryImageLoader(dataset.loader)
    images, _ = next(iter(DataLoader(dataset, batch_size=8, num_workers=0)))
    representatives = large._bank_representatives(manifest)
    represented = sorted({study.canonical(label) for _, label in representatives})
    included = set(labels) | set(represented[:129])
    small_bank = [(path, label) for path, label in representatives
                  if study.canonical(label) in included]
    bank = _bank_features(
        encoder, small_bank, dict(base_cfg, image_batch=cfg["image_batch"],
                                  workers=cfg["workers"]), device,
        out / "smoke_bank.npy", study._hash_file(source))
    optimizer = torch.optim.AdamW([
        {"params": model.parameters(), "lr": cfg["head_lr"]},
        {"params": encoder.parameters(), "lr": cfg["backbone_lr"]}])
    encoder.backbone.eval()
    encoder.projection.train()
    model.train()
    local_cfg = dict(cfg, microbatch=min(2, cfg["microbatch"]), hard_queries=4)
    loss, old_acc, hard_acc = _step(
        model, encoder, images,
        (4, list(range(4)), scans, labels, ref_scales, query_scales),
        bank, "hard16_cross_scale", recipe, local_cfg, optimizer, device,
        np.random.default_rng(cfg["seed"]))
    backbone_grad = encoder.backbone.blocks[-1].attn.qkv.weight.grad
    qkv_grad = model.query_key.weight.grad
    if (backbone_grad is None or qkv_grad is None or
            not torch.isfinite(backbone_grad).all() or
            not torch.isfinite(qkv_grad).all() or
            backbone_grad.norm() <= 0 or qkv_grad.norm() <= 0):
        raise RuntimeError("Hard-shortlist smoke did not reach encoder and QKV")
    out.mkdir(parents=True, exist_ok=True)
    report = {"source_checkpoint_sha256": study._hash_file(source),
              "loss": loss, "episode_r1": old_acc,
              "shortlist_r1": hard_acc, "n_images": 8,
              "n_bank_species": len(bank[0]),
              "encoder_attention_grad_norm": float(backbone_grad.norm()),
              "qkv_grad_norm": float(qkv_grad.norm()),
              "training_checkpoint_saved": False}
    study._json(out / "smoke.json", report)
    print(f"[wood-hard] real-image smoke passed: {report}", flush=True)


def run():
    root = Path(os.environ.get("ROOT_PATH", "/content/drive/MyDrive/NCS")).resolve()
    out = Path(os.environ.get(
        "WOOD_HARD_OUT", root / "results" / "wood_hard_shortlist_study_v1")).resolve()
    ml_out = Path(os.environ.get(
        "WOOD_ML_OUT", root / "results" / "wood_multilevel_study_v1")).resolve()
    base_out = Path(os.environ.get(
        "WOOD_LARGE_BASE_OUT", root / "results" /
        "dinov2s_retrieval_matched_v1")).resolve()
    image_out = Path(os.environ.get(
        "WOOD_LARGE_QKV_OUT", root / "results" /
        "wood_correspondence_image_full_v1")).resolve()
    cfg = _settings()
    mode = os.environ.get("WOOD_HARD_MODE", "preflight")
    if mode not in {"preflight", "smoke", "train", "validate", "select",
                    "scale_stress"}:
        raise ValueError("Invalid WOOD_HARD_MODE")
    manifest = data.load_swi_manifest(root / "swi_manifest.json")
    if mode == "preflight":
        if not (ml_out / "selection_lock.json").is_file():
            raise FileNotFoundError("Prior multilevel validation lock is required")
        ml._locked_selection(ml_out)
        if not (image_out / "qkv" / f"seed_{cfg['seed']}" / "best.pt").is_file():
            raise FileNotFoundError("Image-trained QKV source checkpoint is required")
        audit = {"meta_train_images": len(manifest["meta-train"]),
                 "meta_train_species": len({study.canonical(label) for _, label
                                             in manifest["meta-train"]}),
                 "cross_scale_eligible_species": len(ml._eligible(
                     manifest["meta-train"])),
                 "prior_lock_sha256": study._hash_file(
                     ml_out / "selection_lock.json"),
                 "settings": cfg}
        out.mkdir(parents=True, exist_ok=True)
        study._json(out / "preflight.json", audit)
        print(f"[wood-hard] preflight: {audit}", flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Image training/evaluation requires CUDA")
    device = torch.device("cuda")
    if mode == "smoke":
        _smoke(root, out, manifest, base_out, image_out, cfg, device)
    elif mode == "train":
        _preload_training(manifest, cfg)
        for arm in cfg["arms"]:
            _train(root, out, manifest, base_out, image_out, cfg, arm, device)
    elif mode == "validate":
        for arm in list(COMPARATORS) + cfg["arms"]:
            _validate_arm(root, out, ml_out, manifest, base_out, image_out,
                          cfg, arm, device)
    elif mode == "select":
        _select(out, cfg)
    else:
        _locked(out)
        for arm in list(COMPARATORS) + cfg["arms"]:
            _stress_arm(root, out, ml_out, manifest, base_out, image_out,
                        cfg, arm, device)


if __name__ == "__main__":
    run()
