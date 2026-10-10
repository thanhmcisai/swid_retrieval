"""Isolated, scan-disjoint all-class correspondence study for Colab.

Validation selects a model; meta-test is inaccessible until an explicit lock.
The old submission outputs and the source checkpoint are never overwritten.
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

from . import data, gallery_experiment as study, wood_evidence_experiment as evidence
from .gallery_method import supervised_contrastive_loss
from .wood_correspondence_method import WoodCorrespondence, class_prototypes
from .wood_evidence_method import WoodEvidenceReranker


VARIANTS = {
    "qkv": {},
    "global_only": {"score_mode": "global"},
    "maxsim": {"score_mode": "maxsim"},
    "no_gate": {"gate": False},
    "no_consensus": {"consensus": False},
    "no_global_aux": {"global_aux": 0.0},
    "no_hard_negatives": {"hard_negatives": False},
    "no_cross_scan": {"cross_scan": False},
}


def recipe(variant, stage):
    if variant not in VARIANTS or stage not in {"head", "joint"}:
        raise ValueError("Unknown correspondence variant or stage")
    config = {"variant": variant, "stage": stage, "score_mode": "qkv",
              "gate": True, "consensus": True, "hard_negatives": True,
              "cross_scan": True, "global_aux": 0.5, "supcon": 0.05,
              "ways": [16, 32, 64] if stage == "head" else [16, 32],
              "reference_counts": [1, 2],
              "epochs": int(os.environ.get("WOOD_CORR_HEAD_EPOCHS", "4") if stage == "head"
                            else os.environ.get("WOOD_CORR_JOINT_EPOCHS", "2")),
              "episodes": int(os.environ.get("WOOD_CORR_HEAD_EPISODES", "150") if stage == "head"
                              else os.environ.get("WOOD_CORR_JOINT_EPISODES", "50")),
              "head_lr": float(os.environ.get("WOOD_CORR_HEAD_LR", "3e-4")),
              "backbone_lr": float(os.environ.get("WOOD_CORR_BACKBONE_LR", "1e-5")),
              "last_blocks": int(os.environ.get("WOOD_CORR_LAST_BLOCKS", "2")),
              "image_microbatch": int(os.environ.get("WOOD_CORR_MICROBATCH", "8")),
              "workers": int(os.environ.get("WOOD_CORR_WORKERS", "2")),
              "query_chunk": int(os.environ.get("WOOD_CORR_QUERY_CHUNK", "8")),
              "reference_chunk": int(os.environ.get("WOOD_CORR_REFERENCE_CHUNK", "64"))}
    config.update(VARIANTS[variant])
    if (config["epochs"] < 1 or config["episodes"] < 1 or
            config["last_blocks"] < 1 or config["image_microbatch"] < 1 or
            config["workers"] < 0 or config["head_lr"] <= 0 or
            config["backbone_lr"] <= 0 or config["query_chunk"] < 1 or
            config["reference_chunk"] < 1):
        raise ValueError("Invalid correspondence schedule")
    return config


def _provenance(root, base_path, config, seed, stage, parent=None):
    return {"base_sha256": study._hash_file(base_path),
            "manifest_sha256": study._hash_file(root / "swi_manifest.json"),
            "method_sha256": study._hash_file(Path(__file__).with_name(
                "wood_correspondence_method.py")),
            "experiment_sha256": study._hash_file(__file__),
            "config": config, "seed": seed, "stage": stage,
            "parent_sha256": study._hash_file(parent) if parent else None}


def _paths(out, stage, variant, seed):
    folder = out / stage / variant / f"seed_{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder, folder / "best.pt", folder / "latest.pt"


def _episode(by_class, items, rng, n_way, k, hard, cross_scan):
    if cross_scan:
        return evidence._episode_indices(by_class, items, rng, n_way, k, hard)
    classes = [label for label, groups in by_class.items()
               if sum(len(rows) for rows in groups.values()) >= k + 1]
    if len(classes) < n_way:
        raise ValueError("Not enough species with distinct images for no-cross-scan episode")
    anchor = classes[int(rng.integers(len(classes)))]
    chosen = [anchor] + [x for x in hard[anchor] if x in classes and x != anchor][:n_way // 2]
    remaining = [x for x in classes if x not in chosen]
    chosen += rng.choice(remaining, size=n_way - len(chosen), replace=False).tolist()
    rng.shuffle(chosen)
    refs, queries, labels, scans = [], [], [], []
    for index, label in enumerate(chosen):
        pool = [i for rows in by_class[label].values() for i in rows]
        picked = rng.choice(pool, size=k + 1, replace=False)
        refs.extend(picked[:k].tolist())
        queries.append(int(picked[-1]))
        labels.extend([index] * k)
        scans.extend(study.source_scan_id(items[i][0]) for i in picked[:k])
    return refs, queries, labels, scans


def _transformed(model, embeddings):
    return F.normalize(embeddings.float() + model.global_adapter(embeddings.float()), dim=-1)


def _loss(model, q, qt, r, rt, labels, scans, config):
    label_t = torch.as_tensor(labels, device=q.device)
    main, classes = model(q, qt, r, rt, label_t, scans,
                          mode=config["score_mode"], gate=config["gate"],
                          consensus=config["consensus"],
                          query_chunk=config["query_chunk"],
                          reference_chunk=config["reference_chunk"])
    targets = torch.arange(len(q), device=q.device)
    if not torch.equal(classes, targets):
        raise ValueError("Episode class ordering changed")
    loss = F.cross_entropy(main / 0.07, targets)
    if config["global_aux"]:
        global_scores, _ = model(q, qt, r, rt, label_t, scans, mode="global")
        loss = loss + config["global_aux"] * F.cross_entropy(global_scores / 0.07, targets)
    if config["supcon"]:
        all_embeddings = torch.cat((_transformed(model, r), _transformed(model, q)))
        all_labels = torch.cat((label_t, targets))
        loss = loss + config["supcon"] * supervised_contrastive_loss(
            all_embeddings, all_labels, temperature=0.07)
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite correspondence training loss")
    return loss, float((main.argmax(dim=1) == targets).float().mean().detach())


def _features_for_split(manifest, split, fold, encoder, cfg, cache_out, device, base_hash):
    refs, queries, extras = evidence._validation_sets(manifest, split, fold)
    all_refs = refs + extras
    plans = evidence._gallery_plan(refs, queries, extras, split)
    ref_global, ref_tokens = evidence._features(
        cache_out, f"{split}_fold{fold}_refs", all_refs, encoder, cfg, device,
        base_hash)
    query_global, query_tokens = evidence._features(
        cache_out, f"{split}_fold{fold}_queries", queries, encoder, cfg, device,
        base_hash)
    return all_refs, queries, plans, (ref_global, ref_tokens), (query_global, query_tokens)


def _ranks(scores, truth):
    ordered = torch.argsort(scores, dim=1, descending=True, stable=True)
    return (ordered == truth[:, None]).long().argmax(dim=1) + 1


def _evaluate(root, out, manifest, encoder, cfg, base_hash, model, config, device,
              *, split="meta-val", folds=(0, 1), modes=None):
    if modes is None:
        modes = ("global", "fixed_local", "global_adapt", "maxsim", "qkv_all", "qkv_top128",
                 "qkv_no_gate", "qkv_no_consensus")
    cache_out = out / "joint_feature_cache" if config["stage"] == "joint" else Path(
        os.environ.get("WOOD_CORR_FEATURE_OUT", root / "results" / "wood_evidence_pilot_v1"))
    query_rows, timing_rows = [], []
    model.eval()
    encoder.eval()
    fixed_local = WoodEvidenceReranker().to(device).eval() if "fixed_local" in modes else None
    with torch.no_grad():
        for fold in folds:
            all_refs, queries, plans, ref_features, query_features = _features_for_split(
                manifest, split, fold, encoder, cfg, cache_out, device, base_hash)
            for gallery, (ri, qi) in plans.items():
                if not (gallery.endswith("x5") or gallery.endswith("x1")):
                    continue
                selected_refs = [all_refs[i] for i in ri]
                selected_queries = [queries[i] for i in qi]
                labels = sorted({study.canonical(label) for _, label in selected_refs})
                label_ids = {label: i for i, label in enumerate(labels)}
                reference_labels = torch.tensor(
                    [label_ids[study.canonical(label)] for _, label in selected_refs],
                    device=device)
                scans = [study.source_scan_id(path) for path, _ in selected_refs]
                by_class_scans = defaultdict(set)
                for scan, (_, label) in zip(scans, selected_refs):
                    by_class_scans[study.canonical(label)].add(scan)
                if any(study.source_scan_id(path) in by_class_scans[study.canonical(label)]
                       for path, label in selected_queries):
                    raise ValueError("Reference/query source-scan leakage")
                rg = torch.from_numpy(ref_features[0][ri]).to(device)
                rt = torch.from_numpy(ref_features[1][ri]).to(device)
                qg = torch.from_numpy(query_features[0][qi]).to(device)
                qt = torch.from_numpy(query_features[1][qi]).to(device)
                truth = torch.tensor([label_ids[study.canonical(label)]
                                      for _, label in selected_queries], device=device)
                raw_prototypes, classes, _ = class_prototypes(rg.float(), reference_labels)
                if not torch.equal(classes, torch.arange(len(labels), device=device)):
                    raise ValueError("Gallery class indexing changed")
                raw = qg.float() @ raw_prototypes.T
                raw_ranks = _ranks(raw, truth).cpu().numpy()
                adapted, _ = model(qg, qt, rg, rt, reference_labels, scans,
                                   mode="global")
                adapted_ranks = _ranks(adapted, truth).cpu().numpy()
                chunk = config["query_chunk"]
                for mode in modes:
                    start = time.perf_counter()
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                    parts = []
                    for first in range(0, len(qg), chunk):
                        q, t = qg[first:first + chunk], qt[first:first + chunk]
                        if mode == "global":
                            score = raw[first:first + chunk]
                        elif mode == "fixed_local":
                            score, actual_classes = fixed_local(
                                q, t, rg, rt, reference_labels, scans,
                                top_classes=128, weight_override=0.25)
                            if not torch.equal(actual_classes, classes):
                                raise ValueError("Fixed local score/class alignment changed")
                        else:
                            score_mode = (config["score_mode"] if mode == "trained" else
                                          "global" if mode == "global_adapt" else
                                          "maxsim" if mode == "maxsim" else "qkv")
                            score, actual_classes = model(
                                q, t, rg, rt, reference_labels, scans, mode=score_mode,
                                top_classes=128 if mode == "qkv_top128" else None,
                                gate=config["gate"] if mode == "trained" else
                                     mode != "qkv_no_gate",
                                consensus=config["consensus"] if mode == "trained" else
                                          mode != "qkv_no_consensus",
                                query_chunk=chunk,
                                reference_chunk=config["reference_chunk"])
                            if not torch.equal(actual_classes, classes):
                                raise ValueError("Score/class alignment changed")
                        parts.append(score.detach())
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start
                    scores = torch.cat(parts)
                    ranks = _ranks(scores, truth).cpu().numpy()
                    timing_rows.append({"split": split, "fold": fold, "gallery": gallery,
                                        "mode": mode, "n_queries": len(qg),
                                        "score_ms_per_query": 1000 * elapsed / len(qg)})
                    for j, ((path, label), rank, base_rank, adapt_rank) in enumerate(
                            zip(selected_queries, ranks, raw_ranks, adapted_ranks)):
                        query_rows.append({"split": split, "fold": fold,
                                           "gallery": gallery, "mode": mode,
                                           "query_path": path,
                                           "true_label": study.canonical(label),
                                           "query_scan": study.source_scan_id(path),
                                           "query_scale": study.image_scale(path),
                                           "true_ref_scans": len(by_class_scans[study.canonical(label)]),
                                           "rank": int(rank), "correct": int(rank == 1),
                                           "baseline_rank": int(base_rank),
                                           "adapted_global_rank": int(adapt_rank),
                                           "candidate_r5": int(base_rank <= 5),
                                           "candidate_r32": int(base_rank <= 32),
                                           "candidate_r128": int(base_rank <= 128),
                                           "adapted_candidate_r128": int(adapt_rank <= 128),
                                           "predicted_label": labels[int(scores[j].argmax())],
                                           "same_genus_error": int(
                                               rank != 1 and labels[int(scores[j].argmax())].split("_", 1)[0] ==
                                               study.canonical(label).split("_", 1)[0])})
                    print(f"[wood-corr] {split} fold={fold} {gallery} {mode} "
                          f"R@1={np.mean(ranks == 1):.4f} "
                          f"ms/query={1000 * elapsed / len(qg):.2f}", flush=True)
    frame = pd.DataFrame(query_rows)
    summary = frame.groupby(["split", "gallery", "mode"], as_index=False).agg(
        macro_r1=("correct", lambda s: s.groupby(frame.loc[s.index, "true_label"]).mean().mean()),
        macro_r5=("rank", lambda s: s.le(5).groupby(frame.loc[s.index, "true_label"]).mean().mean()),
        n_queries=("correct", "size"), n_species=("true_label", "nunique"),
        candidate_r5=("candidate_r5", "mean"),
        candidate_r32=("candidate_r32", "mean"),
        candidate_r128=("candidate_r128", "mean"),
        adapted_candidate_r128=("adapted_candidate_r128", "mean"))
    return frame, summary, pd.DataFrame(timing_rows)


def _save_report(out, stem, frame, summary, timing, provenance):
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / f"{stem}_queries.csv", index=False)
    summary.to_csv(out / f"{stem}_summary.csv", index=False)
    folds = frame.groupby(["fold", "gallery", "mode"], as_index=False).agg(
        macro_r1=("correct", lambda s: s.groupby(frame.loc[s.index, "true_label"]).mean().mean()),
        macro_r5=("rank", lambda s: s.le(5).groupby(frame.loc[s.index, "true_label"]).mean().mean()),
        n_queries=("correct", "size"), n_species=("true_label", "nunique"))
    folds.to_csv(out / f"{stem}_folds.csv", index=False)
    timing.to_csv(out / f"{stem}_timing.csv", index=False)
    for column, suffix in (("query_scale", "scale"), ("true_ref_scans", "ref_scans")):
        subgroup = frame.groupby(["gallery", "mode", column], dropna=False).agg(
            n_queries=("correct", "size"), n_species=("true_label", "nunique"),
            micro_r1=("correct", "mean"), same_genus_errors=("same_genus_error", "sum"))
        subgroup.reset_index().to_csv(out / f"{stem}_{suffix}.csv", index=False)
    paired = []
    keys = ["split", "gallery", "fold", "query_path", "true_label"]
    for (gallery, mode), part in frame.groupby(["gallery", "mode"]):
        for baseline_mode in ("global", "global_adapt", "fixed_local"):
            if mode == baseline_mode:
                continue
            baseline = frame[(frame.gallery == gallery) & (frame["mode"] == baseline_mode)]
            if baseline.empty:
                continue
            joined = baseline[keys + ["correct"]].merge(
                part[keys + ["correct"]], on=keys, suffixes=("_baseline", "_new"),
                validate="one_to_one")
            if len(joined) != len(baseline) or len(joined) != len(part):
                raise ValueError("Unpaired or repeated evaluation query")
            delta = joined.assign(delta=joined.correct_new - joined.correct_baseline)
            species = delta.groupby("true_label")["delta"].mean().to_numpy()
            rng = np.random.default_rng(2026)
            boot = species[rng.integers(len(species), size=(2000, len(species)))].mean(axis=1)
            paired.append({"gallery": gallery, "baseline": baseline_mode, "mode": mode,
                           "macro_r1_delta": float(species.mean()),
                           "ci95_low": float(np.quantile(boot, 0.025)),
                           "ci95_high": float(np.quantile(boot, 0.975)),
                           "rescued": int(((joined.correct_baseline == 0) &
                                           (joined.correct_new == 1)).sum()),
                           "harmed": int(((joined.correct_baseline == 1) &
                                          (joined.correct_new == 0)).sum())})
    pd.DataFrame(paired).to_csv(out / f"{stem}_paired.csv", index=False)
    study._json(out / f"{stem}_provenance.json", provenance)


def _selected_score(summary, mode, fold=None):
    # Checkpoint selection is always on fold 0; fold 1 is descriptive only.
    indexed = summary.set_index(["gallery", "mode"])["macro_r1"]
    return (float(indexed["57x5", mode]) + float(indexed["637x1", mode])) / 2


def _head_train(root, out, feature_out, manifest, base_path, base_state, cfg,
                device, config, seed):
    study._seed(seed)
    encoder = evidence._encoder_from_state(base_state, cfg, device)
    items = evidence._representatives(manifest)
    global_np, tokens_np = evidence._features(
        feature_out, "meta_train_representatives", items, encoder, cfg, device,
        study._hash_file(base_path))
    globals_t = torch.from_numpy(global_np).to(device)
    tokens_t = torch.from_numpy(tokens_np).to(device)
    by_class = evidence._episode_layout(items)
    hard = evidence._hard_neighbors(items, global_np, by_class)
    if not config["hard_negatives"]:
        hard = {key: [] for key in hard}
    model = WoodCorrespondence().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["head_lr"], weight_decay=1e-3)
    folder, best, latest = _paths(out, "head", config["variant"], seed)
    provenance = _provenance(root, base_path, config, seed, "head")
    first_epoch, best_score = 1, -float("inf")
    if latest.is_file():
        saved = torch.load(latest, map_location="cpu", weights_only=False)
        if saved["provenance"] != provenance:
            raise ValueError(f"Head checkpoint provenance changed: {latest}")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        first_epoch, best_score = int(saved["epoch"]) + 1, float(saved["best_score"])
    for epoch in range(first_epoch, config["epochs"] + 1):
        rng = np.random.default_rng(seed * 100003 + epoch)
        model.train()
        losses, accuracies = [], []
        for _ in range(config["episodes"]):
            n_way = int(rng.choice(config["ways"]))
            k = int(rng.choice(config["reference_counts"]))
            ri, qi, labels, scans = _episode(
                by_class, items, rng, n_way, k, hard, config["cross_scan"])
            optimizer.zero_grad(set_to_none=True)
            loss, accuracy = _loss(model, globals_t[qi], tokens_t[qi],
                                   globals_t[ri], tokens_t[ri], labels, scans, config)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            accuracies.append(accuracy)
        frame, summary, _ = _evaluate(
            root, out, manifest, encoder, cfg, study._hash_file(base_path), model,
            config, device, folds=(0,), modes=("global", "trained"))
        evidence._assert_base_reproduced(frame, base_state)
        score = _selected_score(summary, "trained")
        payload = {"provenance": provenance, "epoch": epoch, "model": model.state_dict(),
                   "selection_score": score}
        if score > best_score:
            best_score = score
            study._save_checkpoint(best, payload)
        study._save_checkpoint(latest, {**payload, "best_score": best_score,
                                        "optimizer": optimizer.state_dict()})
        print(f"[wood-corr] head {config['variant']} seed={seed} epoch={epoch} "
              f"loss={np.mean(losses):.4f} train_R1={np.mean(accuracies):.4f} "
              f"val_score={score:.4f} best={best_score:.4f}", flush=True)
    if not best.is_file():
        raise RuntimeError("No selected head checkpoint")
    return best


def _encode_images(encoder, images):
    output = encoder.backbone.forward_features(images)
    cls, patches = output["x_norm_clstoken"], output["x_norm_patchtokens"]
    side = int(np.sqrt(patches.shape[1]))
    if side * side != patches.shape[1]:
        raise ValueError("Patch tokens do not form a square grid")
    local = F.adaptive_avg_pool2d(
        patches.transpose(1, 2).reshape(len(images), patches.shape[-1], side, side),
        (4, 4)).flatten(2).transpose(1, 2)
    return encoder.project(cls), encoder.project(local)


def _joint_step(model, encoder, images, ri_count, labels, scans, config, optimizer, device):
    optimizer.zero_grad(set_to_none=True)
    global_parts, token_parts = [], []
    with torch.no_grad():
        for chunk in images.split(config["image_microbatch"]):
            g, t = _encode_images(encoder, chunk.to(device, non_blocking=True))
            global_parts.append(g.float())
            token_parts.append(t.float())
    global_emb = torch.cat(global_parts).detach().requires_grad_(True)
    tokens = torch.cat(token_parts).detach().requires_grad_(True)
    loss, accuracy = _loss(model, global_emb[ri_count:], tokens[ri_count:],
                           global_emb[:ri_count], tokens[:ri_count], labels, scans, config)
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite joint loss; checkpoint not advanced")
    loss.backward()
    global_grad = global_emb.grad.detach()
    token_grad = (tokens.grad.detach() if tokens.grad is not None else
                  torch.zeros_like(tokens))
    if not torch.isfinite(global_grad).all() or not torch.isfinite(token_grad).all():
        raise RuntimeError("Non-finite joint feature gradient")
    offset = 0
    for chunk in images.split(config["image_microbatch"]):
        size = len(chunk)
        g, t = _encode_images(encoder, chunk.to(device, non_blocking=True))
        torch.autograd.backward((g.float(), t.float()),
                                (global_grad[offset:offset + size],
                                 token_grad[offset:offset + size]))
        offset += size
    torch.nn.utils.clip_grad_norm_(
        [p for p in list(model.parameters()) + list(encoder.parameters()) if p.requires_grad], 1.0)
    optimizer.step()
    return float(loss.detach()), accuracy


def _joint_train(root, out, manifest, base_path, base_state, cfg, device,
                 config, seed):
    head_path = _paths(out, "head", config["variant"], seed)[1]
    if not head_path.is_file():
        raise FileNotFoundError(f"Train the matching head first: {head_path}")
    head_state = torch.load(head_path, map_location="cpu", weights_only=False)
    if head_state["provenance"] != _provenance(
            root, base_path, recipe(config["variant"], "head"), seed, "head"):
        raise ValueError("Head checkpoint provenance changed")
    study._seed(seed)
    encoder = evidence._encoder_from_state(base_state, cfg, device)
    for parameter in encoder.backbone.parameters():
        parameter.requires_grad_(False)
    for block in encoder.backbone.blocks[-config["last_blocks"]:]:
        for parameter in block.parameters():
            parameter.requires_grad_(True)
    model = WoodCorrespondence().to(device)
    model.load_state_dict(head_state["model"], strict=True)
    optimizer = torch.optim.AdamW([
        {"params": model.parameters(), "lr": config["head_lr"]},
        {"params": [p for p in encoder.parameters() if p.requires_grad],
         "lr": config["backbone_lr"]}], weight_decay=1e-3)
    folder, best, latest = _paths(out, "joint", config["variant"], seed)
    provenance = _provenance(root, base_path, config, seed, "joint", head_path)
    items = evidence._representatives(manifest)
    by_class = evidence._episode_layout(items)
    dataset = data.ManifestDataset(items, transform=data.get_transforms(224, augment=True))
    feature_out = Path(os.environ.get(
        "WOOD_CORR_FEATURE_OUT", root / "results" / "wood_evidence_pilot_v1"))
    global_np, _ = evidence._features(feature_out, "meta_train_representatives", items,
                                      encoder, cfg, device, study._hash_file(base_path))
    hard = evidence._hard_neighbors(items, global_np, by_class)
    if not config["hard_negatives"]:
        hard = {key: [] for key in hard}
    first_epoch, best_score = 1, -float("inf")
    if latest.is_file():
        saved = torch.load(latest, map_location="cpu", weights_only=False)
        if saved["provenance"] != provenance:
            raise ValueError("Joint checkpoint provenance changed")
        model.load_state_dict(saved["model"], strict=True)
        encoder.load_state_dict(saved["encoder"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        first_epoch, best_score = int(saved["epoch"]) + 1, float(saved["best_score"])
    for epoch in range(first_epoch, config["epochs"] + 1):
        rng = np.random.default_rng(seed * 100003 + epoch)
        layouts, batches = [], []
        for _ in range(config["episodes"]):
            n_way = int(rng.choice(config["ways"]))
            k = int(rng.choice(config["reference_counts"]))
            ri, qi, labels, scans = _episode(
                by_class, items, rng, n_way, k, hard, config["cross_scan"])
            layouts.append((len(ri), labels, scans))
            batches.append(ri + qi)
        loader = DataLoader(dataset, batch_sampler=batches,
                            num_workers=min(config["workers"], os.cpu_count() or 1),
                            pin_memory=True)
        encoder.backbone.eval()
        encoder.projection.train()
        model.train()
        losses, accuracies = [], []
        for (images, _), (ri_count, labels, scans) in zip(loader, layouts):
            loss, accuracy = _joint_step(model, encoder, images, ri_count,
                                         labels, scans, config, optimizer, device)
            losses.append(loss)
            accuracies.append(accuracy)
        encoder.eval()
        model.eval()
        candidate = {"provenance": provenance, "epoch": epoch,
                     "encoder": encoder.state_dict(), "model": model.state_dict()}
        transient = folder / "candidate.pt"
        study._save_checkpoint(transient, candidate)
        frame, summary, _ = _evaluate(
            root, out, manifest, encoder, cfg, study._hash_file(transient), model,
            config, device, folds=(0,), modes=("global", "trained"))
        score = _selected_score(summary, "trained")
        if score > best_score:
            best_score = score
            study._save_checkpoint(best, {**candidate, "selection_score": score})
        study._save_checkpoint(latest, {**candidate, "selection_score": score,
                                        "best_score": best_score,
                                        "optimizer": optimizer.state_dict()})
        print(f"[wood-corr] joint {config['variant']} seed={seed} epoch={epoch} "
              f"loss={np.mean(losses):.4f} train_R1={np.mean(accuracies):.4f} "
              f"val_score={score:.4f} best={best_score:.4f}", flush=True)
    return best


def _smoke(device):
    study._seed(42)
    model = WoodCorrespondence(dimension=32, token_dim=8).to(device)
    q = F.normalize(torch.randn(4, 32, device=device), dim=-1)
    r = F.normalize(torch.randn(8, 32, device=device), dim=-1)
    qt = F.normalize(torch.randn(4, 4, 32, device=device), dim=-1)
    rt = F.normalize(torch.randn(8, 4, 32, device=device), dim=-1)
    labels = torch.arange(4, device=device).repeat_interleave(2)
    scans = [f"scan{i}" for i in range(8)]
    before = model.query_key.weight.detach().clone()
    config = recipe("qkv", "head")
    loss, _ = _loss(model, q, qt, r, rt, labels, scans, config)
    loss.backward()
    if not torch.isfinite(loss) or model.query_key.weight.grad is None:
        raise RuntimeError("QKV smoke loss or gradient failed")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.step()
    if torch.equal(before, model.query_key.weight.detach()):
        raise RuntimeError("QKV smoke did not update attention weights")
    all_scores, _ = model(q, qt, r, rt, labels, scans, mode="qkv")
    shortlist, _ = model(q, qt, r, rt, labels, scans, mode="qkv", top_classes=2)
    if all_scores.shape != (4, 4) or shortlist.shape != (4, 4):
        raise RuntimeError("All-class and shortlist shapes changed")
    print(f"[wood-corr] smoke passed on {device}: loss={float(loss.detach()):.4f}")


def run():
    root = Path(os.environ.get("ROOT_PATH", "/content/drive/MyDrive/NCS")).resolve()
    out = Path(os.environ.get("WOOD_CORR_OUT", root / "results" /
                              "wood_correspondence_study_v1")).resolve()
    study_out = Path(os.environ.get("WOOD_EVIDENCE_STUDY_OUT", root / "results" /
                                    "dinov2s_retrieval_matched_v1")).resolve()
    mode = os.environ.get("WOOD_CORR_MODE", "preflight").lower()
    device = torch.device("cuda" if os.environ.get("DEVICE", "cuda") == "cuda" and
                          torch.cuda.is_available() else "cpu")
    if mode == "smoke":
        return _smoke(device)
    if mode not in {"preflight", "probe", "train_head", "train_joint", "validate", "lock", "final"}:
        raise ValueError("Unknown WOOD_CORR_MODE")
    base_seed = int(os.environ.get("WOOD_CORR_BASE_SEED", "43"))
    base_path, base_state, cfg = evidence._base_checkpoint(root, study_out, base_seed)
    manifest = data.load_swi_manifest(root / "swi_manifest.json")
    variants = [x.strip() for x in os.environ.get("WOOD_CORR_VARIANTS", "qkv").split(",")]
    seeds = [int(x) for x in os.environ.get("WOOD_CORR_SEEDS", "43").split(",")]
    if not variants or any(x not in VARIANTS for x in variants) or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Invalid correspondence variants or seeds")
    print(f"[wood-corr] mode={mode} device={device} out={out} "
          f"variants={variants} seeds={seeds}", flush=True)
    if mode == "preflight":
        audit = {"base_checkpoint_sha256": study._hash_file(base_path),
                 "manifest_sha256": study._hash_file(root / "swi_manifest.json"),
                 "variants": variants, "seeds": seeds,
                 "meta_test_accessed": False, "splits": {}}
        for fold in (0, 1):
            refs, queries, extras = evidence._validation_sets(manifest, "meta-val", fold)
            audit["splits"][str(fold)] = {"references": len(refs),
                                            "queries": len(queries), "extras": len(extras)}
        study._json(out / "preflight.json", audit)
        return audit
    if device.type != "cuda":
        raise RuntimeError("Real correspondence study requires CUDA")
    feature_out = Path(os.environ.get(
        "WOOD_CORR_FEATURE_OUT", root / "results" / "wood_evidence_pilot_v1"))
    if mode in {"train_head", "train_joint"}:
        if os.environ.get("WOOD_CORR_PRELOAD", "0") == "1":
            paths = [path for path, _ in evidence._representatives(manifest)]
            if mode == "train_head":
                for fold in (0, 1):
                    refs, queries, extras = evidence._validation_sets(manifest, "meta-val", fold)
                    paths.extend(path for path, _ in refs + queries + extras)
            stats = data.preload_image_cache(
                sorted(set(paths)), max_workers=int(os.environ.get(
                    "WOOD_CORR_PRELOAD_WORKERS", "2")), desc="Wood correspondence images")
            if stats["bad"]:
                raise RuntimeError("Failed to preload correspondence images")
        for variant in variants:
            for seed in seeds:
                if mode == "train_head":
                    _head_train(root, out, feature_out, manifest, base_path,
                                base_state, cfg, device, recipe(variant, "head"), seed)
                else:
                    _joint_train(root, out, manifest, base_path, base_state,
                                 cfg, device, recipe(variant, "joint"), seed)
        return
    if mode in {"lock", "final"} and (len(variants) != 1 or len(seeds) != 1):
        raise ValueError("Lock/final evaluation requires one variant and seed")
    stage = os.environ.get("WOOD_CORR_STAGE", "head")
    if stage not in {"head", "joint"}:
        raise ValueError("WOOD_CORR_STAGE must be head or joint")
    if mode == "probe":
        stage = "head"
    lock_path = out / "selection_lock.json"
    if mode == "lock":
        variant, seed = variants[0], seeds[0]
        checkpoint = _paths(out, stage, variant, seed)[1]
        report = out / f"validate_{stage}_{variant}_seed{seed}_summary.csv"
        provenance_path = out / f"validate_{stage}_{variant}_seed{seed}_provenance.json"
        if not checkpoint.is_file() or not report.is_file() or not provenance_path.is_file():
            raise FileNotFoundError("Validate the selected checkpoint before locking meta-test")
        with provenance_path.open() as stream:
            validated = json.load(stream)
        checkpoint_hash = study._hash_file(checkpoint)
        if (validated["checkpoint_sha256"] != checkpoint_hash or
                validated["manifest_sha256"] != study._hash_file(root / "swi_manifest.json")):
            raise ValueError("Validation report and selected checkpoint no longer match")
        selected = pd.read_csv(report)
        if not {"57x5", "637x1"}.issubset(set(selected["gallery"])) or not (
                selected["mode"] == "trained").any():
            raise ValueError("Validation report lacks required selection galleries")
        lock = {"stage": stage, "variant": variant, "seed": seed,
                "checkpoint_sha256": checkpoint_hash,
                "validation_summary_sha256": study._hash_file(report),
                "manifest_sha256": validated["manifest_sha256"]}
        if lock_path.is_file():
            with lock_path.open() as stream:
                if json.load(stream) != lock:
                    raise ValueError("Selection is already locked to a different checkpoint")
        else:
            study._json(lock_path, lock)
        print(f"[wood-corr] validation selection locked: {lock_path}", flush=True)
        return lock
    if mode == "final":
        if os.environ.get("WOOD_CORR_APPROVE_FINAL", "0") != "1" or not lock_path.is_file():
            raise ValueError("Explicit approval and selection_lock.json required for meta-test")
        with lock_path.open() as stream:
            lock = json.load(stream)
        checkpoint = _paths(out, stage, variants[0], seeds[0])[1]
        if (lock["stage"], lock["variant"], lock["seed"]) != (stage, variants[0], seeds[0]) or (
                lock["checkpoint_sha256"] != study._hash_file(checkpoint) or
                lock["manifest_sha256"] != study._hash_file(root / "swi_manifest.json") or
                lock["validation_summary_sha256"] != study._hash_file(
                    out / f"validate_{stage}_{variants[0]}_seed{seeds[0]}_summary.csv")):
            raise ValueError("Locked selection, report or dataset changed")
    for variant in variants:
        for seed in seeds:
            config = recipe(variant, stage)
            encoder = evidence._encoder_from_state(base_state, cfg, device)
            model = WoodCorrespondence().to(device)
            checkpoint = None
            if mode != "probe":
                checkpoint = _paths(out, stage, variant, seed)[1]
                if not checkpoint.is_file():
                    raise FileNotFoundError(checkpoint)
                saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
                parent = _paths(out, "head", variant, seed)[1] if stage == "joint" else None
                if saved["provenance"] != _provenance(
                        root, base_path, config, seed, stage, parent):
                    raise ValueError("Selected checkpoint provenance changed")
                model.load_state_dict(saved["model"], strict=True)
                if stage == "joint":
                    encoder.load_state_dict(saved["encoder"], strict=True)
            identity = study._hash_file(checkpoint) if stage == "joint" else study._hash_file(base_path)
            split = "meta-test" if mode == "final" else "meta-val"
            if mode == "final" and os.environ.get("WOOD_CORR_PRELOAD", "0") == "1":
                paths = []
                for fold in (0, 1):
                    refs, queries, extras = evidence._validation_sets(manifest, split, fold)
                    paths.extend(path for path, _ in refs + queries + extras)
                stats = data.preload_image_cache(
                    sorted(set(paths)), max_workers=int(os.environ.get(
                        "WOOD_CORR_PRELOAD_WORKERS", "2")), desc="Locked correspondence test")
                if stats["bad"]:
                    raise RuntimeError("Failed to preload locked test images")
            modes = (("global", "fixed_local", "trained", "global_adapt", "maxsim", "qkv_all", "qkv_top128",
                      "qkv_no_gate", "qkv_no_consensus") if mode != "probe" else
                     ("global", "fixed_local"))
            frame, summary, timing = _evaluate(root, out, manifest, encoder, cfg,
                                               identity, model, config, device,
                                               split=split, modes=modes)
            if stage == "head":
                evidence._assert_base_reproduced(frame, base_state)
            stem = f"{mode}_{stage}_{variant}_seed{seed}"
            provenance = {"base_sha256": study._hash_file(base_path),
                          "checkpoint_sha256": study._hash_file(checkpoint) if checkpoint else None,
                          "manifest_sha256": study._hash_file(root / "swi_manifest.json"),
                          "stage": stage, "variant": variant, "seed": seed,
                          "meta_test_accessed": mode == "final"}
            _save_report(out, stem, frame, summary, timing, provenance)
            print(f"[wood-corr] saved {out / (stem + '_summary.csv')}", flush=True)
