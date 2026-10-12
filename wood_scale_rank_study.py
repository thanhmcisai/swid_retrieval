"""Matched multi-scale episode and rank-loss ablations for wood retrieval.

All training and mining use meta-train. Checkpoint/arm selection uses only
meta-val fold 0; fold 1 is reported after the selection lock is written.
"""

import json
import os
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from . import data, gallery_experiment as study
from . import wood_correspondence_experiment as corr
from . import wood_hard_negative as hard
from . import wood_large_gallery as large
from . import wood_multilevel as ml
from . import wood_reference_budget as budget
from . import wood_scale_rank_method as method
from .wood_large_gallery_methods import class_scores


ARMS = {
    "random": (False, 0, False, False, False),
    "balanced": (True, 0, False, False, False),
    "balanced_cross_scale": (True, 0, False, True, False),
    "balanced_rank32": (True, 32, False, False, False),
    "balanced_rank128": (True, 128, False, False, False),
    "balanced_rank128_allrefs": (True, 128, False, False, True),
    "balanced_rank128_weighted": (True, 128, True, False, False),
    "balanced_rank128_weighted_cross_scale": (True, 128, True, True, False),
}
ENDPOINT = {"budget": 10, "width": 128, "method": "qkv_all_refs"}


def _settings():
    cfg = {
        "arms": [x.strip() for x in os.environ.get(
            "WOOD_SCALE_RANK_ARMS", ",".join(ARMS)).split(",")],
        "seeds": [int(x) for x in os.environ.get(
            "WOOD_SCALE_RANK_SEEDS", "43").split(",")],
        "schedule": [[16, 5], [16, 10]],
        "epochs": int(os.environ.get("WOOD_SCALE_RANK_EPOCHS", "3")),
        "episodes": int(os.environ.get("WOOD_SCALE_RANK_EPISODES", "120")),
        "workers": int(os.environ.get("WOOD_SCALE_RANK_WORKERS", "2")),
        "image_batch": int(os.environ.get("WOOD_SCALE_RANK_IMAGE_BATCH", "64")),
        "microbatch": int(os.environ.get("WOOD_SCALE_RANK_MICROBATCH", "4")),
        "rank_queries": int(os.environ.get("WOOD_SCALE_RANK_QUERIES", "2")),
        "backbone_lr": float(os.environ.get("WOOD_SCALE_RANK_BACKBONE_LR", "1e-6")),
        "head_lr": float(os.environ.get("WOOD_SCALE_RANK_HEAD_LR", "5e-5")),
        "bank_weight": 0.5, "hard_weight": 0.5,
        "rank_weight": float(os.environ.get("WOOD_SCALE_RANK_WEIGHT", "0.2")),
        "contrastive_weight": float(os.environ.get(
            "WOOD_SCALE_RANK_CONTRASTIVE_WEIGHT", "0.2")),
        "selection_budget": 10, "selection_width": 128,
    }
    if (not cfg["arms"] or len(set(cfg["arms"])) != len(cfg["arms"]) or
            any(arm not in ARMS for arm in cfg["arms"]) or
            not cfg["seeds"] or len(set(cfg["seeds"])) != len(cfg["seeds"]) or
            any(seed < 0 for seed in cfg["seeds"]) or
            cfg["epochs"] < 1 or cfg["episodes"] < 1 or cfg["workers"] < 0 or
            cfg["image_batch"] < 1 or cfg["microbatch"] < 1 or
            cfg["rank_queries"] < 1 or cfg["backbone_lr"] <= 0 or
            cfg["head_lr"] <= 0 or cfg["rank_weight"] < 0 or
            cfg["contrastive_weight"] < 0):
        raise ValueError("Invalid scale-rank study settings")
    return cfg


def _paths(root):
    prior = Path(os.environ.get("WOOD_SCALE_RANK_SOURCE",
                 root / "results/wood_hard_shortlist_study_v1")).resolve()
    out = Path(os.environ.get("WOOD_SCALE_RANK_OUT",
               root / "results/wood_scale_rank_study_v1")).resolve()
    base = Path(os.environ.get("WOOD_LARGE_BASE_OUT",
                root / "results/dinov2s_retrieval_matched_v1")).resolve()
    image = Path(os.environ.get("WOOD_LARGE_QKV_OUT",
                 root / "results/wood_correspondence_image_full_v1")).resolve()
    if out == prior or out in prior.parents or prior in out.parents:
        raise ValueError("Scale-rank output must not overwrite the source study")
    return prior, out, base, image


def _source(root, prior, base, image, device):
    old_cfg = json.loads((prior / "preflight.json").read_text())["settings"]
    if old_cfg["seed"] != 43:
        raise ValueError("Unexpected source study seed")
    hard._locked(prior)
    original, encoder, model, base_cfg, recipe = hard._source(
        root, base, image, old_cfg["seed"], device)
    path = prior / "train/random16/best.pt"
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved["provenance"] != hard._provenance(
            root, original, old_cfg, "random16"):
        raise ValueError("Source checkpoint provenance mismatch")
    encoder.load_state_dict(saved["encoder"], strict=True)
    model.load_state_dict(saved["model"], strict=True)
    return path, encoder, model, base_cfg, recipe


def _provenance(root, source, cfg, arm, seed):
    return {"source_sha256": study._hash_file(source),
            "manifest_sha256": study._hash_file(root / "swi_manifest.json"),
            "runner_sha256": study._hash_file(__file__),
            "method_sha256": study._hash_file(method.__file__),
            "arm": arm, "seed": seed, "settings": cfg,
            "train_split": "meta-train", "selection_split": "meta-val/fold0"}


def _preflight(root, prior, out, base, image, manifest, cfg):
    source, _, _, _, _ = _source(root, prior, base, image, torch.device("cpu"))
    items = manifest["meta-train"]
    catalog = method.episode_catalog(items, (5, 10))
    counts = {str(k): {"eligible_species": len(catalog[2][k]),
                       "eligible_queries": sum(map(len, catalog[2][k].values()))}
              for k in (5, 10)}
    if min(v["eligible_species"] for v in counts.values()) < 16:
        raise ValueError("Too few scan-disjoint multi-scale species")
    audit = {"source_checkpoint_sha256": study._hash_file(source),
             "source_validation_sha256": study._hash_file(
                 prior / "validation/random16/summary.csv"),
             "manifest_sha256": study._hash_file(root / "swi_manifest.json"),
             "source": str(source), "settings": cfg, "eligibility": counts,
             "meta_train_images": len(items), "meta_test_evaluated": False}
    path = out / "preflight.json"
    if path.is_file() and json.loads(path.read_text()) != audit:
        raise ValueError("Preflight changed; use a new output folder")
    out.mkdir(parents=True, exist_ok=True)
    study._json(path, audit)
    print(f"[wood-scale-rank] eligibility: {counts}; source={source}", flush=True)
    return catalog


def _check_preflight(root, prior, out, base, image, cfg):
    saved = json.loads((out / "preflight.json").read_text())
    if (saved["settings"] != cfg or
            saved["manifest_sha256"] != study._hash_file(root / "swi_manifest.json") or
            saved["source_checkpoint_sha256"] != study._hash_file(
                prior / "train/random16/best.pt") or
            saved["source_validation_sha256"] != study._hash_file(
                prior / "validation/random16/summary.csv")):
        raise ValueError("Scale-rank preflight is stale")


def _first_bank_rows(ref_labels, count):
    rows = []
    for label in range(count):
        matches = (ref_labels == label).nonzero(as_tuple=True)[0]
        if not len(matches):
            raise ValueError("Missing class in reference bank")
        rows.append(matches[0])
    return torch.stack(rows)


def _rank_aux(model, q, qt, r, rt, labels, scans, species, bank,
              width, weighted, all_bank_refs, query_count, rng, recipe):
    names, bank_labels, bank_g, bank_t, prototypes, bank_scans, first = bank
    lookup = {label: i for i, label in enumerate(names)}
    chosen = rng.choice(len(q), size=min(query_count, len(q)), replace=False)
    losses, ranks = [], []
    for row in chosen:
        positive = lookup[species[int(row)]]
        negatives = hard.mine_negative_classes(
            q[row], prototypes, positive, width, False, rng)
        negative_rows = ((torch.isin(bank_labels, negatives)).nonzero(
            as_tuple=True)[0] if all_bank_refs else first[negatives])
        positive_rows = (torch.as_tensor(labels, device=q.device) == row).nonzero(
            as_tuple=True)[0]
        refs = torch.cat((r[positive_rows], bank_g[negative_rows]), dim=0)
        tokens = torch.cat((rt[positive_rows], bank_t[negative_rows]), dim=0)
        ref_labels = torch.cat((bank_labels.new_full((len(positive_rows),), positive),
                                bank_labels[negative_rows]))
        ref_scans = [scans[int(i)] for i in positive_rows.tolist()] + [
            bank_scans[int(i)] for i in negative_rows.tolist()]
        scores, classes = model(
            q[row:row + 1], qt[row:row + 1], refs, tokens, ref_labels,
            ref_scans, mode="qkv", gate=recipe["gate"],
            consensus=recipe["consensus"], query_chunk=1,
            reference_chunk=recipe["reference_chunk"])
        target = int(torch.searchsorted(classes, classes.new_tensor([positive]))[0])
        if target >= len(classes) or int(classes[target]) != positive:
            raise ValueError("Positive class missing from rank candidates")
        loss, rank = method.position_loss(scores, target, weight_by_rank=weighted)
        losses.append(loss)
        ranks.append(rank)
    return torch.stack(losses).mean(), float(np.mean(ranks))


def _step(model, encoder, images, layout, bank, arm, recipe, cfg,
          optimizer, device, rng):
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
    names, bank_labels, bank_g, bank_t, prototypes, bank_scans, _ = bank
    original, accuracy = corr._loss(
        model, q, all_t[n_refs:], r, all_t[:n_refs], labels, scans, recipe)
    with torch.no_grad():
        adapted_bank, classes, _ = hard.class_prototypes(
            corr._transformed(model, bank_g), bank_labels)
    if len(classes) != len(names):
        raise ValueError("Incomplete adapted bank")
    bank_ce = large._bank_loss(model, q, r, labels, species,
                               names, adapted_bank, 128, "hard_bank")
    hard_ce, _ = hard.shortlist_loss(
        model, q, all_t[n_refs:], r, all_t[:n_refs], labels, scans,
        species, names, bank_labels, bank_g, bank_t, bank_scans,
        prototypes, 16, True, 4, rng, recipe)
    loss = original + cfg["bank_weight"] * bank_ce + cfg["hard_weight"] * hard_ce
    _, width, weighted, contrastive, all_bank_refs = ARMS[arm]
    rank = None
    if width:
        extra, rank = _rank_aux(
            model, q, all_t[n_refs:], r, all_t[:n_refs], labels, scans,
            species, bank, width, weighted, all_bank_refs,
            cfg["rank_queries"], rng, recipe)
        loss = loss + cfg["rank_weight"] * extra
    if contrastive:
        loss = loss + cfg["contrastive_weight"] * ml.cross_scale_loss(
            model, q, r, labels, ref_scales, query_scales)
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite scale-rank training loss")
    loss.backward()
    g_grad, t_grad = all_g.grad.detach(), all_t.grad.detach()
    if not torch.isfinite(g_grad).all() or not torch.isfinite(t_grad).all():
        raise RuntimeError("Non-finite encoded gradient")
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
    return float(loss.detach()), accuracy, rank


def _train_arm(root, prior, out, base, image, manifest, catalog,
               cfg, arm, seed, device):
    study._seed(seed)
    source, encoder, model, base_cfg, recipe = _source(
        root, prior, base, image, device)
    folder = out / "train" / arm / f"seed_{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    provenance = _provenance(root, source, cfg, arm, seed)
    optimizer = torch.optim.AdamW([
        {"params": model.parameters(), "lr": cfg["head_lr"]},
        {"params": encoder.parameters(), "lr": cfg["backbone_lr"]}],
        weight_decay=1e-3)
    latest = folder / "latest.pt"
    start = 1
    if latest.is_file():
        saved = torch.load(latest, map_location="cpu", weights_only=False)
        if saved["provenance"] != provenance:
            raise ValueError("Scale-rank training provenance changed")
        encoder.load_state_dict(saved["encoder"], strict=True)
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        start = saved["epoch"] + 1
    if start > cfg["epochs"]:
        if not all((folder / f"epoch_{epoch:02d}.pt").is_file()
                   for epoch in range(1, cfg["epochs"] + 1)):
            raise ValueError("Completed training is missing epoch checkpoints")
        print(f"[wood-scale-rank] {arm} seed={seed} already trained", flush=True)
        return
    items = manifest["meta-train"]
    dataset = data.ManifestDataset(
        items, transform=data.get_transforms(224, augment=True))
    dataset.loader = large._RetryImageLoader(dataset.loader)
    representatives = large._bank_representatives(manifest)
    bank_cfg = dict(base_cfg, image_batch=cfg["image_batch"], workers=cfg["workers"])
    balanced = ARMS[arm][0]
    for epoch in range(start, cfg["epochs"] + 1):
        previous = source if epoch == 1 else folder / f"epoch_{epoch - 1:02d}.pt"
        bank_path = (out / "bank_features/shared_source.npy" if epoch == 1 else
                     out / "bank_features" / arm / f"seed_{seed}" / f"epoch_{epoch:02d}.npy")
        bank = hard._bank_features(
            encoder, representatives, bank_cfg, device,
            bank_path, study._hash_file(previous))
        bank = (*bank, _first_bank_rows(bank[1], len(bank[0])))
        batches, layouts = method.plan_episodes(
            items, catalog, cfg["schedule"], cfg["episodes"], seed,
            epoch, balanced)
        loader = DataLoader(
            dataset, batch_sampler=batches,
            num_workers=min(cfg["workers"], len(os.sched_getaffinity(0))
                            if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1),
            pin_memory=True)
        encoder.backbone.eval()
        encoder.projection.train()
        model.train()
        losses, accuracies, ranks = [], [], []
        began = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for step, ((images, _), layout) in enumerate(zip(loader, layouts), 1):
            rng = np.random.default_rng(seed * 1000003 + epoch * 10007 + step)
            loss, accuracy, rank = _step(
                model, encoder, images, layout, bank, arm, recipe,
                cfg, optimizer, device, rng)
            losses.append(loss)
            accuracies.append(accuracy)
            if rank is not None:
                ranks.append(rank)
            if step % 10 == 0 or step == len(layouts):
                print(f"[wood-scale-rank] {arm} seed={seed} epoch={epoch} "
                      f"step={step}/{len(layouts)} loss={np.mean(losses):.4f} "
                      f"train_R1={np.mean(accuracies):.4f} "
                      f"rank={np.mean(ranks) if ranks else float('nan'):.1f} "
                      f"elapsed_s={time.perf_counter()-began:.1f}", flush=True)
        epoch_path = folder / f"epoch_{epoch:02d}.pt"
        payload = {"provenance": provenance, "epoch": epoch,
                   "encoder": encoder.state_dict(), "model": model.state_dict()}
        study._save_checkpoint(epoch_path, payload)
        study._save_checkpoint(latest, {**payload, "optimizer": optimizer.state_dict()})
        ref_scales = [scale for layout in layouts for scale in layout[4]]
        query_scales = [scale for layout in layouts for scale in layout[5]]
        per_class_scales = [len(set(layout[4][i * (layout[0] // 16):
                                              (i + 1) * (layout[0] // 16)]))
                            for layout in layouts for i in range(16)]
        report = {"provenance": provenance, "epoch": epoch,
                  "loss": float(np.mean(losses)),
                  "episode_r1": float(np.mean(accuracies)),
                  "mean_positive_rank": float(np.mean(ranks)) if ranks else None,
                  "unique_images": len({i for batch in batches for i in batch}),
                  "distinct_reference_scales": len(set(ref_scales)),
                  "mean_reference_scales_per_species_episode": float(
                      np.mean(per_class_scales)),
                  "min_reference_scales_per_species_episode": min(per_class_scales),
                  "distinct_query_scales": len(set(query_scales)),
                  "train_seconds": time.perf_counter() - began,
                  "peak_gpu_gb": (torch.cuda.max_memory_allocated() / 2**30
                                  if device.type == "cuda" else None)}
        study._json(folder / f"epoch_{epoch:02d}_training.json", report)
    print(f"[wood-scale-rank] trained {arm} seed={seed}", flush=True)


def _preload_training(manifest, catalog, cfg):
    if os.environ.get("WOOD_SCALE_RANK_PRELOAD", "1") != "1":
        return
    items = manifest["meta-train"]
    needed = set()
    for seed in cfg["seeds"]:
        for epoch in range(1, cfg["epochs"] + 1):
            for balanced in {ARMS[arm][0] for arm in cfg["arms"]}:
                batches, _ = method.plan_episodes(
                    items, catalog, cfg["schedule"], cfg["episodes"],
                    seed, epoch, balanced)
                needed.update(i for batch in batches for i in batch)
    paths = [items[i][0] for i in sorted(needed)]
    paths.extend(path for path, _ in large._bank_representatives(manifest))
    print(f"[wood-scale-rank] preloading {len(set(paths))} unique planned images",
          flush=True)
    data.preload_image_cache(
        paths, max_workers=int(os.environ.get("WOOD_SCALE_RANK_PRELOAD_WORKERS", "8")),
        desc="Scale-rank planned training images")


def _load_epoch(root, prior, out, base, image, cfg, arm, seed, epoch, device):
    source, encoder, model, base_cfg, _ = _source(
        root, prior, base, image, device)
    path = out / "train" / arm / f"seed_{seed}" / f"epoch_{epoch:02d}.pt"
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved["provenance"] != _provenance(root, source, cfg, arm, seed) or saved[
            "epoch"] != epoch:
        raise ValueError("Epoch checkpoint provenance mismatch")
    encoder.load_state_dict(saved["encoder"], strict=True)
    model.load_state_dict(saved["model"], strict=True)
    return path, encoder.eval(), model.eval(), base_cfg


def _evaluate(root, out, manifest, encoder, model, base_cfg, cfg,
              checkpoint, folder, arm, folds, budgets, widths, device):
    folder.mkdir(parents=True, exist_ok=True)
    digest = study._hash_file(checkpoint)
    protocol = {"checkpoint_sha256": digest, "arm": arm, "folds": list(folds),
                "budgets": list(budgets), "widths": list(widths),
                "reference_policy": "scan_scale", "shortlist": "raw_prototype",
                "meta_val_only": True, "query_grouping": "scan_disjoint"}
    protocol_path = folder / "protocol.json"
    if protocol_path.is_file():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("Scale-rank validation protocol changed")
        if all((folder / name).is_file() for name in ("summary.csv", "per_query.csv")):
            print(f"[wood-scale-rank] reusing {folder}", flush=True)
            return
    rows, query_rows = [], []
    for fold in folds:
        _, queries = study.validation_items(
            manifest, group_mode="scan_disjoint", fold=fold)
        refs = budget._reference_pool(manifest, queries)
        class_names = sorted({label for _, label in refs})
        if len(class_names) != 637 or len(queries) != 285:
            raise ValueError("Meta-val cohort changed")
        class_ids = {label: i for i, label in enumerate(class_names)}
        selected = budget.select_indices(refs, max(budgets), "scan_scale")
        items = [refs[i] for i in selected]
        rg, rt, qg, qt = hard._reference_data(
            encoder, items, queries,
            dict(base_cfg, image_batch=cfg["image_batch"], workers=cfg["workers"]),
            device, folder / "features" / f"fold_{fold}", digest)
        positions = {index: i for i, index in enumerate(selected)}
        truth = torch.tensor([class_ids[study.canonical(label)]
                              for _, label in queries], device=device)
        for k in budgets:
            chosen = budget.select_indices(refs, k, "scan_scale")
            ids = torch.tensor([positions[i] for i in chosen], device=device)
            current = [refs[i] for i in chosen]
            ref_labels = torch.tensor([class_ids[label] for _, label in current],
                                      device=device)
            scans = [study.source_scan_id(path) for path, _ in current]
            by_class_scans, by_class_scales = defaultdict(set), defaultdict(set)
            for path, label in current:
                by_class_scans[label].add(study.source_scan_id(path))
                by_class_scales[label].add(study.image_scale(path))
            if any(study.source_scan_id(path) in by_class_scans[
                    study.canonical(label)] for path, label in queries):
                raise ValueError("Meta-val source-scan leakage")
            with torch.inference_mode():
                raw, _ = class_scores(qg, rg[ids], ref_labels)
                adapted, _ = class_scores(corr._transformed(model, qg),
                                          corr._transformed(model, rg[ids]),
                                          ref_labels)
            raw_truth = raw.gather(1, truth[:, None])
            raw_ranks = 1 + (raw > raw_truth).sum(1)
            def record(name, width, scores, candidates=None, elapsed=0.0):
                rows.append(hard._summary_row(
                    arm, fold, k, width, name, scores, truth,
                    candidates, elapsed))
                predictions = scores.argmax(dim=1).tolist()
                hits = ((candidates == truth[:, None]).any(dim=1).tolist()
                        if candidates is not None else [None] * len(queries))
                true_scores = scores.gather(1, truth[:, None])
                ranks = 1 + (scores > true_scores).sum(1)
                gaps = (scores.max(1).values - true_scores.squeeze(1)).tolist()
                for i, (path, label) in enumerate(queries):
                    species = study.canonical(label)
                    present = hits[i] is not False
                    query_rows.append({
                        "arm": arm, "fold": fold, "budget": k, "width": width,
                        "method": name, "query_path": path,
                        "true_label": species,
                        "predicted_label": class_names[predictions[i]],
                        "correct": int(predictions[i] == int(truth[i])),
                        "candidate_hit": hits[i],
                        "raw_true_rank": int(raw_ranks[i]),
                        "true_rank": int(ranks[i]) if present else None,
                        "top1_minus_true_score": gaps[i] if present else None,
                        "query_scale": study.image_scale(path),
                        "same_scale_reference": int(
                            study.image_scale(path) in by_class_scales[species]),
                        "distinct_reference_scales": len(by_class_scales[species])})
            record("raw_prototype", 0, raw)
            record("adapted_prototype", 0, adapted)
            for width in widths:
                candidates = raw.topk(width, dim=1).indices
                started = time.perf_counter()
                with torch.inference_mode():
                    scores = ml.rerank_all_references(
                        model, qg, qt, rg[ids], rt[ids], ref_labels,
                        scans, candidates, adapted)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                record("qkv_all_refs", width, scores, candidates,
                       1000 * (time.perf_counter() - started) / len(qg))
                print(f"[wood-scale-rank] {arm} fold={fold} "
                      f"K={k} top={width} R1={rows[-1]['macro_r1']:.4f} "
                      f"candidate={rows[-1]['candidate_macro_recall']:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(folder / "summary.csv", index=False)
    pd.DataFrame(query_rows).to_csv(folder / "per_query.csv", index=False)
    study._json(protocol_path, protocol)


def _validation_folder(out, arm, seed, epoch):
    return out / "validation" / arm / f"seed_{seed}" / f"epoch_{epoch:02d}"


def _validate(root, prior, out, base, image, manifest, cfg, device):
    for arm in cfg["arms"]:
        for seed in cfg["seeds"]:
            for epoch in range(1, cfg["epochs"] + 1):
                path, encoder, model, base_cfg = _load_epoch(
                    root, prior, out, base, image, cfg, arm, seed, epoch, device)
                _evaluate(root, out, manifest, encoder, model, base_cfg, cfg,
                          path, _validation_folder(out, arm, seed, epoch), arm,
                          (0,), (cfg["selection_budget"],),
                          (cfg["selection_width"],), device)
                del encoder, model
                if device.type == "cuda":
                    torch.cuda.empty_cache()


def _candidate_rows(out, cfg):
    rows, evidence = [], {}
    for arm in cfg["arms"]:
        for seed in cfg["seeds"]:
            for epoch in range(1, cfg["epochs"] + 1):
                folder = _validation_folder(out, arm, seed, epoch)
                protocol = json.loads((folder / "protocol.json").read_text())
                checkpoint = out / "train" / arm / f"seed_{seed}" / f"epoch_{epoch:02d}.pt"
                if (protocol["checkpoint_sha256"] != study._hash_file(checkpoint)
                        or protocol["arm"] != arm or protocol["folds"] != [0]
                        or protocol["budgets"] != [cfg["selection_budget"]]
                        or protocol["widths"] != [cfg["selection_width"]]):
                    raise ValueError(f"Invalid selection evidence: {folder}")
                summary = pd.read_csv(folder / "summary.csv")
                target = summary.loc[
                    (summary["fold"] == 0) &
                    (summary["budget"] == cfg["selection_budget"]) &
                    (summary["width"] == cfg["selection_width"]) &
                    (summary["method"] == "qkv_all_refs")]
                if len(target) != 1:
                    raise ValueError(f"Missing unique selection endpoint: {folder}")
                row = target.iloc[0]
                rows.append({"arm": arm, "seed": seed, "epoch": epoch,
                             "macro_r1": float(row["macro_r1"]),
                             "candidate_macro_recall": float(
                                 row["candidate_macro_recall"]),
                             "checkpoint_sha256": protocol["checkpoint_sha256"]})
                key = f"{arm}/seed_{seed}/epoch_{epoch:02d}"
                evidence[key] = {
                    name: study._hash_file(folder / name)
                    for name in ("protocol.json", "summary.csv", "per_query.csv")}
    return rows, evidence


def _select(out, cfg):
    rows, evidence = _candidate_rows(out, cfg)
    ordered = sorted(rows, key=lambda row: (
        -row["macro_r1"], row["arm"], row["seed"], row["epoch"]))
    per_arm = {}
    for row in ordered:
        per_arm.setdefault((row["arm"], row["seed"]), row)
    lock = {"selection_split": "meta-val/fold0", "selection_endpoint": ENDPOINT,
            "selected": ordered[0], "per_arm_seed": list(per_arm.values()),
            "candidate_evidence_sha256": evidence,
            "meta_test_evaluated": False, "public_id_evaluated": False,
            "fold1_not_used_for_this_selection": True}
    path = out / "selection_lock.json"
    if path.is_file() and json.loads(path.read_text()) != lock:
        raise ValueError("Selection lock changed; use a new output folder")
    study._json(path, lock)
    pd.DataFrame(rows).to_csv(out / "selection_candidates.csv", index=False)
    print(f"[wood-scale-rank] locked: {lock['selected']}", flush=True)


def _locked(out, cfg):
    lock = json.loads((out / "selection_lock.json").read_text())
    rows, evidence = _candidate_rows(out, cfg)
    if (lock["candidate_evidence_sha256"] != evidence or
            lock["selected"] not in rows or lock["meta_test_evaluated"] or
            lock["public_id_evaluated"]):
        raise ValueError("Scale-rank selection evidence changed after lock")
    return lock


def _describe(root, prior, out, base, image, manifest, cfg, device):
    lock = _locked(out, cfg)
    for chosen in lock["per_arm_seed"]:
        arm, seed, epoch = chosen["arm"], chosen["seed"], chosen["epoch"]
        path, encoder, model, base_cfg = _load_epoch(
            root, prior, out, base, image, cfg, arm, seed, epoch, device)
        folder = out / "description" / arm / f"seed_{seed}"
        _evaluate(root, out, manifest, encoder, model, base_cfg, cfg,
                  path, folder, arm, (0, 1), (1, 3, 5, 10), (32, 128), device)
        del encoder, model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _export(root, prior, out, cfg):
    lock = _locked(out, cfg)
    files = [out / "preflight.json", out / "selection_lock.json",
             out / "selection_candidates.csv"]
    for arm in cfg["arms"]:
        for seed in cfg["seeds"]:
            for epoch in range(1, cfg["epochs"] + 1):
                folder = _validation_folder(out, arm, seed, epoch)
                files.extend(folder / name for name in (
                    "protocol.json", "summary.csv", "per_query.csv"))
                files.append(out / "train" / arm / f"seed_{seed}" /
                             f"epoch_{epoch:02d}_training.json")
    for chosen in lock["per_arm_seed"]:
        folder = out / "description" / chosen["arm"] / f"seed_{chosen['seed']}"
        files.extend(folder / name for name in (
            "protocol.json", "summary.csv", "per_query.csv"))
    files.extend(prior / name for name in (
        "preflight.json", "selection_lock.json",
        "validation/random16/summary.csv",
        "validation/random16/per_query.csv",
        "validation/random16/protocol.json"))
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Export incomplete: {missing[:5]}")
    archive = out.parent / f"{out.name}_review.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            prefix = "source" if prior in path.parents else "scale_rank"
            parent = prior if prefix == "source" else out
            zf.write(path, f"{prefix}/{path.relative_to(parent)}")
    print(f"[wood-scale-rank] review archive: {archive} "
          f"({archive.stat().st_size / 2**20:.1f} MiB)", flush=True)
    return archive


def _smoke(device):
    items = []
    for species in range(20):
        for scan in range(4):
            for scale in (256, 512, 1024, 2048):
                path = (f"/tmp/train/species_{species}/scale_{scale}/"
                        f"patch_{scale}_0_0_from_Tw{species * 3 + scan}.jpg")
                items.append((path, f"species_{species}"))
    catalog = method.episode_catalog(items, (5, 10))
    batches, layouts = method.plan_episodes(
        items, catalog, [[16, 5], [16, 10]], 4, 43, 1, True)
    if any(not ({*layout[4][row * k:(row + 1) * k]} -
                 {layout[5][row]}) for row in range(16)
           for layout, k in ((layouts[0], 5), (layouts[1], 10))):
        raise RuntimeError("Balanced smoke lacks cross-scale positives")
    scores = torch.tensor([[0.2, 0.1, 0.0]], device=device,
                          requires_grad=True)
    loss, rank = method.position_loss(scores, 2, weight_by_rank=True)
    loss.backward()
    if not torch.isfinite(scores.grad).all() or rank != 3:
        raise RuntimeError("Rank-loss smoke failed")
    print("[wood-scale-rank] smoke passed", flush=True)


def run():
    root = Path(os.environ.get("ROOT_PATH", "/content/drive/MyDrive/NCS")).resolve()
    prior, out, base, image = _paths(root)
    cfg = _settings()
    mode = os.environ.get("WOOD_SCALE_RANK_MODE", "preflight")
    if mode not in {"smoke", "preflight", "train", "validate", "select",
                    "describe", "export"}:
        raise ValueError("Invalid WOOD_SCALE_RANK_MODE")
    if mode == "smoke":
        _smoke(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        return
    manifest = data.load_swi_manifest(root / "swi_manifest.json")
    if mode == "preflight":
        _preflight(root, prior, out, base, image, manifest, cfg)
        return
    _check_preflight(root, prior, out, base, image, cfg)
    if mode == "export":
        _export(root, prior, out, cfg)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Image training and validation require CUDA")
    device = torch.device("cuda")
    if mode == "train":
        catalog = method.episode_catalog(manifest["meta-train"], (5, 10))
        _preload_training(manifest, catalog, cfg)
        for arm in cfg["arms"]:
            for seed in cfg["seeds"]:
                _train_arm(root, prior, out, base, image, manifest, catalog,
                           cfg, arm, seed, device)
    elif mode == "validate":
        _validate(root, prior, out, base, image, manifest, cfg, device)
    elif mode == "select":
        _select(out, cfg)
    else:
        _describe(root, prior, out, base, image, manifest, cfg, device)
