"""Hard-negative image fine-tuning and locked large-gallery retrieval evaluation."""

import json
import os
import time
import zipfile
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
from . import wood_evidence_experiment as evidence
from .wood_correspondence_method import WoodCorrespondence, class_prototypes
from .wood_large_gallery_methods import (candidate_recall, candidate_union,
                                         class_scores, rerank_candidates)


def _settings():
    cfg = {
        "epochs": int(os.environ.get("WOOD_LARGE_EPOCHS", "3")),
        "episodes": int(os.environ.get("WOOD_LARGE_EPISODES", "60")),
        "workers": int(os.environ.get("WOOD_LARGE_WORKERS", "2")),
        "microbatch": int(os.environ.get("WOOD_LARGE_MICROBATCH", "4")),
        "image_batch": int(os.environ.get("WOOD_LARGE_IMAGE_BATCH", "32")),
        "backbone_lr": float(os.environ.get("WOOD_LARGE_BACKBONE_LR", "1e-6")),
        "head_lr": float(os.environ.get("WOOD_LARGE_HEAD_LR", "5e-5")),
        "bank_weight": float(os.environ.get("WOOD_LARGE_BANK_WEIGHT", "0.5")),
        "bank_negatives": int(os.environ.get("WOOD_LARGE_BANK_NEGATIVES", "128")),
        "hard_fraction": float(os.environ.get("WOOD_LARGE_HARD_FRACTION", "0.5")),
        "max_queries_per_species": int(os.environ.get("WOOD_LARGE_QUERY_LIMIT_PER_SPECIES", "0")),
        "rerank_widths": [int(x) for x in os.environ.get(
            "WOOD_LARGE_RERANK_WIDTHS", "32,128").split(",")],
        "source_seed": os.environ.get("WOOD_LARGE_SOURCE_SEED", "43"),
    }
    if (cfg["epochs"] < 1 or cfg["episodes"] < 1 or cfg["workers"] < 0 or
            cfg["microbatch"] < 1 or cfg["image_batch"] < 1 or
            cfg["bank_negatives"] < 1 or cfg["bank_weight"] <= 0 or
            not 0 <= cfg["hard_fraction"] <= 1 or
            cfg["max_queries_per_species"] < 0 or
            any(k < 1 for k in cfg["rerank_widths"])):
        raise ValueError("Invalid large-gallery configuration")
    if cfg["source_seed"] != "match":
        try:
            cfg["source_seed"] = int(cfg["source_seed"])
        except ValueError as exc:
            raise ValueError("WOOD_LARGE_SOURCE_SEED must be an integer or match") from exc
    return cfg


def _upstream_seed(seed):
    source = os.environ.get("WOOD_LARGE_SOURCE_SEED", "43")
    return seed if source == "match" else int(source)


def _source_checkpoint(root, study_out, image_out, seed, device):
    base_path, base_state, base_cfg = evidence._base_checkpoint(root, study_out, seed)
    path = image_out / "qkv" / f"seed_{seed}" / "best.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing image-trained QKV checkpoint: {path}")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    expected = image_train._provenance(root, base_path, saved["provenance"]["config"], seed)
    if saved["provenance"] != expected or saved["provenance"]["config"]["variant"] != "qkv":
        raise ValueError("Source QKV checkpoint provenance mismatch")
    encoder = evidence._encoder_from_state(base_state, base_cfg, device)
    encoder.load_state_dict(saved["encoder"], strict=True)
    model = WoodCorrespondence().to(device)
    model.load_state_dict(saved["model"], strict=True)
    return path, encoder, model, base_cfg, saved["provenance"]["config"]


def _provenance(root, source_path, arm, seed, cfg):
    return {"source_checkpoint_sha256": study._hash_file(source_path),
            "manifest_sha256": study._hash_file(root / "swi_manifest.json"),
            "runner_sha256": study._hash_file(__file__),
            "method_sha256": study._hash_file(Path(corr.__file__).with_name(
                "wood_correspondence_method.py")),
            "arm": arm, "seed": seed, "settings": cfg}


def _bank(encoder, model, representatives, base_cfg, device):
    encoder.eval()
    model.eval()
    local_cfg = dict(base_cfg, image_batch=int(base_cfg.get("image_batch", 32)))
    embeddings = study.encode_items(encoder, representatives, local_cfg, device)
    labels = sorted({study.canonical(label) for _, label in representatives})
    indices = torch.tensor([labels.index(study.canonical(label))
                            for _, label in representatives], device=device)
    with torch.no_grad():
        transformed = corr._transformed(model, torch.as_tensor(embeddings, device=device))
        prototypes, classes, _ = class_prototypes(transformed, indices)
    if len(classes) != len(labels):
        raise ValueError("Memory bank class count changed")
    return labels, prototypes.detach()


def _bank_representatives(manifest):
    grouped = defaultdict(lambda: defaultdict(list))
    for path, label in manifest["meta-train"]:
        grouped[study.canonical(label)][study.source_scan_id(path)].append(path)
    items = []
    for label, scans in sorted(grouped.items()):
        for scan in sorted(scans, key=lambda s: study._hash_bytes(s.encode()))[:2]:
            paths = sorted(scans[scan], key=lambda p: study._hash_bytes(p.encode()))
            items.extend((path, label) for path in paths[:2])
    return items


def _choose_classes(valid, bank_labels, bank, n_way, arm, fraction, rng):
    eligible = sorted(valid)
    anchor = eligible[int(rng.integers(len(eligible)))]
    selected = [anchor]
    if arm == "hard_bank":
        lookup = {label: i for i, label in enumerate(bank_labels)}
        eligible_ids = torch.tensor([lookup[label] for label in eligible], device=bank.device)
        similarity = bank[lookup[anchor]] @ bank[eligible_ids].T
        order = similarity.argsort(descending=True).tolist()
        hard = [eligible[i] for i in order if eligible[i] != anchor]
        selected.extend(hard[:min(int((n_way - 1) * fraction), len(hard))])
    remaining = [label for label in eligible if label not in selected]
    selected.extend(rng.choice(remaining, size=n_way - len(selected),
                               replace=False).tolist())
    return selected


def _episode_plan(items, by_class, valid, bank_labels, bank, arm, cfg, seed, epoch):
    rng = np.random.default_rng(seed * 100003 + epoch)
    decks = defaultdict(list)
    batches, layouts = [], []
    schedule = ((32, 1), (64, 1), (128, 1), (32, 2), (64, 2), (32, 5))
    for step in range(cfg["episodes"]):
        n_way, k = schedule[step % len(schedule)]
        chosen = _choose_classes(valid[k], bank_labels, bank, n_way, arm,
                                 cfg["hard_fraction"], rng)
        subset = {k: {label: valid[k][label] for label in chosen}}
        refs, queries, labels, scans = image_train._episode(
            items, by_class, subset, rng, n_way, k, decks)
        batches.append(refs + queries)
        layouts.append((len(refs), labels, scans,
                        [study.canonical(items[i][1]) for i in queries]))
    return batches, layouts


def _bank_loss(model, query, refs, episode_labels, species, bank_labels, bank,
               count, arm):
    if bank.is_inference():
        bank = bank.clone()
    q = corr._transformed(model, query)
    r = corr._transformed(model, refs)
    prototypes, classes, _ = class_prototypes(
        r, torch.as_tensor(episode_labels, device=r.device))
    if not torch.equal(classes, torch.arange(len(species), device=r.device)):
        raise ValueError("Episode class order changed")
    lookup = {label: i for i, label in enumerate(bank_labels)}
    bank_ids = torch.tensor([lookup[label] for label in species], device=r.device)
    negative = q @ bank.T
    negative[:, bank_ids] = -torch.inf
    k = min(count, len(bank_labels) - len(species))
    if arm == "hard_bank":
        negatives = negative.topk(k, dim=1).values
    else:
        rng = torch.rand_like(negative)
        rng[:, bank_ids] = -torch.inf
        negatives = negative.gather(1, rng.topk(k, dim=1).indices)
    positive = (q * prototypes).sum(dim=1, keepdim=True)
    logits = torch.cat((positive, negatives), dim=1) / 0.07
    return F.cross_entropy(logits, torch.zeros(len(q), dtype=torch.long, device=q.device))


def _step(model, encoder, images, layout, bank_labels, bank, arm, config,
          settings, optimizer, device):
    n_refs, labels, scans, species = layout
    optimizer.zero_grad(set_to_none=True)
    globals_, tokens_ = [], []
    with torch.no_grad():
        for chunk in images.split(settings["microbatch"]):
            g, t = corr._encode_images(encoder, chunk.to(device, non_blocking=True))
            globals_.append(g.float())
            tokens_.append(t.float())
    all_g = torch.cat(globals_).detach().requires_grad_(True)
    all_t = torch.cat(tokens_).detach().requires_grad_(True)
    q, r = all_g[n_refs:], all_g[:n_refs]
    loss, accuracy = corr._loss(model, q, all_t[n_refs:], r, all_t[:n_refs],
                                labels, scans, config)
    bank_ce = _bank_loss(model, q, r, labels, species, bank_labels, bank,
                         settings["bank_negatives"], arm)
    loss = loss + settings["bank_weight"] * bank_ce
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite hard-negative loss")
    loss.backward()
    g_grad, t_grad = all_g.grad.detach(), all_t.grad.detach()
    if not torch.isfinite(g_grad).all() or not torch.isfinite(t_grad).all():
        raise RuntimeError("Non-finite encoder feature gradient")
    offset = 0
    for chunk in images.split(settings["microbatch"]):
        size = len(chunk)
        g, t = corr._encode_images(encoder, chunk.to(device, non_blocking=True))
        torch.autograd.backward((g.float(), t.float()),
                                (g_grad[offset:offset + size], t_grad[offset:offset + size]))
        offset += size
    torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(encoder.parameters()), 1.0)
    optimizer.step()
    return float(loss.detach()), float(bank_ce.detach()), accuracy


def _train(root, out, manifest, source, encoder, model, base_cfg, original_cfg,
           settings, arm, seed, device):
    if arm not in {"hard_bank", "random_bank"}:
        raise ValueError("Training arm must be hard_bank or random_bank")
    folder = out / arm / f"seed_{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    provenance = _provenance(root, source, arm, seed, settings)
    items = manifest["meta-train"]
    by_class, valid = image_train._layout(items)
    representatives = _bank_representatives(manifest)
    if len(valid[5]) < 128 or len({label for _, label in representatives}) != 557:
        raise ValueError("Not enough scan-disjoint species for 128-way training")
    optimizer = torch.optim.AdamW([
        {"params": model.parameters(), "lr": settings["head_lr"]},
        {"params": encoder.parameters(), "lr": settings["backbone_lr"]}], weight_decay=1e-3)
    latest, best = folder / "latest.pt", folder / "best.pt"
    start, best_score = 1, -float("inf")
    if latest.is_file():
        saved = torch.load(latest, map_location="cpu", weights_only=False)
        if saved["provenance"] != provenance:
            raise ValueError("Cannot resume checkpoint with different provenance")
        encoder.load_state_dict(saved["encoder"], strict=True)
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        start, best_score = saved["epoch"] + 1, saved["best_score"]
        print(f"[wood-large] resuming {arm} seed={seed} epoch={start}", flush=True)
    dataset = data.ManifestDataset(items, transform=data.get_transforms(224, augment=True))
    bank_cfg = dict(base_cfg, image_batch=settings["image_batch"],
                    workers=settings["workers"])
    train_cfg = dict(original_cfg, image_microbatch=settings["microbatch"])
    for epoch in range(start, settings["epochs"] + 1):
        bank_labels, bank = _bank(encoder, model, representatives, bank_cfg, device)
        batches, layouts = _episode_plan(items, by_class, valid, bank_labels, bank,
                                         arm, settings, seed, epoch)
        torch.manual_seed(seed * 100003 + epoch)
        loader = DataLoader(dataset, batch_sampler=batches,
                            num_workers=min(settings["workers"], os.cpu_count() or 1),
                            pin_memory=True)
        encoder.backbone.eval()
        encoder.projection.train()
        model.train()
        losses, bank_losses, accuracies = [], [], []
        started = time.perf_counter()
        for step, ((images, _), layout) in enumerate(zip(loader, layouts), 1):
            loss, bank_ce, accuracy = _step(model, encoder, images, layout,
                                            bank_labels, bank, arm, train_cfg,
                                            settings, optimizer, device)
            losses.append(loss)
            bank_losses.append(bank_ce)
            accuracies.append(accuracy)
            if step % 10 == 0 or step == len(layouts):
                print(f"[wood-large] {arm} seed={seed} epoch={epoch} "
                      f"step={step}/{len(layouts)} loss={np.mean(losses):.4f} "
                      f"bank={np.mean(bank_losses):.4f} train_R1={np.mean(accuracies):.4f} "
                      f"elapsed_s={time.perf_counter()-started:.1f}", flush=True)
        encoder.eval()
        model.eval()
        candidate = folder / "candidate.pt"
        payload = {"provenance": provenance, "epoch": epoch,
                   "encoder": encoder.state_dict(), "model": model.state_dict()}
        study._save_checkpoint(candidate, payload)
        frame, summary, timing = corr._evaluate(
            root, out, manifest, encoder, base_cfg, study._hash_file(candidate),
            model, train_cfg, device, folds=(0,), modes=("global", "trained"))
        score = (0.75 * float(summary.set_index(["gallery", "mode"]).loc[
            ("637x1", "trained"), "macro_r1"]) +
                 0.25 * float(summary.set_index(["gallery", "mode"]).loc[
            ("57x5", "trained"), "macro_r1"]))
        metrics = {"epoch": epoch, "selection_score": score,
                   "mean_loss": float(np.mean(losses)),
                   "mean_bank_loss": float(np.mean(bank_losses)),
                   "mean_train_r1": float(np.mean(accuracies)),
                   "unique_train_images": len(set(i for batch in batches for i in batch)),
                   "selection_fold": 0, "meta_test_evaluated": False}
        study._json(folder / f"epoch_{epoch:02d}.json", metrics)
        corr._save_report(folder, f"epoch_{epoch:02d}", frame, summary, timing,
                          {"provenance": provenance, "metrics": metrics})
        if score > best_score:
            best_score = score
            study._save_checkpoint(best, {**payload, "metrics": metrics})
        study._save_checkpoint(latest, {**payload, "optimizer": optimizer.state_dict(),
                                        "best_score": best_score})
        print(f"[wood-large] {arm} seed={seed} epoch={epoch} val_score={score:.4f} "
              f"best={best_score:.4f}", flush=True)


def _load_arm(root, study_out, image_out, out, arm, seed, device):
    upstream_seed = _upstream_seed(seed)
    if arm == "backbone_base":
        base_path, base_state, base_cfg = evidence._base_checkpoint(
            root, study_out, upstream_seed)
        source, _, _, _, config = _source_checkpoint(
            root, study_out, image_out, upstream_seed, device)
        del source
        encoder = evidence._encoder_from_state(base_state, base_cfg, device)
        model = WoodCorrespondence().to(device)
        return base_path, encoder.eval(), model.eval(), base_cfg, {
            **config, "score_mode": "global"}
    if arm == "global_only_base":
        base_path, base_state, base_cfg = evidence._base_checkpoint(
            root, study_out, upstream_seed)
        global_out = Path(os.environ.get(
            "WOOD_LARGE_GLOBAL_OUT", root / "results" /
            "wood_correspondence_global_only_v1")).resolve()
        path = global_out / "global_only" / f"seed_{upstream_seed}" / "best.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing matched global-only control: {path}")
        saved = torch.load(path, map_location="cpu", weights_only=False)
        cfg = saved["provenance"]["config"]
        if (cfg["variant"] != "global_only" or
                saved["provenance"] != image_train._provenance(
                    root, base_path, cfg, upstream_seed)):
            raise ValueError("Global-only checkpoint provenance mismatch")
        encoder = evidence._encoder_from_state(base_state, base_cfg, device)
        encoder.load_state_dict(saved["encoder"], strict=True)
        model = WoodCorrespondence().to(device)
        model.load_state_dict(saved["model"], strict=True)
        encoder.eval()
        model.eval()
        return path, encoder, model, base_cfg, cfg
    source, encoder, model, base_cfg, original_cfg = _source_checkpoint(
        root, study_out, image_out, upstream_seed, device)
    if arm != "qkv_base":
        path = out / arm / f"seed_{seed}" / "best.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing trained arm checkpoint: {path}")
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved["provenance"] != _provenance(
                root, source, arm, seed, saved["provenance"]["settings"]):
            raise ValueError("Hard-negative checkpoint provenance mismatch")
        encoder.load_state_dict(saved["encoder"], strict=True)
        model.load_state_dict(saved["model"], strict=True)
    else:
        path = source
    encoder.eval()
    model.eval()
    return path, encoder, model, base_cfg, original_cfg


def _cache_features(encoder, items, cfg, device, path, checkpoint_hash, tokens):
    path.parent.mkdir(parents=True, exist_ok=True)
    token_path = path.with_name(path.stem + "_tokens.npy")
    cursor = path.with_suffix(".progress.json")
    signature = study._hash_bytes(json.dumps({
        "checkpoint": checkpoint_hash, "items": items, "tokens": tokens,
        "image_size": 224}, separators=(",", ":")).encode())
    if cursor.is_file():
        status = json.loads(cursor.read_text())
        if (status["signature"] != signature or not path.is_file() or
                (tokens and not token_path.is_file())):
            raise ValueError(f"Stale feature cache: {path}")
        offset = status["offset"]
        globals_ = np.lib.format.open_memmap(path, mode="r+")
        locals_ = np.lib.format.open_memmap(token_path, mode="r+") if tokens else None
    else:
        if path.exists() or (tokens and token_path.exists()):
            raise ValueError(f"Unverified feature cache: {path}")
        globals_ = np.lib.format.open_memmap(
            path, mode="w+", dtype=np.float32, shape=(len(items), 512))
        locals_ = (np.lib.format.open_memmap(
            token_path, mode="w+", dtype=np.float16, shape=(len(items), 16, 512))
            if tokens else None)
        offset = 0
        study._json(cursor, {"signature": signature, "offset": offset})
    if (globals_.shape != (len(items), 512) or not 0 <= offset <= len(items) or
            (tokens and locals_.shape != (len(items), 16, 512))):
        raise ValueError("Feature cache shape/cursor mismatch")
    if offset < len(items):
        loader = study._loader(items[offset:], cfg)
        loader.dataset.loader = _RetryImageLoader(loader.dataset.loader)
        encoder.eval()
        with torch.inference_mode():
            for number, (images, _) in enumerate(loader, 1):
                with study._autocast(device):
                    if tokens:
                        g, t = corr._encode_images(encoder, images.to(device, non_blocking=True))
                    else:
                        g = encoder(images.to(device, non_blocking=True))
                if not torch.isfinite(g).all() or (tokens and not torch.isfinite(t).all()):
                    raise RuntimeError(f"Non-finite extracted features in {path}")
                n = len(images)
                globals_[offset:offset+n] = g.float().cpu().numpy()
                if tokens:
                    locals_[offset:offset+n] = t.float().cpu().numpy().astype(np.float16)
                offset += n
                if number % 100 == 0 or offset == len(items):
                    globals_.flush()
                    if tokens:
                        locals_.flush()
                    study._json(cursor, {"signature": signature, "offset": offset})
                    print(f"[wood-large] {path.name}: {offset}/{len(items)}", flush=True)
    if not np.isfinite(globals_).all() or (tokens and not np.isfinite(locals_).all()):
        raise RuntimeError(f"Non-finite completed feature cache: {path}")
    return np.load(path, mmap_mode="r"), (np.load(token_path, mmap_mode="r") if tokens else None)


class _RetryImageLoader:
    def __init__(self, loader, attempts=3):
        self.loader = loader
        self.attempts = attempts

    def load(self, path):
        for attempt in range(self.attempts):
            try:
                return self.loader.load(path)
            except RuntimeError as exc:
                if "Failed to read image:" not in str(exc):
                    raise
                if attempt + 1 < self.attempts:
                    time.sleep(2 ** attempt)
        source = Path(path)
        cached = self.loader._cache_path(path)
        raise RuntimeError(
            f"Failed to read image after {self.attempts} attempts: {path}; "
            f"source_exists={source.is_file()}, "
            f"source_bytes={source.stat().st_size if source.is_file() else None}, "
            f"cache_exists={cached.is_file()}, "
            f"cache_bytes={cached.stat().st_size if cached.is_file() else None}. "
            "Restore the exact image or its matching local image cache, then "
            "rerun full954; the feature-cache cursor will resume extraction."
        )


def _balanced_gallery(items, limit=5):
    grouped = defaultdict(lambda: defaultdict(list))
    for path, label in items:
        grouped[study.canonical(label)][study.source_scan_id(path)].append((path, label))
    chosen = []
    for label, scans in sorted(grouped.items()):
        per_class = []
        for scan in sorted(scans, key=lambda s: study._hash_bytes(s.encode())):
            per_class.append(sorted(scans[scan], key=lambda x: study._hash_bytes(x[0].encode()))[0])
            if len(per_class) >= limit:
                break
        if len(per_class) < limit:
            all_items = [x for rows in scans.values() for x in rows if x not in per_class]
            per_class.extend(sorted(all_items, key=lambda x: study._hash_bytes(x[0].encode()))
                             [:limit-len(per_class)])
        chosen.extend(per_class)
    if len({study.canonical(label) for _, label in chosen}) != len(grouped):
        raise ValueError("Balanced gallery dropped a species")
    return chosen


def _limit_queries(items, per_species):
    if not per_species:
        return items
    grouped = defaultdict(list)
    for item in items:
        grouped[study.canonical(item[1])].append(item)
    return [item for label in sorted(grouped) for item in sorted(
        grouped[label], key=lambda x: study._hash_bytes(x[0].encode()))[:per_species]]


def _macro(predictions, truth):
    classes = torch.unique(truth)
    return float(torch.stack([(predictions[truth == label] == label).float().mean()
                              for label in classes]).mean())


def _macro_hit(candidates, truth):
    hit = (candidates == truth[:, None]).any(dim=1).long()
    return float(torch.stack([
        hit[truth == label].float().mean() for label in torch.unique(truth)]).mean())


def _macro_mrr(scores, truth):
    order = scores.argsort(dim=1, descending=True)
    hits = order == truth[:, None]
    ranks = hits.float().argmax(dim=1) + 1
    reciprocal = (1.0 / ranks).masked_fill(
        torch.isneginf(scores.gather(1, truth[:, None]).squeeze(1)), 0)
    return float(torch.stack([reciprocal[truth == label].mean()
                              for label in torch.unique(truth)]).mean())


def _evaluate_gallery(root, out, manifest, arm, seed, settings, device,
                      study_out, image_out):
    path, encoder, model, base_cfg, _ = _load_arm(
        root, study_out, image_out, out, arm, seed, device)
    full_items = [(path, study.canonical(label)) for path, label in data.full_swi_items(manifest)]
    frame = study._public_frames(root, manifest)["id"]
    query_items = _limit_queries(list(zip(frame["file_path"], frame["label"])),
                                 settings["max_queries_per_species"])
    source_by_path = dict(zip(frame["file_path"], frame["source_dataset"]))
    if len(full_items) != 176123 or len(set(label for _, label in full_items)) != 954:
        raise ValueError("Full SWI gallery is not the audited 954-species cohort")
    if settings["max_queries_per_species"] == 0 and len(query_items) != 6189:
        raise ValueError("Corrected public ID cohort is not the audited 6,189 images")
    if len(set(label for _, label in query_items)) != 24:
        raise ValueError("Public ID query cohort changed")
    balanced = _balanced_gallery(full_items)
    folder = out / "full954" / arm / f"seed_{seed}"
    cache = folder / "features"
    checkpoint_hash = study._hash_file(path)
    extract_cfg = dict(base_cfg, image_batch=settings["image_batch"],
                       workers=settings["workers"])
    full_g, _ = _cache_features(encoder, full_items, extract_cfg, device,
                               cache / "swi_full.npy", checkpoint_hash, False)
    refs_g, refs_t = _cache_features(encoder, balanced, extract_cfg, device,
                                    cache / "swi_capped5.npy", checkpoint_hash, True)
    query_g, query_t = _cache_features(encoder, query_items, extract_cfg, device,
                                      cache / "public_id.npy", checkpoint_hash, True)
    labels = sorted(set(label for _, label in full_items))
    label_to_id = {label: i for i, label in enumerate(labels)}
    full_labels = torch.tensor([label_to_id[label] for _, label in full_items], device=device)
    ref_labels = torch.tensor([label_to_id[label] for _, label in balanced], device=device)
    truth = torch.tensor([label_to_id[label] for _, label in query_items], device=device)
    full = torch.as_tensor(np.asarray(full_g), device=device)
    refs = torch.as_tensor(np.asarray(refs_g), device=device)
    tokens = torch.as_tensor(np.asarray(refs_t, dtype=np.float32), device=device)
    q_global = torch.as_tensor(np.asarray(query_g), device=device)
    q_tokens = torch.as_tensor(np.asarray(query_t, dtype=np.float32), device=device)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    stage1_start = time.perf_counter()
    with torch.inference_mode():
        raw_proto, raw_max = class_scores(q_global, full, full_labels)
        adapt_full = corr._transformed(model, full)
        adapt_q = corr._transformed(model, q_global)
        adapt_proto, adapt_max = class_scores(adapt_q, adapt_full, full_labels)
        shortlist, fused = candidate_union((adapt_proto, adapt_max, raw_proto),
                                            max(settings["rerank_widths"]))
    if device.type == "cuda":
        torch.cuda.synchronize()
    stage1_seconds = time.perf_counter() - stage1_start
    candidates = {"raw_prototype": raw_proto, "raw_image_max": raw_max,
                  "adapted_prototype": adapt_proto, "adapted_image_max": adapt_max,
                  "rank_fusion": fused}
    rows, query_rows, curve = [], [], []
    for name, scores in candidates.items():
        predictions = scores.argmax(dim=1)
        for k in (1, 5, 16, 32, 64, 128):
            ranked = scores.topk(min(k, len(labels)), dim=1).indices
            rows.append({"arm": arm, "seed": seed, "method": name, "k": k,
                         "candidate_recall": candidate_recall(ranked, truth),
                         "candidate_macro_recall": _macro_hit(ranked, truth),
                         "macro_r1": _macro(predictions, truth),
                         "macro_r5": _macro_hit(scores.topk(5, dim=1).indices, truth),
                         "macro_mrr": _macro_mrr(scores, truth),
                         "n_queries": len(query_items), "n_gallery_species": 954,
                         "n_gallery_images_stage1": len(full_items),
                         "n_gallery_images_stage2": len(balanced)})
        query_rows.extend({"arm": arm, "seed": seed, "method": name, "width": 0,
                           "query_path": path, "true_label": label,
                           "predicted_label": labels[int(predictions[i])],
                           "source_dataset": source_by_path[path],
                           "correct": int(predictions[i] == truth[i]),
                           "candidate_hit": int(predictions[i] == truth[i])}
                          for i, (path, label) in enumerate(query_items))
    target_labels = sorted(set(label for _, label in query_items))
    distractors = [label for label in labels if label not in target_labels]
    distractors.sort(key=lambda label: study._hash_bytes(label.encode()))
    for n_species in (24, 50, 100, 250, 500, 954):
        included = torch.tensor([label_to_id[label] for label in
                                 target_labels + distractors[:n_species-24]], device=device)
        for name in ("raw_prototype", "adapted_prototype", "adapted_image_max"):
            subset = candidates[name][:, included]
            predicted = included[subset.argmax(dim=1)]
            curve.append({"arm": arm, "seed": seed, "method": name,
                          "gallery_species": n_species,
                          "macro_r1": _macro(predicted, truth),
                          "n_queries": len(query_items)})
    scans = [study.source_scan_id(path) for path, _ in balanced]
    rerank_mode = "global" if arm in {"global_only_base", "backbone_base"} else "qkv"
    rerank_name = "global_shortlist" if rerank_mode == "global" else "qkv_union"
    for width in settings["rerank_widths"]:
        selected = shortlist[:, :width]
        if selected.shape[1] != width:
            raise ValueError("Requested rerank width exceeds gallery classes")
        start = time.perf_counter()
        with torch.inference_mode():
            reranked = rerank_candidates(model, q_global, q_tokens, refs, tokens,
                                         ref_labels, scans, selected, adapt_proto,
                                         mode=rerank_mode,
                                         max_references_per_class=3)
        elapsed = time.perf_counter() - start
        predictions = reranked.argmax(dim=1)
        rows.append({"arm": arm, "seed": seed, "method": rerank_name,
                     "k": width, "candidate_recall": candidate_recall(selected, truth),
                     "candidate_macro_recall": _macro_hit(selected, truth),
                     "macro_r1": _macro(predictions, truth),
                     "macro_r5": _macro_hit(reranked.topk(5, dim=1).indices, truth),
                     "macro_mrr": _macro_mrr(reranked, truth),
                     "n_queries": len(query_items), "n_gallery_species": 954,
                     "n_gallery_images_stage1": len(full_items),
                     "n_gallery_images_stage2": len(balanced),
                     "rerank_ms_per_query": 1000 * elapsed / len(query_items)})
        query_rows.extend({"arm": arm, "seed": seed, "width": width,
                           "method": rerank_name,
                           "query_path": path, "true_label": label,
                           "predicted_label": labels[int(predictions[i])],
                           "source_dataset": source_by_path[path],
                           "correct": int(predictions[i] == truth[i]),
                           "candidate_hit": int((selected[i] == truth[i]).any())}
                          for i, (path, label) in enumerate(query_items))
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(folder / "full954_summary.csv", index=False)
    query_frame = pd.DataFrame(query_rows)
    query_frame.to_csv(folder / "full954_queries.csv", index=False)
    pd.DataFrame(curve).to_csv(folder / "cardinality_curve.csv", index=False)
    per_source = query_frame.groupby(
        ["method", "width", "source_dataset"], as_index=False).agg(
            n_queries=("correct", "size"),
            n_species=("true_label", "nunique"),
            macro_r1=("correct", lambda s: s.groupby(
                query_frame.loc[s.index, "true_label"]).mean().mean()))
    per_source.to_csv(folder / "per_source.csv", index=False)
    study._json(folder / "protocol.json", {
        "checkpoint_sha256": checkpoint_hash,
        "manifest_sha256": study._hash_file(root / "swi_manifest.json"),
        "id_csv_sha256": study._hash_file(root / "ID_images_expanded.csv"),
        "query_limit_per_species": settings["max_queries_per_species"],
        "n_queries": len(query_items), "n_stage1_images": len(full_items),
        "n_stage2_images": len(balanced), "n_species": 954,
        "stage1_ms_per_query": 1000 * stage1_seconds / len(query_items),
        "timing_scope": "scoring precomputed embeddings only; image encoding excluded",
        "peak_gpu_gb": (torch.cuda.max_memory_allocated() / 2**30
                        if device.type == "cuda" else None),
        "stage2_cache_rule": "up to 5 images/species, scan-diverse, deterministic",
        "stage2_score_rule": "up to 3 distinct scans/species among cached references",
        "encoder_train_species": 557, "query_species_split": "meta-test",
        "public_id_test_evaluated": True, "selection_from_public_id": False})
    print(f"[wood-large] full954 result: {folder / 'full954_summary.csv'}", flush=True)
    return folder


def _validate(root, out, manifest, study_out, image_out, arms, seeds, device):
    rows, selection_rows = [], []
    for arm in arms:
        for seed in seeds:
            path, encoder, model, base_cfg, config = _load_arm(
                root, study_out, image_out, out, arm, seed, device)
            folder = out / "meta_val" / arm / f"seed_{seed}"
            modes = ("global", "global_adapt", "fixed_local", "trained")
            if config["score_mode"] == "qkv":
                modes += ("maxsim", "qkv_no_gate", "qkv_no_consensus", "qkv_top128")
            frame, summary, timing = corr._evaluate(
                root, out, manifest, encoder, base_cfg, study._hash_file(path),
                model, config, device, folds=(0, 1),
                modes=modes)
            corr._save_report(folder, "selected", frame, summary, timing,
                              {"checkpoint_sha256": study._hash_file(path),
                               "selection_fold": 0, "descriptive_fold": 1,
                               "meta_test_evaluated": False})
            rows.extend({"arm": arm, "seed": seed, **item}
                        for item in summary.to_dict("records"))
            selected = frame[(frame["fold"] == 0) & (frame["mode"] == "trained")]
            by_gallery = selected.groupby("gallery")["correct"].mean()
            selection_rows.append({"arm": arm, "seed": seed,
                                   "r1_57x5": float(by_gallery["57x5"]),
                                   "r1_637x1": float(by_gallery["637x1"]),
                                   "selection_score": float(0.25 * by_gallery["57x5"] +
                                                            0.75 * by_gallery["637x1"]),
                                   "checkpoint_sha256": study._hash_file(path)})
    pd.DataFrame(rows).to_csv(out / "meta_val_all_arms.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(out / "selection_candidates.csv", index=False)
    print(f"[wood-large] meta-val report: {out / 'meta_val_all_arms.csv'}", flush=True)


def _select(out, arms, seeds):
    path = out / "selection_candidates.csv"
    if not path.is_file():
        raise FileNotFoundError("Run meta-val validation before selecting a method")
    frame = pd.read_csv(path)
    expected = {(arm, seed) for arm in arms for seed in seeds}
    actual = set(zip(frame["arm"], frame["seed"]))
    if actual != expected or len(frame) != len(expected):
        raise ValueError("Validation rows do not cover the requested arms and seeds")
    ranked = frame.groupby("arm")["selection_score"].mean().sort_values(ascending=False)
    winner = str(ranked.index[0])
    seed = 43 if 43 in seeds else min(seeds)
    selected = frame[(frame["arm"] == winner) & (frame["seed"] == seed)].iloc[0]
    record = {"selected_arm": winner, "report_seed": seed,
              "selected_checkpoint_sha256": selected["checkpoint_sha256"],
              "selection_score_mean_across_seeds": float(ranked.iloc[0]),
              "selection_fold": 0, "metric": "0.75 R@1(637x1) + 0.25 R@1(57x5)",
              "candidate_csv_sha256": study._hash_file(path),
              "meta_test_evaluated": False, "public_id_test_evaluated": False}
    lock = out / "selection_lock.json"
    if lock.is_file() and json.loads(lock.read_text()) != record:
        raise ValueError("Existing selection lock differs; use a new output directory")
    study._json(lock, record)
    print(f"[wood-large] selected {winner}, seed={seed}: {lock}", flush=True)
    return record


def _locked(out, root, study_out, image_out, device):
    lock = out / "selection_lock.json"
    if not lock.is_file():
        raise FileNotFoundError("Run selection on meta-val before opening test cohorts")
    selected = json.loads(lock.read_text())
    if selected["candidate_csv_sha256"] != study._hash_file(out / "selection_candidates.csv"):
        raise ValueError("Selection input changed after locking")
    path, _, _, _, _ = _load_arm(root, study_out, image_out, out,
                                 selected["selected_arm"], selected["report_seed"], device)
    if study._hash_file(path) != selected["selected_checkpoint_sha256"]:
        raise ValueError("Selected checkpoint changed after locking")
    return selected


def _meta_test(root, out, manifest, study_out, image_out, selected, arms, device):
    rows = []
    seed = selected["report_seed"]
    controls = tuple(arm for arm in ("backbone_base", "global_only_base", "qkv_base")
                     if arm in arms)
    for arm in dict.fromkeys((*controls, selected["selected_arm"])):
        path, encoder, model, base_cfg, config = _load_arm(
            root, study_out, image_out, out, arm, seed, device)
        frame, summary, timing = corr._evaluate(
            root, out, manifest, encoder, base_cfg, study._hash_file(path),
            model, config, device, split="meta-test", folds=(0, 1),
            modes=("global", "global_adapt", "fixed_local", "trained"))
        folder = out / "meta_test" / arm / f"seed_{seed}"
        corr._save_report(folder, "locked", frame, summary, timing,
                          {"selection_lock_sha256": study._hash_file(
                              out / "selection_lock.json"),
                           "checkpoint_sha256": study._hash_file(path),
                           "meta_test_evaluated": True})
        rows.extend({"arm": arm, "seed": seed, **item}
                    for item in summary.to_dict("records"))
    pd.DataFrame(rows).to_csv(out / "meta_test_locked_summary.csv", index=False)


def _compare_full(out, selected, arms):
    seed = selected["report_seed"]
    primary = selected["selected_arm"]
    baselines = [arm for arm in ("backbone_base", "qkv_base", "global_only_base")
                 if arm in arms and arm != primary]
    primary_file = out / "full954" / primary / f"seed_{seed}" / "full954_queries.csv"
    primary_rows = pd.read_csv(primary_file)
    primary_rows = primary_rows[primary_rows["width"] > 0]
    results = []
    for baseline in baselines:
        path = out / "full954" / baseline / f"seed_{seed}" / "full954_queries.csv"
        reference = pd.read_csv(path)
        reference = reference[reference["width"] > 0]
        for width in sorted(primary_rows["width"].unique()):
            left = primary_rows[primary_rows["width"] == width]
            right = reference[reference["width"] == width]
            keys = ["query_path", "true_label", "source_dataset", "width"]
            paired = left.merge(right, on=keys, how="outer", validate="one_to_one",
                                suffixes=("_selected", "_baseline"), indicator=True)
            if len(paired) != len(left) or not paired["_merge"].eq("both").all():
                raise ValueError("Public ID paired comparison changed query identities")
            group = paired.groupby("true_label", sort=True)
            species_delta = (group["correct_selected"].mean() -
                             group["correct_baseline"].mean()).to_numpy()
            rng = np.random.default_rng(2026 + int(width))
            draws = rng.integers(0, len(species_delta), size=(5000, len(species_delta)))
            bootstrap = species_delta[draws].mean(axis=1)
            results.append({"selected_arm": primary, "baseline_arm": baseline,
                            "seed": seed, "width": int(width),
                            "n_queries": len(paired), "n_species": len(species_delta),
                            "selected_macro_r1": float(group["correct_selected"].mean().mean()),
                            "baseline_macro_r1": float(group["correct_baseline"].mean().mean()),
                            "delta": float(species_delta.mean()),
                            "ci95_low": float(np.quantile(bootstrap, 0.025)),
                            "ci95_high": float(np.quantile(bootstrap, 0.975)),
                            "rescued": int(((paired["correct_selected"] == 1) &
                                            (paired["correct_baseline"] == 0)).sum()),
                            "harmed": int(((paired["correct_selected"] == 0) &
                                           (paired["correct_baseline"] == 1)).sum()),
                            "ci_scope": "fixed checkpoints; species bootstrap only"})
    if results:
        pd.DataFrame(results).to_csv(out / "full954_paired.csv", index=False)


def _smoke(device):
    torch.manual_seed(7)
    query = F.normalize(torch.randn(4, 8, device=device), dim=1)
    refs = F.normalize(torch.randn(8, 8, device=device), dim=1)
    labels = torch.arange(4, device=device).repeat_interleave(2)
    proto, nearest = class_scores(query, refs, labels)
    chosen, _ = candidate_union((proto, nearest), 3)
    model = WoodCorrespondence(dimension=8, token_dim=4).to(device)
    qt = torch.randn(4, 16, 8, device=device)
    rt = torch.randn(8, 16, 8, device=device)
    scored = rerank_candidates(model, query, qt, refs, rt, labels,
                               [f"s{i}" for i in range(8)], chosen, proto)
    if scored.shape != (4, 4) or not torch.isfinite(scored.gather(1, chosen)).all():
        raise RuntimeError("Candidate reranking smoke test failed")
    bank_labels = [f"species_{i:03d}" for i in range(140)]
    bank = F.normalize(torch.randn(140, 8, device=device), dim=1)
    for arm in ("hard_bank", "random_bank"):
        loss = _bank_loss(model, query, refs, labels.tolist(), bank_labels[:4],
                          bank_labels, bank, 32, arm)
        if not torch.isfinite(loss):
            raise RuntimeError("Memory-bank loss smoke test failed")
    with torch.inference_mode():
        inference_bank = bank.clone()
    differentiable_query = query.detach().requires_grad_(True)
    _bank_loss(model, differentiable_query, refs, labels.tolist(), bank_labels[:4],
               bank_labels, inference_bank, 32, "hard_bank").backward()
    if (differentiable_query.grad is None or
            not torch.isfinite(differentiable_query.grad).all()):
        raise RuntimeError("Memory-bank gradient smoke test failed")
    items = [(f"/scale_256/patch_{j}_from_Tw{i*4+s:05d}.jpg", label)
             for i, label in enumerate(bank_labels) for s in range(2) for j in range(6)]
    by_class, valid = image_train._layout(items)
    for arm in ("hard_bank", "random_bank"):
        batches, layouts = _episode_plan(items, by_class, valid, bank_labels, bank,
                                         arm, {"episodes": 6, "hard_fraction": 0.5}, 43, 1)
        for batch, (n_refs, episode_labels, scans, species) in zip(batches, layouts):
            if len(species) != len(batch) - n_refs:
                raise RuntimeError("Episode query/class mapping failed")
            for class_id, query_id in enumerate(batch[n_refs:]):
                if study.canonical(items[query_id][1]) != species[class_id]:
                    raise RuntimeError("Episode species order failed")
                if any(scan == study.source_scan_id(items[query_id][0])
                       for scan, label in zip(scans, episode_labels) if label == class_id):
                    raise RuntimeError("Scan leakage in hard-negative episode")
    print(f"[wood-large] smoke passed on {device}", flush=True)


def _export(out):
    target = out / "wood_large_gallery_review.zip"
    count = 0
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6) as archive:
        for path in sorted(out.rglob("*")):
            if (path.is_file() and path != target and path.suffix in {".csv", ".json"}
                    and "feature_cache" not in path.parts and
                    "features" not in path.parts and not path.name.endswith(".progress.json")):
                archive.write(path, path.relative_to(out))
                count += 1
    if not count:
        raise RuntimeError("No review artifacts found")
    print(f"[wood-large] review ZIP: {target} ({count} files, "
          f"{target.stat().st_size/2**20:.1f} MiB)", flush=True)
    return target


def run():
    root = Path(os.environ.get("ROOT_PATH", "/content/drive/MyDrive/NCS")).resolve()
    out = Path(os.environ.get("WOOD_LARGE_OUT", root / "results" /
                              "wood_large_gallery_v1")).resolve()
    study_out = Path(os.environ.get("WOOD_LARGE_BASE_OUT", root / "results" /
                                    "dinov2s_retrieval_matched_v1")).resolve()
    image_out = Path(os.environ.get("WOOD_LARGE_QKV_OUT", root / "results" /
                                    "wood_correspondence_image_full_v1")).resolve()
    mode = os.environ.get("WOOD_LARGE_MODE", "preflight").lower()
    if mode not in {"smoke", "preflight", "train", "validate", "select", "meta_test",
                    "full954", "export"}:
        raise ValueError("Invalid WOOD_LARGE_MODE")
    arms = [x.strip() for x in os.environ.get(
        "WOOD_LARGE_ARMS", "backbone_base,qkv_base,global_only_base,hard_bank,random_bank").split(",")]
    seeds = [int(x) for x in os.environ.get("WOOD_LARGE_SEEDS", "43").split(",")]
    if not arms or not seeds or any(x not in {"backbone_base", "qkv_base", "global_only_base",
                                            "hard_bank", "random_bank"}
                                     for x in arms):
        raise ValueError("Invalid large-gallery arms or seeds")
    settings = _settings()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if mode == "smoke":
        _smoke(device)
        return
    if mode == "export":
        return _export(out)
    manifest = data.load_swi_manifest(root / "swi_manifest.json")
    out.mkdir(parents=True, exist_ok=True)
    plan = {"arms": arms, "seeds": seeds, "settings": settings,
            "selection_split": "meta-val fold 0",
            "test_splits": ["meta-test", "corrected public ID against full SWI"]}
    plan_path = out / "study_plan.json"
    if plan_path.is_file() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Study plan changed; use a new output directory")
    study._json(plan_path, plan)
    if mode in {"train", "validate"} and (out / "selection_lock.json").is_file():
        raise ValueError("Selection is locked; use a new output directory for further training")
    if mode == "preflight":
        train_items = manifest["meta-train"]
        full_items = data.full_swi_items(manifest)
        id_frame = study._public_frames(root, manifest)["id"]
        if (len(train_items) != 124577 or
                len({study.canonical(label) for _, label in train_items}) != 557 or
                len(full_items) != 176123 or
                len({study.canonical(label) for _, label in full_items}) != 954 or
                len(id_frame) != 6189 or
                len({study.canonical(label) for label in id_frame["label"]}) != 24):
            raise ValueError("Input cohorts differ from the audited 557/954/24-species protocol")
        print("[wood-large] cohorts: train=124577/557, SWI=176123/954, "
              "public-ID=6189/24", flush=True)
        for seed in seeds:
            source, _, _, _, _ = _load_arm(
                root, study_out, image_out, out, "qkv_base", seed, device)
            print(f"[wood-large] source seed={seed}: {source}", flush=True)
            if "global_only_base" in arms:
                control, _, _, _, _ = _load_arm(
                    root, study_out, image_out, out, "global_only_base", seed, device)
                print(f"[wood-large] global control seed={seed}: {control}", flush=True)
        print(f"[wood-large] preflight complete: {out}", flush=True)
        return
    if device.type != "cuda":
        raise RuntimeError("Large-gallery image study requires CUDA")
    if mode == "train":
        for seed in seeds:
            for arm in arms:
                if arm in {"backbone_base", "qkv_base", "global_only_base"}:
                    continue
                source, encoder, model, base_cfg, original_cfg = _source_checkpoint(
                    root, study_out, image_out, _upstream_seed(seed), device)
                _train(root, out, manifest, source, encoder, model, base_cfg,
                       original_cfg, settings, arm, seed, device)
    elif mode == "validate":
        _validate(root, out, manifest, study_out, image_out, arms, seeds, device)
    elif mode == "select":
        _select(out, arms, seeds)
    elif mode == "meta_test":
        selected = _locked(out, root, study_out, image_out, device)
        _meta_test(root, out, manifest, study_out, image_out, selected, arms, device)
    else:
        selected = _locked(out, root, study_out, image_out, device)
        seed = selected["report_seed"]
        for arm in dict.fromkeys((*(arm for arm in ("backbone_base", "qkv_base",
                                                   "global_only_base") if arm in arms),
                                  selected["selected_arm"])):
            _evaluate_gallery(root, out, manifest, arm, seed, settings, device,
                              study_out, image_out)
        _compare_full(out, selected, arms)
