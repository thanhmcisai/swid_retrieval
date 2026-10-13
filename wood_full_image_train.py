"""Full-query-coverage ArcFace, contrastive, and QKV training on meta-train."""

import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from . import data, gallery_experiment as study
from . import wood_correspondence_experiment as corr
from . import wood_evidence_experiment as evidence
from .gallery_method import ArcFaceClassifier
from .wood_correspondence_method import WoodCorrespondence


ARMS = ("arcface", "arcface_qkv", "arcface_pair_nce", "arcface_supcon",
        "arcface_packed", "arcface_multipos", "arcface_multipos_qkv")


def _config():
    cfg = {
        "arms": [x.strip() for x in os.environ.get(
            "WOOD_FULL_ARMS", "arcface,arcface_qkv").split(",")],
        "seeds": [int(x) for x in os.environ.get("WOOD_FULL_SEEDS", "43").split(",")],
        "epochs": int(os.environ.get("WOOD_FULL_EPOCHS", "3")),
        "batch": int(os.environ.get("WOOD_FULL_BATCH", "64")),
        "microbatch": int(os.environ.get("WOOD_FULL_MICROBATCH", "16")),
        "workers": int(os.environ.get("WOOD_FULL_WORKERS", "2")),
        "backbone_lr": float(os.environ.get("WOOD_FULL_BACKBONE_LR", "2e-6")),
        "head_lr": float(os.environ.get("WOOD_FULL_HEAD_LR", "1e-4")),
        "arcface_weight": float(os.environ.get("WOOD_FULL_ARCFACE_WEIGHT", "1")),
        "qkv_weight": float(os.environ.get("WOOD_FULL_QKV_WEIGHT", "0.5")),
        "contrastive_weight": float(os.environ.get("WOOD_FULL_CONTRASTIVE_WEIGHT", "0.2")),
        "contrastive_temperature": float(os.environ.get("WOOD_FULL_CONTRASTIVE_TEMPERATURE", "0.07")),
        "source_seed": int(os.environ.get("WOOD_FULL_SOURCE_SEED", "43")),
        "print_every": int(os.environ.get("WOOD_FULL_PRINT_EVERY", "200")),
    }
    if (not cfg["arms"] or len(set(cfg["arms"])) != len(cfg["arms"]) or
            any(x not in ARMS for x in cfg["arms"]) or not cfg["seeds"] or
            cfg["epochs"] < 1 or cfg["batch"] < 2 or cfg["microbatch"] < 1 or
            cfg["workers"] < 0 or cfg["backbone_lr"] <= 0 or cfg["head_lr"] <= 0 or
            cfg["arcface_weight"] <= 0 or cfg["qkv_weight"] <= 0 or
            cfg["contrastive_weight"] <= 0 or cfg["contrastive_temperature"] <= 0 or
            cfg["print_every"] < 1):
        raise ValueError("Invalid full-image study configuration")
    return cfg


def reference_plan(items, seed, epoch):
    """Each source image is a query once; select a distinct same-species reference."""
    groups = defaultdict(list)
    for index, (_, label) in enumerate(items):
        groups[study.canonical(label)].append(index)
    rng = np.random.default_rng(seed * 100003 + epoch)
    scans = [study.source_scan_id(path) for path, _ in items]
    scales = [study.image_scale(path) for path, _ in items]
    result = np.empty(len(items), dtype=np.int64)
    cross_scan = cross_scale = 0
    for indices in groups.values():
        if len(indices) < 2:
            raise ValueError("Every meta-train species needs two distinct images")
        by_scan = defaultdict(list)
        for index in indices:
            by_scan[scans[index]].append(index)
        for index in indices:
            eligible = [j for scan, rows in by_scan.items() if scan != scans[index]
                        for j in rows]
            if not eligible:
                eligible = [j for j in indices if j != index]
            other_scale = [j for j in eligible if scales[j] != scales[index]]
            chosen = int(rng.choice(other_scale if other_scale else eligible))
            result[index] = chosen
            cross_scan += scans[index] != scans[chosen]
            cross_scale += scales[index] != scales[chosen]
    return result, {"query_images": len(items), "cross_scan_pairs": cross_scan,
                    "cross_scale_pairs": cross_scale,
                    "same_scan_fallback_pairs": len(items) - cross_scan}


def class_balanced_batches(items, batch_size, seed):
    """Visit every image once while maximizing distinct species per batch."""
    groups = defaultdict(list)
    for index, (_, label) in enumerate(items):
        groups[study.canonical(label)].append(index)
    rng = np.random.default_rng(seed)
    for indices in groups.values():
        rng.shuffle(indices)
    active = list(groups)
    batches = []
    while active:
        selected = []
        while len(selected) < batch_size and active:
            rng.shuffle(active)
            for label in active:
                selected.append(groups[label].pop())
                if len(selected) == batch_size:
                    break
            active = [label for label in active if groups[label]]
        batches.append(selected)
    if sorted(i for batch in batches for i in batch) != list(range(len(items))):
        raise RuntimeError("Full-image batch plan omitted or repeated queries")
    return batches


def multi_positive_batches(items, batch_size, seed):
    """Cover every query once, grouping pairs across scans/scales when possible."""
    if batch_size < 4:
        raise ValueError("Multi-positive batches need at least four images")
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    scans = [study.source_scan_id(path) for path, _ in items]
    scales = [study.image_scale(path) for path, _ in items]
    for index, (_, label) in enumerate(items):
        groups[study.canonical(label)].append(index)
    units = {}
    for label, indices in groups.items():
        rng.shuffle(indices)
        pairs = []
        while indices:
            first = indices.pop()
            if not indices:
                pairs.append([first])
                break
            scores = [(int(scans[first] != scans[j]), int(scales[first] != scales[j]))
                      for j in indices]
            best = max(scores)
            candidates = [pos for pos, score in enumerate(scores) if score == best]
            second = indices.pop(int(rng.choice(candidates)))
            pairs.append([first, second])
        units[label] = pairs
    active = list(units)
    batches = []
    while active:
        batch = []
        rng.shuffle(active)
        for label in active:
            if units[label] and len(batch) + len(units[label][-1]) <= batch_size:
                batch.extend(units[label].pop())
        active = [label for label in active if units[label]]
        if not batch:
            raise RuntimeError("Multi-positive batch packing made no progress")
        batches.append(batch)
    if sorted(i for batch in batches for i in batch) != list(range(len(items))):
        raise RuntimeError("Multi-positive batches omitted or repeated queries")
    return batches


def batch_positive_audit(items, batches):
    scans = [study.source_scan_id(path) for path, _ in items]
    scales = [study.image_scale(path) for path, _ in items]
    pairs = cross_scan = cross_scale = 0
    for batch in batches:
        groups = defaultdict(list)
        for index in batch:
            groups[study.canonical(items[index][1])].append(index)
        for indices in groups.values():
            for offset, first in enumerate(indices):
                for second in indices[offset + 1:]:
                    pairs += 1
                    cross_scan += scans[first] != scans[second]
                    cross_scale += scales[first] != scales[second]
    return {"query_query_positive_pairs": pairs,
            "cross_scan_pairs": cross_scan, "cross_scale_pairs": cross_scale}


def _pair_nce(query, references, labels, temperature):
    query = F.normalize(query.float(), dim=1)
    references = F.normalize(references.float(), dim=1)
    logits = query @ references.T / temperature
    same = labels[:, None] == labels[None, :]
    diagonal = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    logits = logits.masked_fill(same & ~diagonal, -torch.inf)
    return F.cross_entropy(logits, torch.arange(len(labels), device=labels.device))


def _multi_positive_nce(query, references, labels, temperature):
    features = F.normalize(torch.cat((query, references), dim=0).float(), dim=1)
    all_labels = torch.cat((labels, labels))
    self_mask = torch.eye(len(features), dtype=torch.bool, device=features.device)
    positive = (all_labels[:, None] == all_labels[None, :]) & ~self_mask
    logits = (features @ features.T / temperature).masked_fill(self_mask, -torch.inf)
    log_prob = F.log_softmax(logits, dim=1)
    return -(log_prob.masked_fill(~positive, 0).sum(1) /
             positive.sum(1).clamp_min(1)).mean()


class PairDataset(Dataset):
    def __init__(self, items, references, image_size=224):
        self.items = items
        self.references = references
        self.loader = data.CachedImageLoader(use_ram_cache=False)
        self.transform = data.get_transforms(image_size, augment=True)
        names = sorted({study.canonical(label) for _, label in items})
        self.lookup = {name: i for i, name in enumerate(names)}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, label = self.items[index]
        ref_path, ref_label = self.items[int(self.references[index])]
        if index == int(self.references[index]) or study.canonical(label) != study.canonical(ref_label):
            raise ValueError("Invalid positive reference")
        q = self.transform(image=self.loader.load(path))["image"]
        r = self.transform(image=self.loader.load(ref_path))["image"]
        return q, r, self.lookup[study.canonical(label)], index


def _loss(model, classifier, globals_, tokens, labels, arm, weight,
          extra_label=None, contrastive_weight=0.2, temperature=0.07,
          arcface_weight=1.0):
    batch = len(labels)
    q, r = globals_[:batch], globals_[batch:]
    qt, rt = tokens[:batch], tokens[batch:]
    reference_labels = (labels if extra_label is None else
                        torch.cat((labels, labels.new_tensor([extra_label]))))
    arcface = (classifier(q, labels) + classifier(r, reference_labels)) / 2
    contrastive = None
    if arm == "arcface_pair_nce":
        contrastive = _pair_nce(q, r[:batch], labels, temperature)
    elif arm in {"arcface_supcon", "arcface_multipos", "arcface_multipos_qkv"}:
        contrastive = _multi_positive_nce(q, r[:batch], labels, temperature)
    loss = arcface_weight * arcface
    if contrastive is not None:
        loss = loss + contrastive_weight * contrastive
    metrics = {"arcface": float(arcface.detach()),
               "contrastive": None if contrastive is None else float(contrastive.detach()),
               "qkv": None, "rank1": None,
               "candidate_classes": int(reference_labels.unique().numel()),
               "positive_views_per_anchor": float((
                   2 * torch.bincount(labels).index_select(0, labels) - 1
               ).float().mean())}
    if arm not in {"arcface_qkv", "arcface_multipos_qkv"}:
        return loss, metrics
    # Every query competes with all reference classes in the current batch.
    # The paired reference is always present; no label is injected at inference.
    scores, classes = model(q, qt, r, rt, reference_labels, list(range(len(r))),
                            mode="qkv", consensus=False, query_chunk=8,
                            reference_chunk=32)
    target = torch.searchsorted(classes, labels)
    if not torch.equal(classes[target], labels):
        raise RuntimeError("Positive reference absent from QKV batch")
    qkv = F.cross_entropy(scores / 0.07, target)
    metrics.update(qkv=float(qkv.detach()),
                   rank1=float((scores.argmax(1) == target).float().mean().detach()),
                   candidate_classes=len(classes))
    return loss + weight * qkv, metrics


def _step(encoder, model, classifier, batch, arm, cfg, optimizer, device,
          extra_reference=None):
    query, reference, labels, _ = batch
    labels = labels.to(device, non_blocking=True)
    extra_label = None
    if extra_reference is not None:
        image, extra_label = extra_reference
        reference = torch.cat((reference, image[None]), dim=0)
    images = torch.cat((query, reference), dim=0)
    optimizer.zero_grad(set_to_none=True)
    global_parts, token_parts = [], []
    with torch.no_grad():
        for chunk in images.split(cfg["microbatch"]):
            with study._autocast(device):
                g, t = corr._encode_images(encoder, chunk.to(device, non_blocking=True))
            global_parts.append(g.float())
            token_parts.append(t.float())
    globals_ = torch.cat(global_parts).detach().requires_grad_(True)
    tokens = torch.cat(token_parts).detach().requires_grad_(True)
    loss, metrics = _loss(model, classifier, globals_, tokens, labels,
                          arm, cfg["qkv_weight"], extra_label,
                          cfg.get("contrastive_weight", 0.2),
                          cfg.get("contrastive_temperature", 0.07),
                          cfg.get("arcface_weight", 1.0))
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite full-image loss")
    loss.backward()
    gg = globals_.grad.detach()
    tg = tokens.grad.detach() if tokens.grad is not None else torch.zeros_like(tokens)
    if not torch.isfinite(gg).all() or not torch.isfinite(tg).all():
        raise RuntimeError("Non-finite encoder gradient")
    offset = 0
    for chunk in images.split(cfg["microbatch"]):
        size = len(chunk)
        with study._autocast(device):
            g, t = corr._encode_images(encoder, chunk.to(device, non_blocking=True))
        torch.autograd.backward((g.float(), t.float()),
                                (gg[offset:offset + size], tg[offset:offset + size]))
        offset += size
    torch.nn.utils.clip_grad_norm_(
        list(encoder.parameters()) + list(model.parameters()) + list(classifier.parameters()), 1.0)
    optimizer.step()
    return float(loss.detach()), metrics


def _validate_fold(manifest, encoder, model, base_cfg, arm, device, fold):
    refs, queries, extras = evidence._validation_sets(manifest, "meta-val", fold)
    plans = evidence._gallery_plan(refs, queries, extras, "meta-val")
    items = refs + extras
    extract = dict(base_cfg, image_batch=64, workers=2)
    rg, rt = evidence._extract(encoder, items, extract, device)
    qg, qt = evidence._extract(encoder, queries, extract, device)
    results = {}
    with torch.no_grad():
        for gallery in ("57x5", "57x1", "637x1"):
            ri, qi = plans[gallery]
            selected = [items[i] for i in ri]
            names = sorted({study.canonical(label) for _, label in selected})
            lookup = {name: i for i, name in enumerate(names)}
            labels = torch.tensor([lookup[study.canonical(label)] for _, label in selected],
                                  device=device)
            scans = [study.source_scan_id(path) for path, _ in selected]
            truth = torch.tensor([lookup[study.canonical(queries[i][1])] for i in qi],
                                 device=device)
            r = torch.from_numpy(rg[ri]).to(device)
            rtensor = torch.from_numpy(rt[ri]).to(device)
            q = torch.from_numpy(qg[qi]).to(device)
            qtensor = torch.from_numpy(qt[qi]).to(device)
            predictions, global_predictions, recall128 = [], [], []
            for start in range(0, len(q), 8):
                global_score, global_classes = model(
                    q[start:start + 8], qtensor[start:start + 8], r, rtensor,
                    labels, scans, mode="global")
                global_predictions.append(global_classes[global_score.argmax(1)])
                recall128.append((global_classes[global_score.topk(
                    min(128, len(names)), dim=1).indices] ==
                    truth[start:start + 8, None]).any(dim=1))
                score, classes = model(
                    q[start:start + 8], qtensor[start:start + 8], r, rtensor,
                    labels, scans, mode="qkv" if arm in {
                        "arcface_qkv", "arcface_multipos_qkv"} else "global",
                    top_classes=128 if arm in {"arcface_qkv", "arcface_multipos_qkv"}
                    and len(names) > 128 else None,
                    query_chunk=8, reference_chunk=64)
                predictions.append(classes[score.argmax(1)])
            results[gallery] = float((torch.cat(predictions) == truth).float().mean())
            results[f"{gallery}_global"] = float(
                (torch.cat(global_predictions) == truth).float().mean())
            results[f"{gallery}_candidate_recall128"] = float(
                torch.cat(recall128).float().mean())
    return results


def _validate(manifest, encoder, model, base_cfg, arm, device):
    folds = [_validate_fold(manifest, encoder, model, base_cfg, arm, device, fold)
             for fold in (0, 1)]
    return {key: float(np.mean([part[key] for part in folds])) for key in folds[0]}


def run():
    cfg = _config()
    root = Path(os.environ.get("ROOT_PATH", "/content/drive/MyDrive/NCS")).resolve()
    out = Path(os.environ.get("WOOD_FULL_OUT", root / "results/wood_full_image_ablation_v2"))
    mode = os.environ.get("WOOD_FULL_MODE", "train")
    device = torch.device(os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    if mode == "export":
        from .wood_full_image_eval import export_review
        return export_review(out)
    manifest = data.load_swi_manifest(root / "swi_manifest.json")
    items = manifest["meta-train"]
    if mode == "preflight":
        from .wood_full_image_eval import _reference_pool
        print(f"[wood-full] meta-train: {len(items)} images / "
              f"{len(set(label for _, label in items))} species; arms={cfg['arms']}")
        for seed in cfg["seeds"]:
            _, audit = reference_plan(items, seed, 1)
            print(f"[wood-full] seed={seed} pair audit: {audit}")
        for fold in (0, 1):
            _, queries, _ = evidence._validation_sets(manifest, "meta-val", fold)
            pool = _reference_pool(manifest, queries)
            print(f"[wood-full] fold={fold} K=10 eligible: "
                  f"{len(queries)} queries / {len({label for _, label in pool})} species")
        return
    if mode not in {"train", "validate"}:
        raise ValueError("WOOD_FULL_MODE must be preflight, train, validate, or export")
    source_root = Path(os.environ.get(
        "WOOD_FULL_SOURCE_OUT", root / "results/dinov2s_retrieval_matched_v1"))
    base_path, base_state, base_cfg = evidence._base_checkpoint(
        root, source_root, cfg["source_seed"])
    if mode == "train" and os.environ.get("WOOD_FULL_PRELOAD", "1") == "1":
        data.preload_image_cache(
            (path for path, _ in items),
            max_workers=int(os.environ.get("WOOD_FULL_PRELOAD_WORKERS", "8")),
            desc="Full-image meta-train preload")
    recipe = {key: value for key, value in cfg.items()
              if key not in {"arms", "seeds", "epochs", "print_every", "workers"}}
    provenance = {"manifest_sha256": study._hash_file(root / "swi_manifest.json"),
                  "source_sha256": study._hash_file(base_path),
                  "runner_sha256": study._hash_file(__file__),
                  "correspondence_sha256": study._hash_file(Path(corr.__file__).with_name(
                      "wood_correspondence_method.py")),
                  "recipe": recipe}
    out.mkdir(parents=True, exist_ok=True)
    existing_provenance = out / "provenance.json"
    if existing_provenance.is_file():
        import json
        if json.loads(existing_provenance.read_text()) != provenance:
            raise ValueError("Full-image study provenance changed; choose a new output folder")
    else:
        study._json(existing_provenance, provenance)
    if mode == "validate":
        from . import wood_full_image_eval as evaluation
        for arm in cfg["arms"]:
            for seed in cfg["seeds"]:
                checkpoint = out / arm / f"seed_{seed}" / "best.pt"
                if not checkpoint.is_file():
                    raise FileNotFoundError(checkpoint)
                saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
                if (saved["provenance"] != provenance or saved["arm"] != arm or
                        saved["seed"] != seed):
                    raise ValueError(f"Checkpoint provenance mismatch: {checkpoint}")
                encoder = evidence._encoder_from_state(base_state, base_cfg, device)
                model = WoodCorrespondence().to(device)
                encoder.load_state_dict(saved["encoder"], strict=True)
                model.load_state_dict(saved["model"], strict=True)
                encoder.eval()
                model.eval()
                evaluation.evaluate(manifest, encoder, model, base_cfg, arm, device,
                                    out / "validation" / arm / f"seed_{seed}",
                                    study._hash_file(checkpoint),
                                    provenance["manifest_sha256"])
        return
    for arm in cfg["arms"]:
        for seed in cfg["seeds"]:
            folder = out / arm / f"seed_{seed}"
            folder.mkdir(parents=True, exist_ok=True)
            study._seed(seed)
            encoder = evidence._encoder_from_state(base_state, base_cfg, device)
            model = WoodCorrespondence().to(device)
            n_classes = len({study.canonical(label) for _, label in items})
            classifier = ArcFaceClassifier(512, n_classes).to(device)
            representative = {}
            for index, (_, label) in enumerate(items):
                representative.setdefault(study.canonical(label), index)
            optimizer = torch.optim.AdamW([
                {"params": encoder.backbone.parameters(), "lr": cfg["backbone_lr"]},
                {"params": encoder.projection.parameters(), "lr": cfg["head_lr"]},
                {"params": model.parameters(), "lr": cfg["head_lr"]},
                {"params": classifier.parameters(), "lr": cfg["head_lr"]},
            ], weight_decay=0.01)
            latest = folder / "latest.pt"
            start_epoch, best_score = 1, -1.0
            if latest.is_file():
                saved = torch.load(latest, map_location="cpu", weights_only=False)
                if saved["provenance"] != provenance or saved["arm"] != arm or saved["seed"] != seed:
                    raise ValueError("Full-image checkpoint provenance mismatch")
                encoder.load_state_dict(saved["encoder"])
                model.load_state_dict(saved["model"])
                classifier.load_state_dict(saved["classifier"])
                optimizer.load_state_dict(saved["optimizer"])
                study._restore_rng(saved["rng_state"])
                start_epoch = saved["epoch"] + 1
                best_score = saved["best_score"]
                print(f"[wood-full] resuming {arm} seed={seed} epoch={start_epoch}", flush=True)
            for epoch in range(start_epoch, cfg["epochs"] + 1):
                refs, audit = reference_plan(items, seed, epoch)
                dataset = PairDataset(items, refs)
                sampler = (multi_positive_batches if arm in {
                    "arcface_packed", "arcface_multipos", "arcface_multipos_qkv"
                } else class_balanced_batches)
                batches = sampler(items, cfg["batch"], seed * 100003 + epoch)
                batch_audit = batch_positive_audit(items, batches)
                loader = DataLoader(dataset, batch_sampler=batches,
                                    num_workers=cfg["workers"],
                                    pin_memory=device.type == "cuda",
                                    persistent_workers=cfg["workers"] > 0)
                encoder.train()
                model.train()
                classifier.train()
                seen = np.zeros(len(items), dtype=np.uint8)
                losses = []
                candidate_classes = []
                positive_views = []
                started = time.perf_counter()
                for step, batch in enumerate(loader, 1):
                    indices = batch[3].numpy()
                    if seen[indices].any():
                        raise RuntimeError("Duplicate query within full-image epoch")
                    seen[indices] = 1
                    extra = None
                    if len(batch[2].unique()) == 1:
                        current = study.canonical(items[int(indices[0])][1])
                        other = next(label for label in representative if label != current)
                        extra_index = representative[other]
                        extra_image = dataset.transform(
                            image=dataset.loader.load(items[extra_index][0]))["image"]
                        extra = (extra_image, dataset.lookup[other])
                    loss, metrics = _step(encoder, model, classifier, batch, arm,
                                          cfg, optimizer, device, extra)
                    losses.append(loss)
                    candidate_classes.append(metrics["candidate_classes"])
                    positive_views.append(metrics["positive_views_per_anchor"])
                    if step % cfg["print_every"] == 0 or step == len(loader):
                        print(f"[wood-full] {arm} seed={seed} epoch={epoch}/{cfg['epochs']} "
                              f"step={step}/{len(loader)} query_coverage={int(seen.sum())}/{len(items)} "
                              f"loss={np.mean(losses):.4f} contrastive={metrics['contrastive']} "
                              f"qkv={metrics['qkv']} "
                              f"candidate_classes={metrics['candidate_classes']} "
                              f"positive_views={metrics['positive_views_per_anchor']:.2f} "
                              f"elapsed_s={time.perf_counter()-started:.1f}", flush=True)
                if not seen.all():
                    raise RuntimeError("Incomplete meta-train query coverage")
                encoder.eval()
                model.eval()
                scores = _validate(manifest, encoder, model, base_cfg, arm, device)
                selection = (scores["57x5"] + scores["57x1"] + scores["637x1"]) / 3
                record = {"epoch": epoch, "arm": arm, "seed": seed,
                          "query_coverage": int(seen.sum()), "pair_audit": audit,
                          "batch_positive_audit": batch_audit,
                          "mean_candidate_classes": float(np.mean(candidate_classes)),
                          "mean_positive_views_per_anchor": float(np.mean(positive_views)),
                          "mean_train_loss": float(np.mean(losses)),
                          "meta_val": scores, "selection_score": selection}
                study._json(folder / f"epoch_{epoch:02d}.json", record)
                improved = selection > best_score + 1e-12
                best_score = max(best_score, selection)
                state = {**record, "provenance": provenance,
                         "encoder": encoder.state_dict(), "model": model.state_dict(),
                         "classifier": classifier.state_dict(),
                         "optimizer": optimizer.state_dict(), "best_score": best_score,
                         "rng_state": study._rng_state()}
                if improved:
                    study._save_checkpoint(folder / "best.pt", state)
                study._save_checkpoint(latest, state)
                print(f"[wood-full] {arm} seed={seed} epoch={epoch} "
                      f"meta-val={scores} best={best_score:.4f}", flush=True)
    print(f"[wood-full] complete: {out}", flush=True)
