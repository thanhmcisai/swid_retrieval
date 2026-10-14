"""Full-image, gallery-aligned metric learning on meta-train species."""

import hashlib
import json
import math
import os
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from . import data, gallery_experiment as study
from . import wood_correspondence_experiment as corr
from . import wood_evidence_experiment as evidence
from . import wood_full_image_eval as image_eval
from . import wood_full_image_train as full
from .gallery_method import ArcFaceClassifier
from .wood_correspondence_method import WoodCorrespondence


ARMS = ("arcface_supcon", "gallery_full", "gallery_hard", "gallery_hard_scale")
FRACTIONS = (0.25, 0.5, 1.0)
GALLERIES = ("24x5", "57x5", "57x1", "637x1")


def config():
    base = full._config()
    arms = [x.strip() for x in os.environ.get(
        "WOOD_LISTWISE_ARMS", ",".join(ARMS)).split(",")]
    cfg = {
        "arms": arms,
        "seed": int(os.environ.get("WOOD_LISTWISE_SEED", "43")),
        "base": base,
        "bank_references": int(os.environ.get("WOOD_LISTWISE_BANK_REFERENCES", "5")),
        "bank_batch": int(os.environ.get("WOOD_LISTWISE_BANK_BATCH", "64")),
        "bank_momentum": float(os.environ.get("WOOD_LISTWISE_BANK_MOMENTUM", "0.5")),
        "gallery_weight": float(os.environ.get("WOOD_LISTWISE_WEIGHT", "0.25")),
        "hard_weight": float(os.environ.get("WOOD_LISTWISE_HARD_WEIGHT", "0.2")),
        "hard_classes": int(os.environ.get("WOOD_LISTWISE_HARD_CLASSES", "32")),
        "hard_margin": float(os.environ.get("WOOD_LISTWISE_HARD_MARGIN", "0.05")),
        "temperature": float(os.environ.get("WOOD_LISTWISE_TEMPERATURE", "0.07")),
        "retention_tolerance": float(os.environ.get("WOOD_LISTWISE_RETENTION_TOL", "0.03")),
    }
    if (not arms or len(set(arms)) != len(arms) or any(x not in ARMS for x in arms)
            or base["step_mode"] != "direct" or base["epochs"] != 1
            or cfg["bank_references"] < 5 or cfg["bank_batch"] < 1
            or not 0 <= cfg["bank_momentum"] < 1
            or cfg["gallery_weight"] <= 0 or cfg["hard_weight"] < 0
            or cfg["hard_classes"] < 1 or cfg["hard_margin"] < 0
            or cfg["temperature"] <= 0 or cfg["retention_tolerance"] < 0):
        raise ValueError("Invalid gallery-listwise pilot configuration")
    return cfg


def bank_plan(items, seed, width=5):
    """Prefer independent scans and scales without reducing batch class diversity."""
    groups = defaultdict(list)
    for index, (path, label) in enumerate(items):
        groups[study.canonical(label)].append(index)
    labels = sorted(groups)
    rng = np.random.default_rng(seed)
    selected = []
    for label in labels:
        remaining = list(groups[label])
        rng.shuffle(remaining)
        scans, scales, chosen = set(), set(), []
        for _ in range(min(width, len(remaining))):
            best = max(range(len(remaining)), key=lambda pos: (
                study.source_scan_id(items[remaining[pos]][0]) not in scans,
                study.image_scale(items[remaining[pos]][0]) not in scales,
                -pos))
            index = remaining.pop(best)
            chosen.append(index)
            scans.add(study.source_scan_id(items[index][0]))
            scales.add(study.image_scale(items[index][0]))
        selected.append(chosen)
    if len(labels) < 2 or any(not group for group in selected):
        raise ValueError("Bank requires at least two nonempty meta-train species")
    return labels, selected


def bank_plan_audit(items, plan):
    scan_counts = [len({study.source_scan_id(items[index][0]) for index in group})
                   for group in plan]
    scale_counts = [len({study.image_scale(items[index][0]) for index in group})
                    for group in plan]
    return {"species": len(plan), "reference_images": sum(map(len, plan)),
            "species_with_five_references": sum(len(group) >= 5 for group in plan),
            "species_with_multiple_scans": sum(value >= 2 for value in scan_counts),
            "species_with_multiple_scales": sum(value >= 2 for value in scale_counts),
            "mean_distinct_scans": float(np.mean(scan_counts)),
            "mean_distinct_scales": float(np.mean(scale_counts))}


class ReferenceBank:
    def __init__(self, items, labels, plan, embeddings, device):
        self.labels = labels
        self.width = max(map(len, plan))
        self.indices = torch.full((len(labels), self.width), -1,
                                  dtype=torch.long, device=device)
        self.slots = torch.full_like(self.indices, -1)
        self.image_to_slot = torch.full((len(items),), -1,
                                        dtype=torch.long, device=device)
        flat = []
        for class_id, group in enumerate(plan):
            for offset, image_id in enumerate(group):
                slot = len(flat)
                flat.append(image_id)
                self.indices[class_id, offset] = image_id
                self.slots[class_id, offset] = slot
                self.image_to_slot[image_id] = slot
        if embeddings.shape != (len(flat), 512):
            raise ValueError("Bank embeddings do not match selected reference plan")
        self.features = F.normalize(embeddings.float().detach().clone().to(device), dim=-1)
        if not torch.isfinite(self.features).all():
            raise ValueError("Non-finite bank embeddings")

    def scores(self, query, reference, labels, query_indices, k, temperature):
        if k not in (1, 3, 5) or k > self.width:
            raise ValueError("Invalid gallery reference budget")
        slots = self.slots[:, :k]
        selected = self.features[slots.clamp_min(0)]
        valid = slots >= 0
        sums = (selected * valid[..., None]).sum(dim=1)
        counts = valid.sum(dim=1)
        prototypes = F.normalize(sums / counts[:, None], dim=-1)
        q = F.normalize(query.float(), dim=-1)
        r = F.normalize(reference.float(), dim=-1)
        logits = (q @ prototypes.T) / temperature

        own_features = selected[labels]
        own_valid = valid[labels]
        own_indices = self.indices[labels, :k]
        # Always insert the live reference; remove any other bank view of the query.
        self_view = own_valid & (own_indices == query_indices[:, None])
        self_view[:, 0] = False
        own_sum = sums[labels] - own_features[:, 0] + r
        own_sum = own_sum - (own_features * self_view[..., None]).sum(dim=1)
        own_count = counts[labels] - self_view.sum(dim=1)
        positive = F.normalize(own_sum / own_count[:, None], dim=-1)
        positive_logit = (q * positive).sum(dim=-1) / temperature
        logits = torch.where(F.one_hot(labels, len(self.labels)).bool(),
                             positive_logit[:, None], logits)
        return logits

    @torch.no_grad()
    def update(self, image_indices, embeddings, momentum):
        slots = self.image_to_slot[image_indices]
        valid = slots >= 0
        if not valid.any():
            return 0
        ids, inverse = torch.unique(slots[valid], return_inverse=True)
        mean = self.features.new_zeros((len(ids), self.features.shape[1]))
        mean.index_add_(0, inverse, F.normalize(embeddings[valid].float(), dim=-1))
        counts = torch.bincount(inverse, minlength=len(ids))
        mean = F.normalize(mean / counts[:, None], dim=-1)
        updated = F.normalize(momentum * self.features[ids] + (1 - momentum) * mean,
                              dim=-1)
        self.features.index_copy_(0, ids, updated)
        return len(ids)


def gallery_loss(logits, labels, hard_classes, margin, hard_weight):
    full_loss = F.cross_entropy(logits, labels)
    positive = logits.gather(1, labels[:, None])
    negatives = logits.masked_fill(F.one_hot(labels, logits.shape[1]).bool(), -torch.inf)
    hardest = negatives.topk(min(hard_classes, logits.shape[1] - 1), dim=1).values
    hard_loss = F.softplus(torch.logsumexp(hardest - positive + margin, dim=1)).mean()
    total = full_loss + hard_weight * hard_loss
    return total, {"gallery_ce": float(full_loss.detach()),
                   "gallery_hard": float(hard_loss.detach()),
                   "gallery_r1": float((logits.argmax(1) == labels).float().mean().detach())}


def _bank_embeddings(encoder, items, plan, base_cfg, cfg, device):
    indices = [index for group in plan for index in group]
    selected = [items[index] for index in indices]
    loader = study._loader(selected, dict(base_cfg, image_batch=cfg["bank_batch"],
                                          workers=cfg["base"]["workers"]))
    chunks = []
    encoder.eval()
    with torch.inference_mode():
        for images, _ in loader:
            with study._autocast(device):
                vectors = encoder(images.to(device, non_blocking=True))
            chunks.append(vectors.float().cpu().numpy())
    return torch.from_numpy(np.concatenate(chunks)).to(device).clone()


def _step_with_bank(encoder, model, classifier, bank, batch, references,
                    cfg, optimizer, scaler, device, step, arm):
    query, reference, labels, indices = batch
    labels = labels.to(device, non_blocking=True)
    indices = indices.to(device, non_blocking=True)
    images = torch.cat((query, reference), dim=0)
    optimizer.zero_grad(set_to_none=True)
    globals_, tokens = [], []
    for chunk in images.split(cfg["base"]["microbatch"]):
        with study._autocast(device):
            g, t = corr._encode_images(encoder, chunk.to(device, non_blocking=True))
        globals_.append(g.float())
        tokens.append(t.float())
    globals_, tokens = torch.cat(globals_), torch.cat(tokens)
    batch_size = len(labels)
    base_loss, details = full._loss(
        model, classifier, globals_, tokens, labels, "arcface_supcon", 0,
        contrastive_weight=cfg["base"]["contrastive_weight"],
        temperature=cfg["base"]["contrastive_temperature"],
        arcface_weight=cfg["base"]["arcface_weight"])
    k = (1, 3, 5)[(step - 1) % 3] if arm == "gallery_hard_scale" else 1
    logits = bank.scores(globals_[:batch_size], globals_[batch_size:],
                         labels, indices, k, cfg["temperature"])
    extra, gallery_details = gallery_loss(
        logits, labels, cfg["hard_classes"],
        cfg["hard_margin"] / cfg["temperature"],
        cfg["hard_weight"] if arm in {"gallery_hard", "gallery_hard_scale"} else 0)
    loss = base_loss + cfg["gallery_weight"] * extra
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite gallery-listwise training loss")
    (scaler.scale(loss) if scaler is not None else loss).backward()
    if scaler is not None:
        scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(
        list(encoder.parameters()) + list(classifier.parameters()), 1.0)
    scale_before = scaler.get_scale() if scaler is not None else None
    if scaler is None:
        optimizer.step()
    else:
        scaler.step(optimizer)
        scaler.update()
    skipped = scaler is not None and scaler.get_scale() < scale_before
    if not skipped:
        ref_indices = references[indices]
        bank.update(torch.cat((indices, ref_indices)),
                    torch.cat((globals_[:batch_size].detach(),
                               globals_[batch_size:].detach())),
                    cfg["bank_momentum"])
    details.update(gallery_details, gallery_k=k, amp_skip=bool(skipped))
    return float(loss.detach()), details


def _validation_paths(manifest):
    paths = set()
    for fold in (0, 1):
        refs, queries, extras = evidence._validation_sets(manifest, "meta-val", fold)
        paths.update(path for path, _ in refs + queries + extras)
    return sorted(paths)


def _cache_fingerprint(paths):
    digest = hashlib.sha256()
    for path in paths:
        cached = data.CACHE_DIR / (hashlib.md5(str(path).encode()).hexdigest() + ".jpg")
        if not cached.is_file():
            raise FileNotFoundError(f"Validation image is not cached: {path}")
        digest.update(str(path).encode())
        digest.update(study._hash_file(cached).encode())
    return digest.hexdigest()


def _preload_checked(paths, workers, description):
    stats = data.preload_image_cache(paths, max_workers=workers, desc=description)
    if stats["bad"]:
        raise RuntimeError(f"{stats['bad']} images could not be cached: {description}")
    return stats


def _provenance(root, source_path, cfg, validation_fingerprint,
                bank_fingerprint, plan):
    return {
        "source_sha256": study._hash_file(source_path),
        "manifest_sha256": study._hash_file(root / "swi_manifest.json"),
        "runner_sha256": study._hash_file(__file__),
        "full_train_sha256": study._hash_file(Path(full.__file__)),
        "bank_plan_sha256": study._hash_bytes(
            json.dumps(plan, separators=(",", ":")).encode()),
        "validation_cache_sha256": validation_fingerprint,
        "bank_cache_sha256": bank_fingerprint,
        "config": cfg,
        "selection": ("max meta-val 637x1 R@1; 24x5 no more than "
                      f"{cfg['retention_tolerance']} absolute below source"),
        "data_splits": ["meta-train", "meta-val"],
    }


def _milestones(total):
    if total < 4:
        raise ValueError("At least four training batches are required")
    return {math.ceil(total * fraction): fraction for fraction in FRACTIONS}


def _training_items(root):
    manifest = data.load_swi_manifest(root / "swi_manifest.json")
    items = manifest["meta-train"]
    if len(items) < 1000 or len({study.canonical(label) for _, label in items}) < 500:
        raise ValueError("Expected full SmartWoodID meta-train split")
    return manifest, items


def _source_report(out, manifest, base_state, base_cfg, provenance, device):
    path = out / "source.json"
    if path.is_file():
        report = json.loads(path.read_text())
        if report["provenance"] != provenance:
            raise ValueError("Source report provenance changed; use a new output folder")
        return report
    encoder = evidence._encoder_from_state(base_state, base_cfg, device).eval()
    model = WoodCorrespondence().to(device).eval()
    scores = full._validate(manifest, encoder, model, base_cfg,
                            "arcface_supcon", device, galleries=GALLERIES)
    report = {"meta_val": scores, "provenance": provenance,
              "selected_score": scores["637x1"]}
    study._json(path, report)
    del encoder, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


def _train_one(out, manifest, items, plan, bank_embeddings,
               base_state, base_cfg, source_report, provenance, cfg, arm, device):
    folder = out / arm / f"seed_{cfg['seed']}"
    completed = folder / "complete.json"
    if completed.is_file():
        report = json.loads(completed.read_text())
        if report["provenance"] != provenance or study._hash_file(
                folder / "best.pt") != report["best_sha256"]:
            raise ValueError(f"Completed arm changed: {completed}")
        print(f"[wood-listwise] verified {arm}; skipping", flush=True)
        return report
    if any(folder.glob("fraction_*.json")):
        print(f"[wood-listwise] restarting incomplete {arm} from source checkpoint; "
              "mid-epoch worker RNG cannot be resumed exactly", flush=True)

    study._seed(cfg["seed"])
    encoder = evidence._encoder_from_state(base_state, base_cfg, device)
    model = WoodCorrespondence().to(device)
    classifier = ArcFaceClassifier(512, len(plan[0])).to(device)
    optimizer = torch.optim.AdamW([
        {"params": encoder.backbone.parameters(), "lr": cfg["base"]["backbone_lr"]},
        {"params": encoder.projection.parameters(), "lr": cfg["base"]["head_lr"]},
        {"params": classifier.parameters(), "lr": cfg["base"]["head_lr"]},
    ], weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    references, pair_audit = full.reference_plan(items, cfg["seed"], 1)
    reference_indices = torch.as_tensor(references, device=device)
    dataset = full.PairDataset(items, references)
    batches = full.class_balanced_batches(items, cfg["base"]["batch"],
                                          cfg["seed"] * 100003 + 1)
    milestones = _milestones(len(batches))
    loader = DataLoader(dataset, batch_sampler=batches,
                        num_workers=cfg["base"]["workers"],
                        pin_memory=device.type == "cuda",
                        **({"prefetch_factor": 2} if cfg["base"]["workers"] > 0 else {}))
    bank = None if arm == "arcface_supcon" else ReferenceBank(
        items, plan[0], plan[1], bank_embeddings, device)
    seen = np.zeros(len(items), dtype=np.uint8)
    losses, records = [], []
    best_key = (-1, -1.0)
    started = time.perf_counter()
    encoder.train()
    model.train()
    classifier.train()
    for step, batch in enumerate(loader, 1):
        indices = batch[3].numpy()
        if seen[indices].any():
            raise RuntimeError("Duplicate meta-train query in listwise pilot")
        seen[indices] = 1
        if bank is None:
            loss, details = full._step(encoder, model, classifier, batch,
                                       "arcface_supcon", cfg["base"], optimizer,
                                       device, scaler=scaler)
        else:
            loss, details = _step_with_bank(
                encoder, model, classifier, bank, batch,
                reference_indices, cfg, optimizer, scaler,
                device, step, arm)
        losses.append(loss)
        if step % cfg["base"]["print_every"] == 0 or step in milestones:
            print(f"[wood-listwise] {arm} step={step}/{len(batches)} "
                  f"queries={int(seen.sum())}/{len(items)} "
                  f"loss={float(np.mean(losses)):.4f} "
                  f"gallery_r1={details.get('gallery_r1')} "
                  f"elapsed_s={time.perf_counter()-started:.1f}", flush=True)
        if step not in milestones:
            continue
        encoder.eval()
        model.eval()
        scores = full._validate(manifest, encoder, model, base_cfg,
                                "arcface_supcon", device, galleries=GALLERIES)
        if not all(np.isfinite(value) for value in scores.values()):
            raise RuntimeError("Non-finite meta-validation metric")
        fraction = milestones[step]
        record = {"arm": arm, "fraction": fraction, "step": step,
                  "total_steps": len(batches), "query_coverage": int(seen.sum()),
                  "mean_train_loss": float(np.mean(losses)),
                  "meta_val": scores, "primary_score": scores["637x1"],
                  "retention_ok": scores["24x5"] >=
                  source_report["meta_val"]["24x5"] - cfg["retention_tolerance"],
                  "elapsed_s": time.perf_counter() - started,
                  "pair_audit": pair_audit}
        records.append(record)
        study._json(folder / f"fraction_{int(100 * fraction):03d}.json", record)
        candidate_key = (int(record["retention_ok"]), scores["637x1"])
        if candidate_key > best_key:
            best_key = candidate_key
            study._save_checkpoint(folder / "best.pt", {
                "encoder": encoder.state_dict(), "model": model.state_dict(),
                "classifier": classifier.state_dict(), "arm": arm,
                "fraction": fraction, "seed": cfg["seed"],
                "provenance": provenance, "meta_val": scores})
        print(f"[wood-listwise] {arm} fraction={fraction:.2f} "
              f"val24={scores['24x5']:.4f} val57={scores['57x1']:.4f} "
              f"val637={scores['637x1']:.4f} "
              f"eligible={record['retention_ok']} best637={best_key[1]:.4f}",
              flush=True)
        encoder.train()
        model.train()
    if not seen.all():
        raise RuntimeError("Incomplete full-image query coverage")
    best = max(records, key=lambda row: (int(row["retention_ok"]),
                                         row["primary_score"]))
    report = {"arm": arm, "seed": cfg["seed"], "records": records,
              "best_fraction": best["fraction"], "best_meta_val": best["meta_val"],
              "best_sha256": study._hash_file(folder / "best.pt"),
              "provenance": provenance}
    study._json(completed, report)
    return report


def _selection(out, reports, source, provenance, cfg):
    eligible = [("source", source["meta_val"]["637x1"],
                 source["meta_val"], provenance["source_sha256"])]
    for report in reports:
        scores = report["best_meta_val"]
        if scores["24x5"] >= source["meta_val"]["24x5"] - cfg["retention_tolerance"]:
            eligible.append((report["arm"], scores["637x1"], scores,
                             report["best_sha256"]))
    arm, primary, scores, digest = max(eligible, key=lambda row: row[1])
    lock = {"selected_arm": arm, "selected_sha256": digest,
            "meta_val": scores, "primary_metric": "637x1 macro R@1",
            "primary_score": primary, "retention_metric": "24x5 macro R@1",
            "retention_tolerance": cfg["retention_tolerance"],
            "meta_test_evaluated": False, "public_id_evaluated": False,
            "provenance": provenance}
    study._json(out / "selection_lock.json", lock)
    return lock


def _read_lock(out, provenance):
    lock = json.loads((out / "selection_lock.json").read_text())
    if lock["provenance"] != provenance:
        raise ValueError("Selection provenance changed")
    return lock


def _per_query(manifest, encoder, model, base_cfg, device, output,
               checkpoint_hash, manifest_hash):
    all_rows = []
    for fold in (0, 1):
        refs, queries, extras = evidence._validation_sets(manifest, "meta-val", fold)
        plans = evidence._gallery_plan(refs, queries, extras, "meta-val")
        items, lookup = image_eval._unique_items(refs, extras, queries)
        embeddings, tokens = evidence._extract(
            encoder, items, dict(base_cfg, image_batch=64, workers=2), device)
        for gallery in GALLERIES:
            ri, qi = plans[gallery]
            references = [(refs + extras)[int(i)] for i in ri]
            chosen_queries = [queries[int(i)] for i in qi]
            all_rows.extend(image_eval._score_gallery(
                model, "arcface_supcon", fold, gallery, "standard", references,
                chosen_queries, lookup, embeddings, tokens, device))
    summaries = image_eval._summaries(all_rows)
    image_eval._write_csv(output / "per_query.csv", all_rows)
    image_eval._write_csv(output / "summary.csv", summaries)
    study._json(output / "protocol.json", {
        "split": "meta-val", "folds": [0, 1],
        "galleries": list(GALLERIES), "checkpoint_sha256": checkpoint_hash,
        "manifest_sha256": manifest_hash,
        "validation_cache_sha256": _cache_fingerprint(_validation_paths(manifest)),
        "evaluator_sha256": study._hash_file(Path(image_eval.__file__)),
        "meta_test_evaluated": False, "public_id_evaluated": False,
    })
    return summaries


def _export(out):
    if not (out / "selection_lock.json").is_file():
        raise ValueError("Train and lock the selection before exporting")
    files = sorted(path for path in out.rglob("*") if path.is_file() and
                   path.suffix in {".json", ".csv"})
    archive = out / "wood_gallery_listwise_review.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as stream:
        for path in files:
            stream.write(path, path.relative_to(out))
    if archive.stat().st_size < 1024:
        raise RuntimeError("Review archive unexpectedly small")
    print(f"[wood-listwise] review archive: {archive}", flush=True)
    return archive


def run():
    cfg = config()
    root = Path(os.environ.get("ROOT_PATH", "/content/drive/MyDrive/NCS")).resolve()
    out = Path(os.environ.get(
        "WOOD_LISTWISE_OUT", root / "results/wood_gallery_listwise_pilot_v1"))
    mode = os.environ.get("WOOD_LISTWISE_MODE", "preflight")
    if mode == "export":
        return _export(out)
    if mode not in {"preflight", "train", "validate"}:
        raise ValueError("WOOD_LISTWISE_MODE must be preflight, train, validate, or export")
    manifest, items = _training_items(root)
    source_root = Path(os.environ.get(
        "WOOD_LISTWISE_SOURCE_OUT", root / "results/dinov2s_retrieval_matched_v1"))
    source_path, source_state, base_cfg = evidence._base_checkpoint(
        root, source_root, cfg["base"]["source_seed"])
    plan = bank_plan(items, cfg["seed"], cfg["bank_references"])
    if plan[0] != sorted({study.canonical(label) for _, label in items}):
        raise ValueError("Bank class order differs from training labels")
    if mode == "preflight":
        batches = full.class_balanced_batches(items, cfg["base"]["batch"],
                                              cfg["seed"] * 100003 + 1)
        print(f"[wood-listwise] {len(items)} meta-train queries; "
              f"{len(plan[0])} species; {sum(map(len, plan[1]))} bank images; "
              f"{len(batches)} steps; milestones={_milestones(len(batches))}; "
              f"arms={cfg['arms']}", flush=True)
        return
    device = torch.device(os.environ.get("DEVICE", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if os.environ.get("WOOD_LISTWISE_PRELOAD_TRAIN", "1") == "1" and mode == "train":
        _preload_checked((path for path, _ in items),
                         int(os.environ.get("WOOD_LISTWISE_PRELOAD_WORKERS", "8")),
                         "Gallery-listwise meta-train images")
    validation_paths = _validation_paths(manifest)
    _preload_checked(validation_paths,
                     int(os.environ.get("WOOD_LISTWISE_PRELOAD_WORKERS", "8")),
                     "Gallery-listwise meta-val images")
    fingerprint = _cache_fingerprint(validation_paths)
    bank_paths = [items[index][0] for group in plan[1] for index in group]
    _preload_checked(bank_paths,
                     int(os.environ.get("WOOD_LISTWISE_PRELOAD_WORKERS", "8")),
                     "Gallery-listwise bank identity check")
    bank_fingerprint = _cache_fingerprint(bank_paths)
    provenance = _provenance(root, source_path, cfg, fingerprint,
                             bank_fingerprint, plan[1])
    out.mkdir(parents=True, exist_ok=True)
    provenance_path = out / "provenance.json"
    if provenance_path.is_file():
        if json.loads(provenance_path.read_text()) != provenance:
            raise ValueError("Pilot provenance changed; choose a new output directory")
    else:
        study._json(provenance_path, provenance)
        study._json(out / "bank_plan.json", {
            "labels": plan[0], "meta_train_image_indices": plan[1],
            "source_split": "meta-train", "seed": cfg["seed"],
            "audit": bank_plan_audit(items, plan[1])})
    source = _source_report(out, manifest, source_state, base_cfg, provenance, device)
    if mode == "validate":
        lock = _read_lock(out, provenance)
        for name in dict.fromkeys(("source", lock["selected_arm"])):
            report_dir = out / "validation" / name
            if name == "source":
                checkpoint_hash = provenance["source_sha256"]
                encoder = evidence._encoder_from_state(source_state, base_cfg, device)
                model = WoodCorrespondence().to(device)
            else:
                checkpoint = out / name / f"seed_{cfg['seed']}" / "best.pt"
                checkpoint_hash = study._hash_file(checkpoint)
                if checkpoint_hash != lock["selected_sha256"]:
                    raise ValueError("Selected checkpoint changed")
                state = torch.load(checkpoint, map_location="cpu", weights_only=False)
                if state["provenance"] != provenance:
                    raise ValueError("Selected checkpoint provenance changed")
                encoder = evidence._encoder_from_state(source_state, base_cfg, device)
                encoder.load_state_dict(state["encoder"], strict=True)
                model = WoodCorrespondence().to(device)
                model.load_state_dict(state["model"], strict=True)
            encoder.eval()
            model.eval()
            summaries = _per_query(manifest, encoder, model, base_cfg, device,
                                   report_dir, checkpoint_hash,
                                   provenance["manifest_sha256"])
            expected = (source["meta_val"] if name == "source"
                        else lock["meta_val"])
            for gallery in GALLERIES:
                row = next(part for part in summaries if part["gallery"] == gallery
                           and part["policy"] == "standard" and part["mode"] == "global")
                if abs(row["r1"] - expected[gallery]) > 1e-6:
                    raise ValueError(f"Per-query R@1 changed for {name}/{gallery}: "
                                     f"{row['r1']} versus {expected[gallery]}")
            print(f"[wood-listwise] per-query meta-val: {report_dir}", flush=True)
        return
    bank_embeddings = None
    if any(arm != "arcface_supcon" and not (
            out / arm / f"seed_{cfg['seed']}" / "complete.json").is_file()
            for arm in cfg["arms"]):
        encoder = evidence._encoder_from_state(source_state, base_cfg, device)
        bank_embeddings = _bank_embeddings(encoder, items, plan[1], base_cfg, cfg, device)
        del encoder
        if device.type == "cuda":
            torch.cuda.empty_cache()
    reports = [_train_one(out, manifest, items, plan, bank_embeddings,
                          source_state, base_cfg, source, provenance, cfg, arm, device)
               for arm in cfg["arms"]]
    lock = _selection(out, reports, source, provenance, cfg)
    study._json(out / "summary.json", {
        "source": source["meta_val"],
        "selected": {"arm": lock["selected_arm"], "meta_val": lock["meta_val"]},
        "variants": [{"arm": report["arm"],
                      "best_fraction": report["best_fraction"],
                      "best_meta_val": report["best_meta_val"]}
                     for report in reports],
        "provenance": provenance})
    print(f"[wood-listwise] complete: {out / 'summary.json'}", flush=True)


if __name__ == "__main__":
    run()
