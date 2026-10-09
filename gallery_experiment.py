"""Isolated end-to-end gallery-adaptive retrieval study for Colab.

Nothing in this module changes the frozen submission run or its embedding cache.
Model selection uses SWI meta-val only; corrected public images are read solely
in the explicit final evaluation stage.
"""

import hashlib
import json
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import BatchSampler, DataLoader, WeightedRandomSampler

from . import data
from .gallery_method import (ArcFaceClassifier, GalleryEncoder, GalleryMemory, GalleryScorer,
                             episode_objective, supervised_contrastive_loss)
from .wood_encoder import WoodPatternNet

DINO_REPO_REF = "facebookresearch/dinov2:7764ea0f912e53c92e82eb78a2a1631e92725fc8"
BACKBONES = ("dinov2_vitb14", "dinov2_vits14", "convnext_tiny",
             "woodpattern_tiny", "woodpattern_no_attention", "woodpattern_single_scale")


VARIANTS = {
    "dinov2_pretrained": {"objective": "pretrained", "embedding_dim": 768,
                           "train_backbone": False, "scorer_mode": "prototype"},
    "supervised_warmup": {"objective": "supervised_warmup", "scorer_mode": "nearest",
                          "memory_size": 0, "memory_min_classes": 0},
    "metric_retrieval": {"scorer_mode": "nearest", "variable_gallery": False,
                         "stability_weight": 0.0, "pseudo_ood_weight": 0.0},
    "metric_no_memory": {"scorer_mode": "nearest", "variable_gallery": False,
                         "stability_weight": 0.0, "pseudo_ood_weight": 0.0,
                         "memory_size": 0, "memory_min_classes": 0},
    "prototype_retrieval": {"scorer_mode": "prototype", "variable_gallery": False,
                            "stability_weight": 0.0, "pseudo_ood_weight": 0.0},
    "metric_expansion": {"scorer_mode": "nearest"},
    "gallery_adaptive": {},
    "fixed_episode": {"variable_gallery": False, "ways": (16,)},
    "without_expansion_loss": {"stability_weight": 0.0},
    "without_pseudo_ood": {"pseudo_ood_weight": 0.0},
    "without_hard_negatives": {"hard_negative_probability": 0.0},
    "prototype_only": {"scorer_mode": "prototype"},
    "fixed_evidence": {"scorer_mode": "fixed"},
    "without_count_normalization": {"normalize_evidence": False},
    "frozen_encoder": {"train_backbone": False},
    "without_memory": {"memory_size": 0, "memory_min_classes": 0},
    "memory_512": {"memory_size": 512},
    "top_m_32": {"top_m": 32},
    "top_m_128": {"top_m": 128, "memory_min_classes": 160},
    "supcon_finetuned": {"objective": "supcon", "scorer_mode": "prototype"},
    "arcface_finetuned": {"objective": "arcface", "scorer_mode": "prototype"},
}


def _hash_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_sha256(module):
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def canonical(label):
    return data.canonical_label(label)


_SCAN_ID = re.compile(r"_from_(tw[0-9]+)(?:_|$)", re.IGNORECASE)


def source_scan_id(path):
    match = _SCAN_ID.search(Path(path).stem)
    if match is None:
        raise ValueError(f"Cannot recover SmartWoodID source-scan ID from patch: {path}")
    return match.group(1).lower()


def source_scan_audit(manifest):
    summary = {}
    seen = {}
    for split in ("meta-train", "meta-val", "meta-test"):
        by_species = defaultdict(set)
        for path, label in manifest[split]:
            species, scan = canonical(label), source_scan_id(path)
            if scan in seen and seen[scan] != species:
                raise ValueError(f"Source scan {scan} has conflicting species labels")
            seen[scan] = species
            by_species[species].add(scan)
        summary[split] = {"species": len(by_species),
                          "source_scans": len(set().union(*by_species.values())),
                          "species_with_two_or_more_scans": sum(
                              len(scans) >= 2 for scans in by_species.values()),
                          "species_with_one_scan": sum(
                              len(scans) == 1 for scans in by_species.values())}
    return summary


def variant_config(name, *, pilot=False):
    if name not in VARIANTS:
        raise ValueError(f"Unknown variant {name}; choose from {sorted(VARIANTS)}")
    backbone = os.environ.get("GALLERY_STUDY_BACKBONE", "dinov2_vitb14").lower()
    default_backbone_lr = (5e-4 if backbone.startswith("woodpattern_") else
                           3e-5 if backbone in {"dinov2_vits14", "convnext_tiny"} else 1e-5)
    config = {
        "variant": name,
        "backbone": backbone,
        "image_size": 224,
        "embedding_dim": 512,
        "ways": (4, 8, 16),
        "group_mode": "scan_disjoint",
        "support": 2,
        "queries": 1,
        "top_m": 64,
        "memory_size": int(os.environ.get("GALLERY_STUDY_MEMORY_SIZE", "256")),
        "memory_min_classes": int(os.environ.get("GALLERY_STUDY_MEMORY_MIN_CLASSES", "80")),
        "temperature": 0.07,
        "epochs": int(os.environ.get("GALLERY_STUDY_EPOCHS", "2" if pilot else "20")),
        "episodes_per_epoch": int(os.environ.get(
            "GALLERY_STUDY_EPISODES_PER_EPOCH", "30" if pilot else "500")),
        "microbatch": int(os.environ.get("GALLERY_STUDY_MICROBATCH", "8")),
        "image_batch": int(os.environ.get("GALLERY_STUDY_IMAGE_BATCH", "32")),
        "workers": int(os.environ.get("GALLERY_STUDY_WORKERS", "4")),
        "backbone_lr": float(os.environ.get("GALLERY_STUDY_BACKBONE_LR", default_backbone_lr)),
        "head_lr": 3e-4,
        "weight_decay": 0.01,
        "stability_weight": 0.2,
        "pseudo_ood_weight": 0.1,
        "hard_negative_probability": 0.5,
        "train_backbone": True,
        "variable_gallery": True,
        "normalize_evidence": True,
        "scorer_mode": "learned",
        "objective": "episode",
    }
    config.update(VARIANTS[name])
    if config["objective"] == "supervised_warmup":
        config.update({
            "warmup_epochs": int(os.environ.get("GALLERY_STUDY_WARMUP_EPOCHS", "5")),
            "warmup_steps": int(os.environ.get("GALLERY_STUDY_WARMUP_STEPS", "300")),
            "warmup_batch": int(os.environ.get("GALLERY_STUDY_WARMUP_BATCH", "64")),
            "warmup_lr": float(os.environ.get("GALLERY_STUDY_WARMUP_LR", "5e-4")),
        })
    init_path = os.environ.get("GALLERY_STUDY_INIT_CHECKPOINT", "").strip()
    if init_path and config["objective"] not in {"pretrained", "supervised_warmup"}:
        config["init_checkpoint"] = str(Path(init_path).resolve())
        config["init_checkpoint_sha256"] = _hash_file(init_path)
    if config["backbone"] not in BACKBONES:
        raise ValueError(f"Unknown backbone {config['backbone']}; choose from {BACKBONES}")
    if config["objective"] == "pretrained" and config["backbone"] != "dinov2_vitb14":
        raise ValueError("The pretrained 768-dimensional control requires dinov2_vitb14")
    if config["objective"] == "supervised_warmup":
        if not config["backbone"].startswith("woodpattern_"):
            raise ValueError("Supervised warm-up currently supports WoodPatternNet only")
        if (config["warmup_epochs"] < 1 or config["warmup_steps"] < 1 or
                config["warmup_batch"] < 2 or config["warmup_lr"] <= 0):
            raise ValueError("Invalid supervised warm-up schedule")
    if (config["microbatch"] < 1 or config["workers"] < 0 or
            config["epochs"] < 1 or config["episodes_per_epoch"] < 1 or
            config["backbone_lr"] <= 0 or config["head_lr"] <= 0):
        raise ValueError("Invalid microbatch or workers")
    if not 0 <= config["memory_min_classes"] <= config["memory_size"]:
        raise ValueError("Invalid memory warmup/capacity")
    return config


class EpisodeSampler(BatchSampler):
    """Samples labelled support and query images without replacement."""

    def __init__(self, labels, ways, support, queries, episodes, seed,
                 hard_negative_probability=0.5, groups=None):
        self.labels = np.asarray([canonical(x) for x in labels])
        self.ways = tuple(int(x) for x in ways)
        self.support = int(support)
        self.queries = int(queries)
        self.episodes = int(episodes)
        self.seed = int(seed)
        self.hard_negative_probability = float(hard_negative_probability)
        self.by_class = {c: np.flatnonzero(self.labels == c) for c in sorted(set(self.labels))}
        self.by_group = {}
        self.group_options = {}
        if groups is not None:
            if len(groups) != len(labels):
                raise ValueError("Source-scan group list must match labelled image list")
            groups = np.asarray(groups)
            for species, indices in self.by_class.items():
                by_group = {group: indices[groups[indices] == group]
                            for group in sorted(set(groups[indices]))}
                self.by_group[species] = by_group
                self.group_options[species] = [
                    group for group, query_pool in by_group.items()
                    if len(query_pool) >= self.queries and
                    sum(len(other) for name, other in by_group.items() if name != group) >= self.support]
            self.eligible = [c for c in self.by_class if self.group_options[c]]
        else:
            self.eligible = [c for c, idx in self.by_class.items()
                             if len(idx) >= support + queries]
        self.by_genus = defaultdict(list)
        for c in self.eligible:
            self.by_genus[c.split("_", 1)[0]].append(c)
        self.hard_genera = [g for g, classes in self.by_genus.items() if len(classes) >= 2]
        if len(self.eligible) < max(self.ways):
            raise ValueError("Not enough eligible meta-train species for requested episode size")

    def __iter__(self):
        rng = np.random.RandomState(self.seed)
        for _ in range(self.episodes):
            n_way = int(rng.choice(self.ways))
            selected = []
            if self.hard_genera and rng.rand() < self.hard_negative_probability:
                genus = rng.choice(self.hard_genera)
                same_genus = self.by_genus[genus]
                selected = rng.choice(same_genus, min(4, n_way, len(same_genus)), replace=False).tolist()
            available = [c for c in self.eligible if c not in selected]
            selected.extend(rng.choice(available, n_way - len(selected), replace=False).tolist())
            rng.shuffle(selected)
            support_indices, query_indices = [], []
            for species in selected:
                if self.by_group:
                    query_group = rng.choice(self.group_options[species])
                    query_pool = self.by_group[species][query_group]
                    support_pool = np.concatenate([
                        indices for name, indices in self.by_group[species].items()
                        if name != query_group])
                    support_indices.extend(rng.choice(
                        support_pool, self.support, replace=False).tolist())
                    query_indices.extend(rng.choice(
                        query_pool, self.queries, replace=False).tolist())
                else:
                    draw = rng.choice(self.by_class[species],
                                      self.support + self.queries, replace=False)
                    support_indices.extend(draw[:self.support].tolist())
                    query_indices.extend(draw[self.support:].tolist())
            yield support_indices + query_indices

    def __len__(self):
        return self.episodes


def validation_items(manifest, support=5, queries=5, group_mode="image"):
    by_class = defaultdict(list)
    for path, label in manifest["meta-val"]:
        by_class[canonical(label)].append((path, canonical(label)))
    references, probes = [], []
    for label, items in sorted(by_class.items()):
        if group_mode == "scan_disjoint":
            by_scan = defaultdict(list)
            for item in items:
                by_scan[source_scan_id(item[0])].append(item)
            options = [scan for scan, candidates in by_scan.items()
                       if len(candidates) >= queries and
                       sum(len(other) for key, other in by_scan.items() if key != scan) >= support]
            if not options:
                continue
            query_scan = sorted(options, key=lambda scan: _hash_bytes(scan.encode()))[0]
            ref_pool = [item for scan, entries in by_scan.items() if scan != query_scan
                        for item in entries]
            query_pool = by_scan[query_scan]
            references.extend(sorted(ref_pool,
                                     key=lambda item: _hash_bytes(item[0].encode()))[:support])
            probes.extend(sorted(query_pool,
                                 key=lambda item: _hash_bytes(item[0].encode()))[:queries])
        elif group_mode == "image":
            if len(items) < support + queries:
                continue
            chosen = sorted(items, key=lambda item: _hash_bytes(item[0].encode()))[:support + queries]
            references.extend(chosen[:support])
            probes.extend(chosen[support:])
        else:
            raise ValueError(f"Unknown validation group mode: {group_mode}")
    if len({x[1] for x in references}) < 24:
        raise ValueError("Meta-val lacks 24 classes with sufficient images")
    return references, probes


def _loader(items, cfg, augment=False, shuffle=False):
    dataset = data.ManifestDataset(items, transform=data.get_transforms(cfg["image_size"], augment=augment))
    return DataLoader(dataset, batch_size=cfg["image_batch"], shuffle=shuffle,
                      num_workers=cfg["workers"], pin_memory=torch.cuda.is_available(),
                      persistent_workers=cfg["workers"] > 0)


def _model(cfg, device, pretrained=True):
    kind = cfg["backbone"]
    if kind.startswith("dinov2_"):
        backbone = torch.hub.load(DINO_REPO_REF, kind, pretrained=pretrained)
        feature_dim = 384 if kind == "dinov2_vits14" else 768
    elif kind == "convnext_tiny":
        import timm
        backbone = timm.create_model("convnext_tiny", pretrained=pretrained, num_classes=0)
        feature_dim = backbone.num_features
    else:
        backbone = WoodPatternNet(use_attention=kind != "woodpattern_no_attention",
                                  use_multiscale=kind != "woodpattern_single_scale")
        feature_dim = backbone.num_features
    encoder = GalleryEncoder(backbone, feature_dim=feature_dim,
                             embedding_dim=cfg["embedding_dim"]).to(device)
    if cfg["objective"] == "pretrained":
        encoder.projection = torch.nn.Identity()
    scorer = GalleryScorer(cfg["top_m"], cfg["temperature"], cfg["scorer_mode"],
                           cfg["normalize_evidence"]).to(device)
    return encoder, scorer


def _autocast(device):
    return torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda")


def encode_items(encoder, items, cfg, device):
    encoder.eval()
    outputs = []
    with torch.inference_mode():
        for images, _ in _loader(items, cfg):
            with _autocast(device):
                output = encoder(images.to(device, non_blocking=True))
            if not torch.isfinite(output).all():
                raise RuntimeError("Non-finite validation embeddings; checkpoint was not advanced")
            outputs.append(output.float().cpu().numpy())
    return np.concatenate(outputs) if outputs else np.empty((0, cfg["embedding_dim"]), np.float32)


def _episode_step(images, batch_labels, cfg, encoder, scorer, optimizer, scaler, device,
                  classifier=None, memory=None):
    images = images.to(device, non_blocking=True)
    n_way = len(images) // (cfg["support"] + cfg["queries"])
    n_support = n_way * cfg["support"]
    support_labels = batch_labels[:n_support].to(device)
    if memory is not None:
        support_grid = support_labels.reshape(n_way, cfg["support"])
        query_grid = batch_labels[n_support:].to(device).reshape(n_way, cfg["queries"])
        if (bool(support_grid.ne(support_grid[:, :1]).any()) or
                bool(query_grid.ne(support_grid[:, :1]).any()) or
                len(support_grid[:, 0].unique()) != n_way):
            raise ValueError("Episode support/query class layout changed")
    optimizer.zero_grad(set_to_none=True)
    # All supported backbones use sample-independent normalization here. Eval
    # mode keeps stochastic layers identical across feature and gradient replay.
    encoder.backbone.eval()
    encoder.projection.train()
    scorer.train()
    features = []
    with torch.no_grad():
        for chunk in images.split(cfg["microbatch"]):
            with _autocast(device):
                features.append(encoder.backbone(chunk).float())
    features = torch.cat(features).detach().requires_grad_(cfg["train_backbone"])
    if not torch.isfinite(features).all():
        raise RuntimeError("Non-finite backbone features before gallery loss; checkpoint was not advanced")
    embeddings = encoder.project(features)
    if not torch.isfinite(embeddings).all():
        raise RuntimeError("Non-finite projected embeddings before gallery loss; checkpoint was not advanced")
    background = None
    if cfg["objective"] == "arcface":
        loss = classifier(embeddings, batch_labels.to(device))
        details = {"arcface": float(loss.detach())}
    elif cfg["objective"] == "supcon":
        labels = torch.arange(n_way, device=device).repeat_interleave(cfg["support"])
        query_labels = torch.arange(n_way, device=device).repeat_interleave(cfg["queries"])
        background = (memory.distractors(support_labels, n_way, device)
                      if memory is not None else None)
        loss = supervised_contrastive_loss(
            embeddings, torch.cat((labels, query_labels)),
            background=None if background is None else background[0])
        details = {"supcon": float(loss.detach()),
                   "memory_negatives": 0 if background is None else len(background[0])}
    else:
        background = (memory.distractors(support_labels, n_way, device)
                      if memory is not None else None)
        if background is not None and not torch.isfinite(background[0]).all():
            raise RuntimeError("Non-finite gallery memory before loss; checkpoint was not advanced")
        loss, details = episode_objective(
            embeddings, scorer, n_way, cfg["support"], cfg["queries"],
            cfg["stability_weight"], cfg["variable_gallery"],
            cfg["pseudo_ood_weight"], background=background)
    if not torch.isfinite(loss):
        feature_finite = bool(torch.isfinite(features).all())
        embedding_finite = bool(torch.isfinite(embeddings).all())
        memory_finite = (None if background is None else
                         bool(torch.isfinite(background[0]).all()))
        raise RuntimeError(
            f"Non-finite gallery training loss: feature_finite={feature_finite}, "
            f"embedding_finite={embedding_finite}, memory_finite={memory_finite}, "
            f"episode_ways={n_way}; checkpoint was not advanced")
    scaler.scale(loss).backward()
    if cfg["train_backbone"]:
        feature_grad = features.grad.detach()
        offset = 0
        for chunk in images.split(cfg["microbatch"]):
            with _autocast(device):
                replayed = encoder.backbone(chunk).float()
            torch.autograd.backward(replayed, feature_grad[offset:offset + len(chunk)])
            offset += len(chunk)
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for group in optimizer.param_groups for p in group["params"]], 1.0)
    scale_before = float(scaler.get_scale())
    scaler.step(optimizer)
    scaler.update()
    scale_after = float(scaler.get_scale())
    details["grad_norm"] = float(grad_norm.detach())
    details["amp_scale"] = scale_after
    details["optimizer_step_skipped"] = bool(scaler.is_enabled() and scale_after < scale_before)
    if memory is not None and not details["optimizer_step_skipped"]:
        memory.add(embeddings[:n_support], support_labels)
        details["memory_classes"] = len(memory)
    return float(loss.detach()), details


def score_queries(scorer, query, references, reference_labels, device,
                  chunk_size=32, rank_k=100):
    scorer.eval()
    g = torch.as_tensor(references, dtype=torch.float32, device=device)
    classes, encoded_labels = np.unique(np.asarray(reference_labels), return_inverse=True)
    gl = torch.as_tensor(encoded_labels, dtype=torch.long, device=device)
    counts = torch.bincount(gl, minlength=len(classes))
    sums = torch.zeros((len(classes), g.shape[1]), device=device).index_add_(0, gl, g)
    prototypes = F.normalize(sums / counts[:, None], dim=1)
    scores, distances, nearest_predictions, prototype_predictions, ranked_labels = [], [], [], [], []
    with torch.inference_mode():
        for start in range(0, len(query), chunk_size):
            q = torch.as_tensor(query[start:start + chunk_size], dtype=torch.float32, device=device)
            similarity = q @ g.T
            current, columns = scorer(q, g, gl, similarity=similarity)
            if not torch.equal(columns, torch.arange(len(classes), device=device)):
                raise ValueError("Gallery scorer omitted a class")
            if not torch.isfinite(current).all() or not torch.isfinite(similarity).all():
                raise RuntimeError("Non-finite gallery scores or image similarities")
            scores.append(current.float().cpu().numpy())
            ranked_sims, ranked = similarity.topk(min(rank_k, len(g)), dim=1)
            nearest_sim, nearest_index = ranked_sims[:, 0], ranked[:, 0]
            ranked_labels.append(gl[ranked].cpu().numpy())
            distances.append((1 - nearest_sim).float().cpu().numpy())
            nearest_predictions.append(classes[gl[nearest_index].cpu().numpy()])
            prototype_predictions.append(classes[(q @ prototypes.T).argmax(dim=1).cpu().numpy()])
    return (np.concatenate(scores), classes, np.concatenate(distances),
            np.concatenate(nearest_predictions), np.concatenate(prototype_predictions),
            np.concatenate(ranked_labels))


def nearest_distances(query, references, device, chunk_size=32):
    g = torch.as_tensor(references, dtype=torch.float32, device=device)
    output = []
    with torch.inference_mode():
        for start in range(0, len(query), chunk_size):
            q = torch.as_tensor(query[start:start + chunk_size], dtype=torch.float32, device=device)
            output.append((1 - (q @ g.T).amax(dim=1)).cpu().numpy())
    return np.concatenate(output)


def _validation(encoder, scorer, manifest, cfg, device):
    reference_items, probe_items = validation_items(
        manifest, group_mode=cfg["group_mode"])
    refs = encode_items(encoder, reference_items, cfg, device)
    probes = encode_items(encoder, probe_items, cfg, device)
    reference_spread = float(np.linalg.norm(refs.std(axis=0)))
    ref_labels = np.asarray([canonical(y) for _, y in reference_items])
    probe_labels = np.asarray([canonical(y) for _, y in probe_items])
    scores, classes, *_ = score_queries(scorer, probes, refs, ref_labels, device)
    full = float(np.mean([np.mean(classes[scores.argmax(1)][probe_labels == c] == c)
                          for c in sorted(set(probe_labels))]))
    subset = set(sorted(set(ref_labels), key=lambda label: _hash_bytes(label.encode()))[:24])
    mask_r = np.asarray([c in subset for c in ref_labels])
    mask_q = np.asarray([c in subset for c in probe_labels])
    small_scores, small_classes, *_ = score_queries(
        scorer, probes[mask_q], refs[mask_r], ref_labels[mask_r], device)
    small_labels = probe_labels[mask_q]
    small = float(np.mean([np.mean(small_classes[small_scores.argmax(1)][small_labels == c] == c)
                           for c in sorted(subset)]))
    # Predeclared, equal-weight model selection over two gallery cardinalities.
    return {"meta_val_r1_all": full, "meta_val_r1_24": small,
            "selection_score": (full + small) / 2,
            "reference_embedding_spread": reference_spread,
            "validation_species": len(set(ref_labels)),
            "group_mode": cfg["group_mode"],
            "small_gallery_species": [str(label) for label in sorted(subset)],
            "validation_reference_images": len(reference_items), "validation_query_images": len(probe_items)}


def _save_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.pt")
    torch.save(payload, temporary)
    temporary.replace(path)


def _run_signature(root, cfg, seed):
    payload = {"config": cfg, "seed": seed,
               "dino_repo_ref": DINO_REPO_REF,
               "manifest_sha256": _hash_file(root / "swi_manifest.json"),
               "runner_sha256": _hash_file(__file__),
               "model_sha256": _hash_file(Path(__file__).with_name("gallery_method.py")),
               "wood_encoder_sha256": _hash_file(Path(__file__).with_name("wood_encoder.py"))}
    return _hash_bytes(json.dumps(payload, sort_keys=True).encode())


def _supervised_step(images, labels, encoder, classifier, optimizer, scaler, device):
    optimizer.zero_grad(set_to_none=True)
    encoder.train()
    classifier.train()
    with _autocast(device):
        features = encoder.backbone(images.to(device, non_blocking=True))
    with torch.autocast(device_type=device.type, enabled=False):
        projected = encoder.projection(features.float())
        logits = classifier(projected)
        loss = F.cross_entropy(logits, labels.to(device, non_blocking=True))
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite supervised warm-up loss; checkpoint was not advanced")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
    torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
    scale_before = float(scaler.get_scale())
    scaler.step(optimizer)
    scaler.update()
    skipped = bool(scaler.is_enabled() and float(scaler.get_scale()) < scale_before)
    return {"loss": float(loss.detach()),
            "accuracy": float((logits.argmax(1) == labels.to(device)).float().mean().detach()),
            "grad_norm": float(grad_norm.detach()), "amp_skipped": skipped}


def train_supervised_warmup(root, out, manifest, cfg, seed, device):
    if cfg["objective"] != "supervised_warmup":
        raise ValueError("Expected supervised_warmup variant")
    _seed(seed)
    run_dir = out / cfg["backbone"] / cfg["variant"] / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    signature = _run_signature(root, cfg, seed)
    manifest_sha256 = _hash_file(root / "swi_manifest.json")
    latest, best = run_dir / "latest.pt", run_dir / "best.pt"
    train_items = [(path, canonical(label)) for path, label in manifest["meta-train"]]
    if os.environ.get("GALLERY_STUDY_PRELOAD", "0") == "1":
        val_refs, val_queries = validation_items(manifest, group_mode=cfg["group_mode"])
        data.preload_image_cache(
            [path for path, _ in train_items + val_refs + val_queries],
            max_workers=int(os.environ.get("GALLERY_STUDY_PRELOAD_WORKERS", "16")),
            desc="Gallery study supervised warm-up images")
    dataset = data.ManifestDataset(
        train_items, transform=data.get_transforms(cfg["image_size"], augment=True))
    labels = torch.tensor(dataset.get_labels(), dtype=torch.long)
    counts = torch.bincount(labels, minlength=len(dataset.class_to_idx)).float()
    sample_weights = counts[labels].rsqrt()
    encoder, scorer = _model(cfg, device, pretrained=False)
    classifier = torch.nn.Linear(cfg["embedding_dim"], len(dataset.class_to_idx)).to(device)
    initial_backbone_sha256 = _state_sha256(encoder.backbone)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=cfg["warmup_lr"], weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda", init_scale=1024.0)
    epoch_start, best_score = 1, -float("inf")
    if latest.exists():
        state = torch.load(latest, map_location="cpu", weights_only=False)
        if (state["signature"] != signature or
                state["initial_backbone_sha256"] != initial_backbone_sha256):
            raise ValueError(f"Supervised warm-up provenance changed at {latest}")
        encoder.load_state_dict(state["encoder"])
        classifier.load_state_dict(state["classifier"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        _restore_rng(state["rng_state"])
        epoch_start = int(state["epoch"]) + 1
        best_score = float(state["best_score"])
        print(f"[gallery] Resuming supervised warm-up seed={seed} at epoch {epoch_start}", flush=True)
    print(f"[gallery] supervised warm-up: {len(dataset.class_to_idx)} meta-train species, "
          f"{cfg['warmup_steps']} steps/epoch, batch={cfg['warmup_batch']}", flush=True)
    for epoch in range(epoch_start, cfg["warmup_epochs"] + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        generator = torch.Generator().manual_seed(seed + epoch * 100003)
        sampler = WeightedRandomSampler(
            sample_weights, num_samples=cfg["warmup_steps"] * cfg["warmup_batch"],
            replacement=True, generator=generator)
        loader = DataLoader(dataset, batch_size=cfg["warmup_batch"], sampler=sampler,
                            num_workers=cfg["workers"], pin_memory=device.type == "cuda",
                            persistent_workers=cfg["workers"] > 0)
        steps = [_supervised_step(images, labels, encoder, classifier, optimizer, scaler, device)
                 for images, labels in loader]
        del loader
        skipped_steps = sum(step["amp_skipped"] for step in steps)
        if skipped_steps == len(steps):
            raise RuntimeError("Every supervised warm-up optimizer step was skipped")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_seconds = time.perf_counter() - start
        validation = _validation(encoder, scorer, manifest, cfg, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        peak_gb = (torch.cuda.max_memory_allocated(device) / 1024**3
                   if device.type == "cuda" else None)
        metric = float(validation["selection_score"])
        if not np.isfinite(metric):
            raise RuntimeError("Non-finite supervised warm-up validation score")
        print(f"[gallery] supervised_warmup seed={seed} epoch={epoch}/{cfg['warmup_epochs']} "
              f"loss={np.mean([step['loss'] for step in steps]):.4f} "
              f"train_acc={np.mean([step['accuracy'] for step in steps]):.4f} "
              f"val24={validation['meta_val_r1_24']:.4f} "
              f"val{validation['validation_species']}={validation['meta_val_r1_all']:.4f} "
              f"ref_spread={validation['reference_embedding_spread']:.4f} "
              f"amp_skips={skipped_steps}/{len(steps)} "
              f"train_s={train_seconds:.1f} peak_gb={peak_gb}", flush=True)
        improved = metric > best_score
        best_score = max(best_score, metric)
        payload = {"checkpoint_kind": "supervised_warmup", "signature": signature,
                   "manifest_sha256": manifest_sha256, "config": cfg, "seed": seed,
                   "epoch": epoch, "best_score": best_score, "validation": validation,
                   "initial_backbone_sha256": initial_backbone_sha256,
                   "encoder": encoder.state_dict(), "classifier": classifier.state_dict(),
                   "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                   "rng_state": _rng_state()}
        if improved:
            _save_checkpoint(best, {key: payload[key] for key in (
                "checkpoint_kind", "signature", "manifest_sha256", "config", "seed", "epoch",
                "best_score", "validation", "initial_backbone_sha256", "encoder", "classifier")})
            _json(run_dir / "best_validation.json", {"signature": signature, "epoch": epoch,
                                                       "validation": validation})
        _save_checkpoint(latest, payload)
        _json(run_dir / "progress.json", {
            "signature": signature, "epoch": epoch, "best_score": best_score,
            "last_validation": validation, "mean_loss": float(np.mean([step["loss"] for step in steps])),
            "train_accuracy": float(np.mean([step["accuracy"] for step in steps])),
            "amp_skipped_steps": skipped_steps, "train_seconds": train_seconds,
            "peak_gpu_memory_gb": peak_gb})
    return best


def _load_warmup_encoder(root, cfg, seed, encoder):
    path = cfg.get("init_checkpoint")
    if not path:
        return
    state = torch.load(path, map_location="cpu", weights_only=False)
    source_cfg = state.get("config", {})
    if (state.get("checkpoint_kind") != "supervised_warmup" or
            state.get("manifest_sha256") != _hash_file(root / "swi_manifest.json") or
            source_cfg.get("backbone") != cfg["backbone"] or
            source_cfg.get("embedding_dim") != cfg["embedding_dim"] or
            state.get("seed") != seed):
        raise ValueError(f"Warm-up checkpoint is not aligned with this metric run: {path}")
    if _hash_file(path) != cfg["init_checkpoint_sha256"]:
        raise ValueError(f"Warm-up checkpoint changed after configuration: {path}")
    encoder.load_state_dict(state["encoder"], strict=True)
    print(f"[gallery] Initialized {cfg['variant']} seed={seed} from supervised warm-up "
          f"epoch {state['epoch']} ({Path(path).name})", flush=True)


def train_variant(root, out, manifest, cfg, seed, device):
    _seed(seed)
    run_dir = out / cfg["backbone"] / cfg["variant"] / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    signature = _run_signature(root, cfg, seed)
    latest, best = run_dir / "latest.pt", run_dir / "best.pt"
    train_items = [(path, canonical(label)) for path, label in manifest["meta-train"]]
    if os.environ.get("GALLERY_STUDY_PRELOAD", "0") == "1":
        val_refs, val_queries = validation_items(manifest, group_mode=cfg["group_mode"])
        data.preload_image_cache(
            [path for path, _ in train_items + val_refs + val_queries],
            max_workers=int(os.environ.get("GALLERY_STUDY_PRELOAD_WORKERS", "16")),
            desc="Gallery study training images")
    dataset = data.ManifestDataset(train_items, transform=data.get_transforms(cfg["image_size"], augment=True))
    labels = [canonical(label) for _, label in train_items]
    train_groups = ([source_scan_id(path) for path, _ in train_items]
                    if cfg["group_mode"] == "scan_disjoint" else None)
    _seed(seed)
    encoder, scorer = _model(cfg, device)
    _load_warmup_encoder(root, cfg, seed, encoder)
    initial_backbone_sha256 = _state_sha256(encoder.backbone)
    print(f"[gallery] backbone={cfg['backbone']} params="
          f"{sum(p.numel() for p in encoder.backbone.parameters()):,} "
          f"memory={cfg['memory_size']} top_m={cfg['top_m']}", flush=True)
    if cfg["objective"] == "pretrained":
        if best.exists():
            existing = torch.load(best, map_location="cpu", weights_only=False)
            if (existing["signature"] != signature or
                    existing.get("initial_backbone_sha256") != initial_backbone_sha256):
                raise ValueError(f"Pretrained-control provenance changed at {best}")
            return best
        validation = _validation(encoder, scorer, manifest, cfg, device)
        _save_checkpoint(best, {"signature": signature, "config": cfg, "seed": seed,
                                "epoch": 0, "best_score": validation["selection_score"],
                                "initial_backbone_sha256": initial_backbone_sha256,
                                "validation": validation, "encoder": encoder.state_dict(),
                                "scorer": scorer.state_dict(), "classifier": None})
        print(f"[gallery] Saved same-resolution pretrained DINOv2 control: {best}", flush=True)
        return best
    classifier = (ArcFaceClassifier(cfg["embedding_dim"], len(dataset.class_to_idx)).to(device)
                  if cfg["objective"] == "arcface" else None)
    memory = (GalleryMemory(cfg["memory_size"], cfg["memory_min_classes"])
              if cfg["objective"] in {"episode", "supcon"} and cfg["memory_size"] else None)
    for p in encoder.backbone.parameters():
        p.requires_grad_(cfg["train_backbone"])
    head_parameters = list(encoder.projection.parameters()) + [
        p for p in scorer.parameters() if p.requires_grad]
    if classifier is not None:
        head_parameters += list(classifier.parameters())
    groups = [{"params": head_parameters,
               "lr": cfg["head_lr"]}]
    if cfg["train_backbone"]:
        groups.append({"params": encoder.backbone.parameters(), "lr": cfg["backbone_lr"]})
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda", init_scale=1024.0)
    epoch_start, best_score = 1, -float("inf")
    if latest.exists():
        state = torch.load(latest, map_location="cpu", weights_only=False)
        if state["signature"] != signature:
            raise ValueError(f"Recipe changed for {latest}; use a new output directory")
        encoder.load_state_dict(state["encoder"])
        scorer.load_state_dict(state["scorer"])
        if classifier is not None:
            classifier.load_state_dict(state["classifier"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        if state.get("initial_backbone_sha256") != initial_backbone_sha256:
            raise ValueError("Initial backbone weights changed since this run began")
        if memory is not None:
            memory.load_state_dict(state["gallery_memory"])
        _restore_rng(state["rng_state"])
        epoch_start = int(state["epoch"]) + 1
        best_score = float(state["best_score"])
        print(f"[gallery] Resuming {cfg['variant']} seed={seed} at epoch {epoch_start}", flush=True)
    for epoch in range(epoch_start, cfg["epochs"] + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        epoch_start_time = time.perf_counter()
        sampler = EpisodeSampler(labels, cfg["ways"], cfg["support"], cfg["queries"],
                                 cfg["episodes_per_epoch"], seed + epoch * 100003,
                                 cfg["hard_negative_probability"], groups=train_groups)
        if epoch == epoch_start:
            print(f"[gallery] scan-disjoint eligible meta-train species="
                  f"{len(sampler.eligible)}", flush=True)
        loader = DataLoader(dataset, batch_sampler=sampler, num_workers=cfg["workers"],
                            pin_memory=device.type == "cuda", persistent_workers=cfg["workers"] > 0)
        losses, train_r1, top_m_episodes, max_gallery, skipped_steps = [], [], 0, 0, 0
        for images, batch_labels in loader:
            loss, details = _episode_step(images, batch_labels, cfg, encoder, scorer,
                                          optimizer, scaler, device, classifier, memory)
            losses.append(loss)
            if "train_episode_r1" in details:
                train_r1.append(details["train_episode_r1"])
            top_m_episodes += int(details.get("top_m_active", False))
            max_gallery = max(max_gallery, details.get("gallery_images", 0))
            skipped_steps += int(details["optimizer_step_skipped"])
        if skipped_steps == len(losses):
            raise RuntimeError(
                f"All {skipped_steps} optimizer steps were skipped by CUDA AMP in epoch {epoch}; "
                f"last gradient norm={details['grad_norm']}, loss scale={details['amp_scale']}")
        del loader
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_seconds = time.perf_counter() - epoch_start_time
        validation = _validation(encoder, scorer, manifest, cfg, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_seconds = time.perf_counter() - epoch_start_time
        peak_gpu_gb = (torch.cuda.max_memory_allocated(device) / 1024**3
                       if device.type == "cuda" else None)
        metric = validation["selection_score"]
        if not np.isfinite(metric):
            raise RuntimeError("Non-finite meta-validation score; checkpoint was not advanced")
        print(f"[gallery] {cfg['variant']} seed={seed} epoch={epoch}/{cfg['epochs']} "
              f"loss={np.mean(losses):.4f} "
              f"train_episode_r1={np.mean(train_r1) if train_r1 else float('nan'):.4f} "
              f"val24={validation['meta_val_r1_24']:.4f} "
              f"val{validation['validation_species']}="
              f"{validation['meta_val_r1_all']:.4f} "
              f"ref_spread={validation['reference_embedding_spread']:.4f} "
              f"topM={top_m_episodes}/{len(losses)} max_gallery={max_gallery} "
              f"amp_skips={skipped_steps}/{len(losses)} "
              f"train_s={train_seconds:.1f} epoch_s={epoch_seconds:.1f} "
              f"peak_gb={peak_gpu_gb}", flush=True)
        improved = metric > best_score
        best_score = max(best_score, metric)
        payload = {"signature": signature, "config": cfg, "seed": seed, "epoch": epoch,
                   "best_score": best_score, "validation": validation,
                   "initial_backbone_sha256": initial_backbone_sha256,
                   "gallery_memory": None if memory is None else memory.state_dict(),
                   "rng_state": _rng_state(),
                   "encoder": encoder.state_dict(), "scorer": scorer.state_dict(),
                   "classifier": None if classifier is None else classifier.state_dict(),
                   "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict()}
        if improved:
            _save_checkpoint(best, {key: payload[key] for key in (
                "signature", "config", "seed", "epoch", "best_score", "validation",
                "initial_backbone_sha256",
                "encoder", "scorer", "classifier")})
            _json(run_dir / "best_validation.json", {"signature": signature, "epoch": epoch,
                                                       "validation": validation})
        _save_checkpoint(latest, payload)
        _json(run_dir / "progress.json", {"signature": signature, "epoch": epoch,
                                            "best_score": best_score, "last_validation": validation,
                                            "train_episode_r1": (float(np.mean(train_r1))
                                                                 if train_r1 else None),
                                            "top_m_episodes": top_m_episodes,
                                            "max_training_gallery_images": max_gallery,
                                            "amp_skipped_steps": skipped_steps,
                                            "memory_classes": 0 if memory is None else len(memory),
                                            "train_seconds": train_seconds,
                                            "epoch_seconds": epoch_seconds,
                                            "peak_gpu_memory_gb": peak_gpu_gb})
    return best


def _public_frames(root, manifest):
    frames = {}
    for side, expected in (("id", 6189), ("ood", 30868)):
        path = root / f"{side.upper()}_images_expanded.csv"
        frame = pd.read_csv(path)
        needed = {"file_path", "label", "source_dataset"}
        if len(frame) != expected or not needed.issubset(frame.columns):
            raise ValueError(f"Expected corrected {side} CSV with {expected} rows and {sorted(needed)}")
        frame["label"] = frame["label"].map(canonical)
        frame["source_dataset"] = frame["source_dataset"].fillna("").astype(str)
        frame["file_path"] = frame["file_path"].astype(str).str.replace(
            data.LOCAL_PREFIX, str(root / "datasets"), regex=False)
        missing_source = frame["source_dataset"].str.len().eq(0)
        if missing_source.any():
            from .experiments.ood_within_source import _infer_source
            frame.loc[missing_source, "source_dataset"] = frame.loc[
                missing_source, "file_path"].map(_infer_source)
        if frame["source_dataset"].str.upper().str.contains("WOODAUTH|WOOD_AUTH|WOOD-AUTH").any():
            raise ValueError("WoodAuth rows remain in corrected public CSV")
        frames[side] = frame
    if len(set(frames["id"]["label"])) != 24 or len(set(frames["ood"]["label"])) != 153:
        raise ValueError("Corrected public species counts do not match the audited cohort")
    if set(frames["id"]["label"]) & set(frames["ood"]["label"]):
        raise ValueError("Public ID/OOD species overlap")
    train_species = {canonical(label) for _, label in manifest["meta-train"]}
    val_species = {canonical(label) for _, label in manifest["meta-val"]}
    test_species = {canonical(label) for _, label in manifest["meta-test"]}
    id_species = set(frames["id"]["label"])
    if id_species & (train_species | val_species) or not id_species <= test_species:
        raise ValueError("Public ID species must be exclusively in SWI meta-test")
    if set(frames["ood"]["label"]) & (train_species | val_species | test_species):
        raise ValueError("Public OOD species overlap the SWI species universe")
    return frames


def _embedding_cache(encoder, items, cfg, device, target, checkpoint_hash):
    """Sequential .npy extraction with a checked resume cursor."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    signature = _hash_bytes(json.dumps({"checkpoint": checkpoint_hash,
                                         "image_size": cfg["image_size"],
                                         "items": items}, separators=(",", ":")).encode())
    progress_path = target.with_suffix(".progress.json")
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if progress["signature"] != signature or not target.exists():
            raise ValueError(f"Stale or incomplete embedding cache at {target}")
        offset = int(progress["offset"])
        matrix = np.lib.format.open_memmap(target, mode="r+")
    else:
        if target.exists():
            raise ValueError(f"Embedding cache {target} has no provenance cursor")
        matrix = np.lib.format.open_memmap(
            target, mode="w+", dtype=np.float32, shape=(len(items), cfg["embedding_dim"]))
        offset = 0
        _json(progress_path, {"signature": signature, "offset": 0,
                              "checkpoint_sha256": checkpoint_hash, "rows": len(items)})
    if matrix.shape != (len(items), cfg["embedding_dim"]) or not 0 <= offset <= len(items):
        raise ValueError(f"Invalid embedding cache dimensions or cursor for {target}")
    if offset == len(items):
        print(f"[gallery] Reusing {target} ({offset} images)", flush=True)
        return np.load(target, mmap_mode="r")
    print(f"[gallery] Extracting {target.name}: {offset}/{len(items)} complete", flush=True)
    encoder.eval()
    loader = _loader(items[offset:], cfg)
    interval = max(1, int(os.environ.get("GALLERY_STUDY_CACHE_INTERVAL", "100")))
    with torch.inference_mode():
        for batch_index, (images, _) in enumerate(loader, start=1):
            with _autocast(device):
                embeddings = encoder(images.to(device, non_blocking=True))
            if not torch.isfinite(embeddings).all():
                raise RuntimeError(f"Non-finite embeddings while extracting {target} at row {offset}")
            n = len(images)
            matrix[offset:offset + n] = embeddings.float().cpu().numpy()
            offset += n
            if batch_index % interval == 0 or offset == len(items):
                matrix.flush()
                _json(progress_path, {"signature": signature, "offset": offset,
                                      "checkpoint_sha256": checkpoint_hash, "rows": len(items)})
    if offset != len(items) or not np.isfinite(matrix).all():
        raise ValueError(f"Incomplete or non-finite embedding cache {target}")
    return np.load(target, mmap_mode="r")


def macro_accuracy(predictions, labels):
    predictions, labels = np.asarray(predictions), np.asarray(labels)
    if len(predictions) != len(labels) or len(labels) == 0:
        raise ValueError("Predictions and labels must have the same non-zero length")
    return float(np.mean([np.mean(predictions[labels == c] == c) for c in sorted(set(labels))]))


def _predict(scorer, queries, q_labels, gallery, g_labels, device):
    scores, classes, distance, nearest, prototype, ranked_indices = score_queries(
        scorer, queries, gallery, g_labels, device)
    predictions = classes[scores.argmax(axis=1)]
    ranked = classes[ranked_indices]
    relevance = ranked == np.asarray(q_labels)[:, None]
    positions = np.arange(1, relevance.shape[1] + 1)
    relevant_counts = pd.Series(g_labels).value_counts()
    denominators = np.minimum(
        np.asarray([relevant_counts.get(label, 0) for label in q_labels]),
        relevance.shape[1]).clip(min=1)
    precision = relevance.cumsum(axis=1) / positions[None]
    ap = (precision * relevance).sum(axis=1) / denominators
    first = relevance.argmax(axis=1) + 1
    rr = np.where(relevance.any(axis=1), 1.0 / first, 0.0)
    q_labels = np.asarray(q_labels)
    image_map_at_100 = float(np.mean([ap[q_labels == label].mean()
                                      for label in sorted(set(q_labels))]))
    image_mrr_at_100 = float(np.mean([rr[q_labels == label].mean()
                                      for label in sorted(set(q_labels))]))
    return {"species_r1": macro_accuracy(predictions, q_labels),
            "nearest_image_r1": macro_accuracy(nearest, q_labels),
            "prototype_r1": macro_accuracy(prototype, q_labels),
            "image_map_at_100": image_map_at_100,
            "image_mrr_at_100": image_mrr_at_100,
            "distance": distance, "predictions": predictions}


def _ood_metrics(id_scores, ood_scores):
    from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
    y = np.r_[np.zeros(len(id_scores)), np.ones(len(ood_scores))]
    scores = np.r_[id_scores, ood_scores]
    fpr, tpr, _ = roc_curve(y, scores)
    idx = min(int(np.searchsorted(tpr, 0.95)), len(fpr) - 1)
    return {"auroc": float(roc_auc_score(y, scores)),
            "aupr_ood": float(average_precision_score(y, scores)), "fpr95": float(fpr[idx]),
            "id_images": len(id_scores), "ood_images": len(ood_scores)}


def _ood_test_mask(labels):
    species = sorted(set(labels))
    rng = np.random.RandomState(42)
    rng.shuffle(species)
    n_val = max(1, int(0.2 * len(species)))
    chosen = set(species[n_val:])
    return np.asarray([x in chosen for x in labels], dtype=bool)


def _within_source_control(id_emb, ood_emb, id_frame, ood_frame, device):
    from .experiments.ood_within_source import _source_protocol_indices
    meta = pd.concat((id_frame[["label", "source_dataset"]],
                      ood_frame[["label", "source_dataset"]]), ignore_index=True)
    public = np.concatenate((id_emb, ood_emb))
    results, pooled_id, pooled_ood = {}, [], []
    for source in sorted(meta["source_dataset"].unique()):
        protocol = _source_protocol_indices(meta, source)
        if protocol is None:
            continue
        gallery = public[protocol["gallery_local"]]
        sid = nearest_distances(public[protocol["id_query_local"]], gallery, device)
        sod = nearest_distances(public[protocol["ood_query_local"]], gallery, device)
        results[source] = {**_ood_metrics(sid, sod),
                           "known_species": len(protocol["id_species"]),
                           "heldout_ood_species": len(protocol["ood_species"]),
                           "gallery_images": len(protocol["gallery_local"])}
        pooled_id.append(sid)
        pooled_ood.append(sod)
    if not results:
        raise ValueError("No source satisfied the established within-source OOD protocol")
    return {"per_source": results,
            "pooled": _ood_metrics(np.concatenate(pooled_id), np.concatenate(pooled_ood))}


def _gallery_expansion(scorer, swi, swi_labels, id_emb, id_labels,
                       ood_emb, ood_labels, ood_frame, device):
    counts = ood_frame.groupby("label").size().sort_values(ascending=False)
    species = counts.head(50).index.tolist()
    rng = np.random.RandomState(42)
    reference_idx, query_idx = [], []
    for label in species:
        indices = np.flatnonzero(ood_labels == label)
        rng.shuffle(indices)
        n_query = min(10, len(indices) // 2)
        query_idx.extend(indices[:n_query].tolist())
        reference_idx.extend(indices[n_query:n_query + 10].tolist())
    gallery = np.concatenate((swi, ood_emb[reference_idx]))
    labels = np.r_[swi_labels, ood_labels[reference_idx]]
    old = _predict(scorer, id_emb, id_labels, gallery, labels, device)
    new = _predict(scorer, ood_emb[query_idx], ood_labels[query_idx], gallery, labels, device)
    return {"old_after_r1": old["species_r1"],
            "new_species_r1": new["species_r1"],
            "old_nearest_r1": old["nearest_image_r1"],
            "new_nearest_r1": new["nearest_image_r1"],
            "new_gallery_images": len(reference_idx), "new_query_images": len(query_idx),
            "new_species": len(species)}


def _public_kshot(scorer, id_emb, id_labels, id_frame, device, repeats=30):
    rng = np.random.RandomState(42)
    splits = {}
    for label in sorted(set(id_labels)):
        indices = np.flatnonzero(id_labels == label)
        rng.shuffle(indices)
        n_query = min(10, len(indices) // 2)
        splits[label] = (indices[:n_query], indices[n_query:])
    query_idx = np.concatenate([part[0] for part in splits.values()])
    results = []
    for k in (1, 5, 10):
        for repeat in range(repeats):
            sample = np.random.RandomState(42 + repeat)
            refs = np.concatenate([sample.choice(pool, min(k, len(pool)), replace=False)
                                   for _, pool in splits.values()])
            result = _predict(scorer, id_emb[query_idx], id_labels[query_idx],
                              id_emb[refs], id_labels[refs], device)
            results.append({"k": k, "repeat": repeat,
                            "r1": result["species_r1"],
                            "nearest_r1": result["nearest_image_r1"],
                            "prototype_r1": result["prototype_r1"]})
    return pd.DataFrame(results)


def _cardinality_curve(scorer, id_emb, id_labels, swi_emb, swi_items, device):
    by_class = defaultdict(list)
    for index, (path, label) in enumerate(swi_items):
        by_class[label].append((path, index))
    target = sorted(set(id_labels))
    distractors = sorted(set(by_class) - set(target))
    if len(target) != 24 or len(distractors) != 930:
        raise ValueError("Unexpected species universe for cardinality curve")
    rng = np.random.RandomState(42)
    rng.shuffle(distractors)
    k_refs = min(5, min(len(entries) for entries in by_class.values()))
    selected_indices = {}
    for label, entries in by_class.items():
        chosen = sorted(entries, key=lambda entry: _hash_bytes(entry[0].encode()))[:k_refs]
        selected_indices[label] = [index for _, index in chosen]
    rows = []
    for n_species in (24, 50, 100, 250, 500, 954):
        species = target + distractors[:n_species - 24]
        indices = [index for label in species for index in selected_indices[label]]
        labels = np.asarray([label for label in species for _ in range(k_refs)])
        result = _predict(scorer, id_emb, id_labels, swi_emb[indices], labels, device)
        rows.append({"species": n_species, "references_per_species": k_refs,
                     "gallery_images": len(indices),
                     "r1": result["species_r1"],
                     "nearest_r1": result["nearest_image_r1"],
                     "prototype_r1": result["prototype_r1"]})
    return pd.DataFrame(rows)


def _vn26_generalization(root, run_dir, encoder, scorer, cfg, checkpoint_hash,
                         matrices, items, device):
    from .final_colab_audit import _vn26_items
    by_path = {}
    for split in ("id", "ood"):
        for index, (path, label) in enumerate(items[split]):
            if path in by_path and by_path[path][1] != label:
                raise ValueError(f"Public image has conflicting labels: {path}")
            by_path[path] = (np.asarray(matrices[split][index]), label)
    vn26 = _vn26_items(root)
    by_mag = {}
    missing_count = 0
    for mag, rows in vn26.items():
        embeddings = [None] * len(rows)
        missing_positions, missing_items = [], []
        for index, (path, label) in enumerate(rows):
            label = canonical(label)
            if path in by_path:
                vector, public_label = by_path[path]
                if label != public_label:
                    raise ValueError(f"VN26/public label conflict for {path}")
                embeddings[index] = vector
            else:
                missing_positions.append(index)
                missing_items.append((path, label))
        if missing_items:
            fresh = _embedding_cache(encoder, missing_items, cfg, device,
                                     run_dir / "embeddings" / f"vn26_{mag}_missing.npy",
                                     checkpoint_hash)
            for position, vector in zip(missing_positions, fresh):
                embeddings[position] = vector
            missing_count += len(missing_items)
        by_mag[mag] = (np.stack(embeddings).astype(np.float32),
                       np.asarray([canonical(label) for _, label in rows]))
    swi_labels = np.asarray([label for _, label in items["swi"]])
    vn_embeddings = np.concatenate([by_mag[mag][0] for mag in ("x10", "x20", "x50")])
    vn_labels = np.concatenate([by_mag[mag][1] for mag in ("x10", "x20", "x50")])
    common = set(swi_labels) & set(vn_labels)
    mask_g = np.isin(swi_labels, list(common))
    mask_q = np.isin(vn_labels, list(common))
    pool = _predict(scorer, vn_embeddings[mask_q], vn_labels[mask_q],
                    matrices["swi"][mask_g], swi_labels[mask_g], device)
    cross = {}
    for gallery_mag in ("x10", "x20", "x50"):
        for query_mag in ("x10", "x20", "x50"):
            if gallery_mag == query_mag:
                continue
            g_emb, g_labels = by_mag[gallery_mag]
            q_emb, q_labels = by_mag[query_mag]
            shared = set(g_labels) & set(q_labels)
            if not shared:
                continue
            gm = np.isin(g_labels, list(shared))
            qm = np.isin(q_labels, list(shared))
            result = _predict(scorer, q_emb[qm], q_labels[qm], g_emb[gm], g_labels[gm], device)
            cross[f"{gallery_mag}_to_{query_mag}"] = {
                "species": len(shared), "queries": int(qm.sum()),
                "r1": result["species_r1"],
                "nearest_r1": result["nearest_image_r1"],
                "prototype_r1": result["prototype_r1"]}
    return {"swi_pool_to_vn26_all": {"species": len(common), "queries": int(mask_q.sum()),
                                       "r1": pool["species_r1"],
                                       "nearest_r1": pool["nearest_image_r1"],
                                       "prototype_r1": pool["prototype_r1"]},
            "cross_magnification": cross,
            "vn26_images_not_reused_from_public": missing_count}


def evaluate_variant(root, out, manifest, cfg, seed, device, repeats=30):
    run_dir = out / cfg["backbone"] / cfg["variant"] / f"seed_{seed}"
    checkpoint = run_dir / "best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing meta-val-selected checkpoint: {checkpoint}")
    frames = _public_frames(root, manifest)
    verified_cache = Path(os.environ.get("GALLERY_STUDY_VERIFIED_CACHE",
                                       root / "embedding_cache_full954_v6_public_row_verified.npz"))
    if not verified_cache.exists():
        raise FileNotFoundError(f"Corrected public-row verification cache is required: {verified_cache}")
    from .final_scurd_retrain import validate_public
    public_audit = validate_public(root, verified_cache, root / "swi_manifest.json")
    checkpoint_hash = _hash_file(checkpoint)
    _seed(seed)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state["config"] != cfg or state["seed"] != seed or state["signature"] != _run_signature(root, cfg, seed):
        raise ValueError(f"Checkpoint recipe mismatch at {checkpoint}")
    encoder, scorer = _model(cfg, device)
    if _state_sha256(encoder.backbone) != state["initial_backbone_sha256"]:
        raise ValueError("Initial backbone weights differ from those used in training")
    encoder.load_state_dict(state["encoder"])
    scorer.load_state_dict(state["scorer"])
    encoder.eval()
    scorer.eval()
    items = {
        "swi": [(path, canonical(label)) for path, label in data.full_swi_items(manifest)],
        "id": list(zip(frames["id"]["file_path"], frames["id"]["label"])),
        "ood": list(zip(frames["ood"]["file_path"], frames["ood"]["label"])),
    }
    matrices = {}
    for split, rows in items.items():
        matrices[split] = _embedding_cache(
            encoder, rows, cfg, device, run_dir / "embeddings" / f"{split}.npy", checkpoint_hash)
    swi_labels = np.asarray([label for _, label in items["swi"]])
    id_labels = frames["id"]["label"].to_numpy()
    ood_labels = frames["ood"]["label"].to_numpy()
    scopes = {}
    for scope in ("id_only", "full_swi"):
        mask = np.isin(swi_labels, np.unique(id_labels)) if scope == "id_only" else np.ones(len(swi_labels), bool)
        if scope == "id_only" and (mask.sum() != 3315 or len(set(swi_labels[mask])) != 24):
            raise ValueError("24-species SWI gallery no longer matches the audited protocol")
        if scope == "full_swi" and (len(swi_labels) != 176123 or len(set(swi_labels)) != 954):
            raise ValueError("954-species SWI gallery no longer matches the audited protocol")
        prediction = _predict(scorer, matrices["id"], id_labels,
                              matrices["swi"][mask], swi_labels[mask], device)
        scopes[scope] = {k: value for k, value in prediction.items()
                         if k not in {"distance", "predictions"}}
        scopes[scope]["gallery_images"] = int(mask.sum())
        scopes[scope]["gallery_species"] = len(set(swi_labels[mask]))
        if scope == "id_only":
            id_distance = prediction["distance"]
            id_mask = mask
        print(f"[gallery] {cfg['variant']} seed={seed} {scope}: "
              f"R@1={prediction['species_r1']:.4f}", flush=True)
    ood_test = _ood_test_mask(ood_labels)
    ood_distance = nearest_distances(matrices["ood"][ood_test],
                                     matrices["swi"][id_mask], device)
    ood_result = _ood_metrics(id_distance, ood_distance)
    within = _within_source_control(
        matrices["id"], matrices["ood"], frames["id"], frames["ood"], device)
    expansion = _gallery_expansion(
        scorer, matrices["swi"][id_mask], swi_labels[id_mask], matrices["id"], id_labels,
        matrices["ood"], ood_labels, frames["ood"], device)
    kshot = _public_kshot(scorer, matrices["id"], id_labels, frames["id"], device, repeats)
    cardinality = _cardinality_curve(scorer, matrices["id"], id_labels,
                                     matrices["swi"], items["swi"], device)
    vn26 = (_vn26_generalization(root, run_dir, encoder, scorer, cfg, checkpoint_hash,
                                 matrices, items, device)
            if os.environ.get("GALLERY_STUDY_EVAL_VN26", "1") == "1" else None)
    kshot.to_csv(run_dir / "kshot_repeats.csv", index=False)
    cardinality.to_csv(run_dir / "cardinality_curve.csv", index=False)
    report = {"variant": cfg["variant"], "seed": seed, "config": cfg,
              "selection": state["validation"], "checkpoint_sha256": checkpoint_hash,
              "run_signature": state["signature"],
              "runner_sha256": _hash_file(__file__),
              "model_sha256": _hash_file(Path(__file__).with_name("gallery_method.py")),
              "wood_encoder_sha256": _hash_file(Path(__file__).with_name("wood_encoder.py")),
              "backbone": cfg["backbone"],
              "backbone_initialization": ("random" if cfg["backbone"].startswith("woodpattern_")
                                           else "pretrained"),
              "dino_repo_ref": DINO_REPO_REF if cfg["backbone"].startswith("dinov2_") else None,
              "backbone_parameters": sum(p.numel() for p in encoder.backbone.parameters()),
              "torch_version": torch.__version__, "device": str(device),
              "manifest_sha256": _hash_file(root / "swi_manifest.json"),
              "public_csv_sha256": {side: _hash_file(root / f"{side.upper()}_images_expanded.csv")
                                     for side in ("id", "ood")},
              "public_audit": public_audit, "scopes": scopes,
              "pooled_ood": ood_result, "within_source_ood": within,
              "enrollment": expansion,
              "vn26_generalization": vn26,
              "cardinality_curve": json.loads(cardinality.to_json(orient="records")),
              "kshot": {str(k): {"mean": float(group["r1"].mean()),
                                  "std": float(group["r1"].std(ddof=1)) if len(group) > 1 else 0.0}
                        for k, group in kshot.groupby("k")},
              "specimen_disjointness": {
                  "swi_training_and_validation": "source-scan-disjoint support/query by patch filename",
                  "public_kshot": "unverified: public source-specimen IDs unavailable"},
              "caution": "Public data informed study design; this is not an untouched confirmatory test."}
    _json(run_dir / "evaluation.json", report)
    return report


def run():
    root = Path(os.environ.get("ROOT_PATH", "/content/drive/MyDrive/NCS")).resolve()
    out = Path(os.environ.get("GALLERY_STUDY_OUT", root / "results" / "metric_retrieval_study")).resolve()
    mode = os.environ.get("GALLERY_STUDY_MODE", "pilot").lower()
    if mode not in {"smoke", "backbone_smoke", "preflight", "pilot", "warmup", "train", "evaluate"}:
        raise ValueError("GALLERY_STUDY_MODE must be smoke, backbone_smoke, preflight, pilot, "
                         "warmup, train, or evaluate")
    device = torch.device("cuda" if os.environ.get("DEVICE", "cuda") == "cuda" and
                          torch.cuda.is_available() else "cpu")
    if mode == "smoke":
        return smoke_test(device)
    if mode == "backbone_smoke":
        if device.type != "cuda":
            raise RuntimeError("Backbone smoke test requires CUDA")
        backbone = os.environ.get("GALLERY_STUDY_BACKBONE", "dinov2_vitb14").lower()
        return smoke_test(device, real_backbone=True, backbone=backbone)
    if device.type != "cuda" and mode in {"pilot", "warmup", "train"}:
        raise RuntimeError("Full-backbone training requires a CUDA runtime")
    pilot = mode == "pilot" or os.environ.get("GALLERY_STUDY_PILOT_CHECKPOINTS", "0") == "1"
    default_variants = ("supervised_warmup" if mode == "warmup" else
                        "metric_retrieval,prototype_retrieval" if mode == "pilot" else
                        "metric_retrieval")
    variants = [name.strip() for name in os.environ.get("GALLERY_STUDY_VARIANTS", default_variants).split(",")]
    seeds = [int(x.strip()) for x in os.environ.get("GALLERY_STUDY_SEEDS", "42" if mode == "pilot" else "42,43,44").split(",")]
    if any(name not in VARIANTS for name in variants) or len(set(variants)) != len(variants):
        raise ValueError(f"Invalid or duplicate variant in {variants}; allowed: {sorted(VARIANTS)}")
    if not variants or not seeds:
        raise ValueError("At least one variant and seed are required")
    if (mode == "warmup" and variants != ["supervised_warmup"]) or (
            mode != "warmup" and "supervised_warmup" in variants):
        raise ValueError("supervised_warmup must run alone in warmup mode")
    manifest = data.load_swi_manifest(root / "swi_manifest.json")
    expected_splits = {"meta-train": (124577, 557), "meta-val": (18018, 80),
                       "meta-test": (33528, 317)}
    species_by_split = {}
    for split, (expected_images, expected_species) in expected_splits.items():
        species = {canonical(label) for _, label in manifest[split]}
        if len(manifest[split]) != expected_images or len(species) != expected_species:
            raise ValueError(f"SWI manifest {split} no longer matches the audited protocol")
        species_by_split[split] = species
    if any(species_by_split[a] & species_by_split[b] for a, b in (
            ("meta-train", "meta-val"), ("meta-train", "meta-test"),
            ("meta-val", "meta-test"))):
        raise ValueError("SWI manifest species splits overlap")
    out.mkdir(parents=True, exist_ok=True)
    print(f"[gallery] mode={mode} device={device} out={out} variants={variants} seeds={seeds}", flush=True)
    scan_audit = source_scan_audit(manifest)
    train_items = manifest["meta-train"]
    train_groups = [source_scan_id(path) for path, _ in train_items]
    eligible_sampler = EpisodeSampler(
        [label for _, label in train_items], (16,), 2, 1, 0, 42,
        groups=train_groups)
    val_references, _ = validation_items(manifest, group_mode="scan_disjoint")
    scan_audit["protocol"] = {
        "train_episode_eligible_species": len(eligible_sampler.eligible),
        "validation_eligible_species": len(val_references) // 5,
        "support_query_grouping": "different Tw source scans"}
    _json(out / "source_scan_audit.json", scan_audit)
    print(f"[gallery] source-scan audit: {scan_audit}", flush=True)
    if mode == "preflight":
        return out
    reports = []
    for variant in variants:
        cfg = variant_config(variant, pilot=pilot)
        for seed in seeds:
            if mode == "warmup":
                train_supervised_warmup(root, out, manifest, cfg, seed, device)
            elif mode in {"pilot", "train"}:
                train_variant(root, out, manifest, cfg, seed, device)
            else:
                reports.append(evaluate_variant(
                    root, out, manifest, cfg, seed, device,
                    repeats=int(os.environ.get("GALLERY_STUDY_KSHOT_REPEATS", "30"))))
    if reports:
        by_run = {(report["backbone"], report["variant"], report["seed"]): report
                  for report in reports}
        for path in sorted(out.glob("*/*/seed_*/evaluation.json")):
            saved = json.loads(path.read_text())
            if (saved.get("runner_sha256") != _hash_file(__file__) or
                    saved.get("model_sha256") != _hash_file(Path(__file__).with_name("gallery_method.py")) or
                    saved.get("wood_encoder_sha256") != _hash_file(Path(__file__).with_name("wood_encoder.py"))):
                raise ValueError(f"Cannot combine evaluation from a different code version: {path}")
            by_run.setdefault((saved["backbone"], saved["variant"], saved["seed"]), saved)
        reports = [by_run[key] for key in sorted(by_run)]
        rows = []
        for report in reports:
            rows.append({"variant": report["variant"], "backbone": report["backbone"],
                         "seed": report["seed"],
                         "meta_val_selection": report["selection"]["selection_score"],
                         "id24_r1": report["scopes"]["id_only"]["species_r1"],
                         "matched954_r1": report["scopes"]["full_swi"]["species_r1"],
                         "id24_map100": report["scopes"]["id_only"]["image_map_at_100"],
                         "matched954_map100": report["scopes"]["full_swi"]["image_map_at_100"],
                         "id24_mrr100": report["scopes"]["id_only"]["image_mrr_at_100"],
                         "matched954_mrr100": report["scopes"]["full_swi"]["image_mrr_at_100"],
                         "pooled_ood_auroc": report["pooled_ood"]["auroc"],
                         "pooled_ood_fpr95": report["pooled_ood"]["fpr95"],
                         "old_after": report["enrollment"]["old_after_r1"],
                         "new_species": report["enrollment"]["new_species_r1"],
                         "vn26_swi_pool_r1": (float("nan") if report["vn26_generalization"] is None else
                                              report["vn26_generalization"]["swi_pool_to_vn26_all"]["r1"])})
        pd.DataFrame(rows).to_csv(out / "comparison.csv", index=False)
        table = pd.DataFrame(rows)
        columns = [name for name in table.columns if name not in {"variant", "backbone", "seed"}]
        summary = table.groupby(["backbone", "variant"])[columns].agg(["mean", "std", "count"])
        summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
        summary.to_csv(out / "comparison_summary.csv")
        if "metric_retrieval" in set(table["variant"]):
            proposed = table[table["variant"] == "metric_retrieval"].set_index(
                ["backbone", "seed"])
            paired = []
            for (backbone, variant), frame in table.groupby(["backbone", "variant"]):
                if variant == "metric_retrieval":
                    continue
                current = frame.set_index(["backbone", "seed"])
                for _, seed in proposed.index.intersection(current.index):
                    pair = {"backbone": backbone, "variant": variant, "seed": int(seed)}
                    for metric in columns:
                        pair[f"delta_{metric}"] = float(
                            proposed.loc[(backbone, seed), metric] -
                            current.loc[(backbone, seed), metric])
                    paired.append(pair)
            if paired:
                pd.DataFrame(paired).to_csv(out / "paired_seed_differences.csv", index=False)
        _json(out / "evaluation_manifest.json", {"reports": [str(out / row["backbone"] /
            row["variant"] /
            f"seed_{row['seed']}" / "evaluation.json") for row in rows], "status": "exploratory"})
    print(f"[gallery] complete: {out}", flush=True)
    return out


def smoke_test(device, real_backbone=False, backbone="dinov2_vitb14"):
    from torch import nn
    _seed(42)
    cfg = variant_config("metric_retrieval", pilot=True)
    cfg.update(microbatch=2, support=2, queries=1, ways=(4,), backbone=backbone,
               top_m=4, memory_size=12, memory_min_classes=8)
    if real_backbone:
        encoder, scorer = _model(cfg, device)
        size = cfg["image_size"]
    else:
        backbone = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 768), nn.LayerNorm(768))
        encoder = GalleryEncoder(backbone, embedding_dim=512).to(device)
        scorer = GalleryScorer(mode="nearest").to(device)
        size = 8
    optimizer = torch.optim.AdamW(list(encoder.parameters()) +
                                  [p for p in scorer.parameters() if p.requires_grad], lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda", init_scale=1024.0)
    images = torch.randn(12, 3, size, size)
    labels = torch.arange(4).repeat_interleave(2).tolist() + list(range(4))
    watched = next(encoder.backbone.parameters())
    initial = watched.detach().clone()
    memory = GalleryMemory(cfg["memory_size"], cfg["memory_min_classes"])
    memory.add(F.normalize(torch.randn(8, 512), dim=1), torch.arange(4, 12))
    skipped_steps = 0
    for _ in range(8):
        loss, details = _episode_step(images, torch.tensor(labels), cfg, encoder, scorer,
                                      optimizer, scaler, device, memory=memory)
        if not np.isfinite(loss):
            raise RuntimeError("Gallery study smoke test returned a non-finite loss")
        if details["top_m_active"]:
            raise RuntimeError("Exact nearest retrieval unexpectedly activated top-M scoring")
        skipped_steps += int(details["optimizer_step_skipped"])
        if not torch.equal(initial, watched.detach()):
            break
    else:
        raise RuntimeError(
            f"Gallery study smoke test did not update the backbone after 8 attempts; "
            f"AMP skipped {skipped_steps}, last gradient norm={details['grad_norm']}, "
            f"loss scale={details['amp_scale']}")
    print(f"[gallery] {backbone if real_backbone else 'synthetic'} smoke test passed "
          f"on {device}; final loss={loss:.4f}, amp_skips={skipped_steps}", flush=True)
    return loss
