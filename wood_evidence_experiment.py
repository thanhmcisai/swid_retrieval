"""Isolated local-evidence study on scan-disjoint wood-image galleries.

The DINOv2-S encoder stays frozen in this diagnostic. Only a small patch-token
adapter and a mixing coefficient are fitted on meta-train representatives.
Meta-test is accessible only in the explicit final mode after model selection.
"""

import json
import inspect
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from . import data, gallery_diagnostics as diagnostics, gallery_experiment as study
from .wood_evidence_method import WoodEvidenceReranker, prototype_scores

_LEGACY_EXTRACTOR_SOURCE_SHA256 = "7a0a40e475bed1ee9d9bfe4ea58f355afe27775d6762efc07db99d1cee3cf9ff"
_LEGACY_EXPERIMENT_SHA256 = "479dc0dece69f5c6f7aafee367d99504c95161e93b60e5b739dbb38f3d336f5e"


def _atomic_json(path, payload):
    study._json(path, payload)


def _base_checkpoint(root, study_out, seed):
    path = study_out / "dinov2_vits14" / "prototype_large" / f"seed_{seed}" / "best.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing selected DINOv2-S checkpoint: {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    cfg = state["config"]
    if (state["seed"] != seed or cfg["variant"] != "prototype_large" or
            cfg["backbone"] != "dinov2_vits14" or cfg["image_size"] != 224 or
            cfg["scorer_mode"] != "prototype" or cfg.get("local_weight", 0) or
            cfg["group_mode"] != "scan_disjoint" or
            cfg["validation_folds"] != 2 or
            cfg["validation_train_distractors"] != 1 or
            "637" not in state["validation"].get("meta_val_gallery_curve", {}) or
            state["signature"] != study._run_signature(root, cfg, seed)):
        raise ValueError("Base checkpoint recipe/code/manifest provenance changed")
    return path, state, cfg


def _representatives(manifest, max_scans=3, images_per_scan=2):
    by_class = defaultdict(lambda: defaultdict(list))
    for path, label in manifest["meta-train"]:
        by_class[study.canonical(label)][study.source_scan_id(path)].append(path)
    rows = []
    for label, scans in sorted(by_class.items()):
        if len(scans) < 2:
            continue
        ordered_scans = sorted(scans, key=lambda s: study._hash_bytes(s.encode()))[:max_scans]
        for scan in ordered_scans:
            chosen = sorted(scans[scan], key=lambda p: study._hash_bytes(p.encode()))
            rows.extend((path, label) for path in chosen[:images_per_scan])
    return rows


def _validation_sets(manifest, split, fold):
    if split == "meta-val":
        refs, queries = study.validation_items(
            manifest, group_mode="scan_disjoint", fold=fold)
    elif split == "meta-test":
        refs, queries = study.validation_items(
            {"meta-val": manifest["meta-test"]}, group_mode="scan_disjoint", fold=fold)
    else:
        raise ValueError("Only meta-val and locked meta-test are supported")
    labels = {study.canonical(label) for _, label in refs}
    extras = diagnostics._extra_references(manifest, labels)
    return refs, queries, extras


def _gallery_plan(refs, queries, extras, split):
    if split == "meta-val":
        return diagnostics.gallery_plan(refs, queries, extras)
    labels = np.asarray([study.canonical(label) for _, label in refs])
    first = {}
    for index, label in enumerate(labels):
        first.setdefault(label, index)
    balanced = np.asarray(list(first.values()), dtype=np.int64)
    if len(balanced) < 200 or len(balanced) >= 637:
        raise ValueError("Unexpected number of scan-disjoint meta-test species")
    needed = 637 - len(balanced)
    if len(extras) < needed:
        raise ValueError("Not enough held-out distractor species")
    ordered = sorted(range(len(extras)),
                     key=lambda index: study._hash_bytes(extras[index][1].encode()))
    full = np.r_[balanced, len(refs) + np.asarray(ordered[:needed], dtype=np.int64)]
    return {f"{len(balanced)}x5": (np.arange(len(refs)), np.arange(len(queries))),
            f"{len(balanced)}x1": (balanced, np.arange(len(queries))),
            "637x1": (full, np.arange(len(queries)))}


def _cache_signature(items, base_hash, extractor_hash):
    payload = {"checkpoint_sha256": base_hash,
               "extractor_sha256": extractor_hash,
               "items": [[str(path), study.canonical(label)] for path, label in items],
               "transform": "get_transforms(224, augment=False)", "patch_grid": 4}
    return study._hash_bytes(json.dumps(payload, sort_keys=True).encode())


def _extract(encoder, items, cfg, device):
    globals_, tokens_ = [], []
    encoder.eval()
    checked = False
    with torch.inference_mode():
        for images, _ in study._loader(items, cfg):
            images = images.to(device, non_blocking=True)
            with study._autocast(device):
                output = encoder.backbone.forward_features(images)
                cls, patches = output["x_norm_clstoken"], output["x_norm_patchtokens"]
                side = int(np.sqrt(patches.shape[1]))
                if side * side != patches.shape[1]:
                    raise ValueError("DINOv2-S tokens do not form a square grid")
                local = F.adaptive_avg_pool2d(
                    patches.transpose(1, 2).reshape(len(images), patches.shape[-1], side, side),
                    (4, 4)).flatten(2).transpose(1, 2)
                global_emb = encoder.project(cls)
                local_emb = encoder.project(local)
                if not checked:
                    direct = encoder(images)
                    if not torch.allclose(direct.float(), global_emb.float(), atol=2e-4, rtol=2e-4):
                        raise ValueError("Extracted CLS embedding differs from checkpoint forward")
                    checked = True
            if not torch.isfinite(global_emb).all() or not torch.isfinite(local_emb).all():
                raise RuntimeError("Non-finite base image features")
            globals_.append(global_emb.float().cpu().numpy())
            tokens_.append(local_emb.float().cpu().numpy().astype(np.float16))
    if not globals_:
        raise ValueError("No images to extract")
    return np.concatenate(globals_), np.concatenate(tokens_)


def _feature_cache(out, cache_name, items, encoder, cfg, device, base_hash):
    extractor_hash = study._hash_bytes(inspect.getsource(_extract).encode())
    signatures = [_cache_signature(items, base_hash, extractor_hash)]
    if extractor_hash == _LEGACY_EXTRACTOR_SOURCE_SHA256:
        signatures.append(_cache_signature(items, base_hash, _LEGACY_EXPERIMENT_SHA256))
    for candidate in signatures:
        cached = out / "feature_cache" / f"{cache_name}_{candidate[:16]}.npz"
        if cached.is_file():
            with np.load(cached, allow_pickle=False) as saved:
                if (str(saved["signature"]) != candidate or
                        len(saved["global_emb"]) != len(items) or
                        len(saved["tokens"]) != len(items)):
                    raise ValueError(f"Feature cache identity mismatch: {cached}")
                print(f"[wood-evidence] reusing {cached.name}", flush=True)
                return saved["global_emb"], saved["tokens"]
    signature = signatures[0]
    path = out / "feature_cache" / f"{cache_name}_{signature[:16]}.npz"
    if encoder is None:
        raise ValueError(f"Feature cache absent; encoder is needed: {path}")
    print(f"[wood-evidence] extracting {cache_name}: {len(items)} images", flush=True)
    global_emb, tokens = _extract(encoder, items, cfg, device)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, signature=np.asarray(signature), global_emb=global_emb,
             tokens=tokens)
    temporary.replace(path)
    return global_emb, tokens


def _encoder_from_state(state, cfg, device):
    encoder, _ = study._model(cfg, device, pretrained=True)
    if study._state_sha256(encoder.backbone) != state["initial_backbone_sha256"]:
        raise ValueError("Pretrained DINOv2-S weights changed")
    encoder.load_state_dict(state["encoder"], strict=True)
    encoder.eval()
    return encoder


def _features(out, name, items, encoder, cfg, device, base_hash):
    extraction = dict(cfg)
    extraction["image_batch"] = int(os.environ.get("WOOD_EVIDENCE_IMAGE_BATCH", "64"))
    requested_workers = int(os.environ.get("WOOD_EVIDENCE_WORKERS", "8"))
    if extraction["image_batch"] < 1 or requested_workers < 0:
        raise ValueError("Invalid image extraction batch or workers")
    available_cpus = (len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity")
                      else os.cpu_count() or 1)
    extraction["workers"] = min(requested_workers, available_cpus)
    embeddings, tokens = _feature_cache(out, name, items, encoder, extraction,
                                        device, base_hash)
    if embeddings.shape != (len(items), 512) or tokens.shape != (len(items), 16, 512):
        raise ValueError("Unexpected global/local feature dimensions")
    return embeddings, tokens


def _episode_layout(items):
    by_class = defaultdict(lambda: defaultdict(list))
    for index, (path, label) in enumerate(items):
        by_class[study.canonical(label)][study.source_scan_id(path)].append(index)
    if len(by_class) < 64:
        raise ValueError("Meta-train representative set has fewer than 64 species")
    return by_class


def _episode_indices(by_class, items, rng, n_way, k, hard_neighbors):
    classes = list(by_class)
    anchor = classes[int(rng.integers(len(classes)))]
    hard = [label for label in hard_neighbors[anchor] if label != anchor]
    chosen = [anchor]
    chosen.extend(hard[:min(n_way // 2, n_way - 1)])
    remaining = [label for label in classes if label not in chosen]
    chosen.extend(rng.choice(remaining, size=n_way - len(chosen), replace=False).tolist())
    rng.shuffle(chosen)
    refs, queries, ref_labels, scans = [], [], [], []
    for class_index, label in enumerate(chosen):
        groups = by_class[label]
        group_names = list(groups)
        query_scan = group_names[int(rng.integers(len(group_names)))]
        query_index = int(rng.choice(groups[query_scan]))
        support_scans = [scan for scan in group_names if scan != query_scan]
        rng.shuffle(support_scans)
        selected = [int(rng.choice(groups[scan])) for scan in support_scans[:k]]
        while len(selected) < k:
            available = [index for scan in support_scans for index in groups[scan]
                         if index not in selected]
            if not available:
                raise ValueError("Not enough different-scan references for episode")
            selected.append(int(rng.choice(available)))
        refs.extend(selected)
        ref_labels.extend([class_index] * k)
        scans.extend(study.source_scan_id(items[index][0]) for index in selected)
        queries.append(query_index)
    return refs, queries, ref_labels, scans


def _hard_neighbors(items, embeddings, by_class):
    classes = list(by_class)
    prototypes = np.stack([embeddings[[idx for rows in by_class[label].values()
                                       for idx in rows]].mean(0) for label in classes])
    prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True).clip(1e-8)
    similarities = prototypes @ prototypes.T
    result = {}
    for index, label in enumerate(classes):
        genus = label.split("_", 1)[0]
        same_genus = [other for other in classes if other != label and
                      other.split("_", 1)[0] == genus]
        nearest = [classes[i] for i in np.argsort(-similarities[index])
                   if classes[i] != label]
        result[label] = list(dict.fromkeys(same_genus + nearest))
    return result


def _metric_rows(scores, classes, queries, name, fold, mode, global_scores,
                 reference_counts=None, scan_counts=None):
    order = np.argsort(-scores, axis=1, kind="stable")
    global_order = np.argsort(-global_scores, axis=1, kind="stable")
    class_index = {label: index for index, label in enumerate(classes)}
    rows = []
    for index, (path, raw_label) in enumerate(queries):
        label = study.canonical(raw_label)
        true = class_index[label]
        rank = int(np.flatnonzero(order[index] == true)[0]) + 1
        global_rank = int(np.flatnonzero(global_order[index] == true)[0]) + 1
        predicted = str(classes[order[index, 0]])
        rows.append({"query_path": path, "true_label": label, "gallery": name,
                     "query_scan": study.source_scan_id(path),
                     "query_scale": study.image_scale(path),
                     "true_reference_images": (reference_counts or {}).get(label),
                     "true_reference_scans": (scan_counts or {}).get(label),
                     "fold": fold, "mode": mode, "predicted_label": predicted,
                     "correct": int(rank == 1),
                     "same_genus_error": int(predicted != label and
                                             predicted.split("_", 1)[0] ==
                                             label.split("_", 1)[0]),
                     "true_rank": rank, "global_true_rank": global_rank,
                     "true_score": float(scores[index, true]),
                     "top_score": float(scores[index, order[index, 0]]),
                     "top128_oracle": int(global_rank <= 128)})
    return rows


def _evaluate_split(manifest, split, out, encoder, cfg, device, base_hash,
                    ranker, modes, folds=(0, 1)):
    rows = []
    fixed_ranker = WoodEvidenceReranker().to(device).eval()
    for fold in folds:
        refs, queries, extras = _validation_sets(manifest, split, fold)
        all_refs = refs + extras
        plans = _gallery_plan(refs, queries, extras, split)
        ref_global, ref_tokens = _features(out, f"{split}_fold{fold}_refs", all_refs,
                                            encoder, cfg, device, base_hash)
        query_global, query_tokens = _features(out, f"{split}_fold{fold}_queries", queries,
                                                encoder, cfg, device, base_hash)
        for gallery, (ref_index, query_index) in plans.items():
            if gallery not in {"57x5", "57x1", "637x1", f"{len(refs) // 5}x5",
                               f"{len(refs) // 5}x1"}:
                continue
            selected_refs = [all_refs[index] for index in ref_index]
            selected_queries = [queries[index] for index in query_index]
            labels = [study.canonical(label) for _, label in selected_refs]
            classes = sorted(set(labels))
            index_map = {label: index for index, label in enumerate(classes)}
            label_tensor = torch.tensor([index_map[label] for label in labels], device=device)
            scans = [study.source_scan_id(path) for path, _ in selected_refs]
            scans_by_class = defaultdict(set)
            counts_by_class = defaultdict(int)
            for label, scan in zip(labels, scans):
                scans_by_class[label].add(scan)
                counts_by_class[label] += 1
            if any(study.source_scan_id(path) in scans_by_class[study.canonical(label)]
                   for path, label in selected_queries):
                raise ValueError("Reference/query source-scan leakage")
            q = torch.from_numpy(query_global[query_index]).to(device)
            qt = torch.from_numpy(query_tokens[query_index]).to(device)
            r = torch.from_numpy(ref_global[ref_index]).to(device)
            rt = torch.from_numpy(ref_tokens[ref_index]).to(device)
            with torch.no_grad():
                global_scores, global_classes = prototype_scores(q, r, label_tensor)
                global_np = global_scores.cpu().numpy()
                for mode in modes:
                    if mode == "global":
                        scores, class_ids = global_scores, global_classes
                    else:
                        active = fixed_ranker if mode == "local_fixed" else ranker
                        scores, class_ids = active(
                            q, qt, r, rt, label_tensor, scans, top_classes=128,
                            consensus=mode != "local_no_consensus",
                            weight_override=0.25 if mode == "local_fixed" else None)
                    ordered_classes = np.asarray([classes[int(i)] for i in class_ids.cpu()])
                    if not torch.equal(class_ids, global_classes):
                        raise RuntimeError("Local and global class orders differ")
                    rows.extend(_metric_rows(scores.cpu().numpy(), ordered_classes,
                                             selected_queries, gallery, fold, mode,
                                             global_np, counts_by_class,
                                             {key: len(value) for key, value in
                                              scans_by_class.items()}))
            recent = pd.DataFrame(rows)
            recent = recent[(recent["gallery"] == gallery) & (recent["fold"] == fold)]
            metrics = recent.groupby("mode")["correct"].mean().to_dict()
            print(f"[wood-evidence] {split} fold={fold} {gallery} "
                  + " ".join(f"{mode}={metrics[mode]:.4f}" for mode in modes),
                  flush=True)
    frame = pd.DataFrame(rows)
    summary_rows = []
    for (gallery, mode), part in frame.groupby(["gallery", "mode"], sort=True):
        by_species = part.groupby("true_label")
        summary_rows.append({
            "gallery": gallery, "mode": mode,
            "macro_r1": float(by_species["correct"].mean().mean()),
            "macro_top128_oracle": float(by_species["top128_oracle"].mean().mean()),
            "n_queries": len(part), "n_species": part["true_label"].nunique()})
    summary = pd.DataFrame(summary_rows)
    return frame, summary


def _save_report(out, stem, frame, summary, metadata):
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / f"{stem}_queries.csv", index=False)
    summary.to_csv(out / f"{stem}_summary.csv", index=False)
    paired = _paired_differences(frame)
    paired.to_csv(out / f"{stem}_paired.csv", index=False)
    _atomic_json(out / f"{stem}_provenance.json", metadata)


def _paired_differences(frame, n_boot=2000):
    keys = ["gallery", "fold", "query_path", "true_label"]
    if frame.duplicated(keys + ["mode"]).any():
        raise ValueError("Repeated query/mode rows in paired comparison")
    global_rows = frame[frame["mode"] == "global"][keys + ["correct"]]
    if global_rows.empty:
        raise ValueError("Global baseline is required for paired comparison")
    results = []
    for mode in sorted(set(frame["mode"]) - {"global"}):
        local_rows = frame[frame["mode"] == mode][keys + ["correct"]]
        paired = global_rows.merge(local_rows, on=keys, suffixes=("_global", "_local"),
                                   validate="one_to_one")
        if len(paired) != len(global_rows) or len(paired) != len(local_rows):
            raise ValueError("Paired query alignment failed")
        paired["delta"] = paired["correct_local"] - paired["correct_global"]
        for gallery, part in paired.groupby("gallery", sort=True):
            species_delta = part.groupby("true_label")["delta"].mean().to_numpy()
            rng = np.random.default_rng(2026)
            draw = rng.integers(0, len(species_delta), size=(n_boot, len(species_delta)))
            boot = species_delta[draw].mean(axis=1)
            results.append({"gallery": gallery, "mode": mode,
                            "n_queries": len(part), "n_species": len(species_delta),
                            "macro_r1_delta": float(species_delta.mean()),
                            "ci95_low": float(np.quantile(boot, 0.025)),
                            "ci95_high": float(np.quantile(boot, 0.975)),
                            "rescued_queries": int((part["delta"] == 1).sum()),
                            "harmed_queries": int((part["delta"] == -1).sum())})
    return pd.DataFrame(results)


def _assert_base_reproduced(frame, state):
    validation = state["validation"]
    expected_folds = validation["fold_metrics"]
    for fold, expected in enumerate(expected_folds):
        available = frame[(frame["fold"] == fold) & (frame["mode"] == "global")]
        if available.empty:
            continue
        for gallery, target in (("57x5", expected["meta_val_r1_all"]),
                                ("637x1", expected["meta_val_gallery_curve"]["637"])):
            part = available[available["gallery"] == gallery]
            actual = float(part.groupby("true_label")["correct"].mean().mean())
            if not np.isclose(actual, target, atol=1e-6):
                raise ValueError(f"Base checkpoint {gallery} fold={fold} R@1 mismatch: "
                                 f"computed {actual:.6f}, stored {target:.6f}")


def _train(root, out, manifest, base_path, state, cfg, device):
    base_hash = study._hash_file(base_path)
    encoder = _encoder_from_state(state, cfg, device)
    items = _representatives(manifest)
    global_np, tokens_np = _features(out, "meta_train_representatives", items,
                                     encoder, cfg, device, base_hash)
    by_class = _episode_layout(items)
    hard = _hard_neighbors(items, global_np, by_class)
    global_t = torch.from_numpy(global_np).to(device)
    tokens_t = torch.from_numpy(tokens_np).to(device)
    seeds = [int(x) for x in os.environ.get("WOOD_EVIDENCE_SEEDS", "43").split(",")]
    epochs = int(os.environ.get("WOOD_EVIDENCE_EPOCHS", "4"))
    episodes = int(os.environ.get("WOOD_EVIDENCE_EPISODES", "150"))
    if not seeds or len(set(seeds)) != len(seeds) or epochs < 1 or episodes < 1:
        raise ValueError("Invalid scorer training schedule")
    for seed in seeds:
        study._seed(seed)
        ranker = WoodEvidenceReranker().to(device)
        optimizer = torch.optim.AdamW(ranker.parameters(), lr=3e-4, weight_decay=1e-3)
        best_score = -float("inf")
        best_path = out / "checkpoints" / f"base_{state['seed']}_scorer_{seed}_best.pt"
        latest_path = out / "checkpoints" / f"base_{state['seed']}_scorer_{seed}_latest.pt"
        best_path.parent.mkdir(parents=True, exist_ok=True)
        provenance = {"base_checkpoint_sha256": base_hash,
                      "base_checkpoint": str(base_path), "base_seed": state["seed"],
                      "manifest_sha256": study._hash_file(root / "swi_manifest.json"),
                      "method_sha256": study._hash_file(Path(__file__).with_name(
                          "wood_evidence_method.py")),
                      "experiment_sha256": study._hash_file(__file__),
                      "scorer_seed": seed, "epochs": epochs, "episodes_per_epoch": episodes,
                      "ways": [16, 32, 64], "reference_counts": [1, 2],
                      "learning_rate": 3e-4, "weight_decay": 1e-3}
        epoch_start = 1
        if latest_path.is_file():
            latest = torch.load(latest_path, map_location="cpu", weights_only=False)
            if any(latest.get(key) != value for key, value in provenance.items()):
                raise ValueError(f"Scorer resume provenance changed: {latest_path}")
            ranker.load_state_dict(latest["ranker"], strict=True)
            optimizer.load_state_dict(latest["optimizer"])
            best_score = float(latest["best_score"])
            epoch_start = int(latest["epoch"]) + 1
            print(f"[wood-evidence] resuming scorer seed={seed} at epoch {epoch_start}",
                  flush=True)
        for epoch in range(epoch_start, epochs + 1):
            ranker.train()
            rng = np.random.default_rng(seed * 100003 + epoch)
            losses = []
            for _ in range(episodes):
                n_way = int(rng.choice([16, 32, 64]))
                k = int(rng.choice([1, 2]))
                ref_idx, query_idx, labels, scans = _episode_indices(
                    by_class, items, rng, n_way, k, hard)
                optimizer.zero_grad(set_to_none=True)
                scores, classes = ranker(
                    global_t[query_idx], tokens_t[query_idx], global_t[ref_idx],
                    tokens_t[ref_idx], torch.tensor(labels, device=device), scans,
                    top_classes=64)
                if not torch.equal(classes, torch.arange(n_way, device=device)):
                    raise RuntimeError("Episode class order changed")
                loss = F.cross_entropy(scores / 0.07,
                                       torch.arange(n_way, device=device))
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite local-evidence training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ranker.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach()))
            ranker.eval()
            val_rows, summary = _evaluate_split(manifest, "meta-val", out, encoder, cfg,
                                                device, base_hash, ranker,
                                                ("global", "local_learned"), folds=(0,))
            _assert_base_reproduced(val_rows, state)
            lookup = summary.set_index(["gallery", "mode"])["macro_r1"]
            score = (float(lookup["57x5", "local_learned"]) +
                     float(lookup["637x1", "local_learned"])) / 2
            baseline = (float(lookup["57x5", "global"]) +
                        float(lookup["637x1", "global"])) / 2
            print(f"[wood-evidence] seed={seed} epoch={epoch}/{epochs} "
                  f"loss={np.mean(losses):.4f} val_score={score:.4f} "
                  f"global={baseline:.4f} local_weight={float(F.softplus(ranker.weight_logit)):.3f}",
                  flush=True)
            payload = {**provenance, "epoch": epoch, "selection_score": score,
                       "global_baseline_score": baseline, "ranker": ranker.state_dict()}
            if score > best_score:
                best_score = score
                study._save_checkpoint(best_path, payload)
            study._save_checkpoint(latest_path, {**payload, "best_score": best_score,
                                                 "optimizer": optimizer.state_dict()})
        if not best_path.is_file():
            raise RuntimeError("Scorer training finished without a selected checkpoint")
        selected = torch.load(best_path, map_location="cpu", weights_only=False)
        ranker.load_state_dict(selected["ranker"], strict=True)
        ranker.eval()
        frame, summary = _evaluate_split(manifest, "meta-val", out, encoder, cfg,
                                         device, base_hash, ranker,
                                         ("global", "local_fixed", "local_no_consensus",
                                          "local_learned"))
        _assert_base_reproduced(frame, state)
        _save_report(out, f"meta_val_scorer_{seed}", frame, summary,
                     {key: value for key, value in selected.items() if key != "ranker"})
    print(f"[wood-evidence] scorer study complete: {out}", flush=True)


def run():
    root = Path(os.environ.get("ROOT_PATH", "/content/drive/MyDrive/NCS")).resolve()
    study_out = Path(os.environ.get(
        "WOOD_EVIDENCE_STUDY_OUT", root / "results" / "dinov2s_retrieval_matched_v1")).resolve()
    out = Path(os.environ.get(
        "WOOD_EVIDENCE_OUT", root / "results" / "wood_evidence_pilot_v1")).resolve()
    mode = os.environ.get("WOOD_EVIDENCE_MODE", "preflight").lower()
    if mode not in {"preflight", "probe", "train", "final"}:
        raise ValueError("WOOD_EVIDENCE_MODE must be preflight, probe, train, or final")
    seed = int(os.environ.get("WOOD_EVIDENCE_BASE_SEED", "43"))
    base_path, state, cfg = _base_checkpoint(root, study_out, seed)
    device = torch.device("cuda" if os.environ.get("DEVICE", "cuda") == "cuda" and
                          torch.cuda.is_available() else "cpu")
    manifest = data.load_swi_manifest(root / "swi_manifest.json")
    representatives = _representatives(manifest)
    audit = {"mode": mode, "device": str(device), "base_checkpoint": str(base_path),
             "base_checkpoint_sha256": study._hash_file(base_path),
             "manifest_sha256": study._hash_file(root / "swi_manifest.json"),
             "meta_train_representative_images": len(representatives),
             "meta_train_representative_species": len(set(label for _, label in representatives)),
             "meta_test_accessed": mode == "final"}
    print(f"[wood-evidence] {mode} base_seed={seed} device={device} out={out}", flush=True)
    if mode == "preflight":
        for fold in range(2):
            refs, queries, extras = _validation_sets(manifest, "meta-val", fold)
            audit[f"val_fold_{fold}"] = {"references": len(refs), "queries": len(queries),
                                         "extra_species": len(extras)}
        out.mkdir(parents=True, exist_ok=True)
        _atomic_json(out / "preflight.json", audit)
        return
    if device.type != "cuda":
        raise RuntimeError("Probe/train/final require CUDA for DINOv2-S token extraction")
    if mode == "final" and os.environ.get("WOOD_EVIDENCE_APPROVE_FINAL", "0") != "1":
        raise ValueError("Set WOOD_EVIDENCE_APPROVE_FINAL=1 only after locking validation selection")
    if os.environ.get("WOOD_EVIDENCE_PRELOAD", "1") == "1":
        paths = []
        if mode == "train":
            paths.extend(path for path, _ in representatives)
        split = "meta-test" if mode == "final" else "meta-val"
        for fold in range(2):
            refs, queries, extras = _validation_sets(manifest, split, fold)
            paths.extend(path for path, _ in refs + queries + extras)
        stats = data.preload_image_cache(
            sorted(set(paths)),
            max_workers=int(os.environ.get("WOOD_EVIDENCE_PRELOAD_WORKERS", "16")),
            desc=f"Wood evidence {mode} images")
        if stats["bad"]:
            raise RuntimeError(f"{stats['bad']} images failed to preload")
    encoder = _encoder_from_state(state, cfg, device)
    if mode == "train":
        _train(root, out, manifest, base_path, state, cfg, device)
        return
    ranker = WoodEvidenceReranker().to(device)
    ranker.eval()
    if mode == "final":
        scorer_seed = int(os.environ.get("WOOD_EVIDENCE_SELECTED_SCORER_SEED", "43"))
        checkpoint = out / "checkpoints" / f"base_{seed}_scorer_{scorer_seed}_best.pt"
        selected = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if (selected["base_checkpoint_sha256"] != audit["base_checkpoint_sha256"] or
                selected["manifest_sha256"] != audit["manifest_sha256"] or
                selected["method_sha256"] != study._hash_file(Path(__file__).with_name(
                    "wood_evidence_method.py")) or
                selected["experiment_sha256"] != study._hash_file(__file__)):
            raise ValueError("Selected scorer provenance changed")
        ranker.load_state_dict(selected["ranker"], strict=True)
        modes = ("global", "local_fixed", "local_no_consensus", "local_learned")
        split, stem = "meta-test", "locked_meta_test"
        audit["selected_scorer_checkpoint"] = str(checkpoint)
        audit["selected_scorer_sha256"] = study._hash_file(checkpoint)
    else:
        modes = ("global", "local_fixed", "local_no_consensus")
        split, stem = "meta-val", "inference_probe"
    frame, summary = _evaluate_split(manifest, split, out, encoder, cfg, device,
                                     audit["base_checkpoint_sha256"], ranker, modes)
    if mode == "probe":
        _assert_base_reproduced(frame, state)
    _save_report(out, stem, frame, summary, audit)
    print(f"[wood-evidence] {mode} complete: {out / (stem + '_summary.csv')}", flush=True)
