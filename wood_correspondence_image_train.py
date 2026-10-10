"""Image-based, scan-disjoint joint training for wood correspondence.

This is separate from the representative-feature pilot. The encoder and
correspondence scorer receive gradients from meta-train images; meta-test is
never read by this runner. Checkpoint selection uses meta-val fold 0 only.
"""

import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import data, gallery_experiment as study
from . import wood_correspondence_experiment as corr
from . import wood_evidence_experiment as evidence
from .wood_correspondence_method import WoodCorrespondence


SCHEDULE = ((16, 1), (32, 1), (64, 1), (16, 2), (32, 2),
            (64, 2), (16, 5), (32, 5))
VARIANTS = ("qkv", "global_only", "maxsim", "no_gate", "no_consensus")


def _config():
    variant = os.environ.get("WOOD_IMAGE_VARIANT", "qkv")
    if variant not in VARIANTS:
        raise ValueError("Unsupported WOOD_IMAGE_VARIANT")
    config = corr.recipe(variant, "joint")
    config.update({
        "stage": "joint", "variant": variant,
        "epochs": int(os.environ.get("WOOD_IMAGE_EPOCHS", "5")),
        "episodes": int(os.environ.get("WOOD_IMAGE_EPISODES", "100")),
        "image_microbatch": int(os.environ.get("WOOD_IMAGE_MICROBATCH", "4")),
        "workers": int(os.environ.get("WOOD_IMAGE_WORKERS", "2")),
        "head_lr": float(os.environ.get("WOOD_IMAGE_HEAD_LR", "1e-4")),
        "backbone_lr": float(os.environ.get("WOOD_IMAGE_BACKBONE_LR", "2e-6")),
        "train_all_blocks": os.environ.get("WOOD_IMAGE_ALL_BLOCKS", "1") == "1",
        "last_blocks": int(os.environ.get("WOOD_IMAGE_LAST_BLOCKS", "2")),
        "schedule": [list(pair) for pair in SCHEDULE],
        "print_every": int(os.environ.get("WOOD_IMAGE_PRINT_EVERY", "10")),
    })
    if (config["epochs"] < 1 or config["episodes"] < 1 or
            config["image_microbatch"] < 1 or config["workers"] < 0 or
            config["head_lr"] <= 0 or config["backbone_lr"] <= 0 or
            config["print_every"] < 1 or config["last_blocks"] < 1):
        raise ValueError("Invalid image-training configuration")
    return config


def _layout(items):
    by_class = defaultdict(lambda: defaultdict(list))
    for index, (path, label) in enumerate(items):
        by_class[study.canonical(label)][study.source_scan_id(path)].append(index)
    valid = {}
    for n_way, k in SCHEDULE:
        if k in valid:
            continue
        valid[k] = {}
        for label, scans in by_class.items():
            query_scans = [scan for scan in scans
                           if sum(len(rows) for other, rows in scans.items()
                                  if other != scan) >= k]
            if query_scans:
                valid[k][label] = query_scans
    for n_way, k in SCHEDULE:
        if len(valid[k]) < n_way:
            raise ValueError(f"Only {len(valid[k])} scan-disjoint species for {n_way}-way K={k}")
    return by_class, valid


def _pool_audit(items, by_class, valid):
    result = {"meta_train_images": len(items), "meta_train_species": len(by_class),
              "episodes_use_original_images": True,
              "representative_cache_used_for_training": False,
              "query_support_must_have_different_source_scans": True,
              "meta_test_evaluated": False, "by_k": {}}
    for k, eligible in valid.items():
        indices = {index for label in eligible for rows in by_class[label].values()
                   for index in rows}
        query_indices = {index for label, allowed in eligible.items()
                         for scan in allowed for index in by_class[label][scan]}
        result["by_k"][str(k)] = {"eligible_species": len(eligible),
                                    "eligible_images": len(indices),
                                    "query_eligible_images": len(query_indices),
                                    "excluded_single_or_insufficient_scan_species":
                                    len(by_class) - len(eligible)}
    return result


def _episode(items, by_class, valid, rng, n_way, k, query_decks):
    classes = list(valid[k])
    anchor = classes[int(rng.integers(len(classes)))]
    genus = anchor.split("_", 1)[0]
    same_genus = [label for label in classes if label != anchor and
                  label.split("_", 1)[0] == genus]
    rng.shuffle(same_genus)
    chosen = [anchor] + same_genus[:min(len(same_genus), n_way // 4)]
    remaining = [label for label in classes if label not in chosen]
    chosen += rng.choice(remaining, size=n_way - len(chosen), replace=False).tolist()
    rng.shuffle(chosen)
    refs, queries, labels, scans = [], [], [], []
    for class_id, label in enumerate(chosen):
        groups = by_class[label]
        scan = valid[k][label][int(rng.integers(len(valid[k][label])))]
        key = (label, scan)
        if not query_decks[key]:
            query_decks[key] = rng.permutation(groups[scan]).tolist()
        query = int(query_decks[key].pop())
        support_scans = [other for other in groups if other != scan]
        rng.shuffle(support_scans)
        selected = [int(rng.choice(groups[other])) for other in support_scans[:k]]
        if len(selected) < k:
            remaining_refs = [i for other in support_scans for i in groups[other]
                              if i not in selected]
            selected += rng.choice(remaining_refs, size=k - len(selected),
                                   replace=False).tolist()
        if len(selected) != k or query in selected or any(
                study.source_scan_id(items[i][0]) == scan for i in selected):
            raise RuntimeError("Scan-disjoint episode construction failed")
        refs.extend(selected)
        queries.append(query)
        labels.extend([class_id] * k)
        scans.extend(study.source_scan_id(items[i][0]) for i in selected)
    return refs, queries, labels, scans


def _epoch_plan(items, by_class, valid, config, seed, epoch):
    rng = np.random.default_rng(seed * 100003 + epoch)
    query_decks = defaultdict(list)
    layouts, batches = [], []
    seen, queried, species, scans = set(), set(), set(), set()
    for step in range(config["episodes"]):
        n_way, k = SCHEDULE[step % len(SCHEDULE)]
        refs, queries, labels, ref_scans = _episode(
            items, by_class, valid, rng, n_way, k, query_decks)
        batch = refs + queries
        batches.append(batch)
        layouts.append((len(refs), labels, ref_scans))
        seen.update(batch)
        queried.update(queries)
        species.update(study.canonical(items[i][1]) for i in batch)
        scans.update(study.source_scan_id(items[i][0]) for i in batch)
    coverage = {"image_draws": sum(map(len, batches)),
                "unique_images": len(seen), "unique_query_images": len(queried),
                "unique_species": len(species), "unique_source_scans": len(scans),
                "pool_images": len(items), "image_fraction": len(seen) / len(items)}
    return layouts, batches, coverage


def _provenance(root, base_path, config, seed):
    return {"base_checkpoint_sha256": study._hash_file(base_path),
            "manifest_sha256": study._hash_file(root / "swi_manifest.json"),
            "runner_sha256": study._hash_file(__file__),
            "correspondence_sha256": study._hash_file(Path(corr.__file__)),
            "method_sha256": study._hash_file(Path(__file__).with_name(
                "wood_correspondence_method.py")),
            "config": config, "seed": seed}


def _gradient_norm(parameters):
    return float(sum(float(p.grad.detach().float().norm()) ** 2
                     for p in parameters if p.grad is not None) ** 0.5)


def _train(root, out, manifest, base_path, base_state, base_cfg, device,
           config, seed):
    study._seed(seed)
    items = manifest["meta-train"]
    by_class, valid = _layout(items)
    audit = _pool_audit(items, by_class, valid)
    folder = out / config["variant"] / f"seed_{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    study._json(folder / "pool_audit.json", audit)
    encoder = evidence._encoder_from_state(base_state, base_cfg, device)
    if not config["train_all_blocks"]:
        for parameter in encoder.backbone.parameters():
            parameter.requires_grad_(False)
        for block in encoder.backbone.blocks[-config["last_blocks"]:]:
            for parameter in block.parameters():
                parameter.requires_grad_(True)
    model = WoodCorrespondence().to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.parameters(), "lr": config["head_lr"]},
        {"params": [p for p in encoder.parameters() if p.requires_grad],
         "lr": config["backbone_lr"]}], weight_decay=1e-3)
    provenance = _provenance(root, base_path, config, seed)
    latest, best = folder / "latest.pt", folder / "best.pt"
    baseline_file = folder / "baseline_meta_val_summary.csv"
    baseline_meta = folder / "baseline_meta_val_provenance.json"
    if baseline_file.is_file() != baseline_meta.is_file():
        raise ValueError("Incomplete baseline validation report")
    if baseline_meta.is_file():
        with baseline_meta.open() as stream:
            if json.load(stream) != provenance:
                raise ValueError("Existing baseline report has different provenance")
    if not baseline_file.is_file():
        _, baseline_summary, _ = corr._evaluate(
            root, out, manifest, encoder, base_cfg, study._hash_file(base_path), model,
            config, device, folds=(0,), modes=("global",))
        baseline_summary.to_csv(baseline_file, index=False)
        study._json(baseline_meta, provenance)
        print(f"[wood-image] base encoder meta-val score="
              f"{corr._selected_score(baseline_summary, 'global'):.4f}", flush=True)
    first_epoch, best_score = 1, -float("inf")
    seen_all, queries_all = set(), set()
    if latest.is_file():
        saved = torch.load(latest, map_location="cpu", weights_only=False)
        if saved["provenance"] != provenance:
            raise ValueError("Existing image-training checkpoint has different provenance")
        encoder.load_state_dict(saved["encoder"], strict=True)
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        seen_all = set(saved["seen_indices"])
        queries_all = set(saved["query_indices"])
        first_epoch, best_score = int(saved["epoch"]) + 1, float(saved["best_score"])
        print(f"[wood-image] resuming at epoch {first_epoch}", flush=True)
    dataset = data.ManifestDataset(items, transform=data.get_transforms(224, augment=True))
    probe_parameter = encoder.backbone.blocks[-1].attn.qkv.weight
    for epoch in range(first_epoch, config["epochs"] + 1):
        layouts, batches, coverage = _epoch_plan(items, by_class, valid, config, seed, epoch)
        seen_all.update(i for batch in batches for i in batch)
        queries_all.update(i for batch, layout in zip(batches, layouts)
                           for i in batch[layout[0]:])
        generator = torch.Generator().manual_seed(seed * 100003 + epoch)
        available_cpus = (len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity")
                          else os.cpu_count() or 1)
        loader = DataLoader(dataset, batch_sampler=batches,
                            num_workers=min(config["workers"], available_cpus),
                            pin_memory=True, generator=generator)
        encoder.backbone.eval()
        encoder.projection.train()
        model.train()
        losses, accuracies, backbone_grads, head_grads, attention_grads = [], [], [], [], []
        started = time.perf_counter()
        for step, ((images, _), (ri_count, labels, scans)) in enumerate(
                zip(loader, layouts), start=1):
            before = probe_parameter.detach().clone() if step == 1 else None
            loss, accuracy = corr._joint_step(model, encoder, images, ri_count,
                                              labels, scans, config, optimizer, device)
            backbone_grad = _gradient_norm(p for p in encoder.parameters() if p.requires_grad)
            head_grad = _gradient_norm(model.parameters())
            attention_grad = _gradient_norm((probe_parameter,))
            qkv_grad = _gradient_norm((model.query_key.weight,))
            if (not np.isfinite(backbone_grad) or not np.isfinite(head_grad) or
                    backbone_grad <= 0 or head_grad <= 0 or attention_grad <= 0):
                raise RuntimeError("Encoder or scorer received no finite training gradient")
            if config["score_mode"] == "qkv" and qkv_grad <= 0:
                raise RuntimeError("QKV correspondence weights received no gradient")
            if step == 1 and torch.equal(before, probe_parameter.detach()):
                raise RuntimeError("Last encoder attention block did not update")
            losses.append(loss)
            accuracies.append(accuracy)
            backbone_grads.append(backbone_grad)
            head_grads.append(head_grad)
            attention_grads.append(attention_grad)
            if step % config["print_every"] == 0 or step == len(layouts):
                print(f"[wood-image] {config['variant']} seed={seed} epoch={epoch} "
                      f"step={step}/{len(layouts)} loss={np.mean(losses):.4f} "
                      f"train_R1={np.mean(accuracies):.4f} "
                      f"backbone_grad={backbone_grad:.3g} "
                      f"attention_grad={attention_grad:.3g} "
                      f"head_grad={head_grad:.3g} "
                      f"elapsed_s={time.perf_counter() - started:.1f}", flush=True)
        encoder.eval()
        model.eval()
        candidate = folder / "candidate.pt"
        payload = {"provenance": provenance, "epoch": epoch,
                   "encoder": encoder.state_dict(), "model": model.state_dict()}
        study._save_checkpoint(candidate, payload)
        frame, summary, timing = corr._evaluate(
            root, out, manifest, encoder, base_cfg, study._hash_file(candidate), model,
            config, device, folds=(0,), modes=("global", "trained"))
        corr._save_report(folder, f"epoch_{epoch:02d}_meta_val", frame, summary,
                          timing, {"provenance": provenance, "epoch": epoch,
                                   "checkpoint_sha256": study._hash_file(candidate),
                                   "selection_fold": 0, "meta_test_evaluated": False})
        score = corr._selected_score(summary, "trained")
        global_score = corr._selected_score(summary, "global")
        metrics = {"epoch": epoch, "mean_loss": float(np.mean(losses)),
                   "mean_train_r1": float(np.mean(accuracies)),
                   "mean_backbone_grad": float(np.mean(backbone_grads)),
                   "mean_last_block_attention_grad": float(np.mean(attention_grads)),
                   "mean_scorer_grad": float(np.mean(head_grads)),
                   "val_selection_score": score, "val_global_score": global_score,
                   "epoch_coverage": coverage,
                   "cumulative_unique_images": len(seen_all),
                   "cumulative_unique_queries": len(queries_all),
                   "cumulative_image_fraction": len(seen_all) / len(items),
                   "training_seconds": time.perf_counter() - started,
                   "meta_test_evaluated": False}
        study._json(folder / f"epoch_{epoch:02d}.json", metrics)
        if score > best_score:
            best_score = score
            study._save_checkpoint(best, {**payload, "selection_score": score,
                                          "metrics": metrics})
        study._save_checkpoint(latest, {**payload, "selection_score": score,
                                        "best_score": best_score,
                                        "optimizer": optimizer.state_dict(),
                                        "seen_indices": sorted(seen_all),
                                        "query_indices": sorted(queries_all)})
        print(f"[wood-image] epoch={epoch} selected_val={score:.4f} "
              f"global_val={global_score:.4f} image_coverage="
              f"{len(seen_all)}/{len(items)} best={best_score:.4f}", flush=True)
    return best


def _validate(root, out, manifest, base_path, base_state, base_cfg,
              device, config, seed):
    checkpoint = out / config["variant"] / f"seed_{seed}" / "best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Train a selected image checkpoint first: {checkpoint}")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    provenance = _provenance(root, base_path, config, seed)
    if saved["provenance"] != provenance:
        raise ValueError("Selected checkpoint provenance changed")
    encoder = evidence._encoder_from_state(base_state, base_cfg, device)
    encoder.load_state_dict(saved["encoder"], strict=True)
    encoder.eval()
    model = WoodCorrespondence().to(device)
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    modes = ("global", "fixed_local", "global_adapt", "trained")
    frame, summary, timing = corr._evaluate(
        root, out, manifest, encoder, base_cfg, study._hash_file(checkpoint),
        model, config, device, folds=(0, 1), modes=modes)
    report = checkpoint.parent / "selected_meta_val"
    corr._save_report(report, "selected", frame, summary, timing,
                      {"provenance": provenance,
                       "checkpoint_sha256": study._hash_file(checkpoint),
                       "selected_epoch": saved["epoch"],
                       "selection_fold": 0, "descriptive_fold": 1,
                       "meta_test_evaluated": False})
    print(f"[wood-image] selected meta-val report: {report}", flush=True)
    return report


def run():
    root = Path(os.environ.get("ROOT_PATH", "/content/drive/MyDrive/NCS")).resolve()
    out = Path(os.environ.get("WOOD_IMAGE_OUT", root / "results" /
                              "wood_correspondence_image_train_v1")).resolve()
    mode = os.environ.get("WOOD_IMAGE_MODE", "preflight").lower()
    if mode not in {"preflight", "train", "validate"}:
        raise ValueError("WOOD_IMAGE_MODE must be preflight, train or validate")
    seed = int(os.environ.get("WOOD_IMAGE_SEED", "43"))
    base_seed = int(os.environ.get("WOOD_IMAGE_BASE_SEED", "43"))
    study_out = Path(os.environ.get("WOOD_IMAGE_BASE_OUT", root / "results" /
                                  "dinov2s_retrieval_matched_v1")).resolve()
    config = _config()
    base_path, base_state, base_cfg = evidence._base_checkpoint(root, study_out, base_seed)
    manifest = data.load_swi_manifest(root / "swi_manifest.json")
    items = manifest["meta-train"]
    by_class, valid = _layout(items)
    audit = _pool_audit(items, by_class, valid)
    audit.update({"base_checkpoint_sha256": study._hash_file(base_path),
                  "manifest_sha256": study._hash_file(root / "swi_manifest.json"),
                  "config": config, "seed": seed})
    study._json(out / "preflight.json", audit)
    print(f"[wood-image] {mode}: {len(items)} meta-train images, "
          f"{len(by_class)} species; K=5 eligible: "
          f"{audit['by_k']['5']['eligible_images']} images / "
          f"{audit['by_k']['5']['eligible_species']} species", flush=True)
    if mode == "preflight":
        return audit
    if not torch.cuda.is_available():
        raise RuntimeError("Image-based correspondence training requires CUDA")
    if mode == "train" and os.environ.get("WOOD_IMAGE_PRELOAD", "0") == "1":
        eligible = {index for label in valid[1] for rows in by_class[label].values()
                    for index in rows}
        refs, queries, extras = evidence._validation_sets(manifest, "meta-val", 0)
        paths = [items[index][0] for index in sorted(eligible)]
        paths.extend(path for path, _ in refs + queries + extras)
        stats = data.preload_image_cache(
            sorted(set(paths)), max_workers=int(os.environ.get(
                "WOOD_IMAGE_PRELOAD_WORKERS", "16")),
            desc="Wood correspondence training images")
        if stats["bad"]:
            raise RuntimeError("Some training images could not be preloaded")
    device = torch.device("cuda")
    if mode == "validate":
        return _validate(root, out, manifest, base_path, base_state, base_cfg,
                         device, config, seed)
    return _train(root, out, manifest, base_path, base_state, base_cfg,
                  device, config, seed)
