"""Fractional-epoch meta-validation of full-image ArcFace and SupCon tuning."""

import json
import math
import os
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import data, gallery_experiment as study
from . import wood_evidence_experiment as evidence
from . import wood_full_image_train as full
from .gallery_method import ArcFaceClassifier
from .wood_correspondence_method import WoodCorrespondence


ARMS = ("arcface", "arcface_supcon")
SCHEDULES = ("constant", "cosine")
FRACTIONS = (0.25, 0.5, 1.0)
GALLERIES = ("24x5", "57x5", "57x1", "637x1")


def milestone_steps(total):
    if total < 4:
        raise ValueError("Fractional-epoch study needs at least four training batches")
    return {max(1, math.ceil(total * fraction)): fraction for fraction in FRACTIONS}


def lr_multiplier(schedule, step, total):
    if schedule == "constant":
        return 1.0
    if schedule != "cosine" or not 1 <= step <= total:
        raise ValueError("Invalid learning-rate schedule or step")
    progress = (step - 1) / max(1, total - 1)
    return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2


def selection_score(scores):
    return float(np.mean([scores[name] for name in ("57x5", "57x1", "637x1")]))


def _config():
    base = full._config()
    cfg = {
        "arms": [x.strip() for x in os.environ.get(
            "WOOD_EARLY_ARMS", ",".join(ARMS)).split(",")],
        "schedules": [x.strip() for x in os.environ.get(
            "WOOD_EARLY_SCHEDULES", ",".join(SCHEDULES)).split(",")],
        "seed": int(os.environ.get("WOOD_EARLY_SEED", "43")),
        "base": base,
    }
    if (not cfg["arms"] or not cfg["schedules"] or
            len(set(cfg["arms"])) != len(cfg["arms"]) or
            len(set(cfg["schedules"])) != len(cfg["schedules"]) or
            any(arm not in ARMS for arm in cfg["arms"]) or
            any(schedule not in SCHEDULES for schedule in cfg["schedules"])):
        raise ValueError("Invalid early-stop arm or schedule")
    return cfg


def _train_one(out, manifest, items, base_state, base_cfg, source_hash,
               provenance, cfg, arm, schedule, device, source_score):
    folder = out / arm / schedule / f"seed_{cfg['seed']}"
    completed = folder / "complete.json"
    if completed.is_file():
        report = json.loads(completed.read_text())
        if report["provenance"] != provenance:
            raise ValueError(f"Completed study provenance mismatch: {completed}")
        checkpoint = (folder / "best.pt" if report["best_fraction"] else None)
        if checkpoint is not None and study._hash_file(checkpoint) != report["best_sha256"]:
            raise ValueError(f"Completed checkpoint changed: {checkpoint}")
        print(f"[wood-early] verified {arm}/{schedule}; skipping", flush=True)
        return report

    base = cfg["base"]
    seed = cfg["seed"]
    study._seed(seed)
    encoder = evidence._encoder_from_state(base_state, base_cfg, device)
    model = WoodCorrespondence().to(device)
    classifier = ArcFaceClassifier(
        512, len({study.canonical(label) for _, label in items})).to(device)
    optimizer = torch.optim.AdamW([
        {"params": encoder.backbone.parameters(), "lr": base["backbone_lr"]},
        {"params": encoder.projection.parameters(), "lr": base["head_lr"]},
        {"params": model.parameters(), "lr": base["head_lr"]},
        {"params": classifier.parameters(), "lr": base["head_lr"]},
    ], weight_decay=0.01)
    initial_lrs = [group["lr"] for group in optimizer.param_groups]
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    refs, pair_audit = full.reference_plan(items, seed, 1)
    dataset = full.PairDataset(items, refs)
    batches = full.class_balanced_batches(items, base["batch"], seed * 100003 + 1)
    milestones = milestone_steps(len(batches))
    representative = {}
    for index, (_, label) in enumerate(items):
        representative.setdefault(study.canonical(label), index)
    loader = DataLoader(dataset, batch_sampler=batches, num_workers=base["workers"],
                        pin_memory=device.type == "cuda",
                        **({"prefetch_factor": 2} if base["workers"] > 0 else {}))
    seen = np.zeros(len(items), dtype=np.uint8)
    losses, records = [], []
    best_score, best_fraction = source_score, None
    started = time.perf_counter()
    encoder.train()
    model.train()
    classifier.train()
    for step, batch in enumerate(loader, 1):
        indices = batch[3].numpy()
        if seen[indices].any():
            raise RuntimeError("Duplicate meta-train query")
        seen[indices] = 1
        factor = lr_multiplier(schedule, step, len(batches))
        for group, initial in zip(optimizer.param_groups, initial_lrs):
            group["lr"] = initial * factor
        extra = None
        if len(batch[2].unique()) == 1:
            current = study.canonical(items[int(indices[0])][1])
            other = next(label for label in representative if label != current)
            extra_index = representative[other]
            extra_image = dataset.transform(
                image=dataset.loader.load(items[extra_index][0]))["image"]
            extra = (extra_image, dataset.lookup[other])
        loss, _ = full._step(encoder, model, classifier, batch, arm, base,
                             optimizer, device, extra, scaler)
        losses.append(loss)
        if step % base["print_every"] == 0:
            print(f"[wood-early] {arm}/{schedule} step={step}/{len(batches)} "
                  f"seen={int(seen.sum())} loss={np.mean(losses):.4f} "
                  f"lr_factor={factor:.3f} elapsed_s={time.perf_counter()-started:.1f}",
                  flush=True)
        if step not in milestones:
            continue
        encoder.eval()
        model.eval()
        scores = full._validate(manifest, encoder, model, base_cfg, arm, device,
                                galleries=GALLERIES)
        score = selection_score(scores)
        fraction = milestones[step]
        record = {"arm": arm, "schedule": schedule, "seed": seed,
                  "fraction": fraction, "step": step, "total_steps": len(batches),
                  "unique_queries_seen": int(seen.sum()), "total_queries": len(items),
                  "train_loss_to_step": float(np.mean(losses)),
                  "lr_multiplier": factor, "meta_val": scores,
                  "selection_score": score, "pair_audit": pair_audit,
                  "elapsed_s": time.perf_counter() - started}
        records.append(record)
        study._json(folder / f"fraction_{int(fraction * 100):03d}.json", record)
        if score > best_score + 1e-12:
            best_score, best_fraction = score, fraction
            study._save_checkpoint(folder / "best.pt", {
                "encoder": encoder.state_dict(), "model": model.state_dict(),
                "arm": arm, "schedule": schedule, "seed": seed,
                "fraction": fraction, "provenance": provenance,
            })
        print(f"[wood-early] {arm}/{schedule} fraction={fraction:.2f} "
              f"meta_val={scores} score={score:.4f} best={best_score:.4f}", flush=True)
        encoder.train()
        model.train()
    if not seen.all():
        raise RuntimeError("Incomplete full-image query coverage")
    report = {"arm": arm, "schedule": schedule, "seed": seed,
              "source_sha256": source_hash, "source_score": source_score,
              "best_fraction": best_fraction, "best_score": best_score,
              "best_sha256": (study._hash_file(folder / "best.pt")
                              if best_fraction else source_hash),
              "records": records, "provenance": provenance}
    study._json(completed, report)
    return report


def _export(out):
    files = sorted(path for path in out.rglob("*.json") if path.is_file())
    if not (out / "summary.json").is_file():
        raise ValueError("Run all early-stop variants before export")
    archive = out / "wood_full_image_early_stop_review.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as stream:
        for path in files:
            stream.write(path, path.relative_to(out))
    print(f"[wood-early] review archive: {archive}", flush=True)
    return archive


def run():
    cfg = _config()
    root = Path(os.environ.get("ROOT_PATH", "/content/drive/MyDrive/NCS")).resolve()
    out = Path(os.environ.get("WOOD_EARLY_OUT", root / "results/wood_full_image_early_stop_v1"))
    mode = os.environ.get("WOOD_EARLY_MODE", "train")
    if mode == "export":
        return _export(out)
    if mode not in {"preflight", "train"}:
        raise ValueError("WOOD_EARLY_MODE must be preflight, train, or export")
    manifest = data.load_swi_manifest(root / "swi_manifest.json")
    items = manifest["meta-train"]
    source_root = Path(os.environ.get(
        "WOOD_FULL_SOURCE_OUT", root / "results/dinov2s_retrieval_matched_v1"))
    source_path, source_state, source_cfg = evidence._base_checkpoint(
        root, source_root, cfg["base"]["source_seed"])
    if mode == "preflight":
        batches = full.class_balanced_batches(
            items, cfg["base"]["batch"], cfg["seed"] * 100003 + 1)
        print(f"[wood-early] {len(items)} queries, {len(batches)} steps, "
              f"milestones={milestone_steps(len(batches))}; "
              f"arms={cfg['arms']} schedules={cfg['schedules']}", flush=True)
        return
    if os.environ.get("WOOD_FULL_PRELOAD", "1") == "1":
        data.preload_image_cache(
            (path for path, _ in items),
            max_workers=int(os.environ.get("WOOD_FULL_PRELOAD_WORKERS", "8")),
            desc="Full-image early-stop preload")
    provenance = {
        "manifest_sha256": study._hash_file(root / "swi_manifest.json"),
        "source_sha256": study._hash_file(source_path),
        "runner_sha256": study._hash_file(__file__),
        "full_runner_sha256": study._hash_file(Path(full.__file__)),
        "arms": cfg["arms"], "schedules": cfg["schedules"],
        "seed": cfg["seed"], "base_recipe": {
            key: value for key, value in cfg["base"].items()
            if key not in {"arms", "seeds", "epochs", "print_every"}},
        "fractions": list(FRACTIONS),
    }
    out.mkdir(parents=True, exist_ok=True)
    provenance_path = out / "provenance.json"
    if provenance_path.is_file():
        if json.loads(provenance_path.read_text()) != provenance:
            raise ValueError("Early-stop provenance changed; use a new output folder")
    else:
        study._json(provenance_path, provenance)
    source_report = out / "source.json"
    if source_report.is_file():
        source = json.loads(source_report.read_text())
        if source["source_sha256"] != provenance["source_sha256"]:
            raise ValueError("Source baseline checkpoint changed")
    else:
        device = torch.device(os.environ.get("DEVICE", "cuda"))
        encoder = evidence._encoder_from_state(source_state, source_cfg, device).eval()
        model = WoodCorrespondence().to(device).eval()
        scores = full._validate(manifest, encoder, model, source_cfg, "arcface",
                                device, galleries=GALLERIES)
        source = {"source_sha256": provenance["source_sha256"],
                  "meta_val": scores, "selection_score": selection_score(scores)}
        study._json(source_report, source)
        del encoder, model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    device = torch.device(os.environ.get("DEVICE", "cuda"))
    reports = []
    for arm in cfg["arms"]:
        for schedule in cfg["schedules"]:
            reports.append(_train_one(
                out, manifest, items, source_state, source_cfg,
                provenance["source_sha256"], provenance, cfg, arm, schedule,
                device, source["selection_score"]))
    study._json(out / "summary.json", {
        "source": source, "variants": [{
            "arm": report["arm"], "schedule": report["schedule"],
            "best_fraction": report["best_fraction"],
            "best_score": report["best_score"],
            "scores_by_fraction": [(row["fraction"], row["selection_score"])
                                   for row in report["records"]],
        } for report in reports], "provenance": provenance})
    print(f"[wood-early] complete: {out / 'summary.json'}", flush=True)


if __name__ == "__main__":
    run()
