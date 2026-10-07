"""Recover public-query row identity from images without retraining backbones.

The corrected v4/v5 caches assumed that a reconstructed expanded CSV had the
same image order as the original v3 CSV. That assumption is false for OOD. We
identify each current image against its own DINOv2 embedding in v3, then use
the verified one-to-one index map for every inherited public feature array.
CE-Full v5 features are already freshly extracted and remain untouched.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ..audit_support import canonical, sha256, write_json
from ..final_colab_audit import _validate_public_rows


SIDES = ("id", "ood")
DEFAULT_TARGET = "embedding_cache_full954_v6_public_row_verified.npz"


def _public_keys(side, files):
    prefixes = (f"embs_{side}_", f"logits_{side}_", f"labels_{side}_")
    return [key for key in files if key.startswith(prefixes)]


def _is_ce_full(key, side):
    return key.startswith((f"embs_{side}_ce_full_", f"logits_{side}_ce_full"))


def validate_correction_cohort(dfs, overrides, expected_relabelled=2901):
    """Count corrected FSDM41 labels, not all FSDM41 images."""
    count = sum(_validate_public_rows(dfs[side], canonical(dfs[side]["label"]),
                                      overrides, side)
                for side in SIDES)
    if count != expected_relabelled:
        raise ValueError(f"Expected {expected_relabelled} FSDM41 relabelled images, found {count}")
    return count


def resolve_feature_equivalent_duplicates(v3, df, side, indices, scores, runner_up,
                                          min_margin=1e-5, max_feature_diff=5e-4):
    """Resolve DINO ties using feature equivalence or a fully anchored ID row order."""
    initial = np.asarray(indices, dtype=np.int64)
    resolved = initial.copy()
    resolved_ties = np.zeros(len(df), dtype=bool)
    ambiguous = np.flatnonzero(np.asarray(scores) - np.asarray(runner_up) < min_margin)
    if not len(ambiguous):
        return resolved, resolved_ties, []
    dino = v3[f"embs_{side}_dinov2"].astype(np.float32)
    dino /= np.maximum(np.linalg.norm(dino, axis=1, keepdims=True), 1e-12)
    old_labels = canonical(v3[f"labels_{side}_dinov2"])
    keys = [key for key in _public_keys(side, v3.files)
            if key.startswith((f"embs_{side}_", f"logits_{side}_"))]
    arrays = {key: v3[key] for key in keys}
    anchors = np.setdiff1d(np.arange(len(df)), ambiguous)
    anchored_id_order = (side == "id" and len(ambiguous) <= max(2, len(df) // 200)
                         and len(anchors) > 0 and np.min(np.asarray(scores)[anchors]) >= 0.999
                         and np.array_equal(initial[anchors], anchors))
    reports = []
    for row in ambiguous:
        if resolved_ties[row]:
            continue
        if float(scores[row]) < 0.99999:
            raise ValueError(f"{side}: ambiguous row {row} lacks an exact DINOv2 match")
        best = int(initial[row])
        nearby = np.flatnonzero(dino @ dino[best] >= 1 - 2e-6)
        candidates = []
        rejected = []
        max_diff = 0.0
        for candidate in nearby:
            if old_labels[candidate] != old_labels[best]:
                continue
            differences = {key: float(np.max(np.abs(arr[candidate].astype(np.float64) -
                                                    arr[best].astype(np.float64))))
                           for key, arr in arrays.items()}
            worst_key = max(differences, key=differences.get)
            difference = differences[worst_key]
            if difference <= max_feature_diff:
                candidates.append(int(candidate))
                max_diff = max(max_diff, difference)
            else:
                rejected.append((int(candidate), worst_key, difference))
        current = np.flatnonzero(np.isin(initial, nearby))
        mode = "feature_equivalent"
        if (anchored_id_order and len(nearby) == 2 and len(current) == 2 and
                set(current) == set(nearby) and
                all(old_labels[i] == canonical([df.iloc[i]["label"]])[0] for i in current) and
                all(float(dino[i] @ dino[best]) >= 1 - 2e-6 for i in current)):
            paths = [Path(p) for p in df.iloc[current]["file_path"].astype(str)]
            stems = [p.stem[:-7] if p.stem.endswith(" - Copy") else p.stem for p in paths]
            if (paths[0].parent == paths[1].parent and paths[0].suffix == paths[1].suffix and
                    stems[0] == stems[1] and paths[0].stem != paths[1].stem):
                candidates = current.tolist()
                mode = "anchored_id_position"
                if rejected:
                    max_diff = max(max_diff, max(item[2] for item in rejected))
        if len(candidates) < 2 or len(current) != len(candidates):
            raise ValueError(f"{side}: ambiguous row {row} has {len(candidates)} equivalent v3 rows "
                             f"for {len(current)} current images; cannot assign one-to-one. "
                             f"First DINO-near but feature-different rows: {rejected[:3]}")
        if resolved_ties[current].any():
            raise ValueError(f"{side}: overlapping duplicate groups at rows {current.tolist()}")
        if len(set(canonical(df.iloc[current]["label"]))) != 1 or \
                len(set(df.iloc[current]["source_dataset"].astype(str))) != 1:
            raise ValueError(f"{side}: equivalent feature group crosses current labels or sources: "
                             f"{current.tolist()}")
        if np.min(np.asarray(scores)[current]) < 0.99999:
            raise ValueError(f"{side}: duplicate group contains a weak match")
        image_hashes = [sha256(path) for path in df.iloc[current]["file_path"].astype(str)]
        if mode == "feature_equivalent" and len(set(image_hashes)) != 1:
            raise ValueError(f"{side}: candidate duplicate images have different file bytes: "
                             f"{current.tolist()}")
        if mode == "anchored_id_position" and len(set(image_hashes)) == 1:
            raise ValueError(
                f"{side}: rows {current.tolist()} have identical image bytes but v3 "
                f"features disagree ({rejected[:3]}); source cache cannot be verified "
                "for these images")
        resolved[current] = np.sort(candidates)
        resolved_ties[current] = True
        reports.append({"current_rows": current.tolist(), "v3_rows": sorted(candidates),
                        "resolution": mode,
                        "label": str(df.iloc[current[0]]["label"]),
                        "source": str(df.iloc[current[0]]["source_dataset"]),
                        "image_sha256": image_hashes,
                        "max_abs_feature_difference": max_diff,
                        "feature_different_candidates": rejected[:3]})
    if not resolved_ties[ambiguous].all():
        raise ValueError(f"{side}: some ambiguous DINOv2 matches remain unresolved")
    return resolved, resolved_ties, reports


def validate_index_map(df, old_labels, indices, scores, runner_up, side,
                       min_cosine=0.999, min_margin=1e-5, resolved_duplicates=None):
    """Refuse incomplete, ambiguous, reused, or cross-label v3 image matches."""
    n = len(df)
    indices = np.asarray(indices, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float32)
    runner_up = np.asarray(runner_up, dtype=np.float32)
    if indices.shape != (n,) or scores.shape != (n,) or runner_up.shape != (n,):
        raise ValueError(f"{side}: incomplete image-to-v3 mapping")
    if not np.isfinite(scores).all() or not np.isfinite(runner_up).all():
        raise ValueError(f"{side}: non-finite DINOv2 match scores")
    if (indices < 0).any() or (indices >= len(old_labels)).any():
        raise ValueError(f"{side}: v3 row index out of range")
    weak = np.flatnonzero(scores < min_cosine)
    resolved_duplicates = (np.zeros(n, dtype=bool) if resolved_duplicates is None
                           else np.asarray(resolved_duplicates, dtype=bool))
    if resolved_duplicates.shape != (n,):
        raise ValueError(f"{side}: duplicate resolution mask has wrong length")
    ambiguous = np.flatnonzero((scores - runner_up < min_margin) & ~resolved_duplicates)
    if len(weak) or len(ambiguous):
        raise ValueError(
            f"{side}: {len(weak)} weak or {len(ambiguous)} ambiguous DINOv2 matches; "
            f"first weak rows={weak[:5].tolist()}, first ambiguous rows={ambiguous[:5].tolist()}")
    if len(np.unique(indices)) != n:
        values, counts = np.unique(indices, return_counts=True)
        raise ValueError(f"{side}: v3 rows reused: {values[counts > 1][:5].tolist()}")
    expected = canonical(df["label"])
    matched = canonical(np.asarray(old_labels)[indices])
    fsdm = df["source_dataset"].astype(str).str.upper().eq("FSDM41")
    fsdm |= df["file_path"].astype(str).str.contains("/FSDM41/", case=False, regex=False)
    fsdm = fsdm.to_numpy()
    bad = np.flatnonzero((expected != matched) & ~fsdm)
    if len(bad):
        examples = [(int(i), str(expected[i]), str(matched[i])) for i in bad[:5]]
        raise ValueError(f"{side}: {len(bad)} non-FSDM41 labels disagree with matched v3 rows: {examples}")
    return {"rows": n, "matched_v3_rows": int(len(np.unique(indices))),
            "min_cosine": float(scores.min()),
            "min_margin": float((scores - runner_up).min()),
            "resolved_ambiguous_rows": int(resolved_duplicates.sum()),
            "fsdm41_rows": int(fsdm.sum()),
            "fsdm41_old_label_differences": int(((expected != matched) & fsdm).sum())}


def rebuild_arrays(v3, v5, dfs, mappings):
    """Copy SWI/CE from v5; align all inherited public arrays to verified paths."""
    out = {key: v5[key] for key in v5.files if not any(
        key.startswith((f"embs_{side}_", f"logits_{side}_", f"labels_{side}_"))
        for side in SIDES)}
    for side in SIDES:
        n = len(dfs[side])
        old_n = len(v3[f"labels_{side}_dinov2"])
        indices = mappings[side]
        for key in _public_keys(side, v5.files):
            if key.startswith(f"labels_{side}_"):
                out[key] = np.asarray(dfs[side]["label"].astype(str), dtype=str)
            elif _is_ce_full(key, side):
                arr = v5[key]
                if len(arr) != n:
                    raise ValueError(f"{key}: v5 has {len(arr)} rows, expected {n}")
                out[key] = arr
            else:
                if key not in v3.files:
                    raise KeyError(f"Cannot repair {key}: absent from original v3 cache")
                arr = v3[key]
                if len(arr) != old_n:
                    raise ValueError(f"{key}: v3 has {len(arr)} rows, expected {old_n}")
                out[key] = arr[indices]
    return out


class _ImageDataset:
    def __init__(self, paths, start, transform):
        self.paths = paths
        self.start = start
        self.transform = transform

    def __len__(self):
        return len(self.paths) - self.start

    def __getitem__(self, index):
        from PIL import Image
        row = self.start + index
        with Image.open(self.paths[row]) as image:
            return self.transform(image.convert("RGB")), row


def _extract_mapping(df, v3_dino, side, partial, signature, model, transform, device):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm

    n = len(df)
    indices = np.empty(n, dtype=np.int64)
    scores = np.empty(n, dtype=np.float32)
    runner_up = np.empty(n, dtype=np.float32)
    start = 0
    if partial.is_file():
        with np.load(partial, allow_pickle=False) as saved:
            if str(saved["signature"].item()) != signature:
                raise ValueError(f"Stale partial mapping: {partial}; use a new output directory")
            start = len(saved["indices"])
            if start > n or len(saved["scores"]) != start or len(saved["runner_up"]) != start:
                raise ValueError(f"Corrupt partial mapping: {partial}")
            indices[:start] = saved["indices"]
            scores[:start] = saved["scores"]
            runner_up[:start] = saved["runner_up"]
        print(f"[repair] Resuming {side} at image {start}/{n}", flush=True)

    old = torch.as_tensor(v3_dino.astype(np.float32), device=device)
    old = F.normalize(old, dim=1)
    paths = df["file_path"].astype(str).tolist()
    batch_size = int(os.environ.get("PUBLIC_REPAIR_BATCH_SIZE", "16"))
    workers = int(os.environ.get("PUBLIC_REPAIR_WORKERS", "4"))
    loader = DataLoader(_ImageDataset(paths, start, transform), batch_size=batch_size,
                        shuffle=False, num_workers=workers, pin_memory=(device == "cuda"),
                        persistent_workers=workers > 0)
    save_every = int(os.environ.get("PUBLIC_REPAIR_SAVE_EVERY", "1024"))
    next_save = start + save_every
    with torch.inference_mode():
        for images, rows in tqdm(loader, desc=f"Recover {side} v3 rows", total=len(loader)):
            image_features = F.normalize(model(images.to(device)).float(), dim=1)
            top = torch.topk(image_features @ old.T, k=2, dim=1)
            positions = rows.numpy()
            indices[positions] = top.indices[:, 0].cpu().numpy()
            scores[positions] = top.values[:, 0].cpu().numpy()
            runner_up[positions] = top.values[:, 1].cpu().numpy()
            done = int(positions[-1]) + 1
            if done >= next_save or done == n:
                temp = partial.with_suffix(".tmp.npz")
                np.savez(temp, indices=indices[:done], scores=scores[:done],
                         runner_up=runner_up[:done], signature=np.asarray(signature))
                temp.replace(partial)
                next_save = done + save_every
    del old
    if device == "cuda":
        torch.cuda.empty_cache()
    return indices, scores, runner_up


def run():
    import torch
    from torchvision import transforms

    root = Path(os.environ["ROOT_PATH"]).resolve()
    source = root / os.environ.get("PUBLIC_REPAIR_SOURCE_CACHE_NAME", "embedding_cache_full954_v3.npz")
    base = root / os.environ.get("PUBLIC_REPAIR_BASE_CACHE_NAME",
                                 "embedding_cache_full954_v5_retrained_ce_corrected_public.npz")
    base_meta_path = base.with_name(base.stem + "_meta.json")
    target = root / os.environ.get("PUBLIC_REPAIR_TARGET_CACHE_NAME", DEFAULT_TARGET)
    audit_dir = Path(os.environ.get("PUBLIC_REPAIR_AUDIT_DIR", root / "results" / "public_row_repair_v1"))
    audit_dir = audit_dir.resolve()
    if target in (source, base):
        raise ValueError("Repair target must not overwrite v3 or v5")
    correction_path = root / "dataset_label_corrections.json"
    for path in (source, base, base_meta_path, correction_path,
                 root / "ID_images_expanded.csv", root / "OOD_images_expanded.csv"):
        if not path.is_file():
            raise FileNotFoundError(path)
    if target.exists():
        raise FileExistsError(f"Verified target already exists: {target}; refusing to overwrite")
    device = os.environ.get("DEVICE", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    audit_dir.mkdir(parents=True, exist_ok=True)
    dfs = {}
    for side in SIDES:
        df = pd.read_csv(root / f"{side.upper()}_images_expanded.csv")
        if df["file_path"].duplicated().any() or not {"label", "source_dataset"}.issubset(df.columns):
            raise ValueError(f"{side} CSV lacks required columns or has duplicate image paths")
        dfs[side] = df
    if len(dfs["id"]) != 6189 or len(dfs["ood"]) != 30868:
        raise ValueError("Unexpected corrected public cohort; refusing v3 row repair")
    if any(dfs[side]["source_dataset"].astype(str).str.upper().eq("WOODAUTH").any()
           for side in SIDES):
        raise ValueError("WoodAuth images remain in corrected public CSVs")
    overrides = json.loads(correction_path.read_text())["FSDM41"]["overrides"]
    relabelled = validate_correction_cohort(dfs, overrides)
    if set(canonical(dfs["id"]["label"])) & set(canonical(dfs["ood"]["label"])):
        raise ValueError("Corrected public ID/OOD species overlap")

    source_hash = sha256(source)
    base_hash = sha256(base)
    signature = json.dumps({
        "v3_sha256": source_hash, "v5_sha256": base_hash,
        "csv_sha256": {side: sha256(root / f"{side.upper()}_images_expanded.csv") for side in SIDES},
        "extractor": "facebookresearch/dinov2:dinov2_vitb14; RGB; Resize518-bicubic; CenterCrop518; ImageNet norm",
    }, sort_keys=True)
    print(f"[repair] v3={source.name} v5={base.name} target={target.name}", flush=True)
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").eval().to(device)
    transform = transforms.Compose([
        transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(518), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    mappings = {}
    summaries = {}
    with np.load(source, allow_pickle=False) as v3, np.load(base, allow_pickle=False) as v5:
        for side in SIDES:
            if v3[f"embs_{side}_dinov2"].shape != (len(v3[f"labels_{side}_dinov2"]), 768):
                raise ValueError(f"{side} v3 DINOv2 embedding/label shape mismatch")
            if not np.array_equal(canonical(dfs[side]["label"]),
                                  canonical(v5[f"labels_{side}_dinov2"])):
                raise ValueError(f"{side} corrected CSV labels do not match v5")
            indices, scores, runner_up = _extract_mapping(
                dfs[side], v3[f"embs_{side}_dinov2"], side,
                audit_dir / f"{side}_mapping.partial.npz", signature, model, transform, device)
            initial_indices = indices.copy()
            audit = dfs[side][["file_path", "label", "source_dataset"]].copy()
            audit["nearest_v3_row"] = initial_indices
            audit["nearest_v3_label"] = np.asarray(v3[f"labels_{side}_dinov2"])[initial_indices]
            audit["cosine"] = scores
            audit["runner_up_cosine"] = runner_up
            audit.to_csv(audit_dir / f"{side}_row_identity.csv", index=False)
            indices, resolved_ties, duplicate_groups = resolve_feature_equivalent_duplicates(
                v3, dfs[side], side, indices, scores, runner_up)
            audit["v3_row"] = indices
            audit["v3_label"] = np.asarray(v3[f"labels_{side}_dinov2"])[indices]
            resolution = np.full(len(audit), "unique_dinov2_match", dtype=object)
            for group in duplicate_groups:
                resolution[group["current_rows"]] = group["resolution"]
            audit["match_resolution"] = resolution
            audit.to_csv(audit_dir / f"{side}_row_identity.csv", index=False)
            summaries[side] = validate_index_map(
                dfs[side], v3[f"labels_{side}_dinov2"], indices, scores, runner_up, side,
                resolved_duplicates=resolved_ties)
            summaries[side]["duplicate_groups"] = duplicate_groups
            mappings[side] = indices
            print(f"[repair] {side}: {summaries[side]}", flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        out = rebuild_arrays(v3, v5, dfs, mappings)
        temp = target.with_suffix(".tmp.npz")
        np.savez_compressed(temp, **out)
        with np.load(temp, allow_pickle=False) as repaired:
            for side in SIDES:
                if not np.array_equal(canonical(repaired[f"labels_{side}_dinov2"]),
                                      canonical(dfs[side]["label"])):
                    raise ValueError(f"{side} repaired cache labels failed post-write validation")
                if len(repaired[f"embs_{side}_dinov2"]) != len(dfs[side]):
                    raise ValueError(f"{side} repaired cache feature count failed post-write validation")
        temp.replace(target)
    meta = json.loads(base_meta_path.read_text())
    meta.pop("old_id_csv", None)
    meta.pop("old_ood_csv", None)
    meta.update({
        "artifact": "public_row_identity_verified", "target": str(target),
        "target_sha256": sha256(target),
        "source_cache": str(base),
        "v3_source": str(source), "v3_sha256": source_hash,
        "v5_base": str(base), "v5_sha256": base_hash,
        "v5_meta_sha256": sha256(base_meta_path),
        "signature": json.loads(signature), "sides": summaries,
        "fsdm41_relabelled": relabelled,
        "public_query_source": "v3 features reindexed by full-image DINOv2 identity",
        "ce_full_public": "copied unchanged from v5 (fresh image extraction)",
        "other_public_features": "reindexed from v3 by exact DINOv2 image identity",
        "swi_features": "copied unchanged from v5",
        "notes": [
            "SWI gallery and CE-Full public arrays copied unchanged from v5.",
            "All other public arrays reindexed from v3 using full-image DINOv2 matches.",
            "The reconstructed pre-correction expanded CSV was not used as a row-order authority.",
        ],
    })
    write_json(target.with_name(target.stem + "_meta.json"), meta)
    write_json(audit_dir / "summary.json", meta)
    print(f"[repair] Verified public-row cache: {target}", flush=True)
    return target


if __name__ == "__main__":
    run()
