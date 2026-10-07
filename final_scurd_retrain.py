"""Isolated, provenance-checked SC-URD head-only rerun for the final paper audit."""

import json
import os
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

from .audit_support import canonical, compare_recipes, sha256, write_json
from . import final_colab_audit as A


SEEDS = (42, 43, 44)
RECIPE = {
    "SCURD_BETA": "0.1", "SCURD_TRAIN_EPOCHS": "20",
    "SCURD_TRAIN_EPISODES": "500", "SCURD_TRAIN_LAMBDA_CONS": "0.5",
    "SCURD_TRAIN_LR": "1e-3", "SCURD_WEIGHT_DECAY": "1e-4",
    "SCURD_N_WAY": "16", "SCURD_K_SUPPORT": "5",
    "SCURD_Q_QUERY": "4", "SCURD_TAU": "0.07",
    "SCURD_TOP_M": "50", "SCURD_MAIN_MODE": "raw",
    "SCURD_TRAIN_SEEDS": "42,43,44", "SCURD_FORCE_RETRAIN_SEEDS": "0",
}


def _meta_path(root, run_root):
    explicit = os.environ.get("FINAL_SCURD_META_CACHE")
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"FINAL_SCURD_META_CACHE does not exist: {path}")
        return path
    name = "urd_v2_meta_dinov2_embeddings_v2.npz"
    candidates = [
        run_root / "deployment" / "research_directions" / name,
        root / "results" / "paper_reframe_full954_final" / "deployment" / "research_directions" / name,
        root / "results" / "paper_reframe_full954_corrected_public" / "deployment" / "research_directions" / name,
    ]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(
        "SC-URD weak/strong meta cache is missing. Set FINAL_SCURD_META_CACHE "
        "to the original urd_v2_meta_dinov2_embeddings_v2.npz. "
        "The v5 full-gallery cache alone has no strong-augmentation embeddings. "
        f"Checked: {[str(p) for p in candidates]}")


def build_meta_from_v5(root, out, cache_path, manifest_path, device):
    """Extract only strong meta-train features; weak/val rows reuse aligned v5 DINOv2."""
    import random
    import torch
    from .training.meta_embeddings import dino_strong_transform, extract_dino_dataset

    path = out / "research_directions" / "urd_v2_meta_dinov2_embeddings_v2.npz"
    if path.exists():
        return path.resolve()
    manifest = json.loads(manifest_path.read_text())
    train_items = manifest["meta-train"]
    val_items = manifest["meta-val"]
    n_train = len(train_items)
    n_val = len(val_items)
    with np.load(cache_path, allow_pickle=False) as cache:
        all_labels = canonical(cache["labels_swi_dinov2"])
        expected = canonical(label for split in ("meta-train", "meta-val", "meta-test")
                             for _, label in manifest[split])
        if not np.array_equal(all_labels, expected):
            raise ValueError("Cannot rebuild meta cache: v5 SWI labels/order differ from manifest")
        swi = cache["embs_swi_dinov2"]
        weak = swi[:n_train].astype(np.float32)
        val = swi[n_train:n_train + n_val].astype(np.float32)
    random.seed(2026)
    np.random.seed(2026)
    torch.manual_seed(2026)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2026)
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").eval().to(device)
    batch = int(os.environ.get("FINAL_SCURD_META_BATCH_SIZE", "16"))
    workers = int(os.environ.get("FINAL_SCURD_META_WORKERS", "4"))
    print(f"[meta] Extracting strong DINOv2 features for {n_train} SWI images; batch={batch}", flush=True)
    strong, strong_labels = extract_dino_dataset(
        model, train_items, dino_strong_transform(), device,
        desc="SC-URD strong meta-train", batch_size=batch, num_workers=workers)
    if not np.array_equal(canonical(strong_labels), canonical(label for _, label in train_items)):
        raise ValueError("Fresh strong embeddings differ from manifest label order")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, train_weak=weak, train_strong=strong,
                        train_labels=canonical(label for _, label in train_items),
                        val_weak=val, val_labels=canonical(label for _, label in val_items))
    tmp.replace(path)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return path.resolve()


def validate_meta(meta_path, manifest_path, cache_path):
    manifest = json.loads(manifest_path.read_text())
    train_labels = canonical(label for _, label in manifest["meta-train"])
    val_labels = canonical(label for _, label in manifest["meta-val"])
    all_labels = canonical(label for split in ("meta-train", "meta-val", "meta-test")
                           for _, label in manifest[split])
    with np.load(meta_path, allow_pickle=False) as meta, np.load(cache_path, allow_pickle=False) as cache:
        required = {"train_weak", "train_strong", "train_labels", "val_weak", "val_labels"}
        if not required.issubset(meta.files):
            raise ValueError(f"Meta cache lacks {sorted(required - set(meta.files))}")
        weak, strong, val = (meta[k] for k in ("train_weak", "train_strong", "val_weak"))
        if weak.shape != (len(train_labels), 768) or strong.shape != weak.shape:
            raise ValueError(f"Unexpected train weak/strong shapes: {weak.shape}, {strong.shape}")
        if val.shape != (len(val_labels), 768):
            raise ValueError(f"Unexpected meta-val shape: {val.shape}")
        if not np.array_equal(canonical(meta["train_labels"]), train_labels):
            raise ValueError("Meta-train labels/order differ from swi_manifest.json")
        if not np.array_equal(canonical(meta["val_labels"]), val_labels):
            raise ValueError("Meta-val labels/order differ from swi_manifest.json")
        if not all(np.isfinite(x).all() for x in (weak, strong, val)):
            raise ValueError("Meta cache contains non-finite embeddings")
        swi = cache["embs_swi_dinov2"]
        if swi.shape != (len(all_labels), 768):
            raise ValueError(f"Unexpected full SWI DINOv2 shape: {swi.shape}")
        if not np.array_equal(canonical(cache["labels_swi_dinov2"]), all_labels):
            raise ValueError("Full SWI cache labels/order differ from manifest")
        max_diff = float(np.max(np.abs(weak.astype(np.float32) - swi[:len(train_labels)].astype(np.float32))))
        if max_diff > 1e-3:
            raise ValueError(f"Meta-train weak embeddings differ from v5 SWI cache (max abs={max_diff:.6g})")
    return {"meta_train_images": len(train_labels), "meta_val_images": len(val_labels),
            "meta_train_species": len(set(train_labels)), "meta_val_species": len(set(val_labels)),
            "train_weak_vs_v5_max_abs_diff": max_diff}


def validate_public(root, cache_path, manifest_path):
    overrides = json.loads((root / "dataset_label_corrections.json").read_text())["FSDM41"]["overrides"]
    manifest = json.loads(manifest_path.read_text())
    meta_train_species = set(canonical(label for _, label in manifest["meta-train"]))
    corrected = 0
    species = {}
    with np.load(cache_path, allow_pickle=False) as cache:
        swi_species = set(canonical(cache["labels_swi_dinov2"]))
        for side, expected_n in (("id", 6189), ("ood", 30868)):
            df = pd.read_csv(root / f"{side.upper()}_images_expanded.csv")
            labels = canonical(cache[f"labels_{side}_dinov2"])
            if len(df) != expected_n or len(labels) != expected_n:
                raise ValueError(f"Unexpected corrected {side} cohort: CSV={len(df)}, cache={len(labels)}")
            corrected += A._validate_public_rows(df, labels, overrides, side)
            species[side] = set(labels)
        if species["id"] & species["ood"]:
            raise ValueError("Corrected ID/OOD species overlap")
        if species["id"] & meta_train_species:
            raise ValueError(f"Public-ID species leaked into SC-URD meta-train: "
                             f"{sorted(species['id'] & meta_train_species)}")
        if species["ood"] & swi_species:
            raise ValueError(f"Public-OOD species overlap SWI: {sorted(species['ood'] & swi_species)}")
    if corrected != 2901 or len(species["id"]) != 24 or len(species["ood"]) != 153:
        raise ValueError(f"Unexpected public correction: FSDM41={corrected}, "
                         f"ID={len(species['id'])}, OOD={len(species['ood'])}")
    return {"id_images": 6189, "ood_images": 30868, "id_species": 24,
            "ood_species": 153, "fsdm41_relabelled": corrected, "woodauth_retained": False,
            "public_id_vs_meta_train_overlap": 0, "public_ood_vs_swi_overlap": 0}


def _embed_images(model, transform, paths, device):
    import torch
    import torch.nn.functional as F
    from PIL import Image
    with torch.inference_mode():
        imgs = torch.stack([transform(Image.open(path).convert("RGB")) for path in paths]).to(device)
        return F.normalize(model(imgs).float(), dim=1).cpu().numpy()


def _nearest_fingerprint(actual, embs, labels=None):
    """Locate an image embedding without trusting the reconstructed CSV row order."""
    matrix = np.asarray(embs, dtype=np.float32)
    matrix = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    scores = matrix @ actual.astype(np.float32)
    index = int(np.argmax(scores))
    result = {"index": index, "cosine": float(scores[index])}
    if labels is not None:
        result["label"] = str(labels[index])
    return result


def image_fingerprints(root, cache_path, exp4_path, device, source_cache_path=None):
    import torch
    from torchvision import transforms

    print("[provenance] Re-extracting sampled public/VN26 DINOv2 images...", flush=True)
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").eval().to(device)
    transform = transforms.Compose([
        transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(518), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    report = []
    source_cache = (np.load(source_cache_path, allow_pickle=False)
                    if source_cache_path is not None and source_cache_path.is_file() else None)
    with np.load(cache_path, allow_pickle=False) as cache:
        manifest = json.loads((root / "swi_manifest.json").read_text())
        swi_embs = cache["embs_swi_dinov2"]
        offset = 0
        for split in ("meta-train", "meta-val", "meta-test"):
            items = manifest[split]
            for relative in sorted(set((0, len(items) // 2, len(items) - 1))):
                index = offset + relative
                path = Path(items[relative][0])
                actual = _embed_images(model, transform, [path], device)[0]
                expected = swi_embs[index].astype(np.float32)
                expected /= max(float(np.linalg.norm(expected)), 1e-12)
                report.append({"side": "swi", "split": split, "row": index,
                               "path": str(path), "source": "SWI",
                               "cosine": float(actual @ expected)})
            offset += len(items)
        for side in ("id", "ood"):
            df = pd.read_csv(root / f"{side.upper()}_images_expanded.csv")
            if not np.array_equal(canonical(df["label"]), canonical(cache[f"labels_{side}_dinov2"])):
                raise ValueError(f"{side} CSV labels are not aligned to v5 cache")
            embs = cache[f"embs_{side}_dinov2"]
            cache_labels = canonical(cache[f"labels_{side}_dinov2"])
            old_embs = None
            old_labels = None
            if source_cache is not None and f"embs_{side}_dinov2" in source_cache:
                old_embs = source_cache[f"embs_{side}_dinov2"]
                old_labels = canonical(source_cache[f"labels_{side}_dinov2"])
            samples = []
            for _, group in df.groupby("source_dataset", sort=True):
                samples.extend([int(group.index[0]), int(group.index[len(group) // 2]),
                                int(group.index[-1])])
            samples = sorted(set(samples))
            for start in range(0, len(samples), 8):
                idx = samples[start:start + 8]
                paths = [Path(df.iloc[i]["file_path"]) for i in idx]
                if not all(p.is_file() for p in paths):
                    raise FileNotFoundError(f"Missing sampled {side} image: {[str(p) for p in paths if not p.is_file()]}")
                actual = _embed_images(model, transform, paths, device)
                expected = embs[idx].astype(np.float32)
                expected /= np.maximum(np.linalg.norm(expected, axis=1, keepdims=True), 1e-12)
                sims = np.sum(actual * expected, axis=1)
                for i, path, sim, fresh in zip(idx, paths, sims, actual):
                    nearest = _nearest_fingerprint(fresh, embs, cache_labels)
                    nearest["same_csv_label"] = nearest["label"] == str(cache_labels[i])
                    nearest["source_at_index"] = str(df.iloc[nearest["index"]]["source_dataset"])
                    row = {"side": side, "row": int(i), "path": str(path),
                           "label": str(cache_labels[i]),
                           "source": str(df.iloc[i]["source_dataset"]),
                           "cosine": float(sim), "nearest_v5": nearest}
                    if old_embs is not None:
                        row["nearest_v3"] = _nearest_fingerprint(fresh, old_embs, old_labels)
                    report.append(row)
    if source_cache is not None:
        source_cache.close()
    with np.load(exp4_path, allow_pickle=False) as exp4:
        vn_items = A._vn26_items(root)
        for mag, items in vn_items.items():
            key = A._e4key("vn26", "DINOv2", mag)
            arr = exp4[key].astype(np.float32)
            arr /= np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12)
            labels = canonical(exp4[key + "_lbl"])
            for i in sorted(set((0, len(items) // 2, len(items) - 1))):
                path, label = items[i]
                actual = _embed_images(model, transform, [Path(path)], device)[0]
                same_label = arr[labels == label]
                if not len(same_label):
                    raise ValueError(f"VN26 {mag}/{label} absent from legacy exp4 cache")
                report.append({"side": f"vn26_{mag}", "path": path, "source": "VN26",
                               "cosine": float(np.max(same_label @ actual))})
    del model
    return {"passed": bool(report and min(row["cosine"] for row in report) >= 0.999),
            "n_samples": len(report), "min_cosine": min(row["cosine"] for row in report),
            "samples": report,
            "limit": "Sampled matches do not prove identity of every image in legacy exp4."}


def run():
    import torch

    root = Path(os.environ["ROOT_PATH"]).resolve()
    run_root = Path(os.environ.get("FINAL_AUDIT_RUN_ROOT",
        root / "results" / "paper_reframe_full954_retrained_ce_corrected_public")).resolve()
    out = Path(os.environ.get("FINAL_SCURD_OUT", root / "results" / "final_scurd_retrain_v1")).resolve()
    old_research = run_root / "deployment" / "research_directions"
    if out == run_root or out == old_research or old_research in out.parents:
        raise ValueError("FINAL_SCURD_OUT must be isolated from the historical run")
    cache_path = root / "embedding_cache_full954_v5_retrained_ce_corrected_public.npz"
    exp4_path = root / "exp4_embedding_cache_v3.npz"
    manifest_path = root / "swi_manifest.json"
    selected_path = run_root / "hyperparameters" / "scurd_hyperparameter_selection.json"
    for path in (cache_path, exp4_path, manifest_path, selected_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    selected = json.loads(selected_path.read_text())["selected"]
    if selected.get("mode") != "raw" or float(selected.get("tau", -1)) != 0.07 or int(selected.get("top_m", -1)) != 50:
        raise ValueError("Saved meta-val selection differs from frozen raw/tau=0.07/top_m=50 protocol")
    device = os.environ.get("DEVICE", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if os.environ.get("FINAL_SCURD_BUILD_META", "0") == "1":
        meta_path = build_meta_from_v5(root, out, cache_path, manifest_path, device)
    else:
        meta_path = _meta_path(root, run_root)
    print(f"FINAL SC-URD HEAD RERUN: {out}\nMETA CACHE: {meta_path}", flush=True)
    alignment = validate_meta(meta_path, manifest_path, cache_path)
    if (alignment["meta_train_images"], alignment["meta_val_images"],
            alignment["meta_train_species"], alignment["meta_val_species"]) != (124577, 18018, 557, 80):
        raise ValueError(f"Unexpected SWI meta-train/val cohort: {alignment}")
    public = validate_public(root, cache_path, manifest_path)
    out.mkdir(parents=True, exist_ok=True)
    fingerprint = None
    if os.environ.get("FINAL_SCURD_IMAGE_FINGERPRINT", "1") == "1":
        source_cache_path = root / "embedding_cache_full954_v3.npz"
        fingerprint = image_fingerprints(root, cache_path, exp4_path, device, source_cache_path)
        write_json(out / "image_fingerprint_diagnostics.json", fingerprint)
        failures = sorted((row for row in fingerprint["samples"] if row["cosine"] < 0.999),
                          key=lambda row: row["cosine"])
        relocated_v5 = sum(row.get("nearest_v5", {}).get("cosine", -1) >= 0.999
                           for row in failures)
        relocated_v3 = sum(row.get("nearest_v3", {}).get("cosine", -1) >= 0.999
                           for row in failures)
        print(f"[provenance] {len(failures)}/{fingerprint['n_samples']} fingerprints failed; "
              f"nearest match in v5={relocated_v5}, v3={relocated_v3}; "
              f"details: {out / 'image_fingerprint_diagnostics.json'}", flush=True)
        for row in failures[:5]:
            print(f"  {row['side']} row={row.get('row')} same-row={row['cosine']:.4f} "
                  f"nearest-v5={row.get('nearest_v5', {}).get('cosine', float('nan')):.4f} "
                  f"nearest-v3={row.get('nearest_v3', {}).get('cosine', float('nan')):.4f} "
                  f"{row['path']}", flush=True)
        if os.environ.get("FINAL_SCURD_DIAGNOSE_ONLY", "0") == "1":
            return out
        if not fingerprint["passed"]:
            raise ValueError("DINOv2 image fingerprint mismatch; head training stopped. "
                             f"Inspect {out / 'image_fingerprint_diagnostics.json'}")
    elif os.environ.get("FINAL_SCURD_DIAGNOSE_ONLY", "0") == "1":
        raise ValueError("FINAL_SCURD_DIAGNOSE_ONLY requires FINAL_SCURD_IMAGE_FINGERPRINT=1")
    meta_hash = sha256(meta_path)
    provenance = {
        "meta_cache": str(meta_path), "meta_cache_sha256": meta_hash,
        "meta_cache_source": "rebuilt_strong_from_images" if os.environ.get("FINAL_SCURD_BUILD_META", "0") == "1"
                             else "existing_weak_strong_cache",
        "full954_cache_sha256": sha256(cache_path), "exp4_cache_sha256": sha256(exp4_path),
        "manifest_sha256": sha256(manifest_path), "selection_sha256": sha256(selected_path),
        "training_engine_sha256": sha256(Path(__file__).parent / "_engines" / "variance_retrieval_evidence_colab.py"),
        "evaluation_code_sha256": sha256(Path(A.__file__)),
        "head_model_sha256": sha256(Path(__file__).parent / "scurd.py"),
        "meta_extraction_code_sha256": sha256(Path(__file__).parent / "training" / "meta_embeddings.py"),
        "runner_sha256": sha256(Path(__file__)), "recipe": RECIPE,
        "torch_version": torch.__version__, "numpy_version": np.__version__,
        "device": device,
        "alignment": alignment, "public": public, "image_fingerprint": fingerprint,
        "selected_legacy_checkpoint_is_not_assumed_same_recipe": True,
    }
    provenance_path = out / "provenance_inputs.json"
    if provenance_path.exists():
        old = json.loads(provenance_path.read_text())
        stable = lambda record: {k: v for k, v in record.items() if k != "image_fingerprint"}
        if stable(old) != stable(provenance) or bool(old.get("image_fingerprint")) != bool(fingerprint):
            raise ValueError("Existing rerun folder has different inputs/recipe; use a new FINAL_SCURD_OUT")
    write_json(provenance_path, provenance)

    module = "swid_retrieval._engines.variance_retrieval_evidence_colab"
    if module in sys.modules:
        raise RuntimeError("Variance engine was already imported; restart runtime or clear swid_retrieval modules")
    os.environ.update(RECIPE)
    os.environ.update({"RESULTS_DIR": str(out), "OUT_DIR": str(out / "training_logs"),
                       "SCURD_META_CACHE_NAME": str(meta_path),
                       "SCURD_META_CACHE_SHA256": meta_hash,
                       "GALLERY_SCOPE": "id_only"})
    from ._engines import variance_retrieval_evidence_colab as engine
    if engine.SCURD_CKPT_DIR.resolve() != (out / "research_directions").resolve():
        raise RuntimeError("Training checkpoint directory was not isolated")
    ckpts = {f"seed{seed}": engine.train_one_scurd_seed(seed, device) for seed in SEEDS}
    recipes = {name: A._checkpoint_metadata(path)[0] for name, path in ckpts.items()}
    comparison = compare_recipes(recipes)
    if not comparison["metadata_recipe_match"]:
        raise ValueError(f"New seed checkpoint recipes disagree: {comparison}")
    if any(torch.load(path, map_location="cpu", weights_only=False).get("meta_cache_sha256") != meta_hash
           for path in ckpts.values()):
        raise ValueError("New seed checkpoint lacks the expected meta-cache hash")

    eval_out = out / "evaluation"
    eval_out.mkdir(parents=True, exist_ok=True)
    args = Namespace(device=device, gallery_repeats=100, ood_kshot_repeats=50)
    p = {"root": root, "cache": cache_path, "exp4": exp4_path, "out": eval_out}
    A.scurd_audit(p, args, {"selected": selected, "recipe_comparison": comparison}, ckpts)
    summary = {
        "checkpoint_sha256": {name: sha256(path) for name, path in ckpts.items()},
        "seed_recipe_metadata_match": True, "meta_cache_sha256": meta_hash,
        "selected_legacy_checkpoint_comparable": False,
        "seed_metrics_csv": str(eval_out / "scurd_raw_centered_seed_audit.csv"),
        "seed_summary_csv": str(eval_out / "scurd_seed_summary.csv"),
        "paper_numbers_updated": False,
    }
    write_json(out / "summary.json", summary)
    print(f"Rerun complete: {out / 'summary.json'}", flush=True)
    return out


if __name__ == "__main__":
    run()
