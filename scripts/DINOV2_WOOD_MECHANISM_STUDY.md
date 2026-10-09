# DINOv2-S retrieval mechanism study (SWI meta-validation only)

This study is exploratory. The previous seed-42 pilot favored DINOv2-S fine-tuned
with prototype scoring, but neither its training budget nor seed count matched a
confirmatory comparison. Do not update manuscript claims from the pilot.

The matched run compares prototype and nearest-reference objectives at the same
8 x 500 episodes and seeds 42/43/44. `prototype_large_frozen` trains the same
projection and loss while freezing the pretrained encoder, isolating the effect
of end-to-end adaptation. All variants use 224 px, 16/32/64-way episodes, two
support references and one query per species, and scan-disjoint sampling.

The separate mechanism pilot tests cross-scale positives and a wood-motivated,
spatially unordered patch-evidence head. DINOv2-S yields the original CLS
embedding and a 4 x 4 pooled set of patch features; a learned 64-d projection
matches local evidence after global scores shortlist 64 species. This does not
detect or label vessels, rays, or parenchyma. The ablations only support a
mechanistic explanation if their differences survive matched-budget seeds and
the final deployment galleries.

For every checkpoint, meta-validation now saves per-species R@1 and a balanced
one-reference-per-species gallery curve at 128, 256 and 637 species. The 24-
and 57-species scan-disjoint endpoints are also retained. The predeclared
selection score remains the mean of 24-species and 637-species R@1 across two
folds. The 637-species stress gallery contains meta-train classes seen during
training; it is **not** a substitute for the 954-species public test.
For patch models, `val24_global_r1`, `val_all_global_r1`, and
`val_stress_global_r1` score the same checkpoint without local reranking;
compare each with its combined counterpart to isolate inference-time patch
benefit. Per-species and fold-level data are in `best_validation.json`.

## Colab cell 1: setup and preflight

Mount Drive first. Do not reuse previous study output directories: code hashes
and checkpoint recipes have changed. Clear stale `swid_retrieval` modules after
pulling. Keep the runtime alive across cells.

```python
%cd /content/drive/MyDrive/NCS
!git -C swid_retrieval pull
!git -C swid_retrieval rev-parse --short HEAD

import os, sys, runpy
from pathlib import Path
sys.path.insert(0, "/content/drive/MyDrive/NCS")
for name in list(sys.modules):
    if name.startswith("swid_retrieval"):
        del sys.modules[name]

root = Path("/content/drive/MyDrive/NCS")
matched = root / "results/dinov2s_retrieval_matched_v1"
mechanism = root / "results/dinov2s_wood_mechanisms_pilot_v1"
os.environ.update({
    "ROOT_PATH": str(root), "DEVICE": "cuda", "RUN_GALLERY_STUDY": "1",
    "RUN_REPAIR_PUBLIC_ROWS": "0", "RUN_FINAL_SCURD_RETRAIN": "0",
    "RUN_FINAL_COLAB_AUDIT": "0", "GALLERY_STUDY_OUT": str(matched),
    "GALLERY_STUDY_BACKBONE": "dinov2_vits14",
    "GALLERY_STUDY_MODE": "preflight", "GALLERY_STUDY_SEEDS": "42",
    "GALLERY_STUDY_IMAGE_SIZE": "224", "GALLERY_STUDY_WAYS": "16,32,64",
    "GALLERY_STUDY_VALIDATION_FOLDS": "2",
    "GALLERY_STUDY_VALIDATION_TRAIN_DISTRACTORS": "1",
    "GALLERY_STUDY_PILOT_CHECKPOINTS": "0", "GALLERY_STUDY_PRELOAD": "0",
    "GALLERY_STUDY_WORKERS": "4", "IMAGE_CACHE_DIR": "/content/cache_images",
})
os.environ.pop("GALLERY_STUDY_INIT_CHECKPOINT", None)
_ = runpy.run_module("swid_retrieval.run_overnight", run_name="__main__")

import torch
from swid_retrieval.gallery_experiment import variant_config, _model
patch_recipe = variant_config("patch_evidence")
encoder, _ = _model(patch_recipe, torch.device("cuda"))
encoder.eval()
with torch.inference_mode():
    example = torch.randn(2, 3, 224, 224, device="cuda")
    cls, tokens = encoder.backbone.forward_with_tokens(example)
    assert torch.allclose(cls, encoder.backbone(example), atol=1e-5)
    global_embedding, local_embedding = encoder.forward_with_tokens(example)
    assert global_embedding.shape == (2, 512)
    assert local_embedding.shape == (2, 16, 64)
    assert torch.isfinite(local_embedding).all()
print("DINOv2-S patch evidence smoke passed")
del encoder, example, cls, tokens, global_embedding, local_embedding
torch.cuda.empty_cache()
```

## Colab cell 2: optional one-time local image cache

Use this on a fresh runtime when repeated Drive reads dominate training. The
cache copies images to `/content/cache_images` and is ephemeral; on a runtime
where it is already populated, skip this cell. It does not change the train/val
split. Allow enough local disk space for the JPEG cache.

```python
from swid_retrieval import data
manifest = data.load_swi_manifest(root / "swi_manifest.json")
paths = [path for split in ("meta-train", "meta-val")
         for path, _ in manifest[split]]
stats = data.preload_image_cache(paths, max_workers=16,
                                 desc="SWI meta-train + meta-val image cache")
assert stats["bad"] == 0, stats
```

## Colab cell 3: matched-budget, three seeds

Run this cell for the main comparison. It can take hours on T4. A repeat with
the same code and recipe resumes from `latest.pt`; do not change the output
directory or hyperparameters mid-run. Each encoder replay microbatch is eight
images, but the 16/32/64-way episode and gallery are not reduced.

```python
os.environ.update({
    "GALLERY_STUDY_OUT": str(matched), "GALLERY_STUDY_BACKBONE": "dinov2_vits14",
    "GALLERY_STUDY_MODE": "screen",
    "GALLERY_STUDY_VARIANTS": "prototype_large,metric_large,prototype_large_frozen",
    "GALLERY_STUDY_EPOCHS": "8", "GALLERY_STUDY_EPISODES_PER_EPOCH": "500",
    "GALLERY_STUDY_MICROBATCH": "8", "GALLERY_STUDY_IMAGE_BATCH": "16",
})
os.environ.pop("GALLERY_STUDY_INIT_CHECKPOINT", None)
for seed in (42, 43, 44):
    os.environ["GALLERY_STUDY_SEEDS"] = str(seed)
    _ = runpy.run_module("swid_retrieval.run_overnight", run_name="__main__")
    print("Completed seed", seed, flush=True)
print(matched / "meta_val_screen_summary.csv")
```

## Colab cell 4: wood-mechanism pilot, seed 42

This is a lower-budget *screen only*. Do not compare its absolute performance
directly to the matched-budget run. The key comparisons are patch vs no patch,
frozen vs full encoder, and cross-scale positive vs the same base objective.
Run full-budget, multi-seed versions only of promising mechanisms afterward.

```python
os.environ.update({
    "GALLERY_STUDY_OUT": str(mechanism),
    "GALLERY_STUDY_BACKBONE": "dinov2_vits14",
    "GALLERY_STUDY_MODE": "screen", "GALLERY_STUDY_SEEDS": "42",
    "GALLERY_STUDY_VARIANTS": (
        "prototype_large,prototype_large_frozen,prototype_large_scale,"
        "patch_evidence,patch_evidence_frozen,patch_evidence_scale"),
    "GALLERY_STUDY_EPOCHS": "5", "GALLERY_STUDY_EPISODES_PER_EPOCH": "150",
    "GALLERY_STUDY_MICROBATCH": "8", "GALLERY_STUDY_IMAGE_BATCH": "16",
})
os.environ.pop("GALLERY_STUDY_INIT_CHECKPOINT", None)
_ = runpy.run_module("swid_retrieval.run_overnight", run_name="__main__")
print(mechanism / "meta_val_screen.csv")
```

## Colab cell 5: small review package

Upload this ZIP for analysis. Do not include full `.pt` checkpoints or `.npy`
embeddings. The archive contains per-species/fold information needed for
paired comparisons. Do not run public-ID/OOD/VN26 evaluation until the
meta-val shortlist and scoring rule are frozen.

```python
from zipfile import ZipFile, ZIP_DEFLATED
archive = root / "results/dinov2s_wood_mechanism_review.zip"
with ZipFile(archive, "w", ZIP_DEFLATED) as z:
    for folder in (matched, mechanism):
        assert (folder / "meta_val_screen.csv").is_file(), folder
        for name in ("meta_val_screen.csv", "meta_val_screen_summary.csv",
                     "meta_val_screen_manifest.json", "source_scan_audit.json"):
            path = folder / name
            if path.is_file():
                z.write(path, path.relative_to(root / "results"))
        for pattern in ("*/*/seed_*/best_validation.json", "*/*/seed_*/progress.json"):
            for path in sorted(folder.glob(pattern)):
                z.write(path, path.relative_to(root / "results"))
print(archive, f"{archive.stat().st_size / 1024**2:.2f} MB")
```
