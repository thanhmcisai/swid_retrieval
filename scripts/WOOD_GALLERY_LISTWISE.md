# Gallery-listwise pilot on SmartWoodID

This isolated pilot uses the existing selected DINOv2-S `prototype_large`
checkpoint. It fine-tunes the encoder on every meta-train image once per arm.
No meta-test or public-ID image is used for fitting or checkpoint selection.

The four matched arms are:

| Arm | Loss beyond ArcFace + SupCon | Bank references |
| --- | --- | --- |
| `arcface_supcon` | None (matched control) | None |
| `gallery_full` | Full 557-class real-reference cross-entropy | K=1 |
| `gallery_hard` | Full loss plus top-32 negative margin | K=1 |
| `gallery_hard_scale` | Full loss plus top-32 negative margin | K cycles 1, 3, 5 |

The bank contains only meta-train images, with distinct scans and scales
preferred. A live positive reference is encoded with gradients and replaces one
bank positive for each query. Bank negatives are detached; bank entries refresh
from observed query/reference embeddings after successful optimizer steps. The
source and validation images are loaded from the same local JPEG cache for the
entire pilot. The cache fingerprint and all recipe/source hashes are recorded.

Checkpoint selection uses two-fold meta-val `637x1` R@1, subject to `24x5`
R@1 not falling by more than 0.03 absolute relative to the source checkpoint.
The selected arm is locked before any meta-test/public-ID evaluation. This is
a pilot, not a claim of improvement on the 954-species cross-domain endpoint.

## Colab cell

Run after mounting Drive. Existing image cache at `/content/cache_images` is
reused; a fresh Colab runtime may need to preload all 124,577 training images.

```python
%cd /content/drive/MyDrive/NCS
!git -C swid_retrieval pull

import os, sys, runpy
from pathlib import Path

sys.path.insert(0, "/content/drive/MyDrive/NCS")
for name in list(sys.modules):
    if name.startswith("swid_retrieval"):
        del sys.modules[name]

for flag in (
    "RUN_WOOD_FULL_IMAGE_EARLY_STOP", "RUN_WOOD_FULL_IMAGE_TRAIN",
    "RUN_WOOD_SCALE_RANK_STUDY", "RUN_WOOD_COVERAGE_STUDY",
    "RUN_WOOD_HARD_SHORTLIST", "RUN_WOOD_MULTILEVEL",
    "RUN_WOOD_REFERENCE_BUDGET", "RUN_WOOD_GALLERY_AUDIT",
    "RUN_WOOD_LARGE_GALLERY", "RUN_WOOD_CORRESPONDENCE_COMPARE",
    "RUN_WOOD_CORRESPONDENCE_IMAGE_TRAIN", "RUN_WOOD_CORRESPONDENCE_STUDY",
    "RUN_WOOD_EVIDENCE_STUDY", "RUN_GALLERY_DIAGNOSTICS",
    "RUN_GALLERY_STUDY", "RUN_REPAIR_PUBLIC_ROWS",
    "RUN_FINAL_SCURD_RETRAIN", "RUN_FINAL_COLAB_AUDIT",
):
    os.environ[flag] = "0"

os.environ.update({
    "ROOT_PATH": "/content/drive/MyDrive/NCS",
    "IMAGE_CACHE_DIR": "/content/cache_images",
    "DEVICE": "cuda",
    "RUN_WOOD_GALLERY_LISTWISE": "1",
    "WOOD_LISTWISE_SOURCE_OUT": "/content/drive/MyDrive/NCS/results/dinov2s_retrieval_matched_v1",
    "WOOD_LISTWISE_OUT": "/content/drive/MyDrive/NCS/results/wood_gallery_listwise_pilot_v1",
    "WOOD_LISTWISE_ARMS": "arcface_supcon,gallery_full,gallery_hard,gallery_hard_scale",
    "WOOD_LISTWISE_SEED": "43",
    "WOOD_LISTWISE_BANK_REFERENCES": "5",
    "WOOD_LISTWISE_BANK_BATCH": "64",
    "WOOD_LISTWISE_BANK_MOMENTUM": "0.5",
    "WOOD_LISTWISE_WEIGHT": "0.25",
    "WOOD_LISTWISE_HARD_WEIGHT": "0.2",
    "WOOD_LISTWISE_HARD_CLASSES": "32",
    "WOOD_LISTWISE_HARD_MARGIN": "0.05",
    "WOOD_LISTWISE_TEMPERATURE": "0.07",
    "WOOD_LISTWISE_RETENTION_TOL": "0.03",
    "WOOD_FULL_ARMS": "arcface_supcon",
    "WOOD_FULL_SEEDS": "43",
    "WOOD_FULL_SOURCE_SEED": "43",
    "WOOD_FULL_EPOCHS": "1",
    "WOOD_FULL_STEP_MODE": "direct",
    "WOOD_FULL_BATCH": "64",
    "WOOD_FULL_MICROBATCH": "32",
    "WOOD_FULL_WORKERS": "2",
    "WOOD_FULL_BACKBONE_LR": "2e-6",
    "WOOD_FULL_HEAD_LR": "1e-4",
    "WOOD_FULL_ARCFACE_WEIGHT": "1",
    "WOOD_FULL_QKV_WEIGHT": "0.5",
    "WOOD_FULL_CONTRASTIVE_WEIGHT": "0.2",
    "WOOD_FULL_CONTRASTIVE_TEMPERATURE": "0.07",
    "WOOD_FULL_PRINT_EVERY": "100",
    "WOOD_LISTWISE_PRELOAD_TRAIN": "1",
    "WOOD_LISTWISE_PRELOAD_WORKERS": "8",
})

for mode in ("preflight", "train", "validate", "export"):
    print(f"\n===== {mode.upper()} =====", flush=True)
    os.environ["WOOD_LISTWISE_MODE"] = mode
    _ = runpy.run_module("swid_retrieval.run_overnight", run_name="__main__")

archive = Path(os.environ["WOOD_LISTWISE_OUT"]) / "wood_gallery_listwise_review.zip"
print(archive, archive.stat().st_size)
from google.colab import files
files.download(str(archive))
```

If the runtime stops, re-run the same cell. Completed arms are verified and
skipped. An interrupted arm restarts from the source checkpoint because
DataLoader worker augmentation RNG cannot be resumed exactly mid-epoch. Keep
the same output path only when code, source checkpoint, manifest, recipe and
validation-cache fingerprint are unchanged; otherwise use a new output path.

The review ZIP contains JSON metrics, the selection lock, provenance and
per-query/summary CSVs for the source and selected checkpoint. It deliberately
excludes large `.pt` checkpoints and image/feature caches.
