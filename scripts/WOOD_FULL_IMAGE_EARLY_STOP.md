# Fractional-epoch full-image fine-tuning diagnosis

This is a targeted follow-up to the full-image ablation, not a new test-set
comparison. It starts every run from the same selected DINOv2-S
`prototype_large` source checkpoint and uses seed 43, all 124,577 meta-train
queries, the same scan-disjoint positive-reference plan, the same 557-class
ArcFace classifier, batch 64, direct gradient path, and identical batch order.

The four runs are ArcFace and ArcFace+SupCon, each with constant LR or a cosine
decay from the original LR to 10% of it over one epoch. Both backbone and head
LRs are multiplied by the same schedule. Source (step 0), 25%, 50%, and 100%
of the epoch are evaluated on the same two scan-disjoint meta-val folds at
24x5, 57x5, 57x1, and 637x1. The locked selection score is the unweighted
mean R@1 of 57x5, 57x1, and 637x1; 24x5 is a retention diagnostic. The actual
batch number and unique-query coverage accompany each measurement. The best
checkpoint is chosen *only* on meta-val. No public-ID, meta-test, or full-954
images enter training or selection.

The constant arm should reproduce the earlier epoch-1 result within normal
backend nondeterminism. A cosine improvement would not by itself distinguish
the schedule shape from a lower integrated LR; that would require a later
constant-low-LR control. The initial source checkpoint was itself selected on
meta-val, so this is exploratory diagnosis, not independent evidence of a
submission claim. Each completed arm/schedule is verified and skipped on
rerun; an interrupted arm restarts from the source rather than claiming exact
mid-epoch resume. Checkpoints stay on Drive. The ZIP contains JSON reports,
not weights.

```python
%cd /content/drive/MyDrive/NCS
!git -C swid_retrieval pull

import os, sys, runpy
sys.path.insert(0, "/content/drive/MyDrive/NCS")
for name in list(sys.modules):
    if name.startswith("swid_retrieval"):
        del sys.modules[name]
for key in list(os.environ):
    if key.startswith(("RUN_", "SKIP_")):
        os.environ.pop(key, None)

os.environ.update({
    "ROOT_PATH": "/content/drive/MyDrive/NCS",
    "DEVICE": "cuda",
    "RUN_WOOD_FULL_IMAGE_EARLY_STOP": "1",
    "WOOD_EARLY_MODE": "preflight",
    "WOOD_EARLY_OUT": "/content/drive/MyDrive/NCS/results/wood_full_image_early_stop_v1",
    "WOOD_EARLY_ARMS": "arcface,arcface_supcon",
    "WOOD_EARLY_SCHEDULES": "constant,cosine",
    "WOOD_EARLY_SEED": "43",
    "WOOD_FULL_SOURCE_OUT": "/content/drive/MyDrive/NCS/results/dinov2s_retrieval_matched_v1",
    "WOOD_FULL_SOURCE_SEED": "43",
    "WOOD_FULL_ARMS": "arcface,arcface_supcon",
    "WOOD_FULL_SEEDS": "43",
    "WOOD_FULL_EPOCHS": "1",
    "WOOD_FULL_BATCH": "64",
    "WOOD_FULL_MICROBATCH": "32",
    "WOOD_FULL_STEP_MODE": "direct",
    "WOOD_FULL_WORKERS": "2",
    "WOOD_FULL_BACKBONE_LR": "2e-6",
    "WOOD_FULL_HEAD_LR": "1e-4",
    "WOOD_FULL_CONTRASTIVE_WEIGHT": "0.2",
    "WOOD_FULL_CONTRASTIVE_TEMPERATURE": "0.07",
    "WOOD_FULL_PRELOAD": "1",
    "WOOD_FULL_PRELOAD_WORKERS": "8",
})
_ = runpy.run_module("swid_retrieval.run_overnight", run_name="__main__")
```

```python
for mode in ("train", "export"):
    os.environ["WOOD_EARLY_MODE"] = mode
    _ = runpy.run_module("swid_retrieval.run_overnight", run_name="__main__")

from google.colab import files
files.download(os.environ["WOOD_EARLY_OUT"] + "/wood_full_image_early_stop_review.zip")
```

Do not change the recipe or source checkpoint inside the output folder. A
completed variant is skipped on rerun; keep the same environment values.
