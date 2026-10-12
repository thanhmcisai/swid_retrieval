# Full-image ArcFace / QKV study

This isolated experiment uses **every one of the 124,577 meta-train images as a
query once per epoch**. It does not read public-ID, meta-test, or the 954-label
CE training pool. Both arms start from the same DINOv2-S retrieval checkpoint,
use the same class-balanced full-coverage batches and the same paired reference
plan. A positive reference is a different image of the same species; another
source scan is preferred and a different crop scale is preferred when available.
Species with only one source scan necessarily use a same-scan positive; this is
counted in the epoch audit.

`arcface` optimizes the shared encoder with a 557-way angular-margin classifier.
`arcface_qkv` adds the existing `WoodCorrespondence` scorer's in-batch QKV loss.
Its query and reference images both backpropagate into the encoder. The QKV
candidate set is **the distinct species in the current batch**, not all 557
species. The ArcFace classifier supplies full-557 discrimination in both arms.
There is no stale feature bank in this study. Validation uses two scan-disjoint
meta-val folds at 57x5, 57x1 and 637x1; QKV reranks at most 128 candidates in
the 637x1 condition. The 954-species test is not touched by this runner.

Run from Colab after Drive mount:

```python
%cd /content/drive/MyDrive/NCS
!git -C swid_retrieval pull

import os, sys, runpy
sys.path.insert(0, "/content/drive/MyDrive/NCS")
for name in list(sys.modules):
    if name.startswith("swid_retrieval"):
        del sys.modules[name]
for key in list(os.environ):
    if key.startswith("RUN_") or key.startswith("SKIP_"):
        os.environ.pop(key, None)
os.environ.update({
    "ROOT_PATH": "/content/drive/MyDrive/NCS",
    "DEVICE": "cuda",
    "RUN_WOOD_FULL_IMAGE_TRAIN": "1",
    "WOOD_FULL_MODE": "preflight",
    "WOOD_FULL_OUT": "/content/drive/MyDrive/NCS/results/wood_full_image_study_v1",
    "WOOD_FULL_SOURCE_OUT": "/content/drive/MyDrive/NCS/results/dinov2s_retrieval_matched_v1",
    "WOOD_FULL_SOURCE_SEED": "43",
    "WOOD_FULL_ARMS": "arcface,arcface_qkv",
    "WOOD_FULL_SEEDS": "43",
    "WOOD_FULL_EPOCHS": "1",
    "WOOD_FULL_BATCH": "64",
    "WOOD_FULL_MICROBATCH": "16",
    "WOOD_FULL_WORKERS": "2",
    "WOOD_FULL_PRELOAD": "1",
    "WOOD_FULL_PRELOAD_WORKERS": "8",
})
_ = runpy.run_module("swid_retrieval.run_overnight", run_name="__main__")
```

Then run the training cell:

```python
os.environ["WOOD_FULL_MODE"] = "train"
_ = runpy.run_module("swid_retrieval.run_overnight", run_name="__main__")
```

First run one epoch to observe throughput and validation. To continue, change
`WOOD_FULL_EPOCHS` to `3` and rerun the training cell. Completed epochs are
resumed from `latest.pt`; batch size, microbatch, source checkpoint and loss
weights are locked by checkpoint provenance. Each arm has `best.pt`,
`latest.pt`, `epoch_XX.json`; `provenance.json` records the source and manifest.
Do not cite this as an independent final result until multiple seeds and the
locked meta-test/full-gallery evaluation are run.
