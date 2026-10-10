# Image-based correspondence training

The previous correspondence pilot fitted a scorer to cached features from a
small representative subset. Its `train_joint` option also used that subset.
This runner instead samples original images from the meta-train manifest. It
updates the DINOv2-S image encoder and the QKV correspondence scorer jointly.
It does **not** train a foundation model from random initialization: the encoder
starts from the selected `prototype_large` DINOv2-S checkpoint.

Episodes use support/query images from different SmartWoodID source scans.
The schedule includes K=1, 2, 5 and 16, 32, 64 candidate species. The K=5
episodes use at most 32 species to keep a T4 run feasible. Species with only
one source scan cannot be used in this strict retrieval loss. The `preflight`
and per-epoch files disclose eligible species/images and actual image coverage;
do not claim that all 124,577 images were trained on unless coverage says so.
The source checkpoint's earlier supervised training is a separate phase.

No meta-test or public-ID/OOD endpoint is used in this runner. Checkpoints are
selected on meta-val fold 0 using the mean of 57-species K=5 and 637-species
K=1 macro R@1. The baseline, raw global score with the trained encoder, and
correspondence score are all recorded. Keep fold 1 and meta-test sealed until
the approach and hyperparameters are frozen.

## Colab

Run from `/content/drive/MyDrive/NCS` after mounting Drive. Clear leftover
mode flags from the current notebook kernel before `runpy`:

```python
%cd /content/drive/MyDrive/NCS
!git -C swid_retrieval pull

import os, sys, runpy
sys.path.insert(0, "/content/drive/MyDrive/NCS")
for name in list(sys.modules):
    if name.startswith("swid_retrieval"):
        del sys.modules[name]
for key in list(os.environ):
    if key.startswith("RUN_") or key.startswith("WOOD_IMAGE_"):
        os.environ.pop(key, None)

os.environ.update({
    "ROOT_PATH": "/content/drive/MyDrive/NCS",
    "DEVICE": "cuda",
    "RUN_WOOD_CORRESPONDENCE_IMAGE_TRAIN": "1",
    "WOOD_IMAGE_OUT": "/content/drive/MyDrive/NCS/results/wood_correspondence_image_v1",
    "WOOD_IMAGE_BASE_SEED": "43",
    "WOOD_IMAGE_SEED": "43",
    "WOOD_IMAGE_VARIANT": "qkv",
    "WOOD_IMAGE_MODE": "preflight",
})
_ = runpy.run_module("swid_retrieval.run_overnight", run_name="__main__")
```

First run a small image-based pilot in a *separate output directory*; it is
still real joint training, just fewer episodes:

```python
os.environ.update({
    "WOOD_IMAGE_OUT": "/content/drive/MyDrive/NCS/results/wood_correspondence_image_pilot_v1",
    "WOOD_IMAGE_MODE": "train",
    "WOOD_IMAGE_EPOCHS": "1",
    "WOOD_IMAGE_EPISODES": "8",
    "WOOD_IMAGE_MICROBATCH": "4",
    "WOOD_IMAGE_WORKERS": "2",
    "WOOD_IMAGE_PRINT_EVERY": "1",
    "WOOD_IMAGE_ALL_BLOCKS": "1",
    # Set to "1" in a fresh Colab runtime if /content/cache_images is empty.
    "WOOD_IMAGE_PRELOAD": "0",
})
_ = runpy.run_module("swid_retrieval.run_overnight", run_name="__main__")
```

If the pilot finishes with finite loss, nonzero encoder/QKV gradients and
`best.pt`, use a new output directory for the full schedule:

```python
os.environ.update({
    "WOOD_IMAGE_OUT": "/content/drive/MyDrive/NCS/results/wood_correspondence_image_full_v1",
    "WOOD_IMAGE_MODE": "train",
    "WOOD_IMAGE_EPOCHS": "5",
    "WOOD_IMAGE_EPISODES": "160",
    "WOOD_IMAGE_PRINT_EVERY": "10",
})
_ = runpy.run_module("swid_retrieval.run_overnight", run_name="__main__")
```

Evaluate the selected checkpoint on the second meta-val fold, without changing
the training configuration or selecting on that fold:

```python
os.environ["WOOD_IMAGE_MODE"] = "validate"
_ = runpy.run_module("swid_retrieval.run_overnight", run_name="__main__")
```

The full call resumes at the next epoch when invoked again with identical
configuration. It cannot continue a partially completed epoch. Training
speed and coverage are reported; a five-epoch schedule does not guarantee
full image coverage. Run a separate `global_only` control in its own output
directory with `WOOD_IMAGE_VARIANT="global_only"` and the same remaining
configuration. Compare matched validation rows, not the training accuracies.

`WOOD_IMAGE_PRELOAD=1` copies eligible training and fold-0 validation images to
`/content/cache_images` before training. This can take tens of minutes but
avoids repeated Google Drive reads; keep it at `0` when that cache is already
populated in the current Colab runtime.
