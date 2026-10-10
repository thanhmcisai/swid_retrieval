# Matched global-only control for image-trained correspondence

The QKV image-trained checkpoint is preserved. This control starts from the
same selected DINOv2-S encoder checkpoint and uses the same seed, image
episodes, augmentation, optimizer rates and epoch count. Its retrieval loss
uses global prototype scoring instead of local QKV correspondence. This is a
matched *training-objective ablation*, not a freeze-encoder baseline.

The comparison reads the selected meta-val reports for both runs. It checks
checkpoint hashes, source code and manifest identity, recipe compatibility,
per-epoch sampling coverage, and exact query identity. Paired 95% intervals
resample species and are conditional on the selected runs. They do not account
for training-seed or model-selection variance. No meta-test is opened.
The shared base encoder had already been selected with both meta-val folds,
so fold 1 is descriptive rather than an independent test of the full pipeline.

## Colab cells

After mounting Drive, update the repository and clear stale mode flags:

```python
%cd /content/drive/MyDrive/NCS
!git -C swid_retrieval pull

import os, sys, runpy
sys.path.insert(0, "/content/drive/MyDrive/NCS")
for name in list(sys.modules):
    if name.startswith("swid_retrieval"):
        del sys.modules[name]
for key in list(os.environ):
    if key.startswith(("RUN_", "WOOD_IMAGE_", "WOOD_COMPARE_")):
        os.environ.pop(key, None)

os.environ.update({
    "ROOT_PATH": "/content/drive/MyDrive/NCS",
    "DEVICE": "cuda",
    "RUN_WOOD_CORRESPONDENCE_IMAGE_TRAIN": "1",
    "WOOD_IMAGE_BASE_SEED": "43",
    "WOOD_IMAGE_SEED": "43",
    "WOOD_IMAGE_VARIANT": "global_only",
    "WOOD_IMAGE_MODE": "train",
    "WOOD_IMAGE_OUT": "/content/drive/MyDrive/NCS/results/wood_correspondence_global_only_v1",
    "WOOD_IMAGE_EPOCHS": "5",
    "WOOD_IMAGE_EPISODES": "160",
    "WOOD_IMAGE_MICROBATCH": "4",
    "WOOD_IMAGE_WORKERS": "2",
    "WOOD_IMAGE_PRINT_EVERY": "10",
    "WOOD_IMAGE_ALL_BLOCKS": "1",
    # Use "1" in a fresh runtime when /content/cache_images is empty.
    "WOOD_IMAGE_PRELOAD": "0",
})
_ = runpy.run_module("swid_retrieval.run_overnight", run_name="__main__")
```

The control resumes from the next completed epoch when rerun unchanged. Once
training finishes, validate its selected checkpoint on both meta-val folds:

```python
os.environ["WOOD_IMAGE_MODE"] = "validate"
_ = runpy.run_module("swid_retrieval.run_overnight", run_name="__main__")
```

Run the paired comparison separately on CPU. It validates the old QKV report
against the control; it will fail if either run has incompatible provenance:

```python
os.environ.update({
    "RUN_WOOD_CORRESPONDENCE_IMAGE_TRAIN": "0",
    "RUN_WOOD_CORRESPONDENCE_COMPARE": "1",
    "WOOD_COMPARE_QKV_OUT": "/content/drive/MyDrive/NCS/results/wood_correspondence_image_full_v1",
    "WOOD_COMPARE_GLOBAL_OUT": "/content/drive/MyDrive/NCS/results/wood_correspondence_global_only_v1",
    "WOOD_COMPARE_OUT": "/content/drive/MyDrive/NCS/results/wood_correspondence_matched_comparison_v1",
    "WOOD_COMPARE_SEED": "43",
})
_ = runpy.run_module("swid_retrieval.run_overnight", run_name="__main__")
```

For review, only the control's `preflight.json`, `pool_audit.json`,
`epoch_*.json`, `baseline_meta_val_summary.csv`, `selected_meta_val/selected_*`
and the comparison's `paired_summary.csv`, `paired_queries.csv`,
`comparison_manifest.json` are needed. Do not transfer `.pt` or `.npz` unless
the provenance audit fails.
