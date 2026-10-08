# Gallery-adaptive end-to-end study (separate from the manuscript run)

This is a prospective method experiment, not an update to the submitted
numbers. It trains all DINOv2 ViT-B/14 backbone parameters by default, with a
small projection and a gallery-conditioned class scorer. A no-grad feature pass
followed by microbatch gradient replay keeps full-backbone fine-tuning feasible
on a 16 GB T4. The backbone is **not frozen** except for the named ablation.
The DINOv2 source commit is pinned in `gallery_experiment.py`; the loaded
pretrained backbone's tensor hash is recorded in each checkpoint.

## Protocol

- Train on SmartWoodID `meta-train` only; choose the best epoch by the mean of
  24-way and 80-way `meta-val` macro R@1. Public images are never used to
  select weights or hyperparameters.
- The training objective contrasts an old-only, one-shot gallery with an
  expanded, multi-shot gallery containing new species. It includes an explicit
  old-species expansion margin. New-episode species act as pseudo-OOD queries
  against the old gallery for a separate distance-margin loss; no public OOD
  labels enter training. Genus-related negatives can be sampled from
  training labels. The learned scorer combines prototypes with per-class
  nearest and count-normalized log-mean-exp evidence. Every gallery class has
  prototype fallback even when absent from the global top-M image candidates.
- `fixed_episode`, `without_expansion_loss`, `without_pseudo_ood`, `without_hard_negatives`,
  `prototype_only`, `fixed_evidence`, `without_count_normalization`, `frozen_encoder` are
  ablations. `supcon_finetuned` and `arcface_finetuned` train the *same*
  backbone/embedding size on the same episodes and 224-pixel images.
  `dinov2_pretrained` is an additional same-resolution zero-shot control.
- The full evaluation reads the corrected public CSVs directly, requires the
  verified public-row cache for label/order audit, and outputs separate
  24-species and 954-species gallery results, a fixed-reference
  gallery-cardinality curve, pooled and within-source OOD, public-only K-shot,
  old/new accuracy after adding 50 species, and SWI-to-VN26 and cross-
  magnification retrieval. VN26 vectors already present in the corrected public
  rows are reused; missing vectors are extracted and cached separately.
- Public data and earlier versions of this benchmark have already informed
  research decisions. Treat this suite as exploratory until an untouched
  external cohort or newly locked test split is available. Do not select a
  variant on public results and then call those same results confirmatory.

## Colab setup

Use a GPU runtime. Mount Drive and ensure `swi_manifest.json`, the corrected
`ID_images_expanded.csv` and `OOD_images_expanded.csv`, and
`embedding_cache_full954_v6_public_row_verified.npz` exist under the root.
The study writes only to `results/gallery_adaptive_study_*` by default; it
does not overwrite the paper cache or figures. Install the project's existing
dependencies, including `torch`, `torchvision`, `numpy`, `pandas`, `opencv-python`,
`albumentations`, `scikit-learn`, and `timm` if the Colab runtime lacks them.

Run this setup cell once (or again after a `git pull`):

```python
%cd /content/drive/MyDrive/NCS
!git -C swid_retrieval pull --ff-only

import os, sys, runpy
sys.path.insert(0, "/content/drive/MyDrive/NCS")
for name in list(sys.modules):
    if name.startswith("swid_retrieval"):
        del sys.modules[name]

os.environ.update({
    "ROOT_PATH": "/content/drive/MyDrive/NCS",
    "DEVICE": "cuda",
    "IMAGE_CACHE_DIR": "/content/cache_images",
    "RUN_GALLERY_STUDY": "1",
    "RUN_REPAIR_PUBLIC_ROWS": "0",
    "RUN_FINAL_SCURD_RETRAIN": "0",
    "RUN_FINAL_COLAB_AUDIT": "0",
    "GALLERY_STUDY_OUT": "/content/drive/MyDrive/NCS/results/gallery_adaptive_study_full",
    "GALLERY_STUDY_WORKERS": "4",
    "GALLERY_STUDY_MICROBATCH": "8",
    "GALLERY_STUDY_IMAGE_BATCH": "32",
})
```

Then run a small synthetic GPU test **without** reading images:

```python
os.environ["GALLERY_STUDY_MODE"] = "smoke"
_ = runpy.run_module("swid_retrieval.run_overnight", run_name="__main__")
```

Check one actual DINOv2 backbone update (downloads pretrained weights once):

```python
os.environ["GALLERY_STUDY_MODE"] = "backbone_smoke"
_ = runpy.run_module("swid_retrieval.run_overnight", run_name="__main__")
```

Next, run a short real-data pilot in a **different output directory**. The
pilot's two-epoch checkpoints must not be passed off as the final model:

```python
os.environ.update({
    "GALLERY_STUDY_MODE": "pilot",
    "GALLERY_STUDY_OUT": "/content/drive/MyDrive/NCS/results/gallery_adaptive_study_pilot",
    "GALLERY_STUDY_VARIANTS": "gallery_adaptive,fixed_episode,supcon_finetuned",
    "GALLERY_STUDY_SEEDS": "42",
    "GALLERY_STUDY_PRELOAD": "0",
})
_ = runpy.run_module("swid_retrieval.run_overnight", run_name="__main__")
```

Monitor peak VRAM during the pilot. On a smaller GPU, reduce
`GALLERY_STUDY_MICROBATCH` from 8 to 4 and/or `GALLERY_STUDY_IMAGE_BATCH`
from 32 to 16 before full training. A changed recipe cannot resume an
existing checkpoint in the same output directory.

For full training, use the original `gallery_adaptive_study_full` output
directory. Train one configuration at a time so each epoch checkpoint can be
resumed after a disconnect. First run the proposed method across three seeds:

```python
os.environ.update({
    "GALLERY_STUDY_MODE": "train",
    "GALLERY_STUDY_OUT": "/content/drive/MyDrive/NCS/results/gallery_adaptive_study_full",
    "GALLERY_STUDY_VARIANTS": "gallery_adaptive",
    "GALLERY_STUDY_SEEDS": "42,43,44",
    "GALLERY_STUDY_PRELOAD": "1",
})
_ = runpy.run_module("swid_retrieval.run_overnight", run_name="__main__")
```

Then change `GALLERY_STUDY_VARIANTS` to each of these and rerun the same cell:
`fixed_episode`, `without_expansion_loss`, `without_pseudo_ood`, `without_hard_negatives`,
`prototype_only`, `fixed_evidence`, `without_count_normalization`, `frozen_encoder`,
`supcon_finetuned`, `arcface_finetuned`, `dinov2_pretrained`. Use seed 42 for
screening; repeat promising configurations with 43 and 44. A complete 9-variant
by 3-seed grid is a large GPU experiment, not a single overnight T4 run.

For final evaluation, keep exactly the same output directory and variant/seed
selection as training:

```python
os.environ.update({
    "GALLERY_STUDY_MODE": "evaluate",
    "GALLERY_STUDY_VARIANTS": "gallery_adaptive,fixed_episode,supcon_finetuned,arcface_finetuned",
    "GALLERY_STUDY_SEEDS": "42",
    "GALLERY_STUDY_PRELOAD": "0",
    "GALLERY_STUDY_KSHOT_REPEATS": "30",
    "GALLERY_STUDY_EVAL_VN26": "1",
})
_ = runpy.run_module("swid_retrieval.run_overnight", run_name="__main__")
```

Only list variants/seeds for which `best.pt` exists. Evaluation extracts new
224-pixel embeddings from all SWI and corrected public images; the `.npy`
arrays and progress files under each seed directory are resumable and may be
large. `comparison.csv` is a cross-variant summary; every `evaluation.json`
records hashes and per-source results. A model at 224 pixels must be compared
with the 224-pixel controls here. The old 518-pixel manuscript DINOv2 results
are contextual, not a resolution-matched causal comparison. The existing
CE-Full model also trained on the public-ID species' SmartWoodID labels,
unlike this meta-train-only study; present that comparison with the asymmetry
explicitly stated.

Run the synthetic tests on Colab before the long training:

```python
!python -m unittest swid_retrieval.tests.test_gallery_method -v
```
