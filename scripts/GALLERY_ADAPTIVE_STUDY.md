# End-to-end metric retrieval study (separate from the manuscript run)

This is a prospective method experiment, not an update to the submitted
numbers. Choose a pretrained DINOv2 ViT-B/14 or ViT-S/14, a pretrained
ConvNeXt-Tiny, or `WoodPatternNet` trained **from random initialization**.
The latter combines depthwise local texture blocks at three resolutions,
cross-scale token attention, and dispersion pooling. These are hypotheses
about wood texture, not validated anatomical-feature detectors or an
established novel method. A no-grad feature pass followed by microbatch
gradient replay trains the whole backbone on a 16 GB T4. The backbone is
**not frozen** except for the named ablation. Initial backbone tensor hashes,
code hashes, checkpoint selection and exact config are recorded.

The primary variant is `metric_retrieval`: train the image encoder and its
embedding projection end-to-end with cross-entropy over the species represented
in an episode gallery. At inference, each class score is the **exact maximum
cosine similarity to any reference image of that class**. There is no trained
post-embedding scoring network; the optional gate is absent from its checkpoint.
`prototype_retrieval` is a nonparametric
prototype control. `metric_expansion` adds old-gallery and expansion/OOD
margins to the same nearest-image scorer. `gallery_adaptive` retains the
learned evidence gate only as an ablation. The historical SC-URD pipeline and
the existing manuscript are not modified by this study.

For numerical stability, the backbone may use CUDA AMP, but embedding
projection/normalization, gallery scores and the loss run in float32. The
runner rejects non-finite validation embeddings/scores and records episode
training R@1 plus AMP skipped steps. A decreasing meta-validation R@1 near
chance is a failed training hypothesis, even if no exception is raised.

## Protocol

- Train on SmartWoodID `meta-train` only. Recover original scan IDs from
  patch filenames (`_from_Tw...`) and require support and query patches to
  come from different scans. Species with only one source scan are excluded
  from training episodes. Choose the best epoch by the mean of 24-way and
  all eligible scan-disjoint `meta-val` macro R@1. The old split CSV suggests
  approximately 57/80 validation species have at least two scans; the
  preflight reports the actual patch-derived count. Public images are never
  used to select weights or hyperparameters.
- `metric_retrieval` optimizes only gallery-conditioned species cross-entropy
  over exact nearest-reference scores. `metric_expansion` additionally contrasts
  old-only and expanded galleries, with an old-species margin and a pseudo-OOD
  distance margin. New-episode species supply pseudo-OOD queries; no public
  OOD images enter training. Genus-related negatives can be sampled from
  training labels. The learned gate in `gallery_adaptive` combines prototype,
  nearest and count-normalized top-M evidence, but is not part of the primary
  method.
- Each episode has 4/8/16 active classes and two live support images per class.
  A FIFO memory of up to 256 **detached meta-train class prototypes** adds
  distractor species after 80 distinct classes have been seen. The primary
  nearest-reference method considers every current support and memory vector;
  `top_m` does not apply to it. Current support/query images and the backbone
  retain gradients; stale memory prototypes do not. Training logs show the
  maximum gallery size and report `topM=0` for exact nearest retrieval.
  This is an approximation to larger galleries, not a full gradient
  through all 954 reference classes; meta-train contains 557 species, fewer
  of which have two or more source scans.
- `metric_expansion` isolates the effect of old-gallery and margin losses
  while keeping exact nearest retrieval. `gallery_adaptive`, `fixed_episode`,
  `without_expansion_loss`, `without_pseudo_ood`, `without_hard_negatives`,
  `prototype_only`, `fixed_evidence`, `without_count_normalization`,
  `without_memory`, `memory_512`, `top_m_32`, `top_m_128`, `frozen_encoder` are
  mostly ablations of the learned-scorer family; `without_memory` and
  `frozen_encoder` also apply to the primary method when configured explicitly.
  The
  `top_m_128` experiment warms up to 160 memory classes so truncation can be
  exercised. `woodpattern_no_attention` and
  `woodpattern_single_scale` isolate the custom encoder's key components.
  `supcon_finetuned` and `arcface_finetuned` train the *same* selected
  backbone/embedding size on the same episodes and 224-pixel images. SupCon
  receives the same detached memory negatives; ArcFace has a 557-class head.
  `dinov2_pretrained` is an additional ViT-B same-resolution zero-shot control.
- The full evaluation reads the corrected public CSVs directly, requires the
  verified public-row cache for label/order audit, and outputs separate
  24-species and 954-species gallery results, a fixed-reference
  gallery-cardinality curve, pooled and within-source OOD, public-only K-shot,
  old/new accuracy after adding 50 species, and SWI-to-VN26 and cross-
  magnification retrieval. VN26 vectors already present in the corrected public
  rows are reused; missing vectors are extracted and cached separately.
  Each main gallery result reports class macro R@1 and image-embedding macro
  mAP@100/MRR@100. The latter are truncated image rankings, not full-gallery
  mAP, and are not the objective optimized by species cross-entropy. AP@100 uses
  `min(number of relevant gallery images, 100)` as its denominator and is
  macro-averaged over query species.
- Public data and earlier versions of this benchmark have already informed
  research decisions. Treat this suite as exploratory until an untouched
  external cohort or newly locked test split is available. Do not select a
  variant on public results and then call those same results confirmatory.
- Selection is the predeclared average of 24-way and all eligible
  scan-disjoint SWI meta-val
  macro R@1. The 954-species stress test is held for evaluation, not used to
  choose a model. Full untruncated retrieval mAP, uncertainty intervals and an
  untouched independent cohort still require separate work before a new
  general-superiority claim can be defended.
- Source-scan disjointness for SWI patches is checked from the filename.
  This is **not** necessarily donor-tree disjointness; multiple scans may
  depict the same wood specimen. Public K-shot specimen IDs are unavailable,
  so that split is also unverified at the specimen level. The evaluation
  JSON marks this limitation. Pooled OOD uses nearest-image distance, not
  class scoring; do not attribute its AUROC gain to the learned gate.

## Diagnostic after unstable scratch training

Do not resume a run that produced non-finite loss or near-chance validation
after many epochs. Checkpoint signatures include code and training config, so
a stability fix requires a fresh output directory. After reusing the warmed
`/content/cache_images`, train one seed for five epochs with and without the
detached prototype memory, keeping all other settings fixed:

```python
os.environ.update({
    "GALLERY_STUDY_MODE": "train",
    "GALLERY_STUDY_OUT": "/content/drive/MyDrive/NCS/results/metric_retrieval_fp32_diagnostic",
    "GALLERY_STUDY_BACKBONE": "woodpattern_tiny",
    "GALLERY_STUDY_VARIANTS": "metric_retrieval,metric_no_memory",
    "GALLERY_STUDY_SEEDS": "42",
    "GALLERY_STUDY_EPOCHS": "5",
    "GALLERY_STUDY_EPISODES_PER_EPOCH": "500",
    "GALLERY_STUDY_MICROBATCH": "48",
    "GALLERY_STUDY_BACKBONE_LR": "5e-4",
    "GALLERY_STUDY_PRELOAD": "0",
})
_ = runpy.run_module("swid_retrieval.run_overnight", run_name="__main__")
```

Inspect `train_episode_r1`, `val24`, `val57`, `ref_spread`, `amp_skips` and
whether both variants remain finite. `ref_spread` is the Euclidean norm of
the per-coordinate standard deviation of the meta-validation reference
embeddings; values near zero indicate collapse. Training R@1 and loss are
not directly comparable between memory/no-memory recipes because they have
different numbers of competitor classes. If validation stays near chance for
both, stop and revisit the scratch-encoder training recipe rather than adding
epochs or choosing a checkpoint on public test results.

## Supervised warm-up of the custom encoder

The scratch WoodPatternNet/no-memory pilot had substantially lower scan-disjoint
meta-validation retrieval than a frozen DINOv2 control under both nearest and
prototype scoring. Before another long metric run, use an isolated supervised
warm-up on **meta-train only**. This stage trains the custom encoder, its
projection, and a temporary 557-way classifier with cross-entropy. Per-image
sampling uses inverse-square-root species-frequency weights. The classifier is
discarded after warm-up; `best.pt` is selected by nearest-image retrieval on
the same scan-disjoint meta-val protocol. No public images are read.

```python
os.environ.update({
    "RUN_GALLERY_STUDY": "1",
    "RUN_REPAIR_PUBLIC_ROWS": "0",
    "RUN_FINAL_SCURD_RETRAIN": "0",
    "RUN_FINAL_COLAB_AUDIT": "0",
    "GALLERY_STUDY_MODE": "warmup",
    "GALLERY_STUDY_OUT": "/content/drive/MyDrive/NCS/results/woodpattern_warmup_pilot",
    "GALLERY_STUDY_BACKBONE": "woodpattern_tiny",
    "GALLERY_STUDY_VARIANTS": "supervised_warmup",
    "GALLERY_STUDY_SEEDS": "42",
    "GALLERY_STUDY_WARMUP_EPOCHS": "5",
    "GALLERY_STUDY_WARMUP_STEPS": "300",
    "GALLERY_STUDY_WARMUP_BATCH": "64",
    "GALLERY_STUDY_WARMUP_LR": "5e-4",
    "GALLERY_STUDY_PRELOAD": "0",
    "GALLERY_STUDY_PILOT_CHECKPOINTS": "0",
    "GALLERY_STUDY_INIT_CHECKPOINT": "",
})
_ = runpy.run_module("swid_retrieval.run_overnight", run_name="__main__")
```

Set `GALLERY_STUDY_PRELOAD=1` if the Colab runtime restarted and its local
`/content/cache_images` no longer contains the SmartWoodID images.

Inspect `val24`, `val57`, `ref_spread`, training accuracy, and AMP skips before
fine-tuning. Do not assume warm-up is successful because training accuracy
increases. `best_validation.json` records the selected epoch without loading
the PyTorch checkpoint. If it is clearly better than the previous no-memory best
(`val24=0.0917`, `val57=0.0456`) but still below the frozen DINOv2 nearest
control (`0.2417`, `0.2456`), it remains an exploratory candidate, not a
superior method. If it does not improve meta-val, stop here.

For a subsequent **separate** metric pilot, set `GALLERY_STUDY_MODE=train`,
`GALLERY_STUDY_VARIANTS=metric_no_memory`, and
`GALLERY_STUDY_INIT_CHECKPOINT` to the warm-up `best.pt`. Use a new
`GALLERY_STUDY_OUT` so the uninitialized checkpoints are never reused. The
runner checks the warm-up seed, backbone, embedding dimension, manifest hash,
and checkpoint file hash before training. The metric run evaluates and saves
the warm-up initialization as epoch 0, so fine-tuning is selected only if it
improves the pre-fine-tuning meta-validation score.
`GALLERY_STUDY_HEAD_LR` separately controls the projection/scorer learning
rate (default `3e-4`); record it alongside `GALLERY_STUDY_BACKBONE_LR` when
testing conservative fine-tuning from a warm-up checkpoint.

## Colab setup

Use a GPU runtime. Mount Drive and ensure `swi_manifest.json`, the corrected
`ID_images_expanded.csv` and `OOD_images_expanded.csv`, and
`embedding_cache_full954_v6_public_row_verified.npz` exist under the root.
Use a new `results/metric_retrieval_study_*` output directory; it
does not overwrite the paper cache or figures. Install the project's existing
dependencies, including `torch`, `torchvision`, `numpy`, `pandas`, `opencv-python`,
`albumentations`, `scikit-learn`, and `timm` if the Colab runtime lacks them.
Do not point this code at checkpoints or `evaluation.json` from the earlier
gallery study: the model and result schemas changed, and the runner checks
their code hashes.

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
    "GALLERY_STUDY_OUT": "/content/drive/MyDrive/NCS/results/metric_retrieval_study_full",
    "GALLERY_STUDY_BACKBONE": "woodpattern_tiny",
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

Before spending GPU time, audit the real manifest's source-scan IDs:

```python
os.environ["GALLERY_STUDY_MODE"] = "preflight"
_ = runpy.run_module("swid_retrieval.run_overnight", run_name="__main__")
```

Inspect `source_scan_audit.json`. If patch names do not encode source scans,
stop and supply verified source-scan metadata; do not silently fall back to
patch-level support/query splitting.

Check one actual `WoodPatternNet` backbone update (no pretrained download):

```python
os.environ["GALLERY_STUDY_MODE"] = "backbone_smoke"
_ = runpy.run_module("swid_retrieval.run_overnight", run_name="__main__")
```

Smoke tests retry initial CUDA AMP skipped steps and report `amp_skips`; a
failure after eight attempts reports the last gradient norm and loss scale.
Training logs and `progress.json` also record skipped optimizer steps. Do not
interpret an epoch with every optimizer step skipped as successful training.

Set `GALLERY_STUDY_BACKBONE` to `dinov2_vits14`, `convnext_tiny`, or
`dinov2_vitb14` and repeat the backbone smoke test before training each
control. The corresponding pretrained weights are fetched on the first use.

Next, run a short real-data pilot in a **different output directory**. The
pilot's two-epoch checkpoints must not be passed off as the final model:

```python
os.environ.update({
    "GALLERY_STUDY_MODE": "pilot",
    "GALLERY_STUDY_OUT": "/content/drive/MyDrive/NCS/results/metric_retrieval_study_pilot",
    "GALLERY_STUDY_BACKBONE": "woodpattern_tiny",
    "GALLERY_STUDY_VARIANTS": "metric_retrieval,prototype_retrieval",
    "GALLERY_STUDY_SEEDS": "42",
    "GALLERY_STUDY_PRELOAD": "0",
})
_ = runpy.run_module("swid_retrieval.run_overnight", run_name="__main__")
```

Monitor peak VRAM during the pilot. On a smaller GPU, reduce
`GALLERY_STUDY_MICROBATCH` from 8 to 4 and/or `GALLERY_STUDY_IMAGE_BATCH`
from 32 to 16 before full training. A changed recipe cannot resume an
existing checkpoint in the same output directory.
On a cold `/content/cache_images`, use `GALLERY_STUDY_PRELOAD=1` once to
move the training/validation images from Drive; the initial copy takes time
but avoids repeatedly waiting on Drive during every episode. Compare
`train_s`, `epoch_s`, and `peak_gb` only after this one-time preload.
Inspect `progress.json`: `max_training_gallery_images` should grow after memory
warmup. `top_m_episodes=0` is expected for exact nearest retrieval, not a
failure. For the learned-scorer ablation, the cap should activate after
memory warmup. Repeat a short pilot with `dinov2_vits14` and
`convnext_tiny` by changing only `GALLERY_STUDY_BACKBONE`. Training/evaluation
outputs are namespaced by backbone, variant, and seed. All comparisons must
use the same image resolution, meta-train split, and evaluation protocol.

For full training, use the `metric_retrieval_study_full` output directory.
Train one configuration at a time so each epoch checkpoint can be resumed
after a disconnect. Begin with the custom-backbone candidate across three
seeds **only if its pilot is stable**:

```python
os.environ.update({
    "GALLERY_STUDY_MODE": "train",
    "GALLERY_STUDY_OUT": "/content/drive/MyDrive/NCS/results/metric_retrieval_study_full",
    "GALLERY_STUDY_VARIANTS": "metric_retrieval",
    "GALLERY_STUDY_BACKBONE": "woodpattern_tiny",
    "GALLERY_STUDY_SEEDS": "42,43,44",
    "GALLERY_STUDY_PRELOAD": "1",
})
_ = runpy.run_module("swid_retrieval.run_overnight", run_name="__main__")
```

For architecture controls, keep `GALLERY_STUDY_VARIANTS=metric_retrieval`
and change `GALLERY_STUDY_BACKBONE` to `woodpattern_no_attention`,
`woodpattern_single_scale`, `dinov2_vits14`, `convnext_tiny`, and
`dinov2_vitb14`. For loss/scorer ablations, restore `woodpattern_tiny`, then
change `GALLERY_STUDY_VARIANTS` to each of these and rerun the same cell:
`prototype_retrieval`, `metric_expansion`, `gallery_adaptive`,
`fixed_episode`, `without_expansion_loss`, `without_pseudo_ood`,
`without_hard_negatives`, `prototype_only`, `fixed_evidence`,
`without_count_normalization`, `without_memory`, `memory_512`, `top_m_32`,
`top_m_128`, `frozen_encoder`,
`supcon_finetuned`, `arcface_finetuned`. `dinov2_pretrained` requires
`dinov2_vitb14`. Use seed 42 for screening; repeat configurations needed
for the final claim with seeds 43 and 44. A full grid is a large GPU study,
not a single overnight T4 run. Scratch training uses a different, declared
backbone learning rate (5e-4) from pretrained ViT-B (1e-5); compare tuned
controls fairly and do not claim an architecture gain from a single recipe.

For final evaluation, keep exactly the same output directory and variant/seed
selection as training:

```python
os.environ.update({
    "GALLERY_STUDY_MODE": "evaluate",
    "GALLERY_STUDY_BACKBONE": "woodpattern_tiny",
    "GALLERY_STUDY_VARIANTS": "metric_retrieval,prototype_retrieval,metric_expansion,gallery_adaptive",
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
large. `comparison.csv` is a cross-variant/backbone summary; every `evaluation.json`
records hashes and per-source results. A model at 224 pixels must be compared
with the 224-pixel controls here. The old 518-pixel manuscript DINOv2 results
are contextual, not a resolution-matched causal comparison. The existing
CE-Full model also trained on the public-ID species' SmartWoodID labels,
unlike this meta-train-only study; present that comparison with the asymmetry
explicitly stated.

Run the synthetic tests on Colab before the long training:

```python
%cd /content/drive/MyDrive/NCS
!PYTHONPATH=/content/drive/MyDrive/NCS python -m unittest discover -s swid_retrieval/tests -p 'test_wood_gallery.py' -v
!PYTHONPATH=/content/drive/MyDrive/NCS python -m unittest discover -s swid_retrieval/tests -p 'test_gallery_method.py' -v
```
