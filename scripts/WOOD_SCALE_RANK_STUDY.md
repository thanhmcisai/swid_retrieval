# Multi-scale reference and rank-loss study

This is a **new, exploratory** experiment. It does not change the submission
manuscript or previously locked results. It starts from the verified
`wood_hard_shortlist_study_v1/train/random16/best.pt` checkpoint and trains the
encoder and QKV reranker end-to-end on SmartWoodID **meta-train only**.

## Questions tested

- Are multi-scale, scan-disjoint K=5/K=10 references in **every** training
  episode better than randomly sampled references at the same K and episode
  count? Balanced sampling covers every available support scale when there are
  at most K such scales, and otherwise maximizes distinct scales. All arms use
  identical species and query draws for each seed/epoch.
- Does cross-scale multi-positive contrastive loss add value after the support
  itself contains multiple scales?
- Does an extra differentiable pairwise inversion penalty help rank the true
  species first? The penalty grows with negative classes above the positive;
  an optional bounded weight also grows with observed positive rank. Observed
  rank is detached, so gradients come from score differences, not from the
  discontinuous `argsort` operation. This is an ablation, not a claim that a
  rank loss necessarily improves retrieval.
- Is training on top-32 sufficient, or must ranking be trained against top-128?
- Does reranking top-128 with one bank reference per negative class differ
  materially from using every available bank reference (up to four per class)?

All arms retain the same original episodic, bank and random-shortlist losses.
The new components are additive, with an explicit arm for each hypothesis.
The negative bank comes from meta-train only. During training the true class is
inserted into the shortlist; **validation does not inject the true class**.
The latter reports candidate recall as well as unconditional Top-1.

Selection is exclusively meta-val fold 0, K=10, raw top-128, QKV with all
selected gallery references. All epoch checkpoints are retained. Only after
`select` writes `selection_lock.json` does `describe` evaluate fold 1 and the
K=1/3/5/10, top-32/128 matrix. No public-ID or meta-test split is consumed.
Fold 1 has been inspected in earlier studies, so it is descriptive rather than
a pristine external test. For a final claim, evaluate the locked model on an
appropriate held-out set later; meta-val improvements alone are not a final
result.

## Colab setup cell

```python
%cd /content/drive/MyDrive/NCS
!git -C swid_retrieval pull

import os, sys, runpy
sys.path.insert(0, "/content/drive/MyDrive/NCS")
for name in list(sys.modules):
    if name.startswith("swid_retrieval"):
        del sys.modules[name]
for name in list(os.environ):
    if name.startswith("RUN_WOOD_") or name in {
        "RUN_FINAL_SCURD_RETRAIN", "RUN_FINAL_COLAB_AUDIT",
        "RUN_REPAIR_PUBLIC_ROWS", "RUN_GALLERY_STUDY",
        "RUN_GALLERY_DIAGNOSTICS",
    }:
        os.environ.pop(name, None)

os.environ.update({
    "ROOT_PATH": "/content/drive/MyDrive/NCS",
    "IMAGE_CACHE_DIR": "/content/cache_images",
    "RUN_WOOD_SCALE_RANK_STUDY": "1",
    "WOOD_SCALE_RANK_OUT": "/content/drive/MyDrive/NCS/results/wood_scale_rank_study_v1",
    "WOOD_SCALE_RANK_SEEDS": "43",
    "WOOD_SCALE_RANK_EPOCHS": "3",
    "WOOD_SCALE_RANK_EPISODES": "120",
    "WOOD_SCALE_RANK_WORKERS": "2",
    "WOOD_SCALE_RANK_PRELOAD": "1",
    "WOOD_SCALE_RANK_PRELOAD_WORKERS": "8",
    "WOOD_SCALE_RANK_IMAGE_BATCH": "64",
    "WOOD_SCALE_RANK_MICROBATCH": "4",
    "WOOD_SCALE_RANK_QUERIES": "2",
})

from pathlib import Path
root = Path(os.environ["ROOT_PATH"])
assert (root / "results/wood_hard_shortlist_study_v1/selection_lock.json").is_file()
assert (root / "results/wood_hard_shortlist_study_v1/train/random16/best.pt").is_file()
assert (root / "swi_manifest.json").is_file()
```

Run the cheap checks first:

```python
for mode in ("smoke", "preflight"):
    os.environ["WOOD_SCALE_RANK_MODE"] = mode
    _ = runpy.run_module("swid_retrieval.run_overnight", run_name="__main__")
```

Then run in the same runtime (or re-run the setup cell after a disconnect):

```python
for mode in ("train", "validate", "select", "describe", "export"):
    os.environ["WOOD_SCALE_RANK_MODE"] = mode
    _ = runpy.run_module("swid_retrieval.run_overnight", run_name="__main__")
```

The review archive is
`results/wood_scale_rank_study_v1_review.zip`. It has CSV/JSON only, no
embeddings or model weights. Download it after `export` completes:

```python
from google.colab import files
files.download("/content/drive/MyDrive/NCS/results/wood_scale_rank_study_v1_review.zip")
```

The full run has eight arms and multiple image-forward passes per episode;
the all-reference top-128 arm is particularly expensive. A low-cost pilot can
use `WOOD_SCALE_RANK_ARMS=random,balanced,balanced_rank128`,
`WOOD_SCALE_RANK_EPOCHS=1`, `WOOD_SCALE_RANK_EPISODES=20`, and a **different**
`WOOD_SCALE_RANK_OUT`. The pilot is a functionality/cost check, not a model
selection result. Do not change the arm list, epochs, episodes or learning
rates in an existing output folder; provenance checks intentionally reject it.

If a run is interrupted, execute the same mode again with the same settings.
Completed training epochs and validation files are reused. Preloading can be
disabled via `WOOD_SCALE_RANK_PRELOAD=0` only if local images are already
cached; it does not change the scientific configuration.
