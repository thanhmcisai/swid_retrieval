# Shortlist-mined wood retrieval study

This exploratory study starts every trainable arm from the same image-trained
DINOv2-S/QKV checkpoint. Both encoder and QKV receive image-level gradients.
It leaves the earlier multilevel checkpoints and results unchanged.

## Protocol

- `random16`: sample 15 negative species uniformly from the query's raw
  top-128 meta-train bank; one positive species is always included.
- `hard16`: use the 15 highest-scoring negative species in that pool.
- `hard16_cross_scale`: hard16 plus a scan-disjoint positive reference with a
  different source-scan crop width.
- `hard32_cross_scale`: same sampler, 31 hard negatives.
- The old `source` and `cross_scale` checkpoints are matched validation
  comparators for the evaluation protocol. They are not retrained or
  re-extracted for the main validation. Their training budgets differ from the
  new arms. Use `random16` versus `hard16` as the compute-matched mining
  contrast, and `hard16` versus `hard16_cross_scale` as the sampler contrast.
- Each episode retains the previous QKV, global and bank objectives, then
  adds QKV cross-entropy over selected candidate species. Negatives are
  cached meta-train images; the query and positive episode references carry
  encoder gradients. The bank refreshes each epoch. No meta-val or test
  images enter mining.
- Selection uses only the two scan-disjoint meta-val folds (57 target species,
  580 distractors, 285 queries/fold). The reference gallery and raw-prototype
  candidate rule match the prior multilevel study. The `scale_stress` mode is
  **descriptive only** and runs after selection is locked. It changes one
  reference per species according to the observed query crop width for *all*
  637 species; true labels are used only to calculate metrics and to flag
  queries where both policies were possible for the true species. Because
  distractor references also change, policy differences are not attributable
  solely to changing the positive reference.
- `scale_*` means crop width, not physical magnification. Meta-val has been
  inspected repeatedly, so no confidence interval on these selected results
  should be described as an independent confirmatory test. This runner does
  not evaluate public-ID or meta-test.
- `hard32_cross_scale` performs more pair comparisons per training query than
  `hard16_cross_scale`; report compute alongside any performance gain.

At 120 episodes/epoch, the run is **not** a complete pass over 124,577
meta-train images. Four trainable arms and 3 epochs can still cost substantial
Colab time. The first epoch bank is shared across arms. On a two-core T4
runtime use 2 workers, not 8 or 16. `train` resumes from each arm's
`latest.pt`; changing train-critical settings requires a new output folder.
The runner preloads only the images scheduled by the selected episodes and
bank (`WOOD_HARD_PRELOAD=1`); `WOOD_HARD_PRELOAD_WORKERS` controls concurrent
Drive reads and defaults to 8. This can be reduced if Drive throttles reads.

## Colab cell 1: update, preflight, real-image smoke

Mount Drive first. Run this cell before the expensive training cell.

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
for key in list(os.environ):
    if key.startswith(("RUN_", "SKIP_")):
        os.environ.pop(key)

os.environ.update({
    "ROOT_PATH": "/content/drive/MyDrive/NCS",
    "RUN_WOOD_HARD_SHORTLIST": "1",
    "WOOD_HARD_OUT": "/content/drive/MyDrive/NCS/results/wood_hard_shortlist_study_v1",
    "WOOD_ML_OUT": "/content/drive/MyDrive/NCS/results/wood_multilevel_study_v1",
    "WOOD_LARGE_BASE_OUT": "/content/drive/MyDrive/NCS/results/dinov2s_retrieval_matched_v1",
    "WOOD_LARGE_QKV_OUT": "/content/drive/MyDrive/NCS/results/wood_correspondence_image_full_v1",
    "WOOD_HARD_SEED": "43",
    "WOOD_HARD_EPOCHS": "3",
    "WOOD_HARD_EPISODES": "120",
    "WOOD_HARD_QUERIES": "4",
    "WOOD_HARD_ARMS": "random16,hard16,hard16_cross_scale,hard32_cross_scale",
    "WOOD_HARD_WORKERS": "2",
    "WOOD_HARD_PRELOAD_WORKERS": "8",
    "WOOD_HARD_IMAGE_BATCH": "64",
    "WOOD_HARD_MICROBATCH": "4",
    "WOOD_HARD_FOLDS": "0,1",
    "WOOD_HARD_BUDGETS": "1,3,10,20",
    "WOOD_HARD_WIDTHS": "32,128",
})
assert Path(os.environ["WOOD_ML_OUT"], "selection_lock.json").is_file()
assert Path(os.environ["WOOD_LARGE_QKV_OUT"], "qkv/seed_43/best.pt").is_file()

for mode in ("preflight", "smoke"):
    os.environ["WOOD_HARD_MODE"] = mode
    _ = runpy.run_module("swid_retrieval.run_overnight", run_name="__main__")
```

## Colab cell 2: training and validation

```python
for mode in ("train", "validate", "select", "scale_stress"):
    print(f"===== {mode.upper()} =====", flush=True)
    os.environ["WOOD_HARD_MODE"] = mode
    _ = runpy.run_module("swid_retrieval.run_overnight", run_name="__main__")
```

For a **technical pilot only**, use a separate output folder and set
`WOOD_HARD_ARMS=random16,hard16`, `WOOD_HARD_EPISODES=10`,
`WOOD_HARD_EPOCHS=1`, `WOOD_HARD_BUDGETS=10`, and `WOOD_HARD_WIDTHS=128`.
Do not compare the pilot's best checkpoint to the full study as if compute
budgets were matched.

## Download a small review bundle

No `.pt`, `.npy`, or image files are needed for review.

```python
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from google.colab import files

root = Path(os.environ["WOOD_HARD_OUT"])
assert (root / "selection_lock.json").is_file()
bundle = root.parent / "wood_hard_shortlist_review.zip"
paths = [root / "preflight.json", root / "smoke.json",
         root / "selection_lock.json"]
paths += list(root.glob("train/*/epoch_*_summary.csv"))
paths += list(root.glob("train/*/epoch_*_provenance.json"))
paths += list(root.glob("validation/*/*.csv"))
paths += list(root.glob("validation/*/protocol.json"))
paths += list(root.glob("scale_stress/*/*.csv"))
paths += list(root.glob("scale_stress/*/protocol.json"))
with ZipFile(bundle, "w", ZIP_DEFLATED) as archive:
    for path in sorted(set(paths)):
        if path.is_file():
            archive.write(path, path.relative_to(root.parent))
print(bundle, f"{bundle.stat().st_size / 2**20:.1f} MiB")
files.download(str(bundle))
```
