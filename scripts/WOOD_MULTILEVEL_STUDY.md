# Cross-scale, end-to-end wood retrieval study

This is a new exploratory study. It does not edit the manuscript or overwrite
the locked large-gallery/public-ID results. It starts all trainable arms from
the same image-trained DINOv2-S + QKV checkpoint, then updates both encoder and
correspondence scorer on **meta-train** images. The source arm is unmodified.

## Ablation matrix

| Arm | Scan-disjoint query/reference | Different crop scale required | Extra loss |
| --- | --- | --- | --- |
| `source` | upstream recipe | upstream recipe | none (unchanged checkpoint) |
| `scan_control` | yes | no | baseline QKV + global + bank |
| `cross_scale` | yes | yes | baseline QKV + global + bank |
| `cross_scale_contrastive` | yes | yes | multi-positive cross-scale InfoNCE |
| `cross_scale_contrastive_margin` | yes | yes | InfoNCE + hardest-impostor QKV margin |

`scale_*` denotes the crop width from a SmartWoodID source scan, **not optical
magnification**. The contrastive positive is same species, different source
scan, different crop width. Same-species/same-scale references are masked out of
the InfoNCE denominator, not treated as negatives. This does not assume that
tokens from different scans are pixel-aligned. All crops are resized to the
model input and may be further augmented, so crop-width labels are a proxy
for anatomical field of view, not a calibrated physical scale. Each epoch logs image coverage
and the number of cross-scale positive pairs. An epoch is *not* a complete pass
over 124,577 training images; the number of episodes controls exposure.
The trainable arms use the same eligible species pool and the same seeded
episode class lists; the sampler/loss is the controlled difference.

Validation uses two scan-disjoint meta-val folds: 57 target species plus 580
meta-train distractors (637 species, 285 queries/fold). Reference budgets
1/3/10/20 are chosen deterministically across source scans and crop widths.
For each budget, the first-stage **raw-prototype** shortlist has width 32 or
128, matching the branch that previously yielded high meta-val recall@128.
Adapted-prototype recall@32/128 is reported as a diagnostic, not used to pick
the QKV shortlist. `qkv_all_refs` scores every selected reference for each candidate; at
the largest budget/width, `qkv_scan3` is the old distinct-scan control. The
summary separates candidate recall, conditional R@1 given inclusion, final
macro R@1, and per-query timing. `by_scale.csv` is descriptive, not a basis
for changing the locked model after public-ID evaluation.
The QKV scorer still aggregates within each source scan before combining up
to two scans; multiple crop widths from one scan increase the evidence choices,
not the number of independent scans.
`WOOD_ML_WIDTHS` can include `637` for all-class QKV on meta-val, but this is
far more expensive than top-128 and should be a separately budgeted analysis.

**Important limitation:** The selected checkpoint is from the earlier
exploratory line for which public-ID has already been viewed. The meta-val
selection below is a prospective *new ablation* but does not turn public-ID
into a fresh untouched test set. The `train`/`validate`/`select` stages touch
neither meta-test nor public-ID. An optional `public_id` stage can be run after
locking; label it as exploratory replication, not independent test evidence.
Do not present meta-val improvements as independent test gains.

## Colab cells

Run the first cell after mounting Drive. Keep the same output directory and
training hyperparameters when resuming. Start with the source/one-arm pilot
if compute is limited, then use a new output folder for a changed experiment
matrix. A T4 with only 2 CPU cores should use 2 workers, not 8 or 16.
The `smoke` mode reads eight real images, runs one encoder+QKV backward step,
checks both gradient paths, and saves no trained checkpoint.

```python
%cd /content/drive/MyDrive/NCS
!git -C swid_retrieval pull

import os, sys, runpy
sys.path.insert(0, "/content/drive/MyDrive/NCS")
for m in list(sys.modules):
    if m.startswith("swid_retrieval"):
        del sys.modules[m]
for key in list(os.environ):
    if key.startswith("RUN_") or key.startswith("SKIP_"):
        os.environ.pop(key, None)

os.environ.update({
    "ROOT_PATH": "/content/drive/MyDrive/NCS",
    "RUN_WOOD_MULTILEVEL": "1",
    "WOOD_ML_OUT": "/content/drive/MyDrive/NCS/results/wood_multilevel_study_v1",
    "WOOD_LARGE_BASE_OUT": "/content/drive/MyDrive/NCS/results/dinov2s_retrieval_matched_v1",
    "WOOD_LARGE_QKV_OUT": "/content/drive/MyDrive/NCS/results/wood_correspondence_image_full_v1",
    "WOOD_ML_SEED": "43",
    "WOOD_ML_SOURCE_SEED": "43",
    "WOOD_ML_ARMS": "source,scan_control,cross_scale,cross_scale_contrastive,cross_scale_contrastive_margin",
    "WOOD_ML_EPOCHS": "3",
    "WOOD_ML_EPISODES": "60",
    "WOOD_ML_WORKERS": "2",
    "WOOD_ML_IMAGE_BATCH": "64",
    "WOOD_ML_MICROBATCH": "4",
    "WOOD_ML_FOLDS": "0,1",
    "WOOD_ML_BUDGETS": "1,3,10,20",
    "WOOD_ML_WIDTHS": "32,128",
})

from pathlib import Path
assert Path(os.environ["WOOD_LARGE_BASE_OUT"]).is_dir()
assert Path(os.environ["WOOD_LARGE_QKV_OUT"] + "/qkv/seed_43/best.pt").is_file()
assert Path(os.environ["ROOT_PATH"] + "/swi_manifest.json").is_file()

os.environ["WOOD_ML_MODE"] = "preflight"
_ = runpy.run_module("swid_retrieval.run_overnight", run_name="__main__")
os.environ["WOOD_ML_MODE"] = "smoke"
_ = runpy.run_module("swid_retrieval.run_overnight", run_name="__main__")
```

```python
for mode in ("train", "validate", "select"):
    print(f"===== {mode.upper()} =====", flush=True)
    os.environ["WOOD_ML_MODE"] = mode
    _ = runpy.run_module("swid_retrieval.run_overnight", run_name="__main__")
```

`train` resumes each arm from `latest.pt`; it will reject changed training
settings or changed source artifacts. The validation feature caches also check
checkpoint and item identities. The `select` step writes an immutable
`selection_lock.json`; a second call only succeeds if the evidence is unchanged.

After reviewing `selection_lock.json`, the optional 954-species public-ID pass
uses the **locked** budget/width/method for both the source checkpoint and the
selected arm. Unlike the old full-reference run, this is a capped gallery
with 1/3/10/20 references per species. If QKV is selected, scores are saved
every 16 queries and resume after a disconnect; extraction uses the existing
verified feature-cache cursor. This stage can still take substantial Colab
time at 128 candidates x 20 references, so do not start it merely because
meta-val candidate recall is high.

```python
os.environ["WOOD_ML_MODE"] = "public_id"
_ = runpy.run_module("swid_retrieval.run_overnight", run_name="__main__")
```

For the first cost-controlled pilot, use a **different output folder**, set
`WOOD_ML_ARMS=source,scan_control,cross_scale_contrastive`, and use one fold,
one budget, one width. That pilot is for debugging/feasibility only, not model
selection. Do not use pilot meta-val results to tune a claimed independent
public-ID test.

## Small review bundle

Do not zip feature `.npy` or checkpoint `.pt` files. After the full run:

```python
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

root = Path(os.environ["WOOD_ML_OUT"])
bundle = root.parent / "wood_multilevel_review.zip"
paths = [root / "preflight.json", root / "selection_lock.json"]
paths += [root / "smoke.json"]
paths += list(root.glob("train/*/epoch_*.json"))
paths += list(root.glob("validation/*/*.csv"))
paths += list(root.glob("validation/*/protocol.json"))
paths += list(root.glob("public_id/*/*.json"))
paths += list(root.glob("public_id/*/per_query.csv"))
paths += list(root.glob("public_id/*/per_source.csv"))
paths += list(root.glob("public_id_comparison.csv"))
paths += list(root.glob("public_id_paired.csv"))
paths += list(root.glob("public_id_paired_summary.json"))
with ZipFile(bundle, "w", ZIP_DEFLATED) as archive:
    for path in paths:
        if path.is_file():
            archive.write(path, path.relative_to(root.parent))
print(bundle, bundle.stat().st_size / 2**20, "MiB")
from google.colab import files
files.download(str(bundle))
```
