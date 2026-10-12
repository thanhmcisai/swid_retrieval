# Training-exposure check for wood retrieval

This study tests one explanation for weak large-gallery ranking: the previous
random-16 shortlist model saw too few images per epoch. The low-exposure result
is the already locked `wood_hard_shortlist_study_v1/random16` run (120 episodes
per epoch, three epochs, seed 43). The new run uses the same source checkpoint,
sampler, losses, learning rates, seed, and meta-val protocol, but 600 episodes
per epoch. It writes into a new folder and never edits the previous study.

The `preflight` mode plans both schedules before paying for GPU training. It
reports distinct train images, query images, species and source scans, including
cumulative coverage across epochs. `compare` fixes the endpoint at K=10 and
top-128 QKV, aligns the same 570 meta-val queries, and reports a paired
species-cluster bootstrap interval. Its interval is exploratory because the
baseline was previously selected on meta-val; it is not a test-set CI. This
study does not evaluate meta-test, public-ID, or full-954.

## Colab cell 1: pull and preflight

Mount Drive first. Run the entire cell after a fresh runtime or a `git pull`.

```python
%cd /content/drive/MyDrive/NCS
!git -C swid_retrieval pull

import os, sys, runpy
from pathlib import Path
sys.path.insert(0, "/content/drive/MyDrive/NCS")
for name in list(sys.modules):
    if name.startswith("swid_retrieval"):
        del sys.modules[name]
for name in list(os.environ):
    if name.startswith("RUN_"):
        os.environ.pop(name)

os.environ.update({
    "ROOT_PATH": "/content/drive/MyDrive/NCS",
    "RUN_WOOD_COVERAGE_STUDY": "1",
    "WOOD_COVERAGE_MODE": "preflight",
    "WOOD_COVERAGE_OUT": "/content/drive/MyDrive/NCS/results/wood_coverage_random16_v1",
    "WOOD_COVERAGE_BASELINE_OUT": "/content/drive/MyDrive/NCS/results/wood_hard_shortlist_study_v1",
    "WOOD_COVERAGE_EPISODES": "600",
    "WOOD_HARD_SEED": "43",
    "WOOD_HARD_ARMS": "random16,hard16,hard16_cross_scale,hard32_cross_scale",
    "WOOD_HARD_EPOCHS": "3",
    "WOOD_HARD_QUERIES": "4",
    "WOOD_HARD_WORKERS": "2",
    "WOOD_HARD_IMAGE_BATCH": "64",
    "WOOD_HARD_MICROBATCH": "4",
    "WOOD_HARD_BACKBONE_LR": "1e-6",
    "WOOD_HARD_HEAD_LR": "5e-5",
    "WOOD_HARD_WEIGHT": "0.5",
    "WOOD_HARD_BANK_WEIGHT": "0.5",
    "WOOD_HARD_FOLDS": "0,1",
    "WOOD_HARD_BUDGETS": "1,3,10,20",
    "WOOD_HARD_WIDTHS": "32,128",
    "WOOD_HARD_PRELOAD": "1",
})
assert Path(os.environ["WOOD_COVERAGE_BASELINE_OUT"], "selection_lock.json").is_file()
_ = runpy.run_module("swid_retrieval.run_overnight", run_name="__main__")
```

Check the printed `planned images` count before continuing. There is no
benefit to running the long training step if distinct-image coverage barely
increases. Keep `WOOD_COVERAGE_EPISODES` unchanged after preflight, or choose a
new output folder and rerun preflight. The reported *unique images* include
references and queries; *unique query images* is the tighter measure.

## Colab cell 2: train and validate

The training checkpoint is resumable. Run the same cell again after a runtime
disconnect; the completed train mode skips and completed validation files are
reused. On a new runtime, rerun cell 1 first to restore environment variables.

```python
import os, runpy
for mode in ("train", "validate", "compare"):
    os.environ["WOOD_COVERAGE_MODE"] = mode
    print(f"\n===== COVERAGE {mode.upper()} =====", flush=True)
    _ = runpy.run_module("swid_retrieval.run_overnight", run_name="__main__")
```

Do not alter old checkpoint files. The expected result is
`results/wood_coverage_random16_v1/coverage_comparison.json`. A higher
meta-val score alone does not establish generalization. If exposure clearly
helps, lock the training settings and run an untouched meta-test/full-gallery
evaluation next. If it does not, prioritize an aligned 128-candidate ranking
loss rather than increasing episodes again.

## Colab cell 3: download a small review bundle

```python
import json, zipfile
from pathlib import Path
from google.colab import files

out = Path("/content/drive/MyDrive/NCS/results/wood_coverage_random16_v1")
old = Path("/content/drive/MyDrive/NCS/results/wood_hard_shortlist_study_v1")
assert (out / "coverage_comparison.json").is_file()
bundle = out.parent / "wood_coverage_review.zip"
with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.suffix in {".json", ".csv"} and \
                "features" not in path.parts:
            z.write(path, path.relative_to(out.parent))
    for name in ("preflight.json", "selection_lock.json",
                 "validation/random16/summary.csv",
                 "validation/random16/per_query.csv",
                 "validation/random16/protocol.json"):
        path = old / name
        assert path.is_file(), path
        z.write(path, path.relative_to(old.parent))
print(bundle, round(bundle.stat().st_size / 1e6, 2), "MB")
files.download(str(bundle))
```

The bundle excludes `.pt`, `.npy`, and image files. Keep the checkpoint on Drive
for later held-out evaluation.

## Audit the final epoch without retraining

The initial run saved `best.pt` from epoch 1 and `latest.pt` from epoch 3.
Epoch 2's `candidate.pt` was overwritten, so epoch 2 cannot be recovered from
this run. The `audit_latest` mode evaluates only epoch 3 on the same meta-val
queries at the predeclared K=10/top-128 endpoint. It does not replace the
selected checkpoint and does not evaluate meta-test or public-ID. The result
is diagnostic, not an independent model-selection result.

On a fresh runtime, rerun cell 1 to restore the configuration, then:

```python
import os, runpy
os.environ["WOOD_COVERAGE_MODE"] = "audit_latest"
_ = runpy.run_module("swid_retrieval.run_overnight", run_name="__main__")
```

To download the small audit files:

```python
import zipfile
from pathlib import Path
from google.colab import files

out = Path("/content/drive/MyDrive/NCS/results/wood_coverage_random16_v1")
audit = out / "checkpoint_audit"
assert (audit / "comparison.json").is_file()
bundle = out.parent / "wood_coverage_latest_audit_review.zip"
with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
    for path in sorted(audit.rglob("*")):
        if path.is_file() and path.suffix in {".json", ".csv"}:
            z.write(path, path.relative_to(out.parent))
    for name in ("preflight.json", "coverage_comparison.json",
                 "validation/random16/protocol.json",
                 "validation/random16/summary.csv"):
        path = out / name
        assert path.is_file(), path
        z.write(path, path.relative_to(out.parent))
print(bundle, round(bundle.stat().st_size / 1e6, 2), "MB")
files.download(str(bundle))
```
