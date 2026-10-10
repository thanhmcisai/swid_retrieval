# All-class wood correspondence study

This is a separate research study, not a replacement for the manuscript's
reported results. It uses the selected fine-tuned DINOv2-S retrieval encoder
and its 512-dimensional global and 4x4 local descriptors. The proposed scorer
learns symmetric query/reference QKV token correspondence, a bounded residual
to global prototype scores, and independent-scan evidence pooling. Its primary
`qkv_all` score evaluates every enrolled class; `qkv_top128` is a genuine
candidate-limited cost control. These pooled tokens are a wood-pattern
hypothesis, **not** anatomically annotated vessels or rays.

Head training uses listwise all-class retrieval loss, a global-score auxiliary
loss and supervised contrastive loss. Episodes have 16/32/64 species, one or
two references per species, hard negatives, and different Tw source scans for
query and reference. Joint training continues from the selected head and
updates the projection and last two DINOv2 blocks by gradient replay. It is
not training a new encoder from scratch. All selection uses meta-validation;
meta-test requires an explicit selection lock. The old paper outputs and
checkpoints are never modified.

This protocol measures species ranking in scan-disjoint SWI galleries. It does
not yet measure public-image cross-domain retrieval, OOD rejection, or
image-level mAP. Those require separately locked evaluations before any
general superiority claim or manuscript replacement. A positive meta-test
result here is still conditional on the chosen gallery size and enrollment
scheme.

The `fixed_local` row is the prior fixed-weight, candidate-limited local
scorer on the *same* gallery. `global` is the original base representation for
head training, but the updated encoder's unadapted representation for joint
training. `global_adapt` isolates the learned global residual. `maxsim`
removes QKV attention, `no_gate` removes learned token salience,
`no_consensus` replaces the distinct-scan pooling with nearest-reference
pooling, and `no_global_aux`, `no_hard_negatives`, `no_cross_scan` ablate the
training protocol. The latter is a deliberately less stringent train-only
control; evaluation always remains scan-disjoint. `qkv_top128` can never
repair a true class outside the baseline top 128, so candidate recall is
saved for each query.

The validation report includes macro R@1/R@5, paired species-bootstrap 95%
intervals versus raw global, adapted global, and fixed local; query-level
rescues/harm, same-genus errors, per-scale and per-reference-scan summaries,
and scoring latency. Bootstrap intervals only quantify uncertainty for the
fixed checkpoint and queries; they do not estimate training-seed variance.
For seed variability, repeat the head stage with 42, 43 and 44. Timing is
scorer-only on the given GPU and feature tensors, not end-to-end image latency.

## Colab cell 1: setup, smoke, provenance

```python
%cd /content/drive/MyDrive/NCS
!git -C swid_retrieval pull

import os, sys, runpy, torch
from pathlib import Path
root = Path('/content/drive/MyDrive/NCS')
sys.path.insert(0, str(root))
for name in list(sys.modules):
    if name.startswith('swid_retrieval'):
        del sys.modules[name]
for flag in ('RUN_WOOD_EVIDENCE_STUDY', 'RUN_GALLERY_STUDY',
             'RUN_GALLERY_DIAGNOSTICS', 'RUN_REPAIR_PUBLIC_ROWS',
             'RUN_FINAL_SCURD_RETRAIN', 'RUN_FINAL_COLAB_AUDIT'):
    os.environ[flag] = '0'
os.environ.update({
    'ROOT_PATH': str(root), 'DEVICE': 'cuda',
    'RUN_WOOD_CORRESPONDENCE_STUDY': '1',
    'WOOD_EVIDENCE_STUDY_OUT': str(root / 'results/dinov2s_retrieval_matched_v1'),
    'WOOD_CORR_FEATURE_OUT': str(root / 'results/wood_evidence_pilot_v1'),
    'WOOD_CORR_OUT': str(root / 'results/wood_correspondence_study_v1'),
    'WOOD_CORR_BASE_SEED': '43', 'WOOD_CORR_SEEDS': '43',
    'WOOD_CORR_VARIANTS': 'qkv', 'WOOD_CORR_WORKERS': '2',
    'WOOD_CORR_HEAD_EPOCHS': '4', 'WOOD_CORR_HEAD_EPISODES': '150',
    'WOOD_CORR_JOINT_EPOCHS': '2', 'WOOD_CORR_JOINT_EPISODES': '50',
    'WOOD_CORR_MICROBATCH': '8',
})
assert torch.cuda.is_available(), 'Enable a GPU runtime'
assert (root / 'swi_manifest.json').is_file()
assert (root / 'results/dinov2s_retrieval_matched_v1/dinov2_vits14/'
        'prototype_large/seed_43/best.pt').is_file()
for mode in ('smoke', 'preflight', 'probe'):
    os.environ['WOOD_CORR_MODE'] = mode
    _ = runpy.run_module('swid_retrieval.run_overnight', run_name='__main__')
```

The probe reuses or extracts validation descriptors and checks that the raw
global result matches the selected base checkpoint. Stop if it fails. No
public-ID, public-OOD or meta-test data are read in these stages.

## Colab cell 2: head-only pilot

```python
import os, runpy
os.environ.update({'WOOD_CORR_MODE': 'train_head',
                   'WOOD_CORR_VARIANTS': 'qkv,global_only,maxsim',
                   'WOOD_CORR_SEEDS': '43', 'WOOD_CORR_PRELOAD': '0'})
_ = runpy.run_module('swid_retrieval.run_overnight', run_name='__main__')
os.environ.update({'WOOD_CORR_MODE': 'validate', 'WOOD_CORR_STAGE': 'head'})
_ = runpy.run_module('swid_retrieval.run_overnight', run_name='__main__')
```

Inspect `validate_head_*_summary.csv`, `*_folds.csv`, `*_paired.csv`, `*_queries.csv` and
`*_timing.csv`. The decisive comparison is QKV versus *both* global-adapted
and fixed-local on 57x5 and 637x1, including the harmed-query count and the
second validation fold. Do not choose a model using public or meta-test scores.

## Colab cell 3: required ablations and seeds

Run this after the pilot if QKV is stable enough to warrant the cost. It
resumes completed epochs only when their full provenance matches. Keep head
schedule environment variables unchanged when resuming or proceeding to
joint training.

```python
import os, runpy
os.environ.update({'WOOD_CORR_MODE': 'train_head',
                   'WOOD_CORR_VARIANTS': 'no_gate,no_consensus,no_global_aux,'
                                         'no_hard_negatives,no_cross_scan',
                   'WOOD_CORR_SEEDS': '43'})
_ = runpy.run_module('swid_retrieval.run_overnight', run_name='__main__')
os.environ.update({'WOOD_CORR_MODE': 'validate', 'WOOD_CORR_STAGE': 'head'})
_ = runpy.run_module('swid_retrieval.run_overnight', run_name='__main__')
os.environ.update({'WOOD_CORR_MODE': 'train_head',
                   'WOOD_CORR_VARIANTS': 'qkv', 'WOOD_CORR_SEEDS': '42,44'})
_ = runpy.run_module('swid_retrieval.run_overnight', run_name='__main__')
os.environ.update({'WOOD_CORR_MODE': 'validate', 'WOOD_CORR_STAGE': 'head'})
_ = runpy.run_module('swid_retrieval.run_overnight', run_name='__main__')
```

## Colab cell 4: joint fine-tuning and matched controls

Only run variants whose matching head checkpoints exist. This stage is
costlier because it reads training images and replays the encoder gradient.
With Drive I/O, `WOOD_CORR_PRELOAD=1` may be worth it after checking free
local storage; it preloads representative images only, not the full 124k set.
The training transform is augmented; validation uses the fixed non-augmented
transform. If the pilot does not beat the fixed local baseline, do not spend
GPU credit on this step.

```python
import os, runpy
os.environ.update({'WOOD_CORR_MODE': 'train_joint', 'WOOD_CORR_STAGE': 'joint',
                   'WOOD_CORR_VARIANTS': 'qkv,global_only',
                   'WOOD_CORR_SEEDS': '43', 'WOOD_CORR_PRELOAD': '1',
                   'WOOD_CORR_PRELOAD_WORKERS': '2'})
_ = runpy.run_module('swid_retrieval.run_overnight', run_name='__main__')
os.environ['WOOD_CORR_MODE'] = 'validate'
_ = runpy.run_module('swid_retrieval.run_overnight', run_name='__main__')
```

## Locked test (run only after reviewing validation)

Select **one** stage, variant and seed using validation only. Then run lock
and final in that order. The lock file records the exact checkpoint, manifest
and validation summary hashes. A changed selection fails rather than silently
opening meta-test repeatedly.

```python
import os, runpy
os.environ.update({'WOOD_CORR_MODE': 'lock', 'WOOD_CORR_STAGE': 'head',
                   'WOOD_CORR_VARIANTS': 'qkv', 'WOOD_CORR_SEEDS': '43'})
_ = runpy.run_module('swid_retrieval.run_overnight', run_name='__main__')
os.environ.update({'WOOD_CORR_MODE': 'final', 'WOOD_CORR_APPROVE_FINAL': '1'})
_ = runpy.run_module('swid_retrieval.run_overnight', run_name='__main__')
```

Change `WOOD_CORR_STAGE` to `joint` only if validation selected that stage.
Do not run `final` for every variant or seed.

## Small review archive

This excludes checkpoints and NPZ feature caches. Run after the stages you
want reviewed.

```python
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
out = Path('/content/drive/MyDrive/NCS/results/wood_correspondence_study_v1')
archive = out.parent / 'wood_correspondence_review.zip'
with ZipFile(archive, 'w', ZIP_DEFLATED) as zf:
    for path in out.rglob('*'):
        if path.is_file() and path.suffix in {'.csv', '.json'} and 'feature_cache' not in path.parts:
            zf.write(path, path.relative_to(out))
print(archive, round(archive.stat().st_size / 1048576, 2), 'MB')
```
