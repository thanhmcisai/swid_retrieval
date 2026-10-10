# Diagnose the DINOv2-S 57-species result

This is an **exploratory, meta-validation-only** diagnostic. It does not update
the manuscript or any public-test result. The existing matched run in
`results/dinov2s_retrieval_matched_v1` must contain its nine selected `best.pt`
checkpoints and `meta_val_screen.csv`; the downloaded review ZIP alone does not
contain model weights. The diagnostic refuses changed checkpoint/code hashes and
checks that each native R@1 exactly reproduces the stored 24-, 57-, 128-, 256-
and 637-species validation result before saving a report. It never trains.

The galleries isolate two questions that the original curve confounds:

- `57x1` versus `57x5`: same 57 target species and queries, different reference
  count. `57x1`, `128x1`, `256x1`, and `637x1` isolate increasing cardinality at
  one reference per species. The 24-species gallery also has five references.
- `nearest` versus `prototype`: the exact same saved embedding with two
  nonparametric scoring rules. `metric_large` uses nearest natively; the
  `prototype_large` variants use prototype natively. Cross-scoring is a
  **post-hoc diagnostic**, not a validated new operating point.

Per-query files record true-class rank, cosine margin to the strongest wrong
species, top-64 oracle coverage, same-genus errors, query scale and number of
same-scale references. Seed 43 is paired to seeds 42 and 44 on identical query
paths. The species-cluster bootstrap is descriptive only: these folds were
already used to select the best epoch for each seed, so it is not an
independent significance test. Scan-disjoint patches are not necessarily
different donor trees.

## Colab cell 1: inference-only diagnosis

```python
%cd /content/drive/MyDrive/NCS
!git -C swid_retrieval pull

import os, sys, runpy, zipfile
from pathlib import Path
root = Path('/content/drive/MyDrive/NCS')
sys.path.insert(0, str(root))
for name in list(sys.modules):
    if name.startswith('swid_retrieval'):
        del sys.modules[name]
for flag in ('RUN_GALLERY_STUDY', 'RUN_REPAIR_PUBLIC_ROWS',
             'RUN_FINAL_SCURD_RETRAIN', 'RUN_FINAL_COLAB_AUDIT'):
    os.environ[flag] = '0'
os.environ.update({
    'ROOT_PATH': str(root),
    'DEVICE': 'cuda',
    'RUN_GALLERY_DIAGNOSTICS': '1',
    'GALLERY_DIAG_STUDY_OUT': str(root / 'results/dinov2s_retrieval_matched_v1'),
    'GALLERY_DIAG_OUT': str(root / 'results/dinov2s_seed43_diagnostics_v1'),
    'GALLERY_DIAG_VARIANTS': 'prototype_large,metric_large,prototype_large_frozen',
    'GALLERY_DIAG_SEEDS': '42,43,44',
    'GALLERY_DIAG_PRELOAD': '1',
    'GALLERY_DIAG_PRELOAD_WORKERS': '16',
})
assert (root / 'results/dinov2s_retrieval_matched_v1/meta_val_screen.csv').is_file()
assert (root / 'results/dinov2s_retrieval_matched_v1/dinov2_vits14/prototype_large/seed_43/best.pt').is_file()
_ = runpy.run_module('swid_retrieval.run_overnight', run_name='__main__')

folder = Path(os.environ['GALLERY_DIAG_OUT'])
archive = root / 'results/dinov2s_seed43_diagnostics_review.zip'
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
    for path in sorted(folder.glob('*.csv')) + sorted(folder.glob('*.json')):
        bundle.write(path, arcname=path.name)
print(archive, archive.stat().st_size / 2**20, 'MiB')
```

Send `dinov2s_seed43_diagnostics_review.zip`. It contains no checkpoint or
dataset image. If the guard rejects a hash or an R@1 mismatch, stop and send
the traceback; do not force a pass.

## Colab cell 2: independent initialization replicates (optional)

Run this only after cell 1 passes. It keeps the same DINOv2-S recipe and 2-fold
meta-validation protocol but trains three additional seeds. It resumes via
`latest.pt` after a disconnect. It does **not** make the validation folds
independent; it measures training-initialization variability. It can take hours
on a T4 and must not be used to cherry-pick a seed on public test data.

```python
%cd /content/drive/MyDrive/NCS
import os, sys, runpy, torch
from pathlib import Path
root = Path('/content/drive/MyDrive/NCS')
sys.path.insert(0, str(root))
for name in list(sys.modules):
    if name.startswith('swid_retrieval'):
        del sys.modules[name]
from swid_retrieval import gallery_experiment as study

study_out = root / 'results/dinov2s_retrieval_matched_v1'
trusted_checkpoint = study_out / 'dinov2_vits14/prototype_large/seed_43/best.pt'
cfg = torch.load(trusted_checkpoint, map_location='cpu', weights_only=False)['config']
for name in list(os.environ):
    if name.startswith('GALLERY_STUDY_'):
        os.environ.pop(name)
mapping = {
    'BACKBONE': cfg['backbone'], 'EPOCHS': cfg['epochs'],
    'EPISODES_PER_EPOCH': cfg['episodes_per_epoch'],
    'MICROBATCH': cfg['microbatch'], 'IMAGE_BATCH': cfg['image_batch'],
    'WORKERS': cfg['workers'], 'BACKBONE_LR': cfg['backbone_lr'],
    'HEAD_LR': cfg['head_lr'], 'VALIDATION_FOLDS': cfg['validation_folds'],
    'VALIDATION_TRAIN_DISTRACTORS': cfg['validation_train_distractors'],
    'WAYS': ','.join(map(str, cfg['ways'])), 'IMAGE_SIZE': cfg['image_size'],
    'MEMORY_SIZE': cfg['memory_size'], 'MEMORY_MIN_CLASSES': cfg['memory_min_classes'],
}
os.environ.update({f'GALLERY_STUDY_{key}': str(value) for key, value in mapping.items()})
assert study.variant_config('prototype_large') == cfg, 'Recipe differs; stop before training'
os.environ.update({
    'ROOT_PATH': str(root), 'DEVICE': 'cuda',
    'RUN_GALLERY_DIAGNOSTICS': '0', 'RUN_GALLERY_STUDY': '1',
    'RUN_REPAIR_PUBLIC_ROWS': '0', 'RUN_FINAL_SCURD_RETRAIN': '0',
    'RUN_FINAL_COLAB_AUDIT': '0',
    'GALLERY_STUDY_MODE': 'screen', 'GALLERY_STUDY_OUT': str(study_out),
    'GALLERY_STUDY_VARIANTS': 'prototype_large',
    'GALLERY_STUDY_SEEDS': '45,46,47',
    'GALLERY_STUDY_PRELOAD': '1',
    'GALLERY_STUDY_PRELOAD_WORKERS': '16',
})
_ = runpy.run_module('swid_retrieval.run_overnight', run_name='__main__')
print(study_out / 'meta_val_screen.csv')
```

Do not promote a method based on the seed-43 peak alone. Advance the prototype
branch only if additional seeds preserve the 57-species gain, the one-reference
and large-gallery controls are acceptable, and independent public/source-shift
evaluation does not reverse the conclusion. The present `0.5105` was a
selected-checkpoint meta-validation result, not an independent test score.
