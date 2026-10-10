# Scan-consensus local-evidence retrieval pilot

This is an isolated **research pilot**, not a replacement for SC-URD or the
submitted tables. It reuses a provenance-checked, fine-tuned DINOv2-S
`prototype_large` encoder without changing the original checkpoint or its
signature-bearing source files. A 4x4 grid of pooled DINOv2 patch tokens is
projected into the checkpoint's metric space. Global prototype scores shortlist
up to 128 species; a symmetric patch-set match reranks that shortlist. With
multiple source scans, the local score uses the two strongest distinct-scan
views after reducing duplicate views within a scan. This is a structural
**hypothesis** about repeated wood texture, not a proven anatomical detector.

`probe` uses a fixed 0.25 local contribution and no fitting. `train` fits only
the local residual adapter and score coefficient from representative
**meta-train** images. Episodes vary 16/32/64 classes and one/two references,
include globally nearest and same-genus hard negatives, and enforce distinct
source scans for query and reference. The encoder remains frozen; this is not
end-to-end fine-tuning. The method is deliberately screened before any costly
backbone training. The held-out meta-test is unavailable until `final` mode is
explicitly approved. Meta-validation chooses the scorer checkpoint by the
predeclared mean of 57x5 and 637x1 macro R@1 on fold 0; both folds are then
reported. Because the base encoder's epoch was selected using these same
meta-validation folds, this is **exploratory**, not independent confirmation.

## Colab cell 1: preflight and inference probe

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
for flag in ('RUN_GALLERY_STUDY', 'RUN_GALLERY_DIAGNOSTICS',
             'RUN_REPAIR_PUBLIC_ROWS', 'RUN_FINAL_SCURD_RETRAIN',
             'RUN_FINAL_COLAB_AUDIT'):
    os.environ[flag] = '0'
os.environ.update({
    'ROOT_PATH': str(root), 'DEVICE': 'cuda',
    'RUN_WOOD_EVIDENCE_STUDY': '1',
    'WOOD_EVIDENCE_BASE_SEED': '43',
    'WOOD_EVIDENCE_STUDY_OUT': str(root / 'results/dinov2s_retrieval_matched_v1'),
    'WOOD_EVIDENCE_OUT': str(root / 'results/wood_evidence_pilot_v1'),
})
assert torch.cuda.is_available(), 'Enable a GPU runtime'
assert (root / 'swi_manifest.json').is_file()
assert (root / 'results/dinov2s_retrieval_matched_v1/dinov2_vits14/'
        'prototype_large/seed_43/best.pt').is_file()
for mode in ('preflight', 'probe'):
    os.environ['WOOD_EVIDENCE_MODE'] = mode
    _ = runpy.run_module('swid_retrieval.run_overnight', run_name='__main__')
```

The probe compares global prototype, fixed local evidence, and a no-consensus
control for the same queries and galleries. It stops if the recomputed global
57x5 or 637x1 result differs from the stored checkpoint result. Review the
probe summary before proceeding. A non-improving or unstable probe is a reason
to revisit the local representation, not to report an advantage.

## Colab cell 2: scorer-only training

Run only after cell 1 passes. It reuses cached descriptors and does not read
public images or meta-test images. Start with one scorer seed for cost control;
additional seeds can be run by setting `WOOD_EVIDENCE_SEEDS='42,43,44'`.

```python
import os, runpy
os.environ.update({
    'RUN_WOOD_EVIDENCE_STUDY': '1',
    'WOOD_EVIDENCE_MODE': 'train',
    'WOOD_EVIDENCE_SEEDS': '43',
    'WOOD_EVIDENCE_EPOCHS': '4',
    'WOOD_EVIDENCE_EPISODES': '150',
})
_ = runpy.run_module('swid_retrieval.run_overnight', run_name='__main__')
```

Download only `preflight.json`, `inference_probe_*.csv/json`, and
`meta_val_scorer_43_*.csv/json` for review; do not zip the feature cache or
checkpoint. Keep the checkpoint on Drive for the eventual locked evaluation.

## Locked evaluation, only after reviewing validation

Do not run this to choose seeds or hyperparameters. Fix the scorer seed from
meta-validation first, then set `WOOD_EVIDENCE_APPROVE_FINAL=1` and mode
`final`. It evaluates 245+ scan-disjoint meta-test species at five and one
reference per species, plus a 637-species stress gallery. It does **not**
evaluate corrected public-ID/OOD or VN26; those are later external tests if
the held-out signal is credible.

```python
import os, runpy
os.environ.update({
    'WOOD_EVIDENCE_MODE': 'final',
    'WOOD_EVIDENCE_SELECTED_SCORER_SEED': '43',
    'WOOD_EVIDENCE_APPROVE_FINAL': '1',
})
_ = runpy.run_module('swid_retrieval.run_overnight', run_name='__main__')
```

Interpretation limits: distinct SmartWoodID source scans are not necessarily
different trees; local tokens are not labeled anatomical structures; the
global encoder is reused, and training on one/two references does not fully
match a five-reference deployment. Do not claim a new SOTA method until
independent meta-test, public cross-domain, OOD, latency, and seed replication
support it.
