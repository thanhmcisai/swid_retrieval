# Locked full-gallery follow-up audit

This run is inference-only. It reuses the completed `swi_full.npy`,
`swi_capped5.npy`, and `public_id.npy` feature caches for the arm that was
selected on meta-validation. It does not read the source images, retrain, or
modify the locked study. Keep the original `features/` folders on Drive.

The new public-ID score variants are **exploratory diagnostics**. Do not pick a
weight from their test accuracy for a confirmatory manuscript result. First
inspect whether the 5-reference/3-scan reduction explains the collapse; then
build a matched, scan-disjoint meta-validation scorer study if warranted.

## Colab cell

```python
%cd /content/drive/MyDrive/NCS
!git -C swid_retrieval pull

import os, runpy, sys
from pathlib import Path

root = Path("/content/drive/MyDrive/NCS")
sys.path.insert(0, str(root))
for name in list(sys.modules):
    if name.startswith("swid_retrieval"):
        del sys.modules[name]

for flag in (
    "RUN_WOOD_LARGE_GALLERY", "RUN_WOOD_CORRESPONDENCE_COMPARE",
    "RUN_WOOD_CORRESPONDENCE_IMAGE_TRAIN", "RUN_WOOD_CORRESPONDENCE_STUDY",
    "RUN_WOOD_EVIDENCE_STUDY", "RUN_GALLERY_DIAGNOSTICS",
    "RUN_GALLERY_STUDY", "RUN_REPAIR_PUBLIC_ROWS",
    "RUN_FINAL_SCURD_RETRAIN", "RUN_FINAL_COLAB_AUDIT",
):
    os.environ[flag] = "0"

os.environ.update({
    "ROOT_PATH": str(root),
    "RUN_WOOD_GALLERY_AUDIT": "1",
    "WOOD_LARGE_OUT": str(root / "results/wood_large_gallery_final_v1"),
    "WOOD_LARGE_QKV_OUT": str(root / "results/wood_correspondence_image_full_v1"),
    "WOOD_LARGE_GLOBAL_OUT": str(root / "results/wood_correspondence_global_only_v1"),
    "WOOD_GALLERY_AUDIT_OUT": str(root / "results/wood_gallery_audit_v1"),
})

locked = root / "results/wood_large_gallery_final_v1"
assert (locked / "selection_lock.json").is_file()
assert (locked / "study_plan.json").is_file()
_ = runpy.run_module("swid_retrieval.run_overnight", run_name="__main__")

from google.colab import files
files.download(str(root / "results/wood_gallery_audit_v1/wood_gallery_audit_review.zip"))
```

The audit checks the checkpoint, manifest, CSV, and cache signatures, then
replays the original full-prototype R@1. It fails rather than silently
re-extracting images if a cache is missing or incomplete.

The ZIP contains `summary.csv`, `paired_changes.csv`, `per_source.csv`,
`per_species.csv`, `reference_selection.csv`, `query_outcomes.csv`, and
`audit.json`. Only the ZIP is needed
for review; do not download the heavy `.npy` caches.
