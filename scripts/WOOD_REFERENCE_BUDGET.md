# Validation-locked reference budget study

This inference-only study diagnoses whether sparse/mismatched reference sets are
the main large-gallery failure. It compares two deterministic selection policies
(`scan_hash`, `scan_scale`), five budgets (1, 3, 5, 10, 20 images/species), and
three all-class scorers (single prototype, top-two scale prototypes, their
equal-weight mean). The scale-aware scorer is a lightweight alternative to QKV,
not a trained new model. Two scan-disjoint meta-val folds (57 target + 580
distractor species) select one recipe. Public-ID is evaluated only afterwards.

The source checkpoint and public-ID feature cache come from the existing locked
large-gallery run. Validation extracts a new resumable reference feature cache;
re-run the `validate` mode after a Colab disconnect. It reads no meta-test or
public-ID queries during selection. Because earlier exploratory public-ID results
informed this study design, the final public-ID estimate is *not* a fresh
independent confirmatory test.

On Colab, mount Drive first, then run:

```python
%cd /content/drive/MyDrive/NCS
!git -C swid_retrieval pull

import os, sys, runpy
sys.path.insert(0, "/content/drive/MyDrive/NCS")
for name in list(sys.modules):
    if name.startswith("swid_retrieval"):
        del sys.modules[name]

for flag in (
    "RUN_WOOD_GALLERY_AUDIT", "RUN_WOOD_LARGE_GALLERY",
    "RUN_WOOD_CORRESPONDENCE_COMPARE", "RUN_WOOD_CORRESPONDENCE_IMAGE_TRAIN",
    "RUN_WOOD_CORRESPONDENCE_STUDY", "RUN_WOOD_EVIDENCE_STUDY",
    "RUN_GALLERY_DIAGNOSTICS", "RUN_GALLERY_STUDY",
    "RUN_REPAIR_PUBLIC_ROWS", "RUN_FINAL_SCURD_RETRAIN",
    "RUN_FINAL_COLAB_AUDIT",
):
    os.environ[flag] = "0"

os.environ.update({
    "ROOT_PATH": "/content/drive/MyDrive/NCS",
    "RUN_WOOD_REFERENCE_BUDGET": "1",
    "WOOD_LARGE_OUT": "/content/drive/MyDrive/NCS/results/wood_large_gallery_final_v1",
    "WOOD_LARGE_BASE_OUT": "/content/drive/MyDrive/NCS/results/dinov2s_retrieval_matched_v1",
    "WOOD_LARGE_QKV_OUT": "/content/drive/MyDrive/NCS/results/wood_correspondence_image_full_v1",
    "WOOD_LARGE_SOURCE_SEED": "43",
    "WOOD_REF_OUT": "/content/drive/MyDrive/NCS/results/wood_reference_budget_v1",
    "WOOD_REF_IMAGE_BATCH": "64",
    "WOOD_REF_WORKERS": "2",
})

for mode in ("preflight", "validate", "select", "public_id", "export"):
    os.environ["WOOD_REF_MODE"] = mode
    print(f"\n===== {mode.upper()} =====", flush=True)
    _ = runpy.run_module("swid_retrieval.run_overnight", run_name="__main__")

from google.colab import files
files.download("/content/drive/MyDrive/NCS/results/wood_reference_budget_v1/wood_reference_budget_review.zip")
```

`WOOD_REF_WORKERS=2` is appropriate for a Colab T4 runtime with two CPU cores;
raising it does not fix Drive I/O wait. Only the selected public-ID recipe is
run, and no 176,123-image public feature extraction is repeated. The ZIP
contains a validation grid, fold-level query predictions, a SHA-locked selection,
and the selected public-ID report. If the selected method does not materially
improve the 954-species result, the next phase is end-to-end cross-scan/cross-scale
metric training, not further test-set reranker tuning.
