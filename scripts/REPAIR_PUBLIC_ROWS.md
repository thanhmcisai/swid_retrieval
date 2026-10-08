# Repair public-query row identity before final evaluation

The sampled image fingerprint audit found 20/24 mismatched OOD rows. Every
mismatched image has an exact DINOv2 match at another row in the original v3
cache. Some matches cross species and source boundaries. The reconstructed
pre-correction expanded CSV cannot be used as a v3 row-order authority.
The reported 2,901 FSDM41 images are those **relabelled**, not necessarily
the total FSDM41 image count; validation uses the correction marker and JSON.

## Read-only duplicate diagnostic

If the repair stops with `identical image bytes but v3 features disagree`, do
not remove that check. The DINOv2 row mapping can be correct while one or more
other feature arrays in v3 are inconsistent with the present image bytes.
After the failed run has saved `ood_row_identity.csv`, run this diagnostic cell.
It reads the saved row mapping and v3 cache, hashes only the tied image files,
and writes a small JSON report. It does not re-extract DINOv2, train a model,
or create v6.

```python
%cd /content/drive/MyDrive/NCS
!git -C swid_retrieval pull

import os, sys, runpy
sys.path.insert(0, "/content/drive/MyDrive/NCS")
for name in list(sys.modules):
    if name.startswith("swid_retrieval"):
        del sys.modules[name]

os.environ.update({
    "ROOT_PATH": "/content/drive/MyDrive/NCS",
    "RUN_REPAIR_PUBLIC_ROWS": "1",
    "PUBLIC_REPAIR_DIAGNOSE_ONLY": "1",
    "RUN_FINAL_SCURD_RETRAIN": "0",
    "RUN_FINAL_COLAB_AUDIT": "0",
})
_ = runpy.run_module("swid_retrieval.run_overnight", run_name="__main__")
```

Inspect or download
`results/public_row_repair_v1/ood_duplicate_feature_diagnostics.json` before
deciding whether any legacy features can be reused. Keep the original v3/v5
files and partial row mapping unchanged.

This job extracts DINOv2 features for all corrected public ID/OOD images,
matches each image to its original v3 feature, and creates a **new** v6 cache.
It does not train or overwrite any checkpoint, v3, v4, or v5 cache. Non-CE
public embeddings/logits are reindexed from v3; the retrained CE-Full public
embeddings/logits and all SWI gallery arrays are copied unchanged from v5.

## Colab cell

```python
%cd /content/drive/MyDrive/NCS
!git -C swid_retrieval pull

import os, sys, runpy
from pathlib import Path
sys.path.insert(0, "/content/drive/MyDrive/NCS")
for name in list(sys.modules):
    if name.startswith("swid_retrieval"):
        del sys.modules[name]

root = Path("/content/drive/MyDrive/NCS")
for name in ("embedding_cache_full954_v3.npz",
             "embedding_cache_full954_v5_retrained_ce_corrected_public.npz",
             "ID_images_expanded.csv", "OOD_images_expanded.csv"):
    assert (root / name).is_file(), name

os.environ.update({
    "ROOT_PATH": str(root),
    "DEVICE": "cuda",
    "RUN_REPAIR_PUBLIC_ROWS": "1",
    "PUBLIC_REPAIR_DIAGNOSE_ONLY": "0",
    "RUN_FINAL_SCURD_RETRAIN": "0",
    "RUN_FINAL_COLAB_AUDIT": "0",
    "PUBLIC_REPAIR_BATCH_SIZE": "16",
    "PUBLIC_REPAIR_WORKERS": "4",
    "PUBLIC_REPAIR_TARGET_CACHE_NAME": "embedding_cache_full954_v6_public_row_verified.npz",
    "PUBLIC_REPAIR_AUDIT_DIR": str(root / "results/public_row_repair_v1"),
})
_ = runpy.run_module("swid_retrieval.run_overnight", run_name="__main__")
```

Image extraction may be lengthy on Drive. Progress is saved every 1024 images
and resumes from the same audit directory after a Colab disconnect. Do not
change the input CSVs or caches between attempts. If the job reports weak,
unresolved ambiguous, duplicate, or wrong-label matches, stop and inspect the row-level
`id_row_identity.csv` / `ood_row_identity.csv` in the audit directory. Do not
relax thresholds merely to get a cache file.

Exact `- Copy.jpg` image pairs may have identical DINOv2 scores. The repair
assigns them one-to-one only if their file SHA-256 values, source/label, and
every original v3 feature/logit agree within a small numerical tolerance.
For public ID only, a non-equivalent pair can instead be assigned by its
original row position when every non-tied ID row independently maps to that
same position, both tied positions have the expected label and DINOv2 match,
the two filenames are an original/`- Copy` pair in the same folder, and the
files have different bytes. Identical files with conflicting v3 features
instead stop the repair: that is a source-cache inconsistency, not a tie. This
is recorded as `anchored_id_position`, including the feature disagreement;
it is not applied globally to OOD, whose old and current row orders differ.
For OOD, a two-row DINOv2 tie may use a **local** source-row offset only when
five independently matched same-species/same-source neighbours on each side
of both current rows agree on the same offset, both predicted v3 rows have
the expected label and DINOv2 match, neither is already used by another
image, and the two image files are not byte-identical when old features
disagree. The uploaded OOD audit had 12 such pairs, with offsets 2882, 3562
or 5595 depending on the local segment; these are checked afresh, not
hard-coded. The resolution is recorded as `anchored_ood_offset`. Decisions
are recorded under `duplicate_groups` in `summary.json`. A preliminary row
identity CSV is saved before tie resolution, so a later failure still has a
diagnostic file.

Successful output:

- `ROOT_PATH/embedding_cache_full954_v6_public_row_verified.npz`
- `ROOT_PATH/embedding_cache_full954_v6_public_row_verified_meta.json`
- `ROOT_PATH/results/public_row_repair_v1/summary.json`
- `ROOT_PATH/results/public_row_repair_v1/{id,ood}_row_identity.csv`

The first subsequent SC-URD run should use a **new** output directory and set
`FINAL_SCURD_CACHE_NAME=embedding_cache_full954_v6_public_row_verified.npz`.
First use `FINAL_SCURD_DIAGNOSE_ONLY=1` to confirm all sampled fingerprints
pass; only then enable head training and recompute affected experiments in a
new stamped results directory. Set `RUN_REPAIR_PUBLIC_ROWS=0` before any of
those later runs. Do not blend old v5 OOD metrics with v6.
