# Final SC-URD head-only rerun (Colab)

This runs a new three-seed head-only experiment. It does not overwrite the
historical selected checkpoint, the old seed checkpoints, the CE checkpoint,
or the v5/exp4 caches. The original meta-val artifact fixes raw scoring,
`tau=0.07`, and `top_m=50`. The new seeds share one recorded weak/strong
meta-train cache, recipe, and independent seed values (42, 43, 44).

The prior selected checkpoint is **not** assumed comparable to the new seeds.
Use the new three-seed result as a separate controlled rerun until its
training-data/recipe identity with the selected checkpoint is established.

The head trainer caches species-to-image indices once per seed and keeps the
weak/strong feature matrices on the selected device (about 0.71 GiB for
124,577 images and 768-D float32 features). This removes repeated full-label
scans, per-episode feature copies, and avoidable CPU/GPU synchronization
without changing episode draws, optimizer steps, loss, or batch composition.
Low instantaneous GPU utilization is still
normal for the small 16-way head and does not by itself imply an out-of-memory
problem. Do not reduce episode count or enable mixed precision when comparing
against the frozen three-seed recipe.

If a Colab run was started with an earlier training-engine revision, let it
finish on that revision or use a **new** `FINAL_SCURD_OUT` after pulling this
optimization. The provenance check deliberately rejects resuming a mixed-code
run in the same output directory; incomplete seeds cannot resume mid-epoch.

## Cell 1: head rerun and provenance

```python
%cd /content/drive/MyDrive/NCS
!git -C swid_retrieval pull --ff-only

import os, sys, runpy
sys.path.insert(0, "/content/drive/MyDrive/NCS")
for m in list(sys.modules):
    if m.startswith("swid_retrieval"):
        del sys.modules[m]

os.environ["ROOT_PATH"] = "/content/drive/MyDrive/NCS"
os.environ["DEVICE"] = "cuda"
os.environ["RUN_FINAL_SCURD_RETRAIN"] = "1"
os.environ["RUN_FINAL_COLAB_AUDIT"] = "0"
os.environ["RUN_REPAIR_PUBLIC_ROWS"] = "0"
os.environ["FINAL_AUDIT_RUN_ROOT"] = (
    "/content/drive/MyDrive/NCS/results/"
    "paper_reframe_full954_retrained_ce_corrected_public"
)
os.environ["FINAL_SCURD_OUT"] = (
    "/content/drive/MyDrive/NCS/results/final_scurd_retrain_v3_v6_fast"
)
os.environ["FULL954_CACHE_NAME"] = (
    "embedding_cache_full954_v6_public_row_verified.npz"
)
os.environ["FINAL_SCURD_CACHE_NAME"] = os.environ["FULL954_CACHE_NAME"]
os.environ["FINAL_SCURD_IMAGE_FINGERPRINT"] = "1"
os.environ["FINAL_SCURD_BUILD_META"] = "0"
os.environ["FINAL_SCURD_DIAGNOSE_ONLY"] = "0"

# Set only if the auto-detected original weak/strong meta cache is elsewhere:
# os.environ["FINAL_SCURD_META_CACHE"] = "/content/drive/MyDrive/NCS/.../urd_v2_meta_dinov2_embeddings_v2.npz"

_ = runpy.run_module("swid_retrieval.run_overnight", run_name="__main__")
```

The runner fails before training if meta labels/order, weak DINOv2 features,
or sampled public/VN26 image fingerprints do not match the frozen caches.
To diagnose a fingerprint failure without starting training, run Cell 1 with
`os.environ["FINAL_SCURD_DIAGNOSE_ONLY"] = "1"`. The JSON report is written to
`final_scurd_retrain_v3_v6_fast/image_fingerprint_diagnostics.json`; it includes
same-row cosine and nearest-row matches in v6 and, when present, v3. Remove
that variable (or set it to `"0"`) before a real head rerun. A near-1.0
nearest-row match with a low same-row score indicates row-order corruption;
low nearest scores require investigation of image identity and preprocessing.
Existing new-run seed checkpoints are reused only when their recipe and
meta-cache SHA-256 match. A rerun after Colab disconnect resumes remaining seeds.

If the weak/strong meta cache is absent, set
`FINAL_SCURD_BUILD_META=1` and rerun the same cell. This builds only the
strong-augmentation DINOv2 meta-train features from the SWI images; weak and
meta-val features are taken from the verified v6 cache. It can take substantial
time and requires the source SWI images. The new meta cache is saved in the
isolated output folder and reused on subsequent runs. Do not set this flag
if the original meta cache exists: preserving the old cache makes the
controlled comparison more informative.

If sampled image fingerprints fail, do not accept the old exp4/public cache
as identity-verified and do not train or update manuscript numbers until the
cause is resolved. The historical public expanded CSV was regenerated from
folder enumeration after the original file went missing; equal row counts and
labels do not prove that the rebuilt CSV has the original image order.
The public-OOD mismatch was confirmed and a full-row repair is documented in
`scripts/REPAIR_PUBLIC_ROWS.md`. The cell above requires the verified v6 cache.

## Cell 2: small review ZIP

```python
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

out = Path("/content/drive/MyDrive/NCS/results/final_scurd_retrain_v3_v6_fast")
review = out.parent / "final_scurd_retrain_v3_v6_fast_review.zip"
with ZipFile(review, "w", ZIP_DEFLATED) as z:
    for folder in (out, out / "evaluation", out / "training_logs"):
        if folder.exists():
            for path in folder.glob("*.json"):
                z.write(path, path.relative_to(out.parent))
            for path in folder.glob("*.csv"):
                z.write(path, path.relative_to(out.parent))
print(review, round(review.stat().st_size / 1e6, 2), "MB")
```

Send only this ZIP for paper review. Do not upload the large `.npz` cache or
`.pt` checkpoints unless a subsequent row-level audit specifically needs them.
