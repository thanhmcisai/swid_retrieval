# Final SC-URD head-only rerun (Colab)

This runs a new three-seed head-only experiment. It does not overwrite the
historical selected checkpoint, the old seed checkpoints, the CE checkpoint,
or the v5/exp4 caches. The original meta-val artifact fixes raw scoring,
`tau=0.07`, and `top_m=50`. The new seeds share one recorded weak/strong
meta-train cache, recipe, and independent seed values (42, 43, 44).

The prior selected checkpoint is **not** assumed comparable to the new seeds.
Use the new three-seed result as a separate controlled rerun until its
training-data/recipe identity with the selected checkpoint is established.

## Cell 1: head rerun and provenance

```python
%cd /content/drive/MyDrive/NCS
!git -C swid_retrieval pull

import os, sys, runpy
sys.path.insert(0, "/content/drive/MyDrive/NCS")
for m in list(sys.modules):
    if m.startswith("swid_retrieval"):
        del sys.modules[m]

os.environ["ROOT_PATH"] = "/content/drive/MyDrive/NCS"
os.environ["DEVICE"] = "cuda"
os.environ["RUN_FINAL_SCURD_RETRAIN"] = "1"
os.environ["RUN_FINAL_COLAB_AUDIT"] = "0"
os.environ["FINAL_AUDIT_RUN_ROOT"] = (
    "/content/drive/MyDrive/NCS/results/"
    "paper_reframe_full954_retrained_ce_corrected_public"
)
os.environ["FINAL_SCURD_OUT"] = (
    "/content/drive/MyDrive/NCS/results/final_scurd_retrain_v1"
)
os.environ["FULL954_CACHE_NAME"] = (
    "embedding_cache_full954_v5_retrained_ce_corrected_public.npz"
)
os.environ["FINAL_SCURD_IMAGE_FINGERPRINT"] = "1"
os.environ["FINAL_SCURD_BUILD_META"] = "0"

# Set only if the auto-detected original weak/strong meta cache is elsewhere:
# os.environ["FINAL_SCURD_META_CACHE"] = "/content/drive/MyDrive/NCS/.../urd_v2_meta_dinov2_embeddings_v2.npz"

runpy.run_module("swid_retrieval.run_overnight", run_name="__main__")
```

The runner fails before training if meta labels/order, weak DINOv2 features,
or sampled public/VN26 image fingerprints do not match the frozen caches.
Existing new-run seed checkpoints are reused only when their recipe and
meta-cache SHA-256 match. A rerun after Colab disconnect resumes remaining seeds.

If the weak/strong meta cache is absent, set
`FINAL_SCURD_BUILD_META=1` and rerun the same cell. This builds only the
strong-augmentation DINOv2 meta-train features from the SWI images; weak and
meta-val features are taken from the frozen v5 cache. It can take substantial
time and requires the source SWI images. The new meta cache is saved in the
isolated output folder and reused on subsequent runs. Do not set this flag
if the original meta cache exists: preserving the old cache makes the
controlled comparison more informative.

If sampled image fingerprints fail, do not accept the old exp4/public cache
as identity-verified. `FINAL_SCURD_IMAGE_FINGERPRINT=0` permits a head-only
rerun, but the resulting provenance report records the image check as absent.

## Cell 2: small review ZIP

```python
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

out = Path("/content/drive/MyDrive/NCS/results/final_scurd_retrain_v1")
review = out.parent / "final_scurd_retrain_review.zip"
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
