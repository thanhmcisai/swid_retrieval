# Retrieval frontier study (exploratory; do not update the paper from screening)

This study leaves the submission pipeline and verified public embedding cache unchanged.
Model/epoch selection uses SWI meta-val only. Public-ID/OOD/VN26 evaluation is a
separate, explicit final step after the candidate set is frozen. Public results
have previously informed the research design and are not an untouched test.

## Recipes and endpoints

- `pretrained_control`, `pretrained_nearest`: same-resolution frozen DINOv2-S/B,
  ConvNeXt-T, or licensed DINOv3-S controls; no random projection is applied.
- `metric_large`: 16/32/64-way, scan-disjoint, current-reference gallery CE.
  `microbatch` only changes the encoder replay chunk, not the negative count.
- `metric_large_ce`: adds a 557-class auxiliary CE head at weight 0.2.
- `metric_large_scale`: prefers different source-field crop scales across
  scan-disjoint support/query and adds a weak positive consistency term.
- `metric_large_no_hard`, `prototype_large`, `supcon_large`: hard-negative,
  prototype-scoring, and loss-function controls.
- `local_evidence`: trainable 64-d local token projection from WoodPatternNet;
  combines global gallery CE with spatially unordered local matching. At
  inference, global scores shortlist 32 species, up to 3 query-nearest references
  per species are compared by 12 tissue-evidence tokens. This is **not** an
  anatomically annotated detector.
- `local_evidence_ce_scale`: local evidence plus auxiliary CE and cross-scale
  positive training. Its component ablations are the variants above.

The fixed meta-val reference protocol has five references and five queries per
eligible species from different Tw source scans. The 24-species subset is nested
inside the 57-species eligible set, so the two endpoints are not independent.
The optional stress gallery adds **one** reference for every SWI meta-train and
meta-val species (637 species in the audited manifest); only scan-disjoint
meta-val species provide queries. The predeclared selection score for this study
is the mean of 24-species R@1 and stress-gallery R@1, averaged across folds.
Stress-gallery references from meta-train are seen species, not an unseen-954
replacement. Each run saves the complete recipe and its source/checkpoint hashes.

Public evaluation reports species R@1 from the local reranker when enabled.
Image mAP/MRR and OOD nearest-distance remain **global-embedding** endpoints;
the evaluation JSON records this distinction. The public K-shot split has no
verified specimen IDs and must remain exploratory.

## Colab cell 1: setup and preflight

Mount Drive first. Pull only after the new commit is visible on GitHub.

```python
%cd /content/drive/MyDrive/NCS
!git -C swid_retrieval pull
!git -C swid_retrieval rev-parse --short HEAD

import os, sys, runpy
from pathlib import Path
sys.path.insert(0, "/content/drive/MyDrive/NCS")
for name in list(sys.modules):
    if name.startswith("swid_retrieval"):
        del sys.modules[name]

root = Path("/content/drive/MyDrive/NCS")
out = root / "results/retrieval_frontier_study_v1"
for flag in ("RUN_REPAIR_PUBLIC_ROWS", "RUN_FINAL_SCURD_RETRAIN", "RUN_FINAL_COLAB_AUDIT"):
    os.environ[flag] = "0"
os.environ.update({
    "ROOT_PATH": str(root), "DEVICE": "cuda", "RUN_GALLERY_STUDY": "1",
    "GALLERY_STUDY_OUT": str(out), "GALLERY_STUDY_BACKBONE": "woodpattern_tiny",
    "GALLERY_STUDY_MODE": "preflight", "GALLERY_STUDY_SEEDS": "42",
    "GALLERY_STUDY_IMAGE_SIZE": "224", "GALLERY_STUDY_VALIDATION_FOLDS": "2",
    "GALLERY_STUDY_VALIDATION_TRAIN_DISTRACTORS": "1",
    "GALLERY_STUDY_WAYS": "16,32,64", "GALLERY_STUDY_PILOT_CHECKPOINTS": "0",
    "GALLERY_STUDY_PRELOAD": "0", "IMAGE_CACHE_DIR": "/content/cache_images",
})
_ = runpy.run_module("swid_retrieval.run_overnight", run_name="__main__")
```

## Colab cell 2: same-resolution pretrained controls

Do not set `GALLERY_STUDY_INIT_CHECKPOINT` for controls. ConvNeXt-T uses `timm`;
install it if missing. DINOv3 is optional and requires a separately licensed
local checkpoint path (`GALLERY_STUDY_DINOV3_WEIGHTS`). All controls use 224 px.

```python
os.environ.pop("GALLERY_STUDY_INIT_CHECKPOINT", None)
os.environ.update({
    "GALLERY_STUDY_MODE": "screen", "GALLERY_STUDY_VARIANTS": "pretrained_control,pretrained_nearest",
    "GALLERY_STUDY_SEEDS": "42", "GALLERY_STUDY_EPOCHS": "8",
    "GALLERY_STUDY_EPISODES_PER_EPOCH": "500", "GALLERY_STUDY_MICROBATCH": "24",
    "GALLERY_STUDY_WORKERS": "4", "GALLERY_STUDY_IMAGE_BATCH": "32",
})
for backbone in ("dinov2_vits14", "dinov2_vitb14", "convnext_tiny"):
    os.environ["GALLERY_STUDY_BACKBONE"] = backbone
    _ = runpy.run_module("swid_retrieval.run_overnight", run_name="__main__")
# Optional only after acquiring the licensed checkpoint:
# os.environ["GALLERY_STUDY_DINOV3_WEIGHTS"] = str(root / "checkpoints/<licensed_vits16>.pth")
# os.environ["GALLERY_STUDY_BACKBONE"] = "dinov3_vits16"
# _ = runpy.run_module("swid_retrieval.run_overnight", run_name="__main__")
```

## Colab cell 3: WoodPatternNet screening, seed 42

Use the previously saved ten-epoch warm-up if present; otherwise first run
`GALLERY_STUDY_MODE=warmup`, `GALLERY_STUDY_VARIANTS=supervised_warmup`,
`GALLERY_STUDY_WARMUP_EPOCHS=10`, `GALLERY_STUDY_WARMUP_STEPS=1000`,
`GALLERY_STUDY_WARMUP_BATCH=64`, then point to its `best.pt`. Do not mix warm-up
checkpoints from another seed or manifest.

```python
warmup = root / ("results/woodpattern_warmup_extended_b79e432/woodpattern_tiny/"
                 "supervised_warmup/seed_42/best.pt")
assert warmup.is_file(), warmup
os.environ.update({
    "GALLERY_STUDY_BACKBONE": "woodpattern_tiny",
    "GALLERY_STUDY_INIT_CHECKPOINT": str(warmup),
    "GALLERY_STUDY_MODE": "screen", "GALLERY_STUDY_SEEDS": "42",
    "GALLERY_STUDY_VARIANTS": (
        "metric_large,metric_large_ce,metric_large_scale,metric_large_no_hard,"
        "prototype_large,supcon_large,local_evidence,local_evidence_ce_scale"),
})
_ = runpy.run_module("swid_retrieval.run_overnight", run_name="__main__")
print(out / "meta_val_screen.csv")
```

Screening includes eight variants and is intentionally finite; it is not an
exhaustive Cartesian sweep. On a T4, run one variant at a time if the runtime is
short. Rerunning the same cell resumes from each variant's `latest.pt` and
rebuilds the screening summary. A changed recipe must use a different output
directory. Watch for non-finite loss, AMP skips, and reference-spread collapse;
do not select a method solely because its 24-species score is high.

## Colab cell 4: replicate selected candidates

Freeze the shortlist using `meta_val_screen.csv` **before** opening public test
results. Use the same `GALLERY_STUDY_EPOCHS`, episode count, ways, image size,
folds and stress protocol for all seeds. The seed-specific warm-up checkpoints
from the previous study are under `woodpattern_warmup_replicates_10fd4bb`.
Replace `candidates` with the baseline and the two best non-collapsed variants.

```python
candidates = "metric_large,local_evidence"  # Edit from meta-val screening only.
os.environ.update({
    "GALLERY_STUDY_OUT": str(out), "GALLERY_STUDY_BACKBONE": "woodpattern_tiny",
    "GALLERY_STUDY_EPOCHS": "8", "GALLERY_STUDY_EPISODES_PER_EPOCH": "500",
    "GALLERY_STUDY_MICROBATCH": "24", "GALLERY_STUDY_IMAGE_BATCH": "32",
})
for seed in (43, 44):
    warmup = root / ("results/woodpattern_warmup_replicates_10fd4bb/woodpattern_tiny/"
                     f"supervised_warmup/seed_{seed}/best.pt")
    assert warmup.is_file(), warmup
    os.environ.update({"GALLERY_STUDY_INIT_CHECKPOINT": str(warmup),
                       "GALLERY_STUDY_SEEDS": str(seed),
                       "GALLERY_STUDY_VARIANTS": candidates,
                       "GALLERY_STUDY_MODE": "screen"})
    _ = runpy.run_module("swid_retrieval.run_overnight", run_name="__main__")
print(out / "meta_val_screen_summary.csv")
```

Architecture ablations (`woodpattern_no_attention`,
`woodpattern_single_scale`) require their own supervised warm-up checkpoints.
Do not load the `woodpattern_tiny` warm-up under another backbone name. An
optional 518-px DINOv2 control must use a **separate output directory** because
checkpoint identity includes the input resolution.

## Optional Colab cell 4a: end-to-end DINOv2-S control

This is a stronger initialization control, not evidence of novelty. Run it in
a separate output directory with the same meta-val folds and stress gallery.
Use a smaller encoder replay microbatch on a T4; this does **not** reduce the
16/32/64-way episode or its negative count. Start with one seed and 150 episodes
per epoch; only allocate a full matched-budget run if it is competitive. Do
not compare the pilot directly to the 500-episode WoodPatternNet runs.

```python
os.environ.pop("GALLERY_STUDY_INIT_CHECKPOINT", None)
os.environ.update({
    "GALLERY_STUDY_OUT": str(root / "results/retrieval_frontier_dinov2s_pilot_v1"),
    "GALLERY_STUDY_BACKBONE": "dinov2_vits14",
    "GALLERY_STUDY_MODE": "screen",
    "GALLERY_STUDY_VARIANTS": "metric_large,prototype_large,metric_large_ce",
    "GALLERY_STUDY_SEEDS": "42", "GALLERY_STUDY_EPOCHS": "5",
    "GALLERY_STUDY_EPISODES_PER_EPOCH": "150",
    "GALLERY_STUDY_MICROBATCH": "8", "GALLERY_STUDY_IMAGE_BATCH": "16",
})
_ = runpy.run_module("swid_retrieval.run_overnight", run_name="__main__")
print(root / "results/retrieval_frontier_dinov2s_pilot_v1/meta_val_screen.csv")
```

If promising, use a **new** directory with 8 epochs and 500 episodes per
epoch for both the DINOv2-S control and its WoodPatternNet comparator; then
replicate seeds 43 and 44 before evaluating public data. The optional DINOv3-S
fine-tune uses the same recipe after setting `GALLERY_STUDY_DINOV3_WEIGHTS` to
a licensed checkpoint and `GALLERY_STUDY_BACKBONE=dinov3_vits16`.

## Optional Colab cell 4b: WoodPatternNet architecture ablations

Run after the main seed-42 screening. These models train from scratch and
must each have their own supervised warm-up. Keep this in a separate directory;
the warm-up and metric stages remain resumable within it.

```python
arch_out = root / "results/retrieval_frontier_architecture_v1"
os.environ.update({
    "GALLERY_STUDY_OUT": str(arch_out), "GALLERY_STUDY_SEEDS": "42",
    "GALLERY_STUDY_WARMUP_EPOCHS": "10", "GALLERY_STUDY_WARMUP_STEPS": "1000",
    "GALLERY_STUDY_WARMUP_BATCH": "64", "GALLERY_STUDY_EPOCHS": "8",
    "GALLERY_STUDY_EPISODES_PER_EPOCH": "500",
    "GALLERY_STUDY_MICROBATCH": "24", "GALLERY_STUDY_IMAGE_BATCH": "32",
})
for backbone in ("woodpattern_no_attention", "woodpattern_single_scale"):
    os.environ["GALLERY_STUDY_BACKBONE"] = backbone
    os.environ.pop("GALLERY_STUDY_INIT_CHECKPOINT", None)
    os.environ.update({"GALLERY_STUDY_MODE": "warmup",
                       "GALLERY_STUDY_VARIANTS": "supervised_warmup"})
    _ = runpy.run_module("swid_retrieval.run_overnight", run_name="__main__")
    warmup = arch_out / backbone / "supervised_warmup/seed_42/best.pt"
    assert warmup.is_file(), warmup
    os.environ.update({"GALLERY_STUDY_INIT_CHECKPOINT": str(warmup),
                       "GALLERY_STUDY_MODE": "screen",
                       "GALLERY_STUDY_VARIANTS": "metric_large"})
    _ = runpy.run_module("swid_retrieval.run_overnight", run_name="__main__")
print(arch_out / "meta_val_screen.csv")
```

Before cell 5, restore `GALLERY_STUDY_OUT=str(out)`, set
`GALLERY_STUDY_BACKBONE=woodpattern_tiny`, and restore the main-study epoch,
episode, microbatch and image-batch settings. Evaluate an architecture ablation
only if it entered the predeclared meta-val shortlist.

## Colab cell 5: final exploratory public evaluation

Only after freezing the winning recipe and seed set. Keep every environment
setting from training unchanged, including `GALLERY_STUDY_INIT_CHECKPOINT` for
each seed. This step requires the verified corrected-public cache and extracts
new model embeddings; the local method additionally writes aligned FP16 token
arrays. Full 176,123-image SWI extraction is the expensive part. Do not run it
for every screened variant.

```python
os.environ["GALLERY_STUDY_MODE"] = "evaluate"
os.environ["GALLERY_STUDY_VARIANTS"] = candidates  # Frozen meta-val shortlist.
os.environ["GALLERY_STUDY_EVAL_VN26"] = "1"
os.environ["GALLERY_STUDY_OUT"] = str(out)
os.environ["GALLERY_STUDY_BACKBONE"] = "woodpattern_tiny"
os.environ.update({"GALLERY_STUDY_EPOCHS": "8", "GALLERY_STUDY_EPISODES_PER_EPOCH": "500",
                   "GALLERY_STUDY_MICROBATCH": "24", "GALLERY_STUDY_IMAGE_BATCH": "32"})
for seed in (42, 43, 44):
    warmup_dir = ("woodpattern_warmup_extended_b79e432" if seed == 42 else
                  "woodpattern_warmup_replicates_10fd4bb")
    warmup = root / "results" / warmup_dir / "woodpattern_tiny/supervised_warmup" / f"seed_{seed}/best.pt"
    os.environ["GALLERY_STUDY_INIT_CHECKPOINT"] = str(warmup)
    os.environ["GALLERY_STUDY_SEEDS"] = str(seed)
    _ = runpy.run_module("swid_retrieval.run_overnight", run_name="__main__")
print(out / "comparison_summary.csv")
```

Check `evaluation.json`, `comparison.csv`, `comparison_summary.csv`,
`cardinality_curve.csv`, and per-seed `progress.json`. For discussion, export
those files and `meta_val_screen*.csv`; large `.npy` arrays and `.pt`
checkpoints are not required unless debugging an extraction/provenance failure.
