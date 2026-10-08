# Verified-public-row final audit on Colab

This command runs inference only. It does not train, rebuild the full-954 cache,
or overwrite the manuscript or historical export. Run it after mounting Drive
and synchronizing this `swid_retrieval` revision to `NCS/swid_retrieval`.
The audit-only flag routes `run_overnight` directly to the audit. It ignores
the old full-pipeline flags (`RUN_CE_CACHE_UPDATE`, `FORCE_REBUILD_FULL954`,
`RUN_BUILD_FULL954`), so it cannot retrain or rewrite the v5/v6 caches.
Set `FINAL_AUDIT_CACHE_NAME` explicitly for v6. Without that flag, the
standalone audit still defaults to the historical v5 cache.

Required Drive inputs under `/content/drive/MyDrive/NCS`:

- `embedding_cache_full954_v6_public_row_verified.npz` and its repair summary
- `exp4_embedding_cache_v3.npz`
- `checkpoints/ce_954sp_convnext_base.pt`
- `ID_images_expanded.csv`, `OOD_images_expanded.csv`
- `dataset_label_corrections.json`
- `ID_species_public.csv`, `OOD_species_public.csv`
- `swi_manifest.json` and source images
- `results/paper_reframe_full954_retrained_ce_corrected_public/hyperparameters/scurd_hyperparameter_selection.json`
- The selected main SC-URD checkpoint and seed 42/43/44 checkpoints in
  `results/paper_reframe_full954_retrained_ce_corrected_public/deployment/research_directions/`

Use one Colab cell. The new output directory avoids a partial audit from the
earlier subprocess-based attempt:

```python
%cd /content/drive/MyDrive/NCS
!git -C swid_retrieval pull --ff-only
import os, sys, runpy
root = "/content/drive/MyDrive/NCS"
sys.path.insert(0, root)
for name in list(sys.modules):
    if name.startswith("swid_retrieval"):
        del sys.modules[name]
os.environ["ROOT_PATH"] = root
os.environ["RUN_REPAIR_PUBLIC_ROWS"] = "0"
os.environ["RUN_FINAL_SCURD_RETRAIN"] = "0"
os.environ["RUN_FINAL_COLAB_AUDIT"] = "1"
os.environ["FINAL_AUDIT_CACHE_NAME"] = "embedding_cache_full954_v6_public_row_verified.npz"
os.environ["FINAL_AUDIT_PREFLIGHT_ONLY"] = "0"
os.environ["FINAL_AUDIT_RUN_ROOT"] = f"{root}/results/paper_reframe_full954_retrained_ce_corrected_public"
os.environ["FINAL_AUDIT_OUT"] = f"{root}/results/paper_reframe_full954_retrained_ce_corrected_public/final_colab_audit_v6"
os.environ["FINAL_AUDIT_BATCH_SIZE"] = "64"
os.environ["FINAL_AUDIT_WORKERS"] = "4"
os.environ["DEVICE"] = "cuda"
_ = runpy.run_module("swid_retrieval.run_overnight", run_name="__main__")
```

Set `FINAL_AUDIT_PREFLIGHT_ONLY=1` for an input validation pass without
starting extraction. Set it back to `0` for the full audit. The command is
resumable by running the same cell again. It hashes input
artifacts and evaluator source code; if any input or code changes, start with
a **new** `--out` directory. The
stage markers detect altered outputs. An interrupted `native_ce` write may leave
an unfinished `native_ce/` directory; inspect it and choose a new `--out`.

The output folder contains:

- `summary.json`: selected raw deployment metrics and artifact pointers.
- `native_ce/native_ce_audit.json`: native CE macro accuracy from fresh logits,
  with per-row feature/logit comparison to the selected cache.
- `ce_exp4_fresh.npz`, `ce_vn26_fresh.json` and the `ce_rq4_*/` folders: fresh CE-only
  SWI-scale/VN26 outputs and complete RQ4 matrices with CIs. The original exp4
  CE-Full path uses 954-class **logits** (despite its generic embedding key) and
  PIL RGB reading for both image sources. The audit reproduces that protocol in
  `ce_rq4_legacy_logits_954/` and separately reports normalized 512-D
  penultimate features in `ce_rq4_features_512/`; they must not be conflated or
  substituted in the manuscript without revising the protocol description.
- `scurd_raw_centered_seed_audit.json` and `.csv`: main and seed checkpoint
  results for both scoring modes, including the selected main checkpoint on
  the matched-954 gallery.
- `scurd_rq4_raw/` and `scurd_rq4_centered/`: complete selected-checkpoint
  SWI-to-VN26 and cross-magnification matrices from the existing RQ4 evaluator.
- `scurd_seed_summary.csv`: sample standard deviations across the three seeds.
- `scurd_triage_raw_centered.csv`: coverage versus accuracy on the public-ID
  queries, using the same evidence score and thresholds as the existing engine.
- `scurd_gallery_resampling_raw_centered.csv`: the selected checkpoint's
  gallery strategies with 100 finite-K draws on the fixed 240-query subset.
- `raw_gallery_engine/gallery_resampling_variance.csv`: complete 66-row raw
  gallery matrix for six methods. Its SC-URD rows are checked against the
  independent evaluator above.
- `scurd_raw_centered_seed_audit.json` also contains OOD-only K=10 results
  over 50 reference draws for the selected checkpoint.
- `paired_raw_baselines.json`: per-species comparisons and Holm-adjusted tests.
- `inputs.json`, `preflight.json` and `*.done.json`: source hashes and provenance.

The legacy exp4 cache stores no ordered image paths. Matching its label counts
does not prove that its DINOv2 features came from the same individual images.
Checkpoint metadata alone also cannot prove identical training data or
preprocessing. Review these limits before using the output to close the paper.
