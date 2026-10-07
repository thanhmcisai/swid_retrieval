# Corrected-public final audit on Colab

This command runs inference only. It does not train, rebuild the full-954 cache,
or overwrite the manuscript or historical export. Run it after mounting Drive
and synchronizing this `swid_retrieval` revision to `NCS/swid_retrieval`.
Run `!git -C /content/drive/MyDrive/NCS/swid_retrieval pull --ff-only` in Colab
before the cell below. The preflight must finish before starting the full run.

Required Drive inputs under `/content/drive/MyDrive/NCS`:

- `embedding_cache_full954_v5_retrained_ce_corrected_public.npz`
- `exp4_embedding_cache_v3.npz`
- `checkpoints/ce_954sp_convnext_base.pt`
- `ID_images_expanded.csv`, `OOD_images_expanded.csv`
- `dataset_label_corrections.json`
- `ID_species_public.csv`, `OOD_species_public.csv`
- `swi_manifest.json` and source images
- `results/paper_reframe_full954_retrained_ce_corrected_public/hyperparameters/scurd_hyperparameter_selection.json`
- The selected main SC-URD checkpoint and seed 42/43/44 checkpoints in
  `results/paper_reframe_full954_retrained_ce_corrected_public/deployment/research_directions/`

Use one Colab cell after the code is present on Drive:

```python
import subprocess, sys

root = "/content/drive/MyDrive/NCS"
cmd = [
    sys.executable, "-m", "swid_retrieval.final_colab_audit",
    "--root", root,
    "--run-root", f"{root}/results/paper_reframe_full954_retrained_ce_corrected_public",
    "--out", f"{root}/results/paper_reframe_full954_retrained_ce_corrected_public/final_colab_audit",
    "--device", "cuda", "--batch-size", "64", "--workers", "4",
]
subprocess.run(cmd + ["--preflight"], cwd=root, check=True)
subprocess.run(cmd, cwd=root, check=True)
```

The command is resumable by running the same cell again. It hashes all input
artifacts; if any input changes, start with a **new** `--out` directory. The
stage markers detect altered outputs. An interrupted `native_ce` write may leave
an unfinished `native_ce/` directory; inspect it and choose a new `--out`.

The output folder contains:

- `summary.json`: selected raw deployment metrics and artifact pointers.
- `native_ce/native_ce_audit.json`: native CE macro accuracy from fresh logits,
  with per-row feature/logit comparison to the v5 cache.
- `ce_exp4_fresh.npz`, `ce_vn26_fresh.json` and `ce_rq4_fresh/`: fresh CE-only
  SWI-scale/VN26 embeddings and the complete RQ4 matrix with CIs.
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
