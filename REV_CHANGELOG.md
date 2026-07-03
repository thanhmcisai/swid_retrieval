# Revision Changelog

## REV-1 / REV-6 partial — SC-URD training and equation fidelity

- Updated SC-URD training metadata for newly produced seed checkpoints:
  `n_way`, `k_support`, `q_query`, `lr`, `weight_decay`,
  `selection_metric=lowest_training_loss`, and `best_training_loss` are now
  saved with the checkpoint.
- Added central SC-URD hyperparameter mirrors in `config.py` so the package
  entry points and the validated variance engine share the same documented
  defaults.
- Preserved existing default seed-checkpoint names for
  `SCURD_TRAIN_LAMBDA_CONS=0.5`; non-default consistency-loss ablations now
  receive a suffix (`_lc...`) to avoid overwriting default checkpoints.
- Manuscript prose was corrected to match the implemented method:
  episodic N-way/K-shot adapter training, weak/strong consistency KL,
  `SCURD_TRAIN_LR=1e-3`, branch-normalized residual adapter, and centered
  log-sum-exp top-m scoring.
- No reported metric values were changed in this edit.

## Open follow-up tasks requiring new Colab runs

- REV-2: `experiments/scurd_training_ablation.py` now runs the consistency-loss
  ablation (`lambda_cons in {0, 0.5}` by default) through the validated
  variance engine and writes `native_experiments/scurd_consistency_ablation.csv`
  plus `native_experiments/scurd_training_ablation.json`. Pending Colab run and
  manuscript table/prose update.
- REV-1: the same module writes `native_experiments/scurd_lr_ablation.csv` for
  the empirical `lr in {1e-3,1e-4}` audit. Non-default LR seed checkpoints now
  receive an `_lr...` suffix so the audit cannot reuse the default `1e-3`
  checkpoints accidentally.
- REV-3: `experiments/supcon_seed_sensitivity.py` now provides an opt-in
  SupCon multi-seed OOD audit (`RUN_SUPCON_SEED_SENSITIVITY=1`). It trains
  seed-specific ConvNeXt SupCon checkpoints when absent, extracts ID/OOD/SWI
  embeddings from images, and writes `native_experiments/supcon_seed_sensitivity.csv`
  plus `native_experiments/supcon_seed_sensitivity_summary.csv`. It is off by
  default because it is image/GPU-heavy.
- REV-4: `experiments/ood_within_source.py` now adds a source-controlled OOD
  protocol and writes `native_experiments/ood_within_source.csv/json`. Pending
  Colab run and manuscript table/prose update.
- REV-5: add a held-out selection split if configuration-selection claims need
  to be separated from final reporting.
- REV-8: `experiments/class_incremental_baseline.py` now emits
  `native_experiments/class_incremental_baseline.json` for the frozen-old-row
  CE class-incremental control. Run the Colab pipeline and add the resulting row
  to the adaptation table if the baseline is retained.

## REV-13 — CE forgetting framing

- Manuscript wording was softened so the CE old=0 rows are described as a
  naive no-rehearsal / trainable-head fine-tuning baseline, not as proof that
  all classifier-based continual learning must fail.
- No reported metric values were changed in this edit.
