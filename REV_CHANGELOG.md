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

- REV-2: run consistency-loss ablation (`lambda_cons in {0, 0.5}`) and report
  mean/std across seeds.
- REV-3: surface SC-URD seed mean/std in the main tables and retrain SupCon
  across seeds if the OOD AUROC comparison remains central.
- REV-4: add same-source OOD protocol to separate species novelty from source
  domain novelty.
- REV-5: add a held-out selection split if configuration-selection claims need
  to be separated from final reporting.
