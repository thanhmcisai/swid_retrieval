# -*- coding: utf-8 -*-
"""Validated experiment engines, kept as-is and run via exec() by the orchestrator.

These are the original Colab scripts, REUSED verbatim (not re-implemented) to
preserve the exact validated numbers — see swid_retrieval/experiments/__init__.py
for the reuse-not-rewrite rationale. They are exec()'d (not imported), each with
its own env-driven config, so their behaviour is identical to running them
standalone; the orchestrator only sets env vars and the gallery scope.

  - variance_retrieval_evidence_colab.py : RQ1 headline (native/prototype/R@1),
    gallery K-shot resampling, OOD AUROC/FPR95, RQ5 ablation + SC-URD memory modes,
    SC-URD seed train/sensitivity, reviewer-gap. Supports GALLERY_SCOPE in
    {full_swi, ce_train, id_only}.
  - review_evidence_colab.py : per-species accuracy, confusion/failure taxonomy,
    statistical cleanup, narrative (GALLERY_SCOPE-aware).
  - edge_deployment_proxy_colab.py : latency / throughput / search-scaling timing.
"""
