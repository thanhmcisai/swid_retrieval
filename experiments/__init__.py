# -*- coding: utf-8 -*-
"""Experiment logic.

Design decision: the bug-affected experiments are REUSED verbatim from the
already-validated flat scripts rather than re-transcribed here, to preserve the
exact published numbers and avoid transcription risk. The orchestrator exec()s:

  - variance_retrieval_evidence_colab.py  → headline R@1 (Exp1A), gallery K-shot
    strategies (Exp1B), distance OOD AUROC/FPR95, gallery-extension retention
    (Exp3.2), backbone/loss + SC-URD memory-mode ablation (Exp5), retrieval
    mAP/Hit@k, reviewer operating-point/FPR95-CI/OOD-by-source, SC-URD seed
    sensitivity. It already supports GALLERY_SCOPE=full_swi.

  - review_evidence_colab.py (migrated)   → per-species accuracy, confusion /
    failure taxonomy, statistical cleanup, deployment narrative. Now honors
    GALLERY_SCOPE=full_swi (mask + meta-test→954-union path source + SC-URD pool).

  - edge_deployment_proxy_colab.py        → edge/CPU latency + search scaling
    (timing; unaffected by gallery scope).

Experiments deliberately NOT recomputed (not affected by the gallery bug): CE
fine-tune catastrophic forgetting (classification argmax), MSP/ODIN/OpenMax/Energy
OOD baselines (logit-based), and VN26/Exp4 (its own SWI_pool/VN26 galleries).
"""
