# -*- coding: utf-8 -*-
"""swid_retrieval — corrected full-954 SmartWoodID gallery retrieval package.

This package consolidates the (previously flat) Colab scripts behind one correct
gallery code-path. The core fix: the SWI retrieval gallery must span all ~954
SmartWoodID species (meta-train ∪ meta-val ∪ meta-test), not only the 24 public-ID
species (the old bug in final_metric_learning_cea_2026.py:2120) and not only the
~317-species meta-test split that embedding_cache_v3.npz currently holds.

Pipeline:
  1. embeddings.build_full954  → builds embedding_cache_full954_v3.npz (954-species
     SWI gallery side; ID/OOD query side copied unchanged from the meta-test cache).
  2. orchestrator              → points EMB_CACHE_NAME at the new cache, runs the
     validated variance/retrieval/reviewer engine + migrated review taxonomy +
     edge proxy, then asserts the cardinality sanity gate.

Most evaluation logic is REUSED from the validated scripts
(variance_retrieval_evidence_colab.py, review_evidence_colab.py,
edge_deployment_proxy_colab.py) rather than re-implemented, to preserve numbers.
"""

__all__ = ["config"]
