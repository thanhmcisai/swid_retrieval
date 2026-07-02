# -*- coding: utf-8 -*-
"""Opt-in DATA-PREP subpackage — raw images → manifest, self-contained.

Ports the standalone SmartWoodID prep scripts (old_version_claim_and_process_data/)
into clean modules that write into the eval package's expected locations
(config.MANIFEST_PATH / SPLIT_CSV / ID_OOD CSVs / IAWA matrix). Stages:

  crop            full microscopy images → multi-scale quality-filtered patches
  select_diverse  ResNet50+KMeans diverse patches → {train,test} tree
  standardize_public  9 public datasets → GBIF-canonical ID/OOD species CSVs
  split           954 species → meta-train/val/test (genus-stratified)
  iawa            IAWA 29-attribute matrix (Paper-2 supervision)
  manifest        scan patch tree → swi_manifest.json (what train/eval read)

Run once with:  python -m swid_retrieval.dataprep.build_all
or:  bash swid_retrieval/scripts/dataprep.sh

OPT-IN and NON-DESTRUCTIVE: not imported by the eval/train orchestrators; every
stage is skip-if-exists; FORCE_DATAPREP=1 overrides. GBIF defaults to offline
(cache-only). To regenerate without clobbering validated files, point the output
dirs (ROOT_PATH / PSI_DIR) at a fresh location.
"""
