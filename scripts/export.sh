#!/usr/bin/env bash
# Consolidate a run's scattered artifacts into two flat folders for handoff:
#   <run>/export/csv/      all result CSVs (+ table-backing JSONs), named by paper label
#   <run>/export/figures/  all figures (PDF/PNG), named by paper stem
# plus export/INDEX.md and swid_csv.zip / swid_figures.zip for download.
#
#   bash swid_retrieval/scripts/export.sh [run_dir]
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
python -m swid_retrieval.export_results "$@"
echo "Download:  <run>/export/swid_csv.zip  and  <run>/export/swid_figures.zip"
