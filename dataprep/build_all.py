# -*- coding: utf-8 -*-
"""End-to-end data-prep orchestrator: raw images → manifest.

Dependency order (each stage skip-if-exists + gated by its RUN_DATAPREP_* flag;
FORCE_DATAPREP=1 overrides skip):

  CROP    → SELECT          (full images → patch tree {train,test}/<image_id>/scale_*)
  PUBLIC                    (public datasets → ID/OOD species CSVs)  [before SPLIT:
                             split force-assigns ID species to meta-test]
  SPLIT   → IAWA            (species-level meta-train/val/test + IAWA matrix)
  MANIFEST                  (patch tree + split CSV → swi_manifest.json)

Opt-in; NOT called by the eval/train orchestrators.
"""

from . import config as D


def _stage(flag, name, module):
    """Lazily import swid_retrieval.dataprep.<module> and run its run(); fail-soft
    (a missing optional dep, e.g. cv2/torch on a CSV-only box, only skips this stage)."""
    if not flag:
        print(f"\n[skip] {name} (flag off)")
        return
    print(f"\n{'=' * 70}\nSTAGE: {name}\n{'=' * 70}")
    try:
        mod = __import__(f"swid_retrieval.dataprep.{module}", fromlist=["run"])
        mod.run()
    except Exception as e:  # noqa: BLE001
        print(f"⚠️  {name} failed ({type(e).__name__}: {e}); continuing.")


def main():
    print(f"DATAPREP  FORCE={D.FORCE_DATAPREP}  GBIF_OFFLINE={D.GBIF_OFFLINE}")
    _stage(D.RUN_DATAPREP_CROP, "CROP (multi-scale patches)", "crop")
    _stage(D.RUN_DATAPREP_SELECT, "SELECT (diverse KMeans + train/test tree)", "select_diverse")
    _stage(D.RUN_DATAPREP_PUBLIC, "PUBLIC (standardize + ID/OOD)", "standardize_public")
    _stage(D.RUN_DATAPREP_SPLIT, "SPLIT (meta-train/val/test)", "split")
    _stage(D.RUN_DATAPREP_IAWA, "IAWA (attribute matrix)", "iawa")
    _stage(D.RUN_DATAPREP_MANIFEST, "MANIFEST (swi_manifest.json)", "manifest")
    print("\n✅ Data-prep pipeline complete.")


if __name__ == "__main__":
    main()
