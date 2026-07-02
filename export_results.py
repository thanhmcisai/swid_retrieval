# -*- coding: utf-8 -*-
"""Consolidate scattered run artifacts into two flat folders for handoff.

Gathers every result CSV (+ the small table-backing JSONs) into <run>/export/csv/
and every figure into <run>/export/figures/, naming each by the paper label it
backs (reusing coverage.TABLE_SOURCES / FIG_STEMS) so the files map 1:1 to the
paper. Writes export/INDEX.md (label → file → source, plus a MISSING list) and
two download zips. Idempotent.

    python -m swid_retrieval.export_results [run_dir]
"""

import os
import re
import shutil
import sys
from pathlib import Path

from . import config
from . import coverage as COV

# Table-backing JSON result files worth shipping for paper-number checking.
JSON_RESULTS = [
    "paradigm/variance_evidence/rq2_ood_full_gallery.json",
    "deployment/variance_evidence/rq2_ood_full_gallery.json",
    "deployment/variance_evidence/rq3_adaptation_full_gallery_partial.json",
    "deployment/variance_evidence/rq5_ablation_full_gallery_partial.json",
    "native_experiments/rq4_generalization.json",
    "native_experiments/rq2_ood_extended.json",
    "native_experiments/rq2_kshot.json",
    "native_experiments/rq5_inference_calibration.json",
    "native_experiments/scurd_backbone_ablation.json",
    "native_experiments/centering_ablation.json",
    "native_experiments/fill_missing_ablation.json",
    "native_experiments/exp3_ce_finetune.json",
    "native_experiments/deployment_search_cost.json",
    "hyperparameters/scurd_hyperparameter_selection.json",
    "cross_experiment_summary.json",
]


def _run_dir(argv):
    if len(argv) > 1 and argv[1] and not str(argv[1]).endswith(".tex"):
        return Path(argv[1])
    stamp = os.environ.get("FULL954_RUN_STAMP", "overnight")
    return Path(os.environ.get(
        "FULL954_RESULTS_DIR", config.ROOT_PATH / "results" / f"paper_reframe_full954_{stamp}"))


def _safe(label):
    return re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")


def _copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def run(run_root):
    run_root = Path(run_root)
    csv_dir = run_root / "export" / "csv"
    fig_dir = run_root / "export" / "figures"
    csv_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    index, missing, exported_srcs = [], [], set()

    # 1) Paper-label-named tables (the primary, reviewer-facing copies).
    for tab in sorted(COV.TABLE_SOURCES):
        src = COV.TABLE_SOURCES[tab]
        for rel in (src if isinstance(src, list) else [src]):
            p = run_root / rel
            if p.exists():
                dst = csv_dir / f"{_safe(tab)}__{p.name}"
                _copy(p, dst)
                exported_srcs.add(str(p.resolve()))
                index.append((tab, dst.name, rel))
            else:
                missing.append((tab, rel))

    # 2) Paper-label-named figures.
    for stem in COV.FIG_STEMS:
        p = run_root / "figures" / f"{stem}.pdf"
        if p.exists():
            _copy(p, fig_dir / p.name)
            exported_srcs.add(str(p.resolve()))
            index.append((stem, p.name, f"figures/{stem}.pdf"))
        else:
            missing.append((stem, f"figures/{stem}.pdf"))

    # 3) Catch-all: every CSV anywhere (except export/) + the table-backing JSONs.
    export_root = (run_root / "export").resolve()
    for p in sorted(run_root.rglob("*.csv")):
        if export_root in p.resolve().parents or str(p.resolve()) in exported_srcs:
            continue
        rel = p.relative_to(run_root)
        _copy(p, csv_dir / str(rel).replace(os.sep, "__"))
        exported_srcs.add(str(p.resolve()))
    for rel in JSON_RESULTS:
        p = run_root / rel
        if p.exists() and str(p.resolve()) not in exported_srcs:
            _copy(p, csv_dir / str(Path(rel)).replace(os.sep, "__"))
            exported_srcs.add(str(p.resolve()))

    # 3b) Catch-all figures (any extra PDF/PNG in the run-local figures folder
    # and nested review-evidence figure folders).
    if (run_root / "figures").exists():
        for p in sorted(run_root.glob("figures/*.pdf")) + sorted(run_root.glob("figures/*.png")):
            d = fig_dir / p.name
            if not d.exists():
                _copy(p, d)
    for p in sorted(run_root.rglob("figures/*.pdf")) + sorted(run_root.rglob("figures/*.png")):
        if export_root in p.resolve().parents:
            continue
        d = fig_dir / p.name
        if not d.exists():
            _copy(p, d)

    # 4) INDEX.md
    lines = ["# Export index", "",
             f"- run_root: `{run_root}`",
             f"- csv files: {len(list(csv_dir.glob('*')))}  |  figures: {len(list(fig_dir.glob('*')))}",
             "", "## Paper label → exported file → source", "",
             "| paper label | exported file | source |", "|---|---|---|"]
    for label, name, rel in index:
        folder = "figures" if name.lower().endswith((".pdf", ".png")) else "csv"
        lines.append(f"| `{label}` | export/{folder}/{name} | {rel} |")
    if missing:
        lines += ["", "## MISSING (paper references with no artifact in this run)", ""]
        lines += [f"- `{label}`  ⟵ {rel}" for label, rel in missing]
    (run_root / "export" / "INDEX.md").write_text("\n".join(lines) + "\n")

    # 5) Download zips.
    shutil.make_archive(str(run_root / "export" / "swid_csv"), "zip", str(csv_dir))
    shutil.make_archive(str(run_root / "export" / "swid_figures"), "zip", str(fig_dir))

    n_csv = len(list(csv_dir.glob("*")))
    n_fig = len(list(fig_dir.glob("*")))
    print(f"  ✅ export: {n_csv} files → export/csv, {n_fig} → export/figures "
          f"({len(missing)} paper refs missing). Zips: swid_csv.zip, swid_figures.zip")
    return {"csv_dir": str(csv_dir), "fig_dir": str(fig_dir),
            "n_csv": n_csv, "n_fig": n_fig, "missing": missing}


def main(argv=None):
    argv = argv if argv is not None else sys.argv
    return run(_run_dir(argv))


if __name__ == "__main__":
    main(sys.argv)
