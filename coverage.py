# -*- coding: utf-8 -*-
"""Coverage cross-check: every paper table/figure → produced artifact.

Lists each `\\label{tab:*}` and `\\includegraphics{*}` in paper_cea_submission.tex
and maps it to the run artifact (JSON/CSV/figure) that supplies it, printing a
checklist with FOUND/MISSING so nothing is silently unreproduced.

    python -m swid_retrieval.coverage [run_dir] [paper.tex]
"""

import os
import re
import csv
import sys
from pathlib import Path

from . import config


def _run_dir(argv):
    if len(argv) > 1 and not argv[1].endswith(".tex"):
        return Path(argv[1])
    stamp = os.environ.get("FULL954_RUN_STAMP", "overnight")
    return Path(os.environ.get("FULL954_RESULTS_DIR", config.ROOT_PATH / "results" / f"paper_reframe_full954_{stamp}"))


# paper artifact -> relative path(s) under the run dir that supply it.
# NOTE: artifact filenames keep the LEGACY rq1..rq5 numbering (rq2_=OOD, rq3_=adaptation,
# rq4_=VN26, rq5_=ablation); the paper uses RQ1-3 + Experiments 1-6. This map is the
# authoritative paper-label → artifact bridge. A value may be a list when a single paper
# table is assembled from several artifacts (different gallery scopes).
TABLE_SOURCES = {
    "tab:rq1": "paradigm/variance_evidence/headline_recomputed_selected_gallery.csv",
    "tab:rq1_gallery": "deployment/variance_evidence/gallery_resampling_variance.csv",
    "tab:ood": "deployment/variance_evidence/rq2_ood_full_gallery.json",
    "tab:adapt": "deployment/variance_evidence/rq3_adaptation_full_gallery_partial.json",
    "tab:deployment_cost": "native_experiments/exp3_ce_finetune.json",
    "tab:inference_cost": "native_experiments/deployment_search_cost.json",
    "tab:edge_proxy": "edge_deployment_proxy/edge_proxy_latency.csv",
    "tab:vn26": "native_experiments/rq4_generalization.json",
    "tab:vn26_wilcoxon": "native_experiments/rq4_generalization.json",
    "tab:backbone_loss": "deployment/variance_evidence/rq5_backbone_loss_full_gallery.csv",
    "tab:operating_point": "deployment/variance_evidence/deployment_operating_point_scurd_full_gallery.csv",
    "tab:ce_train_robustness": "ce_train_robustness/variance_evidence/headline_recomputed_selected_gallery.csv",
    "tab:matched_deployment": "paradigm/variance_evidence/rq2_ood_full_gallery.json",
    "tab:appendix_scurd_training_modes": "deployment/variance_evidence/rq5_scurd_memory_modes_full_gallery.csv",
    # Multi-scope table: RQ1 rows = matched 954 (review cleanup); OOD CI = deployment-24;
    # VN26 Wilcoxon = own gallery. Assembled from all three.
    "tab:appendix_stats": ["paradigm/review_evidence/main_claim_statistical_cleanup.csv",
                           "deployment/variance_evidence/fpr95_difference_bootstrap_ci_full_gallery.csv",
                           "native_experiments/rq4_generalization.json"],
    "tab:appendix_fpr95_ci": "deployment/variance_evidence/fpr95_difference_bootstrap_ci_full_gallery.csv",
    "tab:appendix_ood_extended": "native_experiments/rq2_ood_extended.json",
    "tab:ood_by_source": "deployment/variance_evidence/ood_by_source_scurd_full_gallery.csv",
    "tab:appendix_ood_kshot": "native_experiments/rq2_kshot.json",
    # Failure taxonomy is discussed as deployment behaviour; use the 24-species
    # deployment review pass, not the matched-954 paradigm statistics.
    "tab:appendix_failure_taxonomy": "deployment/review_evidence/rq1_failure_taxonomy_summary.csv",
    "tab:appendix_retrieval_quality": "deployment/variance_evidence/retrieval_map_mrr_recall.csv",
    # RQ4 SC-URD design (open-set/few-shot reframe): backbone choice + centering generalization.
    "tab:scurd_backbone": "native_experiments/scurd_backbone_ablation.csv",
    "tab:centering_generalization": "native_experiments/centering_ablation.csv",
    "tab:scurd_hyperparameters": "hyperparameters/scurd_hyperparameter_selection.csv",
}
FIG_STEMS = ["fig2a_rq1_paradigm", "fig2b_rq1_gallery_strategy", "fig3a_rq2_ood_detection",
             "fig3b_rq2_openset_retrieval", "fig5a_rq4_swi_to_vn26", "fig5b_rq4_cross_magnification",
             "fig6a_rq5_backbone_loss", "fig6b_rq5_scurd_ablation", "fig6_rq5_scurd_ablation",
             "fig_discussion_rq5_tradeoff", "fig_discussion_rq3_stability_plasticity",
             "fig_discussion_ood_score_distribution"]
STATIC_FIG_FILES = ["sc_urd_architecture_py.png",
                    "scurd_occlusion_case_02.png",
                    "scurd_occlusion_case_03.png",
                    "scurd_occlusion_case_05.png"]


def _rows(path):
    try:
        with open(path, newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def _float(row, key):
    try:
        return float(row.get(key, "nan"))
    except Exception:
        return float("nan")


def _protocol_checks(rd):
    checks = [
        ("paradigm/full_swi", rd / "paradigm" / "variance_evidence" / "protocol_audit.csv", "full_swi", 900),
        ("deployment/id_only", rd / "deployment" / "variance_evidence" / "protocol_audit.csv", "id_only", 24),
        ("ce_train/954", rd / "ce_train_robustness" / "variance_evidence" / "protocol_audit.csv", "ce_train", 900),
    ]
    results = []
    for label, path, expected_scope, min_species in checks:
        rows = _rows(path)
        if not rows:
            results.append((False, label, f"missing {path.relative_to(rd) if path.is_absolute() or rd in path.parents else path}"))
            continue
        row = rows[0]
        scope = str(row.get("main_gallery_scope", row.get("gallery_scope", ""))).strip()
        nsp = _float(row, "main_gallery_species")
        ok = scope == expected_scope and nsp >= min_species
        results.append((ok, label, f"scope={scope or '?'} species={nsp:g} expected={expected_scope}, >= {min_species}"))
    return results


def _edge_checks(rd):
    path = rd / "edge_deployment_proxy" / "edge_proxy_latency.csv"
    rows = _rows(path)
    if not rows:
        return [(False, "edge devices", "missing edge_proxy_latency.csv")]
    devices = {str(r.get("device", "")).strip() for r in rows}
    ok = {"cuda", "cpu"}.issubset(devices)
    return [(ok, "edge devices", f"devices={sorted(devices)} expected both cuda and cpu for Table edge_proxy")]


def main(argv=None):
    argv = argv or sys.argv
    rd = _run_dir(argv)
    tex = Path([a for a in argv[1:] if a.endswith(".tex")][0]) if any(a.endswith(".tex") for a in argv[1:]) \
        else config.ROOT_PATH.parent / "paper_cea_submission.tex"
    if not tex.exists():
        tex = Path("paper_cea_submission.tex")
    labels, includes = set(), set()
    if tex.exists():
        s = tex.read_text(errors="ignore")
        labels = set(re.findall(r"\\label\{(tab:[^}]+)\}", s))
        includes = set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", s))
        print(f"paper: {tex}  ({len(labels)} tables, {len(includes)} includegraphics)")
    else:
        print("paper .tex not found; checking known artifacts only.")

    print(f"\nrun_dir: {rd}\n--- TABLE coverage ---")
    ok = miss = 0
    for tab in sorted(set(TABLE_SOURCES) | labels):
        src = TABLE_SOURCES.get(tab)
        if src is None:
            print(f"  [?      ] {tab:34s} (no source mapping — likely static/manual)"); continue
        srcs = src if isinstance(src, list) else [src]
        missing_srcs = [s for s in srcs if not (rd / s).exists()]
        found = not missing_srcs
        ok += found; miss += (not found)
        shown = srcs[0] if len(srcs) == 1 else f"{len(srcs)} artifacts (" + ", ".join(srcs) + ")"
        tag = "FOUND " if found else "MISSING"
        print(f"  [{tag}] {tab:34s} <- {shown}")
        if missing_srcs and len(srcs) > 1:
            print(f"             missing: {missing_srcs}")

    print("\n--- PROTOCOL sanity ---")
    for found, label, detail in _protocol_checks(rd):
        ok += found; miss += (not found)
        print(f"  [{'PASS  ' if found else 'FAIL  '}] {label:34s} {detail}")

    if "tab:edge_proxy" in labels or (rd / "edge_deployment_proxy" / "edge_proxy_latency.csv").exists():
        print("\n--- EDGE table sanity ---")
        for found, label, detail in _edge_checks(rd):
            ok += found; miss += (not found)
            print(f"  [{'PASS  ' if found else 'FAIL  '}] {label:34s} {detail}")

    print("\n--- FIGURE coverage (figures/) ---")
    for stem in FIG_STEMS:
        found = (rd / "figures" / f"{stem}.pdf").exists()
        ok += found; miss += (not found)
        print(f"  [{'FOUND ' if found else 'MISSING'}] {stem}.pdf")

    for name in STATIC_FIG_FILES:
        found = (rd / "figures" / name).exists()
        ok += found; miss += (not found)
        print(f"  [{'FOUND ' if found else 'MISSING'}] {name}")

    if includes:
        print("\n--- includegraphics coverage from paper ---")
        for inc in sorted(includes):
            name = Path(inc).name
            candidates = [
                rd / "figures" / name,
                rd / "export" / "figures" / name,
                Path(inc),
                tex.parent / inc,
            ]
            found = any(p.exists() for p in candidates)
            ok += found; miss += (not found)
            print(f"  [{'FOUND ' if found else 'MISSING'}] {inc}")

    print(f"\n==> {ok} found, {miss} missing.")
    return 0 if miss == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
