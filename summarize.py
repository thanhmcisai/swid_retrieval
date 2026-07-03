# -*- coding: utf-8 -*-
"""Print one compact block with every number needed to fill the paper.

Reads the full-954 run directory and dumps the key CSVs/JSONs (full_swi +
ce_train) so the whole set can be pasted back in a single message.

    python -m swid_retrieval.summarize
    python -m swid_retrieval.summarize /path/to/paper_reframe_full954_overnight
"""

import json
import os
import sys
from pathlib import Path

from . import config


def _run_dir(argv):
    if len(argv) > 1:
        return Path(argv[1])
    stamp = os.environ.get("FULL954_RUN_STAMP", "overnight")
    return Path(os.environ.get(
        "FULL954_RESULTS_DIR", config.ROOT_PATH / "results" / f"paper_reframe_full954_{stamp}"))


def _csv(path, label, max_rows=80):
    print(f"\n----- {label} -----\n{path}")
    if not Path(path).exists():
        print("  (MISSING)")
        return
    try:
        import pandas as pd
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 200)
        pd.set_option("display.max_rows", max_rows)
        df = pd.read_csv(path)
        print(df.to_string(index=False))
    except Exception as e:  # noqa: BLE001
        print(f"  (error reading: {e})")


def _ood_json(path, label):
    print(f"\n----- {label} -----\n{path}")
    if not Path(path).exists():
        print("  (MISSING)"); return
    obj = json.load(open(path))
    ood = obj.get("ood", obj)
    print(f"  gallery_scope={obj.get('gallery_scope')}")
    for m, v in ood.items():
        if isinstance(v, dict):
            au = v.get("AUROC", {}).get("mean") if isinstance(v.get("AUROC"), dict) else v.get("AUROC")
            fp = v.get("FPR95", {}).get("mean") if isinstance(v.get("FPR95"), dict) else v.get("FPR95")
            print(f"  {m:14s} AUROC={au} FPR95={fp}")


def _adapt_json(path, label):
    print(f"\n----- {label} -----\n{path}")
    if not Path(path).exists():
        print("  (MISSING)"); return
    obj = json.load(open(path))
    g = obj.get("gallery_extension", obj)
    print(f"  gallery_scope={obj.get('gallery_scope')}")
    for m, v in g.items():
        if isinstance(v, dict):
            print(f"  {m:14s} old_after={v.get('old_id_mean')} new_50={v.get('new_50_mean')}")


def main(argv):
    rd = _run_dir(argv)
    para = rd / "paradigm" / "variance_evidence"               # RQ1 @ matched 954
    dep = rd / "deployment" / "variance_evidence"              # deployment @ 24
    ce = rd / "ce_train_robustness" / "variance_evidence"      # robustness @ ce_train
    rev = rd / "paradigm" / "review_evidence"                  # RQ1 per-species (954)
    edge = rd / "edge_deployment_proxy"

    print("=" * 72)
    print(f"SWID SUMMARY  run_dir={rd}")
    print("=" * 72)

    print("\n########## RQ1 PARADIGM (matched 954) — tab:rq1 ##########")
    _csv(para / "headline_recomputed_selected_gallery.csv", "headline PARADIGM 954 (native|proto|R@1)")

    print("\n########## RQ1 CE-TRAIN robustness — tab:ce_train_robustness ##########")
    _csv(ce / "headline_recomputed_selected_gallery.csv", "headline ce_train (proto|R@1)")

    print("\n########## DEPLOYMENT (24 target species) — main-text tables ##########")
    _csv(dep / "headline_recomputed_selected_gallery.csv", "headline DEPLOYMENT 24 (native|proto|R@1|OOD|E3)")
    _ood_json(dep / "rq2_ood_full_gallery.json", "OOD deployment-24 (tab:ood)")
    _adapt_json(dep / "rq3_adaptation_full_gallery_partial.json", "retention deployment-24 (tab:adapt)")
    _csv(dep / "gallery_resampling_variance.csv", "gallery strategies A/B/C deployment-24 (tab:rq1_gallery)", max_rows=80)
    _csv(dep / "retrieval_map_mrr_recall.csv", "retrieval quality deployment-24")
    _csv(dep / "rq5_backbone_loss_full_gallery.csv", "rq5 backbone/loss deployment-24")
    _csv(dep / "rq5_scurd_memory_modes_full_gallery.csv", "rq5 SC-URD memory modes deployment-24")
    _csv(dep / "deployment_operating_point_scurd_full_gallery.csv", "operating point deployment-24")
    _csv(dep / "fpr95_difference_bootstrap_ci_full_gallery.csv", "FPR95 diff CI deployment-24")
    _csv(dep / "ood_by_source_scurd_full_gallery.csv", "OOD by source deployment-24")
    _csv(dep / "scurd_seed_sensitivity_summary.csv", "SC-URD seed sensitivity")

    print("\n########## MATCHED-954 deployment versions — tab:matched_deployment (appendix) ##########")
    _ood_json(para / "rq2_ood_full_gallery.json", "OOD matched-954")
    _adapt_json(para / "rq3_adaptation_full_gallery_partial.json", "retention matched-954")

    print("\n########## Review evidence (RQ1 per-species, 954) ##########")
    _csv(rev / "main_claim_statistical_cleanup.csv", "statistical cleanup (Wilcoxon/CI)")
    _csv(rev / "rq1_failure_taxonomy_summary.csv", "failure taxonomy", max_rows=40)

    print("\n########## NATIVE experiments (ported) ##########")
    nat = rd / "native_experiments"
    _json_block(nat / "rq4_generalization.json", "RQ4/VN26 e4a_wilcoxon (tab:vn26_wilcoxon)", "e4a_wilcoxon")
    _ood_baselines(nat / "rq2_ood_extended.json", "OOD baselines (tab:appendix_ood_extended)")
    _csv(nat / "ood_within_source.csv", "same-source OOD protocol (REV-4)", max_rows=80)
    _kshot(nat / "rq2_kshot.json", "OOD K-shot (tab:appendix_ood_kshot)")
    _json_block(nat / "rq5_inference_calibration.json", "RQ5 inference calibration", "inference_calibration")
    _csv(nat / "centering_ablation.csv", "centering generalization across backbones (tab:centering_generalization)", max_rows=60)
    _csv(nat / "scurd_backbone_ablation.csv", "SC-URD backbone ablation (tab:scurd_backbone)")
    _csv(nat / "scurd_consistency_ablation.csv", "SC-URD consistency-loss ablation (REV-2)", max_rows=80)
    _csv(nat / "scurd_lr_ablation.csv", "SC-URD learning-rate audit (REV-1)", max_rows=80)
    _csv(nat / "supcon_seed_sensitivity_summary.csv", "SupCon OOD seed sensitivity (REV-3)")
    _json_block(nat / "appendix_evidence.json", "Appendix pairwise RQ1", "pairwise_rq1")
    _json_block(nat / "exp3_ce_finetune.json", "CE fine-tune forgetting (tab:adapt CE rows)", None)
    _json_block(nat / "deployment_search_cost.json", "Search-cost scaling (tab:inference_cost)", "search_scaling")

    print("\n########## Edge timing (re-run if DINOv2/SC-URD show 504) ##########")
    _csv(edge / "edge_proxy_latency.csv", "edge latency")

    print("\n########## §J cross-experiment master summary ##########")
    _csv(rd / "cross_experiment_summary.csv", "cross-experiment summary (one row/method across RQ1-5)")

    print("\n" + "=" * 72 + "\nEND SUMMARY — paste everything above.\n" + "=" * 72)


def _json_block(path, label, key, max_items=60):
    print(f"\n----- {label} -----\n{path}")
    if not Path(path).exists():
        print("  (MISSING)"); return
    obj = json.load(open(path))
    sub = obj.get(key, obj) if key else obj
    s = json.dumps(sub, indent=1)
    print(s[:6000] + (" …(truncated)" if len(s) > 6000 else ""))


def _ood_baselines(path, label):
    print(f"\n----- {label} -----\n{path}")
    if not Path(path).exists():
        print("  (MISSING)"); return
    suite = json.load(open(path)).get("ood_baseline_suite", {})
    for name in sorted(suite):
        v = suite[name]
        au = v.get("AUROC", {}).get("mean"); fp = v.get("FPR@95TPR", {}).get("mean")
        print(f"  {name:32s} AUROC={au} FPR95={fp}")


def _kshot(path, label):
    print(f"\n----- {label} -----\n{path}")
    if not Path(path).exists():
        print("  (MISSING)"); return
    ks = json.load(open(path)).get("kshot_retrieval", {})
    for m in sorted(ks):
        kd = ks[m]
        cells = " ".join(f"K{k}={kd[str(k)]['mean']:.3f}" for k in (1, 5, 10) if str(k) in kd)
        print(f"  {m:25s} {cells}")




if __name__ == "__main__":
    main(sys.argv)
