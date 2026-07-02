# -*- coding: utf-8 -*-
"""§J — cross-experiment master summary ("the case for retrieval").

Synthesizes one per-method row across every research question by reading the
artifacts the orchestrator already wrote (paradigm/deployment headline CSVs,
RQ2 OOD, RQ3 retention, RQ5 backbone/loss, RQ4 SWI→VN26). NOTHING is recomputed
— this is a pure join, the package analogue of the monolith §J block
(smartwoodid_experiments_full.py:7266-7532), which assembled the same numbers
from in-memory experiment variables.

Output (into the run root):
  cross_experiment_summary.csv   — one row per method, columns per experiment
  cross_experiment_summary.json  — same rows + headline scalars (CE native @954)

    python -m swid_retrieval.experiments.summary_master [run_dir]
"""

import json
import os
import sys
from pathlib import Path

from .. import config


def _run_dir(argv):
    if len(argv) > 1 and argv[1] and not str(argv[1]).endswith(".tex"):
        return Path(argv[1])
    stamp = os.environ.get("FULL954_RUN_STAMP", "overnight")
    return Path(os.environ.get(
        "FULL954_RESULTS_DIR", config.ROOT_PATH / "results" / f"paper_reframe_full954_{stamp}"))


def _read_csv(p):
    import pandas as pd
    return pd.read_csv(p) if Path(p).exists() else None


def _read_json(p):
    return json.load(open(p)) if Path(p).exists() else None


def _ood_dict(obj):
    """{method: (AUROC_mean, FPR95_mean)} from rq2_ood_full_gallery.json."""
    out = {}
    block = (obj or {}).get("ood", obj or {})
    for m, v in block.items():
        if isinstance(v, dict):
            au = v.get("AUROC", {}).get("mean") if isinstance(v.get("AUROC"), dict) else v.get("AUROC")
            fp = v.get("FPR95", {}).get("mean") if isinstance(v.get("FPR95"), dict) else v.get("FPR95")
            if au is not None:
                out[m] = (float(au), float(fp) if fp is not None else None)
    return out


def _headline_rows(csv_path):
    """{method: (proto, r1)} + native CE scalar from a headline CSV."""
    df = _read_csv(csv_path)
    if df is None:
        return {}, None
    rows = {}
    for _, r in df.iterrows():
        rows[str(r["method"])] = (
            float(r["E1A_prototype_macro"]) if r.get("E1A_prototype_macro") is not None else None,
            float(r["E1A_public_id_macro_R1"]) if r.get("E1A_public_id_macro_R1") is not None else None,
        )
    native = None
    if "E1A_native_ce_macro" in df and df["E1A_native_ce_macro"].notna().any():
        native = float(df["E1A_native_ce_macro"].dropna().iloc[0])
    return rows, native


def build_summary(run_root):
    run_root = Path(run_root)
    para_ve = run_root / "paradigm" / "variance_evidence"
    dep_ve = run_root / "deployment" / "variance_evidence"
    nat = run_root / "native_experiments"

    proto954, native954 = _headline_rows(para_ve / "headline_recomputed_selected_gallery.csv")
    proto24, native24 = _headline_rows(dep_ve / "headline_recomputed_selected_gallery.csv")
    ood = _ood_dict(_read_json(dep_ve / "rq2_ood_full_gallery.json"))
    retention = (_read_json(dep_ve / "rq3_adaptation_full_gallery_partial.json") or {}).get("gallery_extension", {})
    rq4 = (_read_json(nat / "rq4_generalization.json") or {}).get("cross_domain", {})

    rq5 = {}
    df5 = _read_csv(dep_ve / "rq5_backbone_loss_full_gallery.csv")
    if df5 is not None:
        for _, r in df5.iterrows():
            rq5[str(r["method"])] = r

    methods = sorted(set(proto954) | set(proto24) | set(ood) | set(retention) | set(rq4) | set(rq5))
    rows = []
    for m in methods:
        ret = retention.get(m, {}) if isinstance(retention.get(m), dict) else {}
        vn = rq4.get(m, {}).get("SWI_pool", {}).get("VN26_all", {}).get("mean") if m in rq4 else None
        r5 = rq5.get(m)
        rows.append({
            "method": m,
            "paradigm954_prototype": proto954.get(m, (None, None))[0],
            "paradigm954_R1": proto954.get(m, (None, None))[1],
            "deploy24_prototype": proto24.get(m, (None, None))[0],
            "deploy24_R1": proto24.get(m, (None, None))[1],
            "ood_auroc": ood.get(m, (None, None))[0],
            "ood_fpr95": ood.get(m, (None, None))[1],
            "retention_old_after_plus50": ret.get("old_id_mean"),
            "retention_new_species": ret.get("new_50_mean"),
            "rq4_swi_to_vn26_R1": vn,
            "rq5_exp1a": (float(r5["exp1a_mean"]) if r5 is not None and r5.get("exp1a_mean") is not None else None),
            "rq5_ood_auroc": (float(r5["ood_auroc"]) if r5 is not None and r5.get("ood_auroc") is not None else None),
            "rq5_exp3_old": (float(r5["exp3_old"]) if r5 is not None and r5.get("exp3_old") is not None else None),
        })
    return rows, {"ce_native_954way": native954, "ce_native_24way": native24}


def run(run_root):
    run_root = Path(run_root)
    rows, scalars = build_summary(run_root)
    if not rows:
        print("  summary_master: no experiment artifacts found; nothing written.")
        return None
    csv_path = run_root / "cross_experiment_summary.csv"
    json_path = run_root / "cross_experiment_summary.json"
    try:
        import pandas as pd
        pd.DataFrame(rows).to_csv(csv_path, index=False)
    except Exception as e:  # noqa: BLE001
        print(f"  summary_master: pandas CSV write skipped ({e}); JSON only.")
        csv_path = None
    with open(json_path, "w") as f:
        json.dump({"scalars": scalars, "rows": rows}, f, indent=2, sort_keys=True, default=float)
    print(f"  ✅ cross-experiment summary: {len(rows)} methods → {json_path.name}"
          + (f" + {csv_path.name}" if csv_path else ""))
    return {"csv": str(csv_path) if csv_path else None, "json": str(json_path), "rows": rows, "scalars": scalars}


def main(argv=None):
    argv = argv if argv is not None else sys.argv
    return run(_run_dir(argv))


if __name__ == "__main__":
    main(sys.argv)
