# -*- coding: utf-8 -*-
"""SC-URD training-objective ablations.

Runs the validated variance engine in small, isolated result folders so the
default submission artifacts are not overwritten. The main use is:

* REV-2: lambda_cons in {0, 0.5}, seeds 42/43/44, headline axes.
* REV-1 audit: lr in {1e-3, 1e-4}, seed 42, lambda_cons=0.5.

The engine trains only the SC-URD residual head from cached DINOv2 meta-train
embeddings; it does not read raw images or retrain any image backbone.
"""

import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from .. import config


def _csv_env(name, default, cast=str):
    out = []
    for item in str(os.environ.get(name, default)).split(","):
        item = item.strip()
        if item:
            out.append(cast(item))
    return out


def _tag_float(x):
    return str(x).replace("-", "m").replace(".", "p")


def _copy_research_artifacts(dst_research):
    dst_research = Path(dst_research)
    dst_research.mkdir(parents=True, exist_ok=True)
    srcs = [
        config.RESEARCH_DIR,
        config.ROOT_PATH / "results" / "paper_reframe" / "research_directions",
    ]
    patterns = [
        f"urd_v2_meta_dinov2_embeddings_{config.RESEARCH_CACHE_VERSION}.npz",
        f"sc_urd_checkpoint_*_{config.SC_URD_CACHE_VERSION}.pt",
        f"sc_urd_train_log_*_{config.SC_URD_CACHE_VERSION}.json",
        f"sc_urd_eval_embeddings_*_{config.SC_URD_CACHE_VERSION}.npz",
    ]
    for src_dir in srcs:
        if not src_dir.exists():
            continue
        for pattern in patterns:
            for src in src_dir.glob(pattern):
                dst = dst_research / src.name
                if not dst.exists():
                    shutil.copy2(src, dst)


def _run_engine(run_root, subdir, env_updates, force=False):
    run_root = Path(run_root)
    out_dir = run_root / "training_ablation" / subdir / "variance_evidence"
    summary_path = out_dir / "scurd_seed_sensitivity_summary.csv"
    rows_path = out_dir / "scurd_seed_sensitivity.csv"
    if summary_path.exists() and rows_path.exists() and not force:
        print(f"  ✅ ablation exists: {summary_path}")
        return summary_path, rows_path

    results_dir = run_root / "training_ablation" / subdir
    _copy_research_artifacts(results_dir / "research_directions")

    env = os.environ.copy()
    env.update({
        "ROOT_PATH": str(config.ROOT_PATH),
        "RESULTS_DIR": str(results_dir),
        "OUT_DIR": str(out_dir),
        "EMB_CACHE_NAME": config.FULL954_CACHE_NAME,
        "GALLERY_SCOPE": "id_only",
        "RUN_MAP": "0",
        "RUN_HEADLINE_RECOMPUTE": "0",
        "RUN_GALLERY_RESAMPLING": "0",
        "RUN_REVIEWER_GAP_FULL_GALLERY": "0",
        "RUN_RQ5_FULL_GALLERY": "0",
        "RUN_SCURD_SEED_SENSITIVITY": "1",
        "RUN_TRAIN_SCURD_SEEDS": "1",
        "SCURD_FORCE_RETRAIN_SEEDS": env_updates.get("SCURD_FORCE_RETRAIN_SEEDS", "0"),
        "SCURD_MAIN_MODE": os.environ.get("SCURD_MAIN_MODE", "centered"),
        "DEVICE": os.environ.get("DEVICE", "cuda"),
        "EXP4_CACHE_NAME": os.environ.get("EXP4_CACHE_NAME", config.EXP4_CACHE_NAME),
        "MIN_FULL_GALLERY_SPECIES": os.environ.get("MIN_FULL_GALLERY_SPECIES", "900"),
    })
    env.update({k: str(v) for k, v in env_updates.items()})

    script = Path(__file__).resolve().parents[1] / "_engines" / "variance_retrieval_evidence_colab.py"
    print(f"  running SC-URD ablation {subdir}: {env_updates}")
    subprocess.run([sys.executable, str(script)], check=True, env=env)
    return summary_path, rows_path


def _read_summary(path, extra):
    rows = []
    if not Path(path).exists():
        return rows
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            row.update(extra)
            rows.append(row)
    return rows


def run(run_root, out_dir=None):
    """Run configured SC-URD training ablations and write summary CSV/JSON."""
    run_root = Path(run_root)
    out_dir = Path(out_dir or (run_root / "native_experiments"))
    out_dir.mkdir(parents=True, exist_ok=True)
    force = os.environ.get("FORCE_SCURD_TRAINING_ABLATION", "0") == "1"

    seeds = os.environ.get("SCURD_ABLATION_SEEDS", os.environ.get("SCURD_TRAIN_SEEDS", "42,43,44"))
    lambda_values = _csv_env("SCURD_ABLATION_LAMBDAS", "0,0.5", float)
    base_lr = os.environ.get("SCURD_TRAIN_LR", str(config.SCURD_TRAIN_LR))

    consistency_rows = []
    for lam in lambda_values:
        sub = f"lambda_{_tag_float(lam)}_lr_{_tag_float(base_lr)}"
        summary, row_csv = _run_engine(
            run_root,
            sub,
            {
                "SCURD_TRAIN_LAMBDA_CONS": lam,
                "SCURD_TRAIN_LR": base_lr,
                "SCURD_TRAIN_SEEDS": seeds,
            },
            force=force,
        )
        consistency_rows.extend(_read_summary(summary, {
            "ablation": "consistency_lambda",
            "lambda_cons": lam,
            "lr": base_lr,
            "source_summary": str(summary),
            "source_rows": str(row_csv),
        }))

    lr_rows = []
    lr_values = _csv_env("SCURD_LR_ABLATION_LRS", "1e-3,1e-4", str)
    lr_seed = os.environ.get("SCURD_LR_ABLATION_SEEDS", "42")
    lr_lambda = os.environ.get("SCURD_LR_ABLATION_LAMBDA_CONS", "0.5")
    for lr in lr_values:
        sub = f"lr_{_tag_float(lr)}_lambda_{_tag_float(lr_lambda)}"
        summary, row_csv = _run_engine(
            run_root,
            sub,
            {
                "SCURD_TRAIN_LAMBDA_CONS": lr_lambda,
                "SCURD_TRAIN_LR": lr,
                "SCURD_TRAIN_SEEDS": lr_seed,
            },
            force=force,
        )
        lr_rows.extend(_read_summary(summary, {
            "ablation": "learning_rate",
            "lambda_cons": lr_lambda,
            "lr": lr,
            "source_summary": str(summary),
            "source_rows": str(row_csv),
        }))

    cons_df = pd.DataFrame(consistency_rows)
    lr_df = pd.DataFrame(lr_rows)
    cons_csv = out_dir / "scurd_consistency_ablation.csv"
    lr_csv = out_dir / "scurd_lr_ablation.csv"
    cons_df.to_csv(cons_csv, index=False)
    lr_df.to_csv(lr_csv, index=False)
    payload = {
        "protocol": {
            "gallery_scope": "id_only",
            "consistency_lambdas": lambda_values,
            "consistency_seeds": seeds,
            "lr_values": lr_values,
            "lr_seed": lr_seed,
            "note": "All rows are produced by the validated variance engine; only SC-URD residual adapters are trained.",
        },
        "consistency_rows": consistency_rows,
        "lr_rows": lr_rows,
    }
    json_path = out_dir / "scurd_training_ablation.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)
    print(f"saved {cons_csv}")
    print(f"saved {lr_csv}")
    print(f"saved {json_path}")
    return payload
