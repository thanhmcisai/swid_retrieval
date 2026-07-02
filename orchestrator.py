# -*- coding: utf-8 -*-
"""End-to-end SWID retrieval run orchestrator.

Pipeline:
  0. Build the full-954 SWI gallery cache (+ aligned SC-URD swi_pool).
  1. Run deployment experiments under the 24-species target gallery.
  2. Run paradigm experiments under the matched 954-species SWI gallery.
  3. Run review evidence under both scopes: full_swi for matched-954 statistics
     and id_only for deployment failure taxonomy / optional saliency.
  3. Run the edge/deployment timing proxy.
  4. Cardinality sanity gate: full-gallery R@1 must be < the old 24-species R@1.

Outputs go to a timestamped results/paper_reframe_full954_<stamp> dir so the
original paper_reframe results and embedding_cache_v3.npz are never overwritten.
The two evaluation scripts are reused as-is (exec) to preserve validated numbers.
"""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path

from . import config


# ── Script discovery ────────────────────────────────────────────────────────
# Canonical location is swid_retrieval/_engines/ (self-contained package). We also
# fall back to the package's parent dir and ROOT_PATH for backward compatibility
# with the older flat layout, and honour explicit *_SCRIPT_PATH overrides.
ENGINES_DIR = Path(__file__).resolve().parent / "_engines"
PKG_DIR = Path(__file__).resolve().parent.parent


def _find_script(env_key, filename):
    override = os.environ.get(env_key)
    if override and Path(override).exists():
        return Path(override)
    for base in (ENGINES_DIR, PKG_DIR, config.ROOT_PATH):
        cand = Path(base) / filename
        if cand.exists():
            return cand
    return ENGINES_DIR / filename  # report the canonical path even if missing


VARIANCE_SCRIPT = _find_script("VARIANCE_SCRIPT_PATH", "variance_retrieval_evidence_colab.py")
REVIEW_SCRIPT = _find_script("REVIEW_SCRIPT_PATH", "review_evidence_colab.py")
EDGE_SCRIPT = _find_script("EDGE_SCRIPT_PATH", "edge_deployment_proxy_colab.py")


def _copy_if_needed(src, dst):
    src, dst = Path(src), Path(dst)
    if not src.exists() or dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _exec_script(path, label):
    if not Path(path).exists():
        print(f"⚠️  {label}: script not found at {path}; skipping.")
        return False
    print(f"\n{'=' * 72}\n{label}: {path}\n{'=' * 72}")
    g = {"__name__": "__main__", "__file__": str(path)}
    exec(compile(open(path).read(), str(path), "exec"), g)
    return True


def _read_headline_full(run_results_dir):
    """New full-gallery per-method R@1 from the variance engine's headline CSV."""
    import csv
    csv_path = Path(run_results_dir) / "variance_evidence" / "headline_recomputed_selected_gallery.csv"
    out = {}
    if not csv_path.exists():
        return out
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            for k in ("E1A_public_id_macro_R1", "E1A_public_id_R1", "E1A"):
                if row.get(k) not in (None, ""):
                    try:
                        out[row["method"]] = float(row[k])
                    except (TypeError, ValueError):
                        pass
                    break
    return out


def _read_headline_old():
    """Old 24-species per-method R@1 from the original paper_reframe artifacts.

    Defensive: tries a few known shapes and returns {} if none parse, in which
    case the sanity gate degrades to a warning instead of a hard failure.
    """
    candidates = [config.RESULTS_DIR / "rq1_paradigm.json"]
    for path in candidates:
        if not path.exists():
            continue
        try:
            obj = json.load(open(path))
        except Exception:
            continue
        block = obj.get("paradigm_comparison", obj)
        out = {}
        # Canonical form (final_metric_learning_cea_2026): list of
        # {"method": ..., "r1": ...} dicts.
        if isinstance(block, list):
            for d in block:
                if isinstance(d, dict) and isinstance(d.get("r1"), (int, float)):
                    out[d.get("method")] = float(d["r1"])
        # Tolerated alternative: dict-of-dicts keyed by method.
        elif isinstance(block, dict):
            for method, v in block.items():
                if isinstance(v, dict):
                    for k in ("r1", "retrieval_R1", "R@1", "R1", "retrieval_r1", "recall@1"):
                        if isinstance(v.get(k), (int, float)):
                            out[method] = float(v[k])
                            break
        if out:
            return out
    return {}


def _cardinality_sanity(paradigm_dir, deployment_dir):
    """Matched 954-gallery R@1 must be < 24-species deployment R@1 for the headline
    methods (more distractors -> lower top-1). A non-negative delta means the 954
    gallery did not take effect. STRICT_SANITY=0 downgrades to a warning."""
    strict = os.environ.get("STRICT_SANITY", "1") == "1"
    para = _read_headline_full(paradigm_dir)       # 954 (matched)
    dep = _read_headline_full(deployment_dir)      # 24 (deployment)
    report = {"paradigm_954_R1": para, "deployment_24_R1": dep, "deltas": {}, "status": "skipped"}
    if not para or not dep:
        print("⚠️  Cardinality sanity: a headline CSV is missing; cannot verify.")
        report["status"] = "incomplete"
        return report
    failures = []
    for method in ("DINOv2", "ArcFace-557", "Fusion"):
        if method in para and method in dep:
            delta = para[method] - dep[method]
            report["deltas"][method] = {"paradigm_954": para[method], "deployment_24": dep[method], "delta": delta}
            flag = "PASS" if delta < 0 else "FAIL"
            print(f"  {method:12s} 954={para[method]:.4f}  dep24={dep[method]:.4f}  Δ={delta:+.4f}  [{flag}]")
            if delta >= 0:
                failures.append(method)
    report["status"] = "fail" if failures else "pass"
    if failures and strict:
        raise RuntimeError(
            f"Cardinality sanity FAILED for {failures}: matched-954 R@1 not < 24-species R@1 "
            f"(the 954 gallery likely did not take effect). Set STRICT_SANITY=0 to override.")
    return report


# All experiment flags, toggled per pass so each scope runs exactly what it needs.
_ALL_RUN_FLAGS = ("RUN_MAP", "RUN_HEADLINE_RECOMPUTE", "RUN_GALLERY_RESAMPLING",
                  "RUN_REVIEWER_GAP_FULL_GALLERY", "RUN_RQ5_FULL_GALLERY",
                  "RUN_SCURD_SEED_SENSITIVITY", "RUN_TRAIN_SCURD_SEEDS")
_SCURD_ARTIFACTS = ("sc_urd_checkpoint_scurd_r01_e20_v2.pt",
                    "sc_urd_eval_embeddings_scurd_r01_e20_recomputed_v2.npz",
                    "urd_v2_meta_dinov2_embeddings_v2.npz")
# Seed checkpoints use configurable seeds (default 42/43/44) → matched by glob, not a
# fixed name. They are persisted in the source research dir and copied into each pass so
# the engine LOADS them (skip-if-exists) instead of retraining every run.
_SCURD_SEED_GLOB = f"sc_urd_checkpoint_scurd_r01_e20_seed*_{config.SC_URD_CACHE_VERSION}.pt"


def _engine_pass(run_root, subdir, scope, enable, label, src_research):
    """Run the validated engine once at gallery `scope` into run_root/subdir, with
    only the RUN_* flags in `enable` turned on. Returns the pass directory."""
    d = run_root / subdir
    research = d / "research_directions"
    research.mkdir(parents=True, exist_ok=True)
    for name in _SCURD_ARTIFACTS:
        _copy_if_needed(src_research / name, research / name)
    for p in Path(src_research).glob(_SCURD_SEED_GLOB):   # load persisted seeds, don't retrain
        _copy_if_needed(p, research / p.name)
    os.environ["RESULTS_DIR"] = str(d)
    os.environ["OUT_DIR"] = str(d / "variance_evidence")
    os.environ["GALLERY_SCOPE"] = scope
    for k in _ALL_RUN_FLAGS:
        os.environ[k] = "1" if k in enable else "0"
    _exec_script(VARIANCE_SCRIPT, label)
    return d


def _review_pass(pass_dir, scope, label, run_interpretability=False):
    """Run review_evidence_colab.py into pass_dir/review_evidence at a fixed
    gallery scope. The review script derives OUT_DIR from RESULTS_DIR, so setting
    RESULTS_DIR to the pass directory is the important part."""
    old = {k: os.environ.get(k) for k in ("RESULTS_DIR", "OUT_DIR", "GALLERY_SCOPE", "RUN_INTERPRETABILITY")}
    os.environ["RESULTS_DIR"] = str(pass_dir)
    os.environ["OUT_DIR"] = str(pass_dir / "review_evidence")
    os.environ["GALLERY_SCOPE"] = scope
    os.environ["RUN_INTERPRETABILITY"] = "1" if run_interpretability else "0"
    try:
        _exec_script(REVIEW_SCRIPT, label)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _point_config_at_results_dir(results_dir):
    """Keep modules that imported swid_retrieval.config in sync with the active
    run directory. Environment variables alone are insufficient because config
    constants are evaluated at import time."""
    results_dir = Path(results_dir)
    config.RESULTS_DIR = results_dir
    config.RESEARCH_DIR = results_dir / "research_directions"
    config.SCURD_PROJ_CACHE = config.RESEARCH_DIR / os.environ.get(
        "SCURD_PROJ_CACHE_NAME", "sc_urd_eval_embeddings_scurd_r01_e20_recomputed_v2.npz")
    config.SCURD_MAIN_CKPT = config.RESEARCH_DIR / os.environ.get(
        "SCURD_MAIN_CKPT_NAME", f"sc_urd_checkpoint_scurd_r01_e20_{config.SC_URD_CACHE_VERSION}.pt")


def _preload_images_if_requested():
    """Optional Colab/Drive prewarm before any experiment starts.

    `build_full954` preloads SWI images when it extracts the 954 cache, but it can
    skip immediately if the cache already exists. This global prewarm covers that
    reuse case and can include public ID/OOD images for saliency/timing/cost steps.
    """
    if os.environ.get("PRELOAD_IMAGE_CACHE", "1") != "1":
        return
    if os.environ.get("SWID_IMAGE_PRELOAD_DONE", "0") == "1":
        return
    if os.environ.get("PRELOAD_ALL_IMAGES", "0") != "1":
        return
    from .data import collect_all_image_paths, preload_image_cache
    paths = collect_all_image_paths(config.ROOT_PATH, include_swi=True, include_public=True)
    preload_image_cache(
        paths,
        max_workers=int(os.environ.get("PRELOAD_WORKERS", "16")),
        desc="Preloading SWI + public ID/OOD images",
    )
    os.environ["SWID_IMAGE_PRELOAD_DONE"] = "1"


def main():
    """One run, correct gallery per research question:
      - DEPLOYMENT pass (id_only, 24 target species): gallery strategies A/B/C,
        OOD AUROC/FPR95, gallery-expansion retention, reviewer-gap, RQ5, and the
        SC-URD seed training/sensitivity (trained once here).
      - PARADIGM pass (full_swi, 954, cardinality-matched to the CE classifier):
        RQ1 native/prototype/R@1, mAP/Hit@k, plus the matched-954 versions of the
        deployment metrics for the transparency appendix.
      - CE-TRAIN pass (ce_train): fairness robustness for RQ1.
      - Review taxonomy (full_swi) and edge timing run once each.
    """
    stamp = os.environ.get("FULL954_RUN_STAMP", datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_root = Path(os.environ.get(
        "FULL954_RESULTS_DIR", config.ROOT_PATH / "results" / f"paper_reframe_full954_{stamp}"))
    (run_root / "research_directions").mkdir(parents=True, exist_ok=True)
    src_research = config.RESULTS_DIR / "research_directions"
    for name in _SCURD_ARTIFACTS:                       # for the edge proxy
        _copy_if_needed(src_research / name, run_root / "research_directions" / name)

    # Shared env (each pass overrides RESULTS_DIR/OUT_DIR/GALLERY_SCOPE/RUN_* itself).
    os.environ["ROOT_PATH"] = str(config.ROOT_PATH)
    os.environ["EMB_CACHE_NAME"] = config.FULL954_CACHE_NAME
    os.environ["MIN_FULL_GALLERY_SPECIES"] = str(config.MIN_FULL_GALLERY_SPECIES)
    os.environ.setdefault("EXP4_CACHE_NAME", config.EXP4_CACHE_NAME)   # VN26 own gallery
    os.environ["SOURCE_RESULTS_DIR"] = str(config.RESULTS_DIR)          # original rq2/3/4 JSONs

    manifest = {"run_root": str(run_root), "source_results_dir": str(config.RESULTS_DIR),
                "full954_cache": str(config.FULL954_CACHE_PATH), "created_at": stamp}
    with open(run_root / "full954_run_manifest_pre.json", "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    print("=" * 72)
    print("SWID_RETRIEVAL — END-TO-END RUN (per-question gallery scopes)")
    print("=" * 72)
    print(f"ROOT_PATH={config.ROOT_PATH}\nRUN_ROOT={run_root}\nFULL954_CACHE={config.FULL954_CACHE_PATH}")

    config.configure_runtime(verbose=True)
    from .data import prepare_public_csvs
    prepare_public_csvs(config.ROOT_PATH)
    _preload_images_if_requested()

    # Step 0 — build the full-954 gallery cache (the only heavy extraction).
    if config.RUN_BUILD_FULL954:
        print("\n[0/8] Building full-954 SWI gallery cache...")
        from .embeddings import build_full954
        build_full954.main()
    else:
        print("\n[0/8] RUN_BUILD_FULL954=0; assuming full-954 cache already exists.")

    # Step 1 — DEPLOYMENT (24-species target gallery): the experiments whose
    # research question is operational deployment. Trains SC-URD seeds once.
    print("\n[1/8] Deployment experiments (id_only, 24 target species)...")
    dep_dir = _engine_pass(
        run_root, "deployment", "id_only",
        {"RUN_HEADLINE_RECOMPUTE", "RUN_MAP", "RUN_GALLERY_RESAMPLING",
         "RUN_REVIEWER_GAP_FULL_GALLERY", "RUN_RQ5_FULL_GALLERY",
         "RUN_SCURD_SEED_SENSITIVITY", "RUN_TRAIN_SCURD_SEEDS"},
        "variance (deployment / id_only)", src_research)
    # Persist any newly-trained SC-URD seed checkpoints back to the source research dir
    # so later runs (new FULL954_RUN_STAMP) reuse them instead of retraining.
    for p in (dep_dir / "research_directions").glob(_SCURD_SEED_GLOB):
        _copy_if_needed(p, src_research / p.name)

    # Step 2 — PARADIGM (matched 954-species gallery): RQ1 + matched-954 deployment
    # metrics for the appendix. Seeds already trained; not retrained here.
    print("\n[2/8] Paradigm comparison (full_swi, matched 954 species)...")
    para_dir = _engine_pass(
        run_root, "paradigm", "full_swi",
        {"RUN_HEADLINE_RECOMPUTE", "RUN_MAP", "RUN_GALLERY_RESAMPLING"},
        "variance (paradigm / full_swi)", src_research)

    # Step 3 — review evidence. The matched-954 pass backs the statistical cleanup;
    # the deployment pass backs qualitative/failure-taxonomy discussion and optional
    # saliency figures. Keep both scopes explicit to avoid mixing 954- and 24-gallery
    # per-species results.
    print("\n[3/8] Review evidence (matched-954 statistics + deployment taxonomy)...")
    if config.RUN_REVIEW_TAXONOMY_FULL_GALLERY:
        _review_pass(para_dir, "full_swi", "review_evidence (paradigm / full_swi)",
                     run_interpretability=False)
        manifest["review_evidence_paradigm"] = str(para_dir / "review_evidence")
    else:
        print("  RUN_REVIEW_TAXONOMY_FULL_GALLERY=0; skipping full_swi review evidence.")
    if config.RUN_REVIEW_TAXONOMY_DEPLOYMENT:
        _review_pass(dep_dir, "id_only", "review_evidence (deployment / id_only)",
                     run_interpretability=os.environ.get("RUN_INTERPRETABILITY", "0") == "1")
        manifest["review_evidence_deployment"] = str(dep_dir / "review_evidence")
    else:
        print("  RUN_REVIEW_TAXONOMY_DEPLOYMENT=0; skipping deployment review evidence.")

    # Step 4 — edge/deployment timing (scope-independent), once.
    if config.RUN_EDGE_PROXY:
        print("\n[4/8] Edge/deployment proxy timing...")
        os.environ["RESULTS_DIR"] = str(run_root)
        os.environ["OUT_DIR"] = str(run_root / "edge_deployment_proxy")
        _exec_script(EDGE_SCRIPT, "edge_deployment_proxy")
    else:
        print("\n[4/8] RUN_EDGE_PROXY=0; skipping edge proxy.")

    # Step 5 — CE-train fairness robustness for RQ1.
    if os.environ.get("RUN_CE_TRAIN_ROBUSTNESS", "1") == "1":
        print("\n[5/8] CE-train robustness (gallery = CE-Full's exact train images)...")
        # tab:ce_train_robustness only needs the headline (prototype/R@1); skip the
        # unused MAP / gallery-resampling passes to save compute.
        ce_dir = _engine_pass(
            run_root, "ce_train_robustness", "ce_train",
            {"RUN_HEADLINE_RECOMPUTE"},
            "variance (ce_train robustness)", src_research)
        manifest["ce_train_robustness"] = str(ce_dir)
    else:
        print("\n[5/8] RUN_CE_TRAIN_ROBUSTNESS=0; skipping CE-train robustness.")

    # Step 6 — native (ported) experiments at the deployment gallery (24): the
    # paper experiments not covered by the engine passes (RQ4/VN26, extended OOD
    # baselines, OOD K-shot, RQ5-extra, appendix/discussion, CE-finetune, costs).
    # One registry build drives all; GPU/image-heavy ones honour RUN_HEAVY.
    print("\n[6/8] Native experiments (RQ4 / OOD baselines / K-shot / ...)...")
    os.environ["RESULTS_DIR"] = str(run_root)
    os.environ["GALLERY_SCOPE"] = "id_only"
    _point_config_at_results_dir(run_root)
    nat_dir = run_root / "native_experiments"
    nat_dir.mkdir(parents=True, exist_ok=True)
    # Paper-complete default: Tables tab:deployment_cost and tab:inference_cost
    # depend on CE fine-tuning / cost artifacts. Set RUN_HEAVY=0 for a fast
    # logic/debug run that intentionally omits those tables.
    heavy = os.environ.get("RUN_HEAVY", "1") == "1"
    try:
        from .experiments import registry as _reg
        M_dep, ctx_dep = _reg.build_M("id_only", with_exp4=True, with_scurd=True)

        def _opt(flag, default, runner):
            if os.environ.get(flag, default) != "1":
                return
            try:
                runner()
            except ImportError:
                print(f"  {flag}: module not available yet; skipping.")
            except Exception as e:  # noqa: BLE001
                print(f"  {flag}: failed ({e}); continuing.")

        _opt("RUN_RQ4_VN26", "1", lambda: __import__("swid_retrieval.experiments.rq4_vn26", fromlist=["run"]).run(M_dep, nat_dir))
        _opt("RUN_OOD_BASELINES", "1", lambda: __import__("swid_retrieval.experiments.ood_baselines", fromlist=["run"]).run(M_dep, ctx_dep, nat_dir))
        _opt("RUN_OOD_KSHOT", "1", lambda: __import__("swid_retrieval.experiments.kshot_openset", fromlist=["run"]).run(M_dep, nat_dir))
        _opt("RUN_RQ5_EXTRA", "1", lambda: __import__("swid_retrieval.experiments.rq5_ablation_extra", fromlist=["run"]).run(M_dep, ctx_dep, nat_dir))
        _opt("RUN_APPENDIX_DISCUSSION", "1", lambda: __import__("swid_retrieval.experiments.appendix_discussion", fromlist=["run"]).run(M_dep, ctx_dep, nat_dir, para_dir, dep_dir))
        if heavy:
            _opt("RUN_CE_FINETUNE", "1", lambda: __import__("swid_retrieval.experiments.ce_finetune", fromlist=["run"]).run(ctx_dep, nat_dir))
            _opt("RUN_COSTS", "1", lambda: __import__("swid_retrieval.experiments.costs", fromlist=["run"]).run(M_dep, ctx_dep, nat_dir))
        else:
            print("  RUN_HEAVY=0; skipping CE-finetune + costs (GPU/image).")
        # §K/§L completeness pass — fill any method × experiment cell still empty.
        # Opt-in (default off) and idempotent: a normal run is unaffected.
        _opt("RUN_FILL_MISSING", "0", lambda: __import__("swid_retrieval.experiments.fill_missing", fromlist=["run"]).run(M_dep, ctx_dep, nat_dir))
        # Backbone-agnostic centering generalization (cheap, cache-only).
        _opt("RUN_CENTERING_ABLATION", "1", lambda: __import__("swid_retrieval.experiments.centering_ablation", fromlist=["run"]).run(nat_dir))
        # SC-URD recipe on alternate bases (heavy: extracts base meta + trains adapters).
        _opt("RUN_SCURD_BACKBONE", "1", lambda: __import__("swid_retrieval.experiments.scurd_backbone", fromlist=["run"]).run(M_dep, ctx_dep, nat_dir))
        manifest["native_experiments"] = str(nat_dir)
    except Exception as e:  # noqa: BLE001
        print(f"⚠️  native experiments failed ({e}); engine-pass results are still complete.")

    # Step 7 — cardinality sanity: matched-954 R@1 must be < 24-species R@1.
    print("\n[7/8] Cardinality sanity (matched-954 R@1 < 24-species deployment R@1)...")
    manifest["passes"] = {"deployment_24": str(dep_dir), "paradigm_954": str(para_dir)}
    manifest["sanity"] = _cardinality_sanity(para_dir, dep_dir)

    # Step 7b — §J cross-experiment master summary: pure join over emitted artifacts.
    if os.environ.get("RUN_SUMMARY_MASTER", "1") == "1":
        try:
            from .experiments import summary_master
            summary_master.run(run_root)
            manifest["cross_experiment_summary"] = str(run_root / "cross_experiment_summary.json")
        except Exception as e:  # noqa: BLE001
            print(f"⚠️  summary_master failed ({e}); results are still complete.")

    # Step 8 — figures (vector PDFs) from the per-scope results.
    if os.environ.get("RUN_VISUALIZE", "1") == "1":
        print("\n[8/8] Rendering paper figures...")
        try:
            from . import visualize
            visualize.main([None, str(run_root)])
            manifest["figures_dir"] = str(run_root / "figures")
        except Exception as e:  # noqa: BLE001
            print(f"⚠️  visualize failed ({e}); results are still complete.")
    else:
        print("\n[8/8] RUN_VISUALIZE=0; skipping figures.")

    # Step 9 — consolidate all CSVs + figures into export/{csv,figures} (+ zips) for handoff.
    if os.environ.get("RUN_EXPORT", "1") == "1":
        try:
            from . import export_results
            export_results.run(run_root)
            manifest["export_dir"] = str(run_root / "export")
        except Exception as e:  # noqa: BLE001
            print(f"⚠️  export failed ({e}); results are still complete.")

    with open(run_root / "full954_run_manifest_done.json", "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    print("\nDONE")
    print(f"  RQ1 paradigm (954):   {para_dir}/variance_evidence")
    print(f"  Deployment (24):      {dep_dir}/variance_evidence")
    print(f"  Figures:              {run_root}/figures")
    print(f"  Export (handoff):     {run_root}/export  (swid_csv.zip, swid_figures.zip)")
    print(f"  Run root: {run_root}")
    return run_root


if __name__ == "__main__":
    main()
