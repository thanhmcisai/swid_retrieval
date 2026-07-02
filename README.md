# swid_retrieval — corrected full-954 SmartWoodID gallery

A clean package that fixes the core retrieval bug in the SmartWoodID paper and
consolidates the previously-flat Colab scripts behind one correct gallery
code-path.

## The bug it fixes

The evaluation code retrieved against a gallery filtered to only the **24
public-ID species** (`final_metric_learning_cea_2026.py:2120`), instead of the
full **~954-species** SmartWoodID gallery — a cardinality confound that inflates
retrieval/OOD numbers (reviewer task **T3** in `AGENTS.md`).

The previously-written `full_swi` path could not run because
`embedding_cache_v3.npz` holds SWI embeddings for only the **~317-species
meta-test split**, so the `>= 900 species` protocol assertion crashed. This
package adds the missing piece — a **full-954 gallery embedding cache** — and
wires every bug-affected experiment to it.

## What runs (and what doesn't)

| Scope | Source |
|---|---|
| Matched paradigm, full 954-species gallery | `variance_retrieval_evidence_colab.py` for native/prototype/R@1 and retrieval-quality checks |
| Deployment, 24 public-ID target species | `variance_retrieval_evidence_colab.py` for gallery strategies, OOD, retention, operating point, RQ5, FPR95 CI, OOD-by-source, seed sensitivity |
| Review evidence | `review_evidence_colab.py` twice: `full_swi` for matched-954 statistical cleanup, `id_only` for deployment failure taxonomy and optional saliency |
| Edge/CPU latency + search scaling | `edge_deployment_proxy_colab.py` (timing; reused as-is) |

**Not recomputed — not affected by the bug:** CE fine-tune catastrophic
forgetting (classification argmax), MSP/ODIN/OpenMax/Energy OOD baselines
(logit-based), and VN26/Exp4 (its own `SWI_pool`/VN26 galleries).

## Layout

```
swid_retrieval/
  config.py                 # all paths/env/scope/SC-URD hyperparams
  data.py                   # datasets, transforms, manifest, canonical_label
  models.py                 # checkpoint loaders (CE/ArcFace-557/-954/Proto/variants/ImageNet/CLIP/DINOv2)
  gallery.py                # build_gallery_mask(scope)  ← the 24-vs-954 decision
  scurd.py                  # SC-URD adapter, loader, projector
  embeddings/
    extract.py              # extract_embeddings / clip / dinov2
    cache_schema.py         # SWI key plan + ID/OOD copy prefixes
    build_full954.py        # *** builds embedding_cache_full954_v3.npz ***
  experiments/
    __init__.py             # notes: logic reused via orchestrator exec()
    rq1_native.py           # native CE + prototype (anchor-matched) helpers
  _engines/                 # validated experiment scripts, run as-is via exec()
    variance_retrieval_evidence_colab.py   # RQ1/OOD/RQ5/gallery/SC-URD-seeds
    review_evidence_colab.py               # per-species / taxonomy / cleanup
    edge_deployment_proxy_colab.py         # latency / search-scaling timing
  orchestrator.py           # build cache → deployment/paradigm/ce_train/native → review → edge → figures/export
  run_colab.py              # one-cell entry
```

The package is **self-contained**: the three validated engines live under
`_engines/` and are `exec()`'d by the orchestrator (not re-implemented), so only
`swid_retrieval/` (plus `run_overnight.py`) needs to be synced to Drive. The
orchestrator finds them in `_engines/` first, falling back to the package parent
dir / `ROOT_PATH` for the older flat layout (or a `*_SCRIPT_PATH` env override).

## Run on Colab

Place this package and the `*_colab.py` scripts on Drive (e.g. under
`/content/drive/MyDrive/NCS`), then in one cell:

```python
from google.colab import drive; drive.mount("/content/drive")
%cd /content/drive/MyDrive/NCS
import os
os.environ["ROOT_PATH"] = "/content/drive/MyDrive/NCS"
from swid_retrieval import orchestrator
orchestrator.main()
```

This builds the 954 cache once (Step 0, the heavy GPU pass — resumable via
`SAVE_PARTIAL`), then runs **each experiment at the gallery scope its research
question requires**, in one go. Outputs land in a fresh
`results/paper_reframe_full954_<timestamp>/` with one subdir per scope; the
original `paper_reframe` results and `embedding_cache_v3.npz` are never touched.
The default run is **paper-complete**: it includes cost tables, CPU proxy rows,
SC-URD backbone ablation and qualitative saliency assets. Disable the heavy
parts explicitly for a fast debug run.

**Per-question gallery (one run):**
| Subdir | Gallery | Experiments → paper |
|---|---|---|
| `paradigm/` | 954 (matched to CE classifier) | RQ1 native/prototype/R@1 (`tab:rq1`), mAP/Hit@k, matched-954 deployment for the appendix; review taxonomy (RQ1 per-species) |
| `deployment/` | 24 target species (`id_only`) | gallery strategies A/B/C (`tab:rq1_gallery`), OOD (`tab:ood`), retention (`tab:adapt`), reviewer-gap, RQ5, SC-URD seed train+sensitivity, deployment failure taxonomy |
| `ce_train_robustness/` | CE-train images (954 species) | RQ1 fairness robustness (`tab:ce_train_robustness`) |
| `native_experiments/` | task-specific galleries | VN26, extended OOD baselines, OOD-only K-shot, cost and appendix analyses |
| `edge_deployment_proxy/` | — | latency/timing (scope-independent) |
| `figures/` | — | vector-PDF paper figures (RQ1 paradigm, gallery strategy, OOD, retention, RQ5, robustness), each from the correct scope |

The cardinality sanity gate checks `paradigm` R@1 < `deployment` R@1. Figures are
rendered in the final step (`RUN_VISUALIZE=1`); regenerate standalone with
`bash swid_retrieval/scripts/visualize.sh` or `python -m swid_retrieval.visualize <run_dir>`.

### Key env knobs
- `RUN_BUILD_FULL954=0` — reuse an existing 954 cache (skip extraction).
- `FORCE_REBUILD_FULL954=1` — rebuild even if a valid cache exists.
- `RUN_REVIEW_TAXONOMY_FULL_GALLERY=0`, `RUN_REVIEW_TAXONOMY_DEPLOYMENT=0`,
  `RUN_EDGE_PROXY=0` — skip those steps.
- `RUN_HEAVY=0` — skip CE fine-tune and cost artifacts (`tab:deployment_cost`,
  `tab:inference_cost` will be missing).
- `RUN_SCURD_BACKBONE=0` — skip SC-URD alternate-backbone ablation
  (`tab:scurd_backbone` will be missing).
- `RUN_INTERPRETABILITY=0` — do not regenerate occlusion saliency images; the
  visualizer will still copy existing static saliency assets when available.
- `RUN_CE_TRAIN_ROBUSTNESS=0` — skip the CE-train fairness pass (Step 5).
- `STRICT_SANITY=0` — downgrade the cardinality gate from hard-fail to warning.
- `NUM_WORKERS`, `DEVICE` — Colab tuning.
- `PRELOAD_IMAGE_CACHE=1` (default), `PRELOAD_ALL_IMAGES=1` (default in the
  Colab/overnight runners), `PRELOAD_WORKERS=16`, `IMAGE_CACHE_DIR=/content/cache_images`
  — pre-warm Drive images into the local JPEG cache before experiments start.
  With `PRELOAD_ALL_IMAGES=1`, the preloader collects all SWI paths from
  `swi_manifest.json` plus public query/reference paths from
  `ID_images_expanded.csv` and `OOD_images_expanded.csv`, normalising any old
  local-machine prefixes to the active Drive dataset root. Set
  `PRELOAD_ALL_IMAGES=0` to pre-warm only SWI during `build_full954`, or
  `PRELOAD_IMAGE_CACHE=0` to disable image pre-warming entirely.

### Gallery scopes (`GALLERY_SCOPE`)
- `full_swi` (default) — all 954 SWI species; the cardinality-matched protocol.
- `ce_train` — the 954-species gallery **restricted to CE-Full's exact training
  images** (a fairness control answering "does retrieval win just because its
  gallery is bigger than what CE learned from?"). The per-row mask
  `swi_in_ce_train` is computed *during the single full-954 extraction* (no second
  extract) by replaying CE-Full's deterministic 65/15/20 split (seed 42), and
  stored in the cache. Step 5 of the orchestrator runs this pass automatically
  into `<run>/ce_train_robustness/`. Still spans all 954 species (CE keeps ≥1
  train image per species).
- `id_only` — the legacy 24-species gallery (the original bug; kept for reference).

The fairness logic in three layers: (1) **Prototype** = one centroid/species vs
CE's one weight/class (anchor-matched); (2) **CE-native vs CE-NN** uses identical
features + data, so the gap can't be a gallery-size effect; (3) **ce_train**
restricts the gallery to CE's exact training images. See the paper's
`app:ce_train_robustness`.

## Verification (hard-fail gates in the orchestrator)

1. **Cache shaped** — `embedding_cache_full954_v3.npz` exists with `>= 900`
   gallery species; ID/OOD query keys copied (same shapes as the old cache).
2. **Scope assert passes** — the variance engine loads under `full_swi` without
   `RuntimeError`.
3. **Outputs tagged** — JSON/CSV carry the expected `gallery_scope` for their
   subdir (`full_swi`, `id_only`, or `ce_train`).
4. **Review evidence ran** — matched-954 review evidence exists under
   `paradigm/review_evidence/`, and deployment failure taxonomy exists under
   `deployment/review_evidence/`.
5. **Cardinality sanity (the core proof)** — for DINOv2 / ArcFace-557 / Fusion,
   full-gallery R@1 **< 24-species R@1** (read from the old `rq1_paradigm.json`).
   A non-negative delta means the gallery did not expand → the orchestrator
   raises (unless `STRICT_SANITY=0`).
