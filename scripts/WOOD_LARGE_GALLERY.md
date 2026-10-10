# Large-gallery wood retrieval study

This is a prospective experiment, not a reported result. The target of 0.5
macro R@1 for a 954-species gallery is a hypothesis, not a guaranteed outcome.
Do not update the manuscript until the locked tests have run and the protocol
checks pass.

The method fine-tunes the image encoder and correspondence scorer jointly from
the existing image-trained QKV checkpoint. Training adds a class-prototype
memory bank of all 557 meta-train species, including single-scan species as
negatives. Hard-bank episodes sample visually nearby classes; random-bank
episodes are the matched negative-sampling control. Support and query images
remain source-scan disjoint, with 32/64/128-way and K=1/2/5 episodes. The bank
is refreshed once per epoch and detached, while the episode images backpropagate
through the encoder. This does **not** train on every meta-train image in each
epoch. Inspect `unique_train_images` in the epoch reports.

The two-stage full-gallery test scores all 176,123 SWI images from 954 species
for class-level candidates. It compares raw/adapted prototype and image-max
scores plus reciprocal-rank fusion. The second stage receives only the 32 or
128 shortlisted species, uses a deterministic scan-diverse cache of up to five
images per species, and scores at most three distinct scans per candidate.
Candidate recall@K is an upper bound on second-stage R@1; no reranker can fix a
true species absent from its shortlist. Global, adapted, fixed-local, MaxSim,
QKV gate/consensus, hard-bank/random-bank, cardinality and source-stratified
controls are reported, alongside the upstream DINOv2-S retrieval encoder and
matched image-trained global-only baseline. The public-ID cohort is **test
only**. There is no OOD
or new-data enrollment claim from this study.
Summary files include macro R@1, macro R@5, macro MRR and candidate recall;
search timing excludes image encoding.

## Colab: setup and preflight

Mount Drive first. This runner is exposed through the existing `runpy` entry
point. Use a fresh output directory for a changed experiment configuration.
The default source seed is 43. Seeds 42/43/44 below vary only the **new
fine-tuning stage** while sharing the same upstream QKV checkpoint. They are
not three independently pretrained encoders. Set `WOOD_LARGE_SOURCE_SEED=match`
only when matched upstream QKV/global-only checkpoints exist for every seed.

```python
%cd /content/drive/MyDrive/NCS
!git -C swid_retrieval pull

import os, sys, runpy
from pathlib import Path
sys.path.insert(0, "/content/drive/MyDrive/NCS")
for name in list(sys.modules):
    if name.startswith("swid_retrieval"):
        del sys.modules[name]
for key in list(os.environ):
    if key.startswith("RUN_") or key.startswith("WOOD_LARGE_"):
        os.environ.pop(key, None)

root = Path("/content/drive/MyDrive/NCS")
os.environ.update({
    "ROOT_PATH": str(root),
    "DEVICE": "cuda",
    "RUN_WOOD_LARGE_GALLERY": "1",
    "WOOD_LARGE_OUT": str(root / "results/wood_large_gallery_final_v1"),
    "WOOD_LARGE_BASE_OUT": str(root / "results/dinov2s_retrieval_matched_v1"),
    "WOOD_LARGE_QKV_OUT": str(root / "results/wood_correspondence_image_full_v1"),
    "WOOD_LARGE_GLOBAL_OUT": str(root / "results/wood_correspondence_global_only_v1"),
    "WOOD_LARGE_SOURCE_SEED": "43",
    "WOOD_LARGE_SEEDS": "42,43,44",
    "WOOD_LARGE_ARMS": "backbone_base,qkv_base,global_only_base,hard_bank,random_bank",
    "WOOD_LARGE_EPOCHS": "5",
    "WOOD_LARGE_EPISODES": "300",
    "WOOD_LARGE_WORKERS": "2",
    "WOOD_LARGE_MICROBATCH": "4",
    "WOOD_LARGE_IMAGE_BATCH": "32",
    "WOOD_LARGE_BANK_NEGATIVES": "128",
    "WOOD_LARGE_HARD_FRACTION": "0.5",
    "WOOD_LARGE_BANK_WEIGHT": "0.5",
    "WOOD_LARGE_RERANK_WIDTHS": "32,128",
    "WOOD_LARGE_QUERY_LIMIT_PER_SPECIES": "0",
})
for mode in ("smoke", "preflight"):
    os.environ["WOOD_LARGE_MODE"] = mode
    _ = runpy.run_module("swid_retrieval.run_overnight", run_name="__main__")
```

`preflight` must find the existing image-trained QKV and matched global-only
control. If either is absent, train the missing control under the documented
`WOOD_CORRESPONDENCE_IMAGE_TRAIN.md` protocol first; do not silently substitute
a feature-only or different-seed checkpoint.

## Train and validate

The full schedule is expensive. It resumes from `latest.pt` at epoch boundaries
with matching code and configuration. A pilot should use a separate output
directory and **not** be merged with the final study. The image cache at
`/content/cache_images` is local to the current Colab runtime; an empty cache
means Drive image reads will dominate early steps. The runner deliberately does
not force another hours-long preload.

```python
os.environ["WOOD_LARGE_MODE"] = "train"
_ = runpy.run_module("swid_retrieval.run_overnight", run_name="__main__")

os.environ["WOOD_LARGE_MODE"] = "validate"
_ = runpy.run_module("swid_retrieval.run_overnight", run_name="__main__")

os.environ["WOOD_LARGE_MODE"] = "select"
_ = runpy.run_module("swid_retrieval.run_overnight", run_name="__main__")
```

The best epoch and arm are selected only from meta-val fold 0 using 75% weight
on 637-species K=1 R@1 and 25% on 57-species K=5 R@1. Fold 1 is descriptive,
not an independent validation set for the previously trained QKV source. The
selection record is immutable in this output directory. Never change the
method or rerank width after inspecting meta-test or public-ID results and
still call them confirmatory.

## Locked tests and review ZIP

Run these only once the choice above is fixed. The full954 step extracts all
gallery and public-ID embeddings separately for each evaluated encoder; it
can take substantial time and Drive space. The cache is resumable. Do not set
`WOOD_LARGE_QUERY_LIMIT_PER_SPECIES` above zero for the final result.

```python
for mode in ("meta_test", "full954", "export"):
    os.environ["WOOD_LARGE_MODE"] = mode
    _ = runpy.run_module("swid_retrieval.run_overnight", run_name="__main__")

from google.colab import files
files.download(str(root / "results/wood_large_gallery_final_v1/wood_large_gallery_review.zip"))
```

Send only the ZIP. It contains small CSV/JSON artifacts, not `*.npy` features
or `*.pt` weights. If an audit requires weights, share their hashes and paths
first. Review `selection_lock.json`, `selection_candidates.csv`,
`meta_val_all_arms.csv`, `meta_test_locked_summary.csv`,
`full954/*/seed_43/full954_summary.csv`, `cardinality_curve.csv`,
`per_source.csv`, `full954_paired.csv`, and each `protocol.json` before making
claims. Paired CIs are across public-ID species for fixed checkpoints only;
they do not quantify full training-run variance. The 954-species public-ID
macro R@1 is not directly interchangeable with the SWI meta-test 637-way R@1.
