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

If `full954` stops at `Failed to read image`, inspect the exact source path
and its local cache copy before restarting. The extractor retries transient
read failures three times and keeps the feature-cache cursor, so rerunning
`full954` resumes from the last flushed batch. A persistently missing or
unreadable source image must be restored from the original dataset; do not
drop it, replace it with a different patch, or change the gallery cohort to
make extraction finish. The local cache path is
`data.CachedImageLoader()._cache_path(source_path)`; check both files with
`Path.is_file()`, size, and `cv2.imread()`. Colab's `/content/cache_images`
is runtime-local and may disappear after a disconnect.
The inference loader accepts the exact training-runner hash from before the
read-retry change, while continuing to verify the source checkpoint, manifest,
method, arm, seed, settings, selection lock and selected checkpoint hash.

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

## Follow-up diagnosis: why is the true species deep in the ranking?

Do not label this a wood-specific fine-grained effect without controls. In the
earlier matched global-only seed-43 meta-val report
(`wood_correspondence_matched_review.zip`), the same 57 unseen target species
with one reference each went from R@1=0.395 and candidate recall@5=0.681 in
the 57-species gallery to R@1=0.153 and recall@5=0.353 in the 637-species
gallery. At 637 species, candidate recall@32=0.640 and @128=0.804: about 20%
of queries are outside top-128, not all queries. The extra 580 distractors are
557 meta-train species plus 23 meta-val species excluded from the scan-disjoint
targets. This creates a potential seen-distractor/unseen-target asymmetry in
addition to cardinality. K=1, source-scan separation, image-scale differences,
prototype quality and genuine anatomical similarity are further hypotheses.

After the locked run, inspect the following without tuning on meta-test or
public-ID:

- Rank quantiles and candidate recall@1/5/10/32/128 by species and source;
  separate isolated outliers from a general ranking failure.
- Paired cardinality curve with fixed queries and the same target reference
  image; this isolates added distractors. Contrast 57x1 against 57x5 separately
  to measure the reference-count effect.
- Fraction of high-scoring false matches from meta-train species versus other
  distractors. The groups are unequal in the current 637-way protocol, so a
  raw fraction alone is not evidence of known-class bias.
- Same-genus versus different-genus false matches, source scan, magnification,
  and image-quality strata; manually inspect a small pre-specified sample of
  the deepest-rank failures with anatomical expertise.
- Stage-1 candidate recall and stage-2 conditional accuracy. If candidate
  recall@128 is below the target R@1, improve the encoder/candidate search;
  otherwise prioritize reranking and its query-reference loss.
- For K=1, stratify paired query/reference outcomes by crop-level gap
  (`abs(log2(scale_query / scale_reference))`), holding target species,
  query, reference count and source-scan separation fixed where possible.
  Report sample counts, R@1 and candidate recall in each stratum; compare
  against same-level pairs before attributing any failure to scale. The
  `scale_*` directory denotes source-image crop width in pixels, not optical
  magnification. Check how many species have cross-level, cross-scan pairs
  before designing a scale-specific training objective.

Potential next loss ablation, only after the above diagnosis: add a hard-pair
margin loss on QKV scores or distill QKV rankings into the global encoder.
Compare each against the unchanged current recipe (episode cross-entropy,
global auxiliary loss, SupCon and prototype-bank loss), with matched seeds and
meta-val-only selection. Do not add a generic contrastive term and attribute
any gain to novel wood anatomy without a matching control.

If a reproducible cross-level deficit remains after the paired analysis, test
cross-level supervised contrastive learning as a separate next-round ablation.
The existing SupCon pairs patch images by species label but does not enforce
different crop levels; episodes enforce different source scans but do not
stratify levels. Sample same-species positives from different levels and scans,
with different-species negatives matched on level where feasible. Compare
unchanged training, level-aware sampling alone, the added loss alone, and both,
using matched seeds and meta-val-only tuning. Evaluate K=1 and K=5 across
57/637/954-species galleries and by crop-level gap. Avoid forcing token-level
correspondence across scans without anatomical alignment, or treating all
crop levels as interchangeable when they expose different structures. Do not
change the current locked run or claim a benefit before these controls pass.
