# Large-gallery, few-reference retrieval: literature and study decisions

Status: research note (2026-10-13). This does not change the manuscript, a
locked checkpoint, or any reported result. "State of the art" is benchmark-
and protocol-specific; leaderboard numbers below are not wood-image baselines.

## Closest published protocols

| Source | Scale and scoring | Match to the wood task |
| --- | --- | --- |
| [FungiTastic, CVPR FGVC 2025](https://openaccess.thecvf.com/content/CVPR2025W/FGVC/papers/Picek_FungiTastic_A_Multi-Modal_Dataset_and_Benchmark_for_Image_Categorization_CVPRW_2025_paper.pdf) | Few-shot subset of a larger fungi benchmark; compares cross-entropy training, nearest image embeddings and class centroids from CLIP, DINOv2 and BioCLIP. | Direct analogue for many fine-grained biological classes with sparse reference evidence; source photographs and wood macro-patches are different domains. |
| [FungiCLEF 2025 overview](https://ceur-ws.org/Vol-4038/paper_233.pdf) | 2,427 training species, 1-4 *observations* per species (one observation may contain multiple photos); 567 species present in the test split. Predictions are ranked; official metric is top-5 accuracy. BioCLIP centroid and nearest-neighbor baselines use the labelled training set as the reference pool. | Strongest nearby large-class/few-reference protocol found. It is not a fixed K-image gallery benchmark and its top-5 score is not wood R@1. Confirm class/query/reference overlap and image counts before reproducing a specific comparison. |
| [FungiCLEF 2025 winning approach, as summarized by organizers](https://ceur-ws.org/Vol-4038/paper_233.pdf) | Multiple pretrained encoders, image augmentation, a projection network, class/observation prototypes, cross-entropy plus InfoNCE, and ensembling; 78.9% private-test top-5. | Suggests a strong *visual-only* comparator and a controlled CE+contrastive ablation. The reported gain bundles several changes and cannot attribute benefit to any single component. |
| [FungiCLEF 2025 runner-up](https://ceur-ws.org/Vol-4038/paper_247.pdf) | DINOv2-derived features, contrastive feature refinement and a Transformer; its paper describes metadata fusion; 78.1% private-test top-5. | Motivates hard-negative contrastive training, but feature refinement is not evidence that our QKV head alone should work. The organizers' overview calls the top two entries image-only, conflicting with the runner-up's methods section. Verify the submitted pipeline/code before treating metadata fusion as part of the scored method. |
| [2026 wildlife study](https://ieeexplore.ieee.org/abstract/document/11507419) | Reports 1,971 species and cosine matching to a small support set. | Potential comparator only: the accessible abstract does not verify K per species or whether all 1,971 species compete in one gallery. Do not compare its headline accuracy with wood R@1. |

## What is already implemented here

| Idea from the literature | Current implementation | Remaining gap / implication |
| --- | --- | --- |
| Domain-relevant backbone rather than a generic embedding alone | `gallery_experiment.py` tests DINOv2 and a compact wood encoder; `wood_full_image_train.py` starts from a wood-adapted DINOv2-S retrieval checkpoint. Earlier CE-Full is also a strong *but differently exposed* comparator. | No controlled BioCLIP/wood-specific self-supervised pretraining or same-protocol encoder ensemble. Do not treat a 954-label CE model as an unseen-species baseline when it has seen the target species. |
| Full-class discrimination | `wood_full_image_train.py` runs every one of 124,577 meta-train images as a query once per epoch and uses 557-way ArcFace in `arcface` and `arcface_qkv` arms. | Both arms have now completed three epochs for seed 43 and were assessed on meta-val (see below). Neither has a held-out result. |
| Contrastive/ranking objective | `wood_multilevel.py` contains cross-scale multi-positive InfoNCE; `wood_scale_rank_study.py` has rank-32/rank-128 inversion-loss arms; `wood_hard_negative.py` mines hard classes. `wood_full_image_train.py` now has matched ArcFace controls, pair InfoNCE, supervised contrastive, packed multi-positive, and QKV arms. | The new arms are coded but have not yet produced Colab results. Full-image QKV competes against only the distinct classes in the batch, whereas its classifier competes against all 557. The packed sampler has fewer in-batch negative classes and must be compared with its sampler-only control. |
| Sparse, multi-view prototypes | `wood_reference_budget.py` and `wood_multilevel.py` test reference budgets and scan/scale-aware selection. The full-image evaluator now holds scan-disjoint meta-val queries fixed across K=1/3/5/10 and two reference policies. | Full-image training still pairs each query with one positive reference, though the packed sampler brings multiple query/reference views of a species into one contrastive batch. It does not train against an explicit enrolled K=1/3/5/10 reference *set* in the same loss. Distinguish a source scan, an image and a patch: multiple patches from one scan are not independent references. |
| Local matching / QKV reranking | `wood_correspondence_method.py`, `wood_hard_negative.py` and `wood_full_image_train.py` implement QKV and shortlist controls. Full-image validation reranks up to 128 classes in the 637x1 condition. | A reranker cannot recover the correct species when candidate generation misses it. Compare global, QKV and any fusion on the *same* fixed candidate set, and report both candidate recall and correct-given-candidate. |
| Balanced sampling and augmentation | `class_balanced_batches()` in `wood_full_image_train.py` covers each image once while maximizing distinct labels; `PairDataset` augments query and reference separately. | Do not equate this with observation-level multi-crop TTA, CE+InfoNCE or model ensembling used by FungiCLEF. Each addition needs its own ablation and inference-cost accounting. |

## Evidence that constrains the next experiment

- `results/wood_gallery_audit_review.zip` (exploratory public-ID audit, selected
  prior to this audit): with the 954-species/full-reference gallery, full
  prototypes gave macro R@1 = 0.0831. The top-128 candidate recall was 0.7204;
  QKV-only top-128 gave macro R@1 = 0.0221. These numbers describe that old
  checkpoint and protocol, not the new full-image ArcFace/QKV runner. Neither
  a 0.5 R@1 target nor a 0.789 top-5 fungal leaderboard score follows from
  candidate recall.
- `results/wood_coverage_latest_audit_review.zip` compares a previously
  selected 120-episode model with a 600-episode version on 570 meta-val queries
  at K=10/top-128: macro R@1 0.2807 versus 0.2509. This does not prove that
  more image exposure is inherently harmful; it is one exploratory comparison
  with a previously inspected validation split.
- `results/wood_full_image_study_review.zip` contains the new full-image study:
  seed 43 only, three epochs per arm, 124,577 unique query images per epoch,
  557-way ArcFace and about 63 distinct reference classes per QKV batch.
  The source checkpoint hash matches the earlier `prototype_large/seed_43`
  checkpoint in `dinov2s_seed43_diagnostics_review.zip`. Both arms select
  epoch 1 by their recorded mean of 57x5, 57x1 and 637x1 meta-val R@1:

  | Epoch-1 model and scoring | 57x5 R@1 | 57x1 R@1 | 637x1 R@1 | Mean |
  | --- | ---: | ---: | ---: | ---: |
  | ArcFace, global | 0.5070 | 0.4246 | 0.1702 | 0.3673 |
  | ArcFace+QKV, global only | 0.5211 | 0.4175 | 0.1702 | 0.3696 |
  | ArcFace+QKV, QKV score | 0.5158 | 0.4140 | 0.1772 | 0.3690 |

  QKV scoring changes the same encoder's global ranking by -3, -2 and +4
  correct queries respectively across the three 570-query endpoints. The
  final QKV model beats the ArcFace model by +5, -6 and +4 correct queries;
  its mean gain is only 0.00175. QKV's slight large-gallery gain is not a
  consistent gain across gallery sizes. For 637x1, global candidate recall@128
  is 0.8105 (ArcFace) versus 0.8070 (ArcFace+QKV). Both models' meta-val
  selection scores decline by epoch 3 despite falling training losses. The
  package has aggregate metrics but no paired query-level outputs, so do not
  claim significance or robustness. The two validation folds have been used
  repeatedly in earlier studies, and only one seed was run. These are
  exploratory meta-val results, not evidence for a 954-species/public-ID gain.

## Prioritized decisions

1. **Completed on meta-val; preserve the held-out test.** The matched
   `arcface` versus `arcface_qkv` seed-43 comparison is complete, and both
   select epoch 1. Do not extend the current runs just because the training
   loss falls. Retain both `best.pt` files on Drive. Pre-specify any additional
   ablations and select them on meta-val only; evaluate the locked candidates
   on meta-test after method selection is finished. A 954-species/public-ID
   claim additionally needs evaluation at the same gallery/reference budget.
2. **Screen the pre-specified objective arms on meta-val.** Run the newly coded
   full-image ArcFace, pair-InfoNCE, SupCon, packed-sampler-only,
   packed-SupCon, ArcFace+QKV and packed-SupCon+QKV arms with the same source
   checkpoint and query coverage. These arms separate scorer, objective and
   sampler effects; verify the measured cross-scan/cross-scale positive-pair
   audit and inference cost before selecting a finalist.
3. **Test reference coverage separately from representation learning.** On a
   frozen, validation-selected encoder compare K=1/3/5/10, random versus
   scan-and-scale-diverse references, image-nearest versus class prototype,
   and global versus QKV at equal budgets. Report source-scan counts and
   same-scale/cross-scale strata. Do not use public-ID to pick the policy.
4. **Only then consider expensive enhancements.** Multi-encoder fusion and
   multi-crop TTA are credible FungiCLEF-inspired controls, but are less useful
   than a better candidate generator if top-128 recall or correct-given-
   candidate remains low. Report FLOPs or measured latency alongside accuracy.

Lock hyperparameters and checkpoints on meta-val. Then run one held-out
meta-test and corrected public-ID/full-954 evaluation with the same query,
gallery and reference budget for every arm. Report R@1, R@5, candidate
recall@128, conditional R@1, per-species/source/scale results and paired
uncertainty. FungiCLEF's top-5 leaderboard and the older exploratory public-ID
audit must never be used to select a wood-model checkpoint.
