# -*- coding: utf-8 -*-
"""Gallery construction by scope — the single place the 24-vs-954 choice lives.

Replaces the hardcoded 24-species filter (final_metric_learning_cea_2026.py:2120)
and centralizes the scope logic from variance_retrieval_evidence_colab.py:385-404.
"""

import numpy as np

from . import config


def build_gallery_mask(labels_swi, id_species, scope=None, min_species=None, ce_train_mask=None):
    """Return (mask, gal_labels, scope_label).

    full_swi  → all 954 SWI species (the corrected, cardinality-matched protocol).
    ce_train  → the 954-species gallery restricted to CE-Full's exact train images
                (fairness control); requires ce_train_mask (the cache's
                'swi_in_ce_train' boolean array).
    id_only   → the legacy 24-species gallery.

    full_swi and ce_train both raise if the gallery does not actually span
    >= min_species species (e.g. you pointed at the old meta-test cache).
    """
    scope = (scope or config.GALLERY_SCOPE)
    min_species = config.MIN_FULL_GALLERY_SPECIES if min_species is None else min_species
    labels_swi = np.asarray(labels_swi)

    if scope in config.FULL_SCOPES:
        mask = np.ones(len(labels_swi), dtype=bool)
        scope_label = "full_swi"
    elif scope in config.CE_TRAIN_SCOPES:
        if ce_train_mask is None:
            raise ValueError("ce_train scope requires ce_train_mask ('swi_in_ce_train').")
        mask = np.asarray(ce_train_mask, dtype=bool)
        scope_label = "ce_train"
    elif scope in config.ID_SCOPES:
        id_species = set(id_species)
        mask = np.asarray([x in id_species for x in labels_swi])
        scope_label = "id_only"
    else:
        raise ValueError(f"Unknown GALLERY_SCOPE={scope!r}; use full_swi, ce_train or id_only.")

    gal_labels = labels_swi[mask]
    n_species = len(set(gal_labels))
    print(f"Gallery scope={scope_label}: {int(mask.sum())} SWI images, {n_species} species")
    if scope_label in {"full_swi", "ce_train"} and n_species < min_species:
        raise RuntimeError(
            f"Full-gallery protocol violation ({scope_label}): got only {n_species} gallery "
            f"species, expected at least {min_species}. The embedding cache likely holds only "
            f"the meta-test split — build embedding_cache_full954_v3.npz first "
            f"(swid_retrieval.embeddings.build_full954)."
        )
    return mask, gal_labels, scope_label
