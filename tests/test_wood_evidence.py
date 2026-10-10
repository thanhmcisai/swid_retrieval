"""Focused checks for the isolated wood local-evidence scorer."""

import importlib.util
import io
import os
import runpy
import unittest
from contextlib import redirect_stdout
from unittest import mock


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch unavailable")
class WoodEvidenceMethodTest(unittest.TestCase):
    def setUp(self):
        import torch
        from swid_retrieval.wood_evidence_method import (
            WoodEvidenceReranker, local_pair_scores, local_pair_scores_batch,
            prototype_scores, scan_consensus, scan_consensus_batch)
        self.torch = torch
        self.model = WoodEvidenceReranker
        self.local_pair = local_pair_scores
        self.local_pair_batch = local_pair_scores_batch
        self.prototype = prototype_scores
        self.consensus = scan_consensus
        self.consensus_batch = scan_consensus_batch

    def test_local_pair_prefers_matching_patch_set(self):
        torch = self.torch
        query = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.]])
        refs = torch.stack((query.flip(0), torch.tensor([[0., 0., 1., 0.],
                                                        [0., 0., 0., 1.]])))
        pair = self.local_pair(query, refs)
        self.assertAlmostEqual(pair[0].item(), 1.0)
        self.assertAlmostEqual(pair[1].item(), 0.0)

    def test_scan_consensus_reduces_per_scan_before_class_pooling(self):
        torch = self.torch
        pair = torch.tensor([0.9, 0.8, 0.2, 0.5])
        labels = torch.tensor([0, 0, 0, 1])
        scans = ["a", "a", "b", "c"]
        classes = torch.tensor([0, 1])
        pooled = self.consensus(pair, labels, scans, classes)
        nearest = self.consensus(pair, labels, scans, classes, enabled=False)
        self.assertAlmostEqual(pooled[0].item(), 0.55, places=6)
        self.assertAlmostEqual(nearest[0].item(), 0.9, places=6)
        self.assertAlmostEqual(pooled[1].item(), 0.5, places=6)

    def test_batched_evidence_matches_single_query(self):
        torch = self.torch
        query = torch.randn(3, 4, 8)
        refs = torch.randn(5, 4, 8)
        query = torch.nn.functional.normalize(query, dim=-1)
        refs = torch.nn.functional.normalize(refs, dim=-1)
        labels = torch.tensor([0, 0, 1, 1, 2])
        scans = ["a", "b", "c", "c", "d"]
        classes = torch.tensor([0, 1, 2])
        batched = self.local_pair_batch(query, refs)
        expected = torch.stack([self.local_pair(q, refs) for q in query])
        self.assertTrue(torch.allclose(batched, expected))
        pooled = self.consensus_batch(batched, labels, scans, classes)
        single = torch.stack([self.consensus(row, labels, scans, classes)
                              for row in expected])
        self.assertTrue(torch.allclose(pooled, single))

    def test_global_mode_matches_prototype_and_local_training_has_gradient(self):
        torch = self.torch
        q = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.]])
        r = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.],
                          [0.8, 0.6, 0., 0.]])
        qt = torch.stack((q, q), dim=1)
        rt = torch.stack((r, r), dim=1)
        labels = torch.tensor([0, 1, 0])
        ranker = self.model(dimension=4, adapter_width=2)
        global_scores, classes = self.prototype(q, r, labels)
        bypass, bypass_classes = ranker(q, qt, r, rt, labels, ["a", "b", "c"],
                                        local=False)
        self.assertTrue(torch.equal(classes, bypass_classes))
        self.assertTrue(torch.allclose(global_scores, bypass))
        local_scores, _ = ranker(q, qt, r, rt, labels, ["a", "b", "c"],
                                 top_classes=2)
        self.assertTrue(torch.isfinite(local_scores).all())
        local_scores[0, 0].backward()
        self.assertIsNotNone(ranker.weight_logit.grad)
        self.assertIsNotNone(ranker.adapter[-1].weight.grad)

    def test_top_one_does_not_update_non_candidate(self):
        torch = self.torch
        q = torch.tensor([[1., 0., 0., 0.]])
        r = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.]])
        qt = q[:, None, :].repeat(1, 2, 1)
        rt = r[:, None, :].repeat(1, 2, 1)
        labels = torch.tensor([0, 1])
        ranker = self.model(dimension=4, adapter_width=2)
        global_scores, _ = self.prototype(q, r, labels)
        local_scores, _ = ranker(q, qt, r, rt, labels, ["a", "b"],
                                 top_classes=1, weight_override=2.0)
        self.assertEqual(local_scores[0, 1].item(), global_scores[0, 1].item())

    def test_variable_reference_episode_optimizer_step(self):
        torch = self.torch
        torch.manual_seed(3)
        ranker = self.model(dimension=8, adapter_width=4)
        optimizer = torch.optim.AdamW(ranker.parameters(), lr=1e-2)
        for k in (1, 2):
            queries = torch.nn.functional.normalize(torch.randn(16, 8), dim=-1)
            references = queries.repeat_interleave(k, dim=0)
            query_tokens = queries[:, None, :].repeat(1, 4, 1)
            reference_tokens = references[:, None, :].repeat(1, 4, 1)
            labels = torch.arange(16).repeat_interleave(k)
            scans = [f"s{index}_{view}" for index in range(16) for view in range(k)]
            optimizer.zero_grad()
            scores, classes = ranker(queries, query_tokens, references,
                                     reference_tokens, labels, scans, top_classes=64)
            self.assertTrue(torch.equal(classes, torch.arange(16)))
            loss = torch.nn.functional.cross_entropy(scores / 0.07, torch.arange(16))
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            optimizer.step()
        self.assertTrue(torch.isfinite(ranker.weight_logit))


@unittest.skipUnless(all(importlib.util.find_spec(name) is not None for name in
                         ("torch", "numpy", "pandas", "cv2", "albumentations")),
                     "Gallery dependencies unavailable")
class WoodEvidenceExperimentTest(unittest.TestCase):
    def test_runpy_dispatches_evidence_study_only(self):
        from swid_retrieval import wood_evidence_experiment as experiment
        flags = {name: "0" for name in (
            "RUN_GALLERY_STUDY", "RUN_GALLERY_DIAGNOSTICS", "RUN_REPAIR_PUBLIC_ROWS",
            "RUN_FINAL_SCURD_RETRAIN", "RUN_FINAL_COLAB_AUDIT")}
        flags["RUN_WOOD_EVIDENCE_STUDY"] = "1"
        with mock.patch.dict(os.environ, flags, clear=False):
            with mock.patch.object(experiment, "run") as pilot:
                with redirect_stdout(io.StringIO()):
                    runpy.run_module("swid_retrieval.run_overnight", run_name="__main__")
                pilot.assert_called_once_with()

    def test_meta_train_episode_is_scan_disjoint_and_variable_k(self):
        import numpy as np
        from swid_retrieval import wood_evidence_experiment as experiment
        items = []
        for index in range(70):
            for scan in range(3):
                items.append((f"/scale_256/patch_x_from_Tw{index * 3 + scan:04d}.jpg",
                              f"species_{index:03d}"))
        by_class = experiment._episode_layout(items)
        hard = {label: [other for other in by_class if other != label]
                for label in by_class}
        for k in (1, 2):
            refs, queries, labels, scans = experiment._episode_indices(
                by_class, items, np.random.default_rng(42 + k), 16, k, hard)
            self.assertEqual(len(refs), 16 * k)
            self.assertEqual(len(queries), 16)
            self.assertEqual(len(labels), 16 * k)
            for class_index, query_index in enumerate(queries):
                query_scan = experiment.study.source_scan_id(items[query_index][0])
                self.assertTrue(all(scan != query_scan for scan in
                                    scans[class_index * k:(class_index + 1) * k]))
                if k == 2:
                    self.assertEqual(len(set(scans[class_index * k:(class_index + 1) * k])), 2)

    def test_global_oracle_stays_fixed_across_rerank_modes(self):
        import numpy as np
        from swid_retrieval import wood_evidence_experiment as experiment
        queries = [("/scale_256/patch_species_a_from_Tw900.jpg", "species_a")]
        classes = np.asarray(["species_a", "species_b"])
        global_scores = np.asarray([[0.1, 0.2]])
        reranked = np.asarray([[0.9, 0.1]])
        rows = experiment._metric_rows(reranked, classes, queries, "2x1", 0,
                                       "local", global_scores)
        self.assertEqual(rows[0]["correct"], 1)
        self.assertEqual(rows[0]["global_true_rank"], 2)

    def test_evaluation_keeps_query_and_reference_rows_aligned(self):
        import numpy as np
        import torch
        from swid_retrieval import wood_evidence_experiment as experiment
        refs = [(f"/scale_256/patch_{label}_{j}_from_Tw{class_id:03d}.jpg", label)
                for class_id, label in enumerate(("species_a", "species_b"))
                for j in range(5)]
        queries = [(f"/scale_512/patch_{label}_from_Tw{class_id + 10:03d}.jpg", label)
                   for class_id, label in enumerate(("species_a", "species_b"))]

        def fake_features(out, name, items, encoder, cfg, device, base_hash):
            global_emb = np.zeros((len(items), 512), dtype=np.float32)
            for index, (_, label) in enumerate(items):
                global_emb[index, 0 if label == "species_a" else 1] = 1
            return global_emb, np.repeat(global_emb[:, None, :], 16, axis=1)

        plan = {"2x1": (np.asarray([0, 5]), np.asarray([0, 1]))}
        with mock.patch.object(experiment, "_validation_sets", return_value=(refs, queries, [])), \
             mock.patch.object(experiment, "_gallery_plan", return_value=plan), \
             mock.patch.object(experiment, "_features", side_effect=fake_features):
            frame, summary = experiment._evaluate_split(
                {}, "meta-test", None, None, None, torch.device("cpu"), "hash",
                experiment.WoodEvidenceReranker(), ("global", "local_fixed"), folds=(0,))
        self.assertEqual(len(frame), 4)
        self.assertTrue((frame["true_reference_scans"] == 1).all())
        self.assertTrue((summary["macro_r1"] == 1).all())

    def test_paired_report_counts_rescues_and_harms(self):
        import pandas as pd
        from swid_retrieval import wood_evidence_experiment as experiment
        rows = []
        for query, label, global_hit, local_hit in (("a", "species_a", 0, 1),
                                                      ("b", "species_b", 1, 0)):
            for mode, correct in (("global", global_hit), ("local_fixed", local_hit)):
                rows.append({"gallery": "2x1", "fold": 0, "query_path": query,
                             "true_label": label, "mode": mode, "correct": correct})
        result = experiment._paired_differences(pd.DataFrame(rows), n_boot=100)
        self.assertEqual(result.iloc[0]["rescued_queries"], 1)
        self.assertEqual(result.iloc[0]["harmed_queries"], 1)
        self.assertEqual(result.iloc[0]["macro_r1_delta"], 0)
        with self.assertRaisesRegex(ValueError, "Repeated"):
            experiment._paired_differences(pd.DataFrame(rows + [rows[0]]))


if __name__ == "__main__":
    unittest.main()
