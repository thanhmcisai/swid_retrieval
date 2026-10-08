"""Small synthetic checks for the isolated gallery study (run on Colab)."""

import importlib.util
import unittest


DEPS = all(importlib.util.find_spec(name) is not None for name in
           ("torch", "numpy", "pandas", "cv2", "albumentations"))


@unittest.skipUnless(DEPS, "Colab ML dependencies are unavailable locally")
class GalleryMethodTest(unittest.TestCase):
    def setUp(self):
        import numpy as np
        import torch
        from swid_retrieval import gallery_experiment as experiment
        from swid_retrieval import gallery_method as method
        self.np, self.torch, self.experiment, self.method = np, torch, experiment, method

    def test_score_contains_every_gallery_class(self):
        torch = self.torch
        scorer = self.method.GalleryScorer(top_m=1, mode="learned")
        refs = torch.nn.functional.normalize(torch.tensor([
            [1., 0.], [0.9, 0.1], [0., 1.], [-1., 0.]]), dim=1)
        labels = torch.tensor([2, 2, 5, 9])
        query = refs[:1]
        scores, classes = scorer(query, refs, labels)
        self.assertEqual(scores.shape, (1, 3))
        self.assertEqual(classes.tolist(), [2, 5, 9])
        self.assertTrue(torch.isfinite(scores).all())

    def test_variable_gallery_loss_reaches_encoder_features(self):
        torch = self.torch
        embeddings = torch.randn(12, 16, requires_grad=True)
        embeddings = torch.nn.functional.normalize(embeddings, dim=1)
        scorer = self.method.GalleryScorer(top_m=4, mode="learned")
        loss, parts = self.method.episode_objective(embeddings, scorer, 4, 2, 1)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("expansion", parts)
        self.assertIsNotNone(scorer.evidence_gate[-1].weight.grad)

    def test_normalized_evidence_is_invariant_to_identical_reference_copy(self):
        torch = self.torch
        query = torch.tensor([[1., 0.]])
        first = torch.tensor([[1., 0.], [0., 1.]])
        doubled = torch.tensor([[1., 0.], [1., 0.], [0., 1.]])
        corrected = self.method.GalleryScorer(top_m=4, mode="fixed", normalize_evidence=True)
        uncorrected = self.method.GalleryScorer(top_m=4, mode="fixed", normalize_evidence=False)
        a, _ = corrected(query, first, torch.tensor([0, 1]))
        b, _ = corrected(query, doubled, torch.tensor([0, 0, 1]))
        c, _ = uncorrected(query, first, torch.tensor([0, 1]))
        d, _ = uncorrected(query, doubled, torch.tensor([0, 0, 1]))
        self.assertTrue(torch.allclose(a, b, atol=1e-6))
        self.assertGreater(float(d[0, 0]), float(c[0, 0]))

    def test_sampler_is_reproducible_and_disjoint(self):
        labels = [f"genus_{i}" for i in range(8) for _ in range(6)]
        sampler = self.experiment.EpisodeSampler(labels, (4, 8), 2, 1, 5, 42)
        first = list(iter(sampler))
        second = list(iter(sampler))
        self.assertEqual(first, second)
        for indices in first:
            self.assertEqual(len(indices), len(set(indices)))

    def test_meta_val_never_uses_train_or_test(self):
        manifest = {
            "meta-train": [["train.jpg", "train_species"]],
            "meta-val": [[f"val_{species}_{image}.jpg", f"genus_{species}"]
                         for species in range(24) for image in range(10)],
            "meta-test": [["test.jpg", "test_species"]],
        }
        refs, queries = self.experiment.validation_items(manifest)
        self.assertEqual(len(refs), 120)
        self.assertEqual(len(queries), 120)
        self.assertTrue(all(path.startswith("val_") for path, _ in refs + queries))
        self.assertFalse(set(path for path, _ in refs) & set(path for path, _ in queries))

    def test_synthetic_full_backbone_step(self):
        self.experiment.smoke_test(self.torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
