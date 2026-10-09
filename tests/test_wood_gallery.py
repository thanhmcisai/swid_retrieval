"""Synthetic checks that do not require public images or pretrained weights."""

import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, "PyTorch unavailable")
class WoodGalleryTest(unittest.TestCase):
    def test_projection_stays_finite_inside_mixed_precision(self):
        from swid_retrieval.gallery_method import GalleryEncoder
        encoder = GalleryEncoder(torch.nn.Identity(), feature_dim=2, embedding_dim=2)
        torch.nn.init.zeros_(encoder.projection[1].weight)
        torch.nn.init.zeros_(encoder.projection[1].bias)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            embedding = encoder.project(torch.zeros(2, 2))
        self.assertEqual(embedding.dtype, torch.float32)
        self.assertTrue(torch.isfinite(embedding).all())

    def test_nearest_scorer_matches_exact_image_retrieval(self):
        from swid_retrieval.gallery_method import GalleryScorer
        references = torch.nn.functional.normalize(torch.tensor([
            [1., 0.], [0.8, 0.2], [0., 1.], [-1., 0.]]), dim=1)
        labels = torch.tensor([5, 5, 2, 9])
        queries = torch.nn.functional.normalize(torch.tensor([
            [1., 0.], [0., 1.], [-1., 0.]]), dim=1)
        scorer = GalleryScorer(top_m=1, mode="nearest")
        scores, classes = scorer(queries, references, labels)
        cached, cached_classes = scorer(queries, references, labels,
                                        similarity=queries @ references.T)
        expected = torch.stack([
            (queries @ references[labels == cls].T).amax(dim=1) / scorer.temperature
            for cls in classes], dim=1)
        self.assertTrue(torch.allclose(scores, expected))
        self.assertTrue(torch.allclose(cached, scores))
        self.assertTrue(torch.equal(cached_classes, classes))
        self.assertEqual(classes[scores.argmax(dim=1)].tolist(), [5, 2, 9])
        self.assertFalse(any(param.requires_grad for param in scorer.parameters()))

    def test_metric_retrieval_uses_only_gallery_ce_and_backpropagates(self):
        from swid_retrieval.gallery_method import GalleryScorer, episode_objective
        vectors = torch.nn.functional.normalize(torch.randn(12, 16), dim=1)
        vectors.requires_grad_()
        scorer = GalleryScorer(mode="nearest")
        loss, parts = episode_objective(vectors, scorer, 4, 2, 1,
                                        stability_weight=0.0,
                                        use_variable_gallery=False,
                                        pseudo_ood_weight=0.0)
        reference_labels = torch.arange(4).repeat_interleave(2)
        scores, _ = scorer(vectors[8:], vectors[:8], reference_labels)
        expected = torch.nn.functional.cross_entropy(scores, torch.arange(4))
        self.assertTrue(torch.allclose(loss, expected))
        self.assertFalse(parts["top_m_active"])
        self.assertNotIn("old_ce", parts)
        self.assertGreaterEqual(parts["train_episode_r1"], 0.0)
        self.assertLessEqual(parts["train_episode_r1"], 1.0)
        loss.backward()
        self.assertGreater(float(vectors.grad.abs().sum()), 0)

    def test_wood_backbone_shapes_and_gradient(self):
        from swid_retrieval.wood_encoder import WoodPatternNet
        from swid_retrieval.gallery_method import GalleryEncoder
        for attention, multiscale in ((True, True), (False, True), (True, False)):
            backbone = WoodPatternNet(use_attention=attention, use_multiscale=multiscale)
            encoder = GalleryEncoder(backbone, backbone.num_features, 512)
            output = encoder(torch.randn(2, 3, 64, 64))
            self.assertEqual(tuple(output.shape), (2, 512))
            output[:, 0].sum().backward()
            self.assertIsNotNone(backbone.stem[0].weight.grad)
            self.assertTrue(torch.isfinite(backbone.stem[0].weight.grad).all())
            self.assertGreater(float(backbone.stem[0].weight.grad.abs().sum()), 0)

    def test_local_tokens_and_species_scores_backpropagate(self):
        from swid_retrieval.wood_encoder import WoodPatternNet
        from swid_retrieval.gallery_method import GalleryEncoder, local_species_scores
        backbone = WoodPatternNet()
        encoder = GalleryEncoder(backbone, backbone.num_features, 32, local_dim=16)
        global_emb, tokens = encoder.forward_with_tokens(torch.randn(3, 3, 64, 64))
        self.assertEqual(tuple(global_emb.shape), (3, 32))
        self.assertEqual(tuple(tokens.shape), (3, 12, 16))
        self.assertTrue(torch.allclose(tokens.norm(dim=-1), torch.ones(3, 12), atol=1e-5))
        scores, classes = local_species_scores(tokens[2:], tokens[:2],
                                               torch.tensor([3, 8]), 0.07)
        self.assertEqual(classes.tolist(), [3, 8])
        torch.nn.functional.cross_entropy(scores, torch.tensor([0])).backward()
        self.assertGreater(float(encoder.local_projection[1].weight.grad.abs().sum()), 0)
        self.assertGreater(float(backbone.stem[0].weight.grad.abs().sum()), 0)

    def test_memory_triggers_top_m_and_excludes_active_species(self):
        from swid_retrieval.gallery_method import (GalleryMemory, GalleryScorer,
                                                   episode_objective)
        memory = GalleryMemory(capacity=8, min_classes=4)
        vectors = torch.nn.functional.normalize(torch.randn(8, 16), dim=1)
        memory.add(vectors, torch.arange(8))
        background = memory.distractors(torch.tensor([0, 0, 1, 1]), 2, "cpu")
        self.assertEqual(len(background[0]), 6)
        self.assertEqual(background[1].tolist(), list(range(2, 8)))
        current = torch.nn.functional.normalize(torch.randn(6, 16), dim=1)
        current.requires_grad_()
        scorer = GalleryScorer(top_m=3)
        loss, details = episode_objective(current, scorer, 2, 2, 1,
                                          background=background)
        loss.backward()
        self.assertTrue(details["top_m_active"])
        self.assertEqual(details["gallery_species"], 8)
        self.assertEqual(details["gallery_images"], 10)
        self.assertIsNotNone(current.grad)
        self.assertFalse(background[0].requires_grad)

    def test_memory_checkpoint_preserves_fifo(self):
        from swid_retrieval.gallery_method import GalleryMemory
        memory = GalleryMemory(3, 2)
        memory.add(torch.randn(3, 4), torch.tensor([1, 2, 3]))
        memory.add(torch.randn(1, 4), torch.tensor([1]))
        self.assertEqual(list(memory.entries), [2, 3, 1])
        restored = GalleryMemory(3, 2)
        restored.load_state_dict(memory.state_dict())
        self.assertEqual(list(restored.entries), [2, 3, 1])
        for key in memory.entries:
            self.assertTrue(torch.equal(memory.entries[key], restored.entries[key]))

    def test_supcon_uses_same_detached_memory_negatives(self):
        from swid_retrieval.gallery_method import supervised_contrastive_loss
        embeddings = torch.nn.functional.normalize(torch.randn(6, 8), dim=1)
        embeddings.requires_grad_()
        labels = torch.tensor([0, 0, 0, 1, 1, 1])
        background = torch.nn.functional.normalize(torch.randn(5, 8), dim=1)
        plain = supervised_contrastive_loss(embeddings, labels)
        with_memory = supervised_contrastive_loss(embeddings, labels,
                                                  background=background)
        self.assertGreater(float(with_memory), float(plain))
        with_memory.backward()
        self.assertGreater(float(embeddings.grad.abs().sum()), 0)


if __name__ == "__main__":
    unittest.main()
