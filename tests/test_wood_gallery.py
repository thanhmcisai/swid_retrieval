"""Synthetic checks that do not require public images or pretrained weights."""

import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, "PyTorch unavailable")
class WoodGalleryTest(unittest.TestCase):
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
