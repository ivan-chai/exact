#!/usr/bin/env python3
from unittest import TestCase, main

import numpy as np
import torch
from torchvision import transforms

from mdn_metric.dataset import ImageTransform
from mdn_metric.model import Model


class TestModel(TestCase):
    def test_forward(self):
        """Run forward for default model."""
        model = Model(num_classes=100, is_multilabel=False)
        image_transform = ImageTransform()
        transform = transforms.Compose([transforms.ToTensor(), image_transform])
        sample_input = (np.random.random((4, image_transform.image_size, image_transform.image_size, 3)) * 255).astype(np.uint8)
        batch = torch.stack(list(map(transform, sample_input)))
        logits = model(batch)["logits"]
        self.assertEqual(logits.shape, (4, 1, 100))

    def test_scoring(self):
        """Run scoring for default model."""
        model = Model(num_classes=100, is_multilabel=False)
        image_transform = ImageTransform()
        transform = transforms.Compose([transforms.ToTensor(), image_transform])
        sample_input1 = (np.random.random((4, image_transform.image_size, image_transform.image_size, 3)) * 255).astype(np.uint8)
        sample_input2 = (np.random.random((4, image_transform.image_size, image_transform.image_size, 3)) * 255).astype(np.uint8)
        batch1 = torch.stack(list(map(transform, sample_input1)))
        batch2 = torch.stack(list(map(transform, sample_input2)))
        embeddings1 = model.embedder(batch1)  # (B, N, D).
        embeddings2 = model.embedder(batch2)  # (B, N, D).
        scores = model.scorer(embeddings1, embeddings2)
        self.assertEqual(scores.shape, (4, 1))


if __name__ == "__main__":
    main()
