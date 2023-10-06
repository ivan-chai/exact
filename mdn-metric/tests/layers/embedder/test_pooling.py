import math
from unittest import TestCase, main

import numpy as np
import torch

from mdn_metric.layers import DiracDistribution
from mdn_metric.layers.embedder.pooling import *


class TestFlattenPooling(TestCase):
    def test_simple(self):
        x = torch.tensor([
            [0, 1],
            [2, 3]
        ]).reshape(1, 1, 2, 2)  # BCHW.
        pooling = FlattenPooling(1, 2)
        y = pooling(x)  # (1, 4).
        self.assertTrue(y.allclose(x.reshape(1, 4)))
        x = torch.tensor([
            [[0, 0], [1, 1]],
            [[2, 2], [3, 3]]
        ])  # HWC.
        pooling = FlattenPooling(1, 2)
        y = pooling(x.permute(2, 0, 1).reshape(1, 2, 2, 2))  # (1, 4).
        self.assertTrue(y.allclose(x.reshape(1, 8)))


class TestEnsemblePooling(TestCase):
    def test_simple(self):
        x = torch.tensor([
            [0, 1],
            [2, 3]
        ]).reshape(1, 1, 2, 2)  # BCHW.
        pooling = EnsemblePooling(1)
        y = pooling(x)  # (1, 4, 1).
        self.assertTrue(y.allclose(x.reshape(1, 4, 1)))
        x = torch.tensor([
            [[0, 0], [1, 1]],
            [[2, 2], [3, 3]]
        ])  # HWC.
        y = pooling(x.permute(2, 0, 1).reshape(1, 2, 2, 2))  # (1, 4, 2).
        self.assertTrue(y.allclose(x.reshape(1, 4, 2)))


class TestDistributionPooling(TestCase):
    def test_simple(self):
        distribution = DiracDistribution(config={"dim": 2})
        x1 = torch.tensor([
            [[0, 1], [1, 2]],
            [[2, 3], [3, 4]]
        ]).permute(2, 0, 1)  # CHW.
        x2 = torch.tensor([
            [[0, 2], [1, 3]],
            [[2, 4], [3, 5]]
        ]).permute(2, 0, 1)  # CHW.
        x = torch.stack([x1, x2]).float()  # BCHW.
        means_gt = torch.tensor([
            [1.5, 2.5],
            [1.5, 3.5]
        ])
        pooling = DistributionPooling(2, distribution)
        y = pooling(x)  # (2, 2).
        self.assertTrue(y.allclose(means_gt))


if __name__ == "__main__":
    main()
