import numpy as np
import torch
from unittest import TestCase, main

from mdn_metric.metrics.classification import InvertedAccuracyMetric


class TestInvertedAccuracyMetric(TestCase):
    def test_simple(self):
        value = self._compute(
            [[ 0.9, -0.8, -0.7],
             [ 2.3,  0.2, -0.1],
             [-5.0, -2.0,  0.0]],
            [0, 0, 2]
        )
        self.assertAlmostEqual(value, 1.0)

        value = self._compute(
            [[ 0.9, -0.8, -0.7],
             [ 2.3,  0.2, -0.1],
             [-5.0, -2.0,  0.0]],
            [1, 2, 0]
        )
        self.assertAlmostEqual(value, 0.0)

        value = self._compute(
            [[ 0.9, -0.8, -0.7],
             [ 2.3,  0.2, -0.1],
             [-5.0, -2.0,  0.0]],
            [1, 0, 1]
        )
        self.assertAlmostEqual(value, 2 / 3)

    def _compute(self, logits, labels):
        metric = InvertedAccuracyMetric()
        metric.update(torch.as_tensor(logits).float(), torch.as_tensor(labels))
        mean, std = metric.compute()
        return mean



if __name__ == "__main__":
    main()
