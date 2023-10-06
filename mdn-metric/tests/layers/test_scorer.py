from unittest import TestCase, main

import numpy as np
import torch

from mdn_metric.layers.distribution import GMMDistribution, VMFDistribution
from mdn_metric.layers.scorer import FeatureMatchingScorer


class TestFeatureMatchingScorer(TestCase):
    def test_simple(self):
        distribution = GMMDistribution(config={"components": 1, "dim": 1})
        parameters1 = distribution.join_parameters(
            torch.tensor([0]),
            torch.tensor([[1]]),
            distribution._parametrization.ipositive(torch.tensor([[2]]))
        )
        parameters2 = distribution.join_parameters(
            torch.tensor([0]),
            torch.tensor([[2]]),
            distribution._parametrization.ipositive(torch.tensor([[1]]))
        )
        # Product distribution is N(5/3, 2/3) with mode 5/3.
        ubm_probs = torch.tensor([0.2, 0.8])
        ubm_distributions = distribution.join_parameters(
            torch.tensor([[0], [0]]),  # (2, 1).
            torch.tensor([[[0]], [[1]]]),  # (2, 1, 1).
            distribution._parametrization.ipositive(torch.tensor([[[1]], [[2]]]))  # (2, 1, 1).
        )  # (2, P).
        prior = (distribution.logpdf(ubm_distributions, torch.tensor([[5/3], [5/3]])).exp() * ubm_probs).sum()
        fms_gt = distribution.logmls(parameters1, parameters2) - prior.log()
        scorer = FeatureMatchingScorer(distribution, config={"method": "point"})
        with scorer.set_ubm(*distribution.make_mixture(ubm_probs.log(), ubm_distributions)):
            fms = scorer(parameters1, parameters2)
        self.assertAlmostEqual(fms.item(), fms_gt.item(), places=4)

        # Test commutativity with high-dim, multi-component, and different distributions.
        for cls in [GMMDistribution, VMFDistribution]:
            for dim in [2, 4]:
                for c in [1, 2]:
                    distribution = cls(config={"components": c, "dim": dim})
                    parameters1 = torch.randn(5, 1, distribution.num_parameters)
                    parameters2 = torch.randn(1, 3, distribution.num_parameters)
                    scorer = FeatureMatchingScorer(distribution, config={"method": "point"})
                    with scorer.set_ubm(*distribution.make_mixture(torch.randn(4), torch.randn(4, distribution.num_parameters))):
                        scores1 = scorer(parameters1, parameters2)
                        scores2 = scorer(parameters2, parameters1)
                    self.assertTrue(scores1.allclose(scores2))


if __name__ == "__main__":
    main()
