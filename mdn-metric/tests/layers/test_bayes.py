import itertools
import math
from unittest import TestCase, main

import numpy as np
import torch

from mdn_metric.layers.bayes import LinearBayes


class TestLinearBayes(TestCase):
    def test_kld(self):
        for log_sigma2 in np.linspace(-10, 10, 10):#[-10, 10]:
            log_sigma2 = torch.full([1], log_sigma2)
            mean = torch.randn([1])
            kld = LinearBayes(1, 1)._prior_kld(mean, log_sigma2)
            std = (0.5 * log_sigma2).exp()
            kld_gt = 0.5 * (-2 * std.log() + std.square() + mean.square() - 1)
            self.assertTrue(kld.allclose(kld_gt))

    def test_dropout_kld(self):
        for log_alpha in np.linspace(-10, 10, 10):#[-10, 10]:
            log_alpha = torch.full([1], log_alpha)
            kld = LinearBayes(1, 1)._dropout_prior_kld(torch.ones(1), log_alpha)
            mean = 1
            std = mean * (0.5 * log_alpha).exp()
            dist = torch.distributions.normal.Normal(mean, std)
            sample = dist.sample([1000000])
            log_c = -0.635  # Expected log|x| for x from N(0, 1), valid only for mean = 1.
            kld_gt = sample.abs().clip(min=1e-6).log().mean() - 0.5 * log_alpha - log_c
            self.assertTrue(kld.allclose(kld_gt, atol=1e-2))

    def test_forward(self):
        for b, vw, vb, local, dropout, ss, sparsify in itertools.product(
                [True, False], [True, False], [True, False], [True, False], [True, False], [None, 1, 3], [None, 0.5, 3.0]):
            config = {
                "variational_weight": vw,
                "variational_bias": vb,
                "dropout": dropout,
                "inference_sample_size": ss,
                "local_reparametrization": local,
                "inference_sparsify_ratio": sparsify
            }
            layer = LinearBayes(3, 5, bias=b, config=config)
            x = torch.randn(2, 2, 3)
            if (not vw and (not vb or not b)) or (ss is None):
                ss = 1
            self.assertEqual(layer(x).shape, (2, 2, 1, 5))
            layer.eval()
            self.assertEqual(layer(x).shape, (2, 2, ss, 5))


if __name__ == "__main__":
    main()
