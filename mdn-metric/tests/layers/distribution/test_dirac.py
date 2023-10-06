import itertools
import math
from unittest import TestCase, main

import numpy as np
import torch

from mdn_metric.layers.distribution import DiracDistribution


class TestDiracDistribution(TestCase):
    def test_pack_unpack(self):
        """Test split is inverse of join."""
        dims = [2, 3, 5, 1024, 4086]
        components = [1, 2, 5]
        for dim, c in itertools.product(dims, components):
            distribution = DiracDistribution(config={"dim": dim, "components": c})
            with torch.no_grad():
                parameters = torch.randn(2, distribution.num_parameters)
                normalized = distribution.pack_parameters(distribution.unpack_parameters(parameters))
                splitted = distribution.unpack_parameters(normalized)
                joined = distribution.pack_parameters(splitted)
            self.assertTrue(np.allclose(joined.detach().numpy(), normalized.numpy()))

    def test_normalizer(self):
        """Test batch norm."""
        batch_size = 5
        dims = [2, 3, 5, 1024, 4086]
        components = [1, 2, 5]
        for dim, c in itertools.product(dims, components):
            with torch.no_grad():
                distribution = DiracDistribution(config={"dim": dim, "components": c})
                normalizer = distribution.make_normalizer().train()
                log_probs_gt = torch.randn(batch_size, c)
                means_gt = torch.randn(batch_size, c, dim)
                parameters = distribution.pack_parameters({
                    "log_probs": log_probs_gt,
                    "mean": means_gt
                })
                unpacked = distribution.unpack_parameters(normalizer(parameters))
                log_probs, means = unpacked["log_probs"], unpacked["mean"]
                self.assertTrue(np.allclose(log_probs, log_probs_gt - torch.logsumexp(log_probs_gt, dim=-1, keepdim=True), atol=1e-6))
                self.assertTrue(np.allclose(means.mean((0, 1)), 0, atol=1e-5))
                self.assertTrue(np.allclose(means.std((0, 1), unbiased=False), 1, atol=1e-2))  # Atol because of epsilon in batchnorm.

    def test_mean_sampling(self):
        """Test MLS is equal to estimation by sampling."""
        distribution = DiracDistribution(config={"dim": 2})
        parameters = torch.randn((1, 1, 2))
        with torch.no_grad():
            means = distribution.mean(parameters)
            sample, _ = distribution.sample(parameters, [50, 10])
            delta = (sample - means).abs().max()
        self.assertTrue(np.allclose(means, parameters, atol=1e-5))
        self.assertAlmostEqual(delta, 0)

        distribution = DiracDistribution(config={"dim": 2, "components": 2})
        probs = torch.tensor([0.3, 0.7])
        means = torch.tensor([[1, 0], [0, 1]])  # (2, 2).
        parameters = distribution.pack_parameters({
            "log_probs": probs.log(),
            "mean": means
        })
        with torch.no_grad():
            sample, components = distribution.sample(parameters[None], [100000])  # (N, 2), (N).
        self.assertTrue(np.allclose(means.take_along_dim(components[:, None], dim=0), sample))
        self.assertAlmostEqual(components.float().mean().item(), 0.7, places=2)

    def test_make_mixture(self):
        distribution = DiracDistribution(config={"dim": 2, "components": 3})
        log_probs = torch.randn(5)
        components_log_probs = torch.randn(5, 3)
        components_log_probs -= components_log_probs.logsumexp(dim=-1, keepdim=True)
        means = torch.randn(5, 3, 2)
        parameters = distribution.pack_parameters({
            "log_probs": components_log_probs,
            "mean": means
        })
        new_distribution, new_parameters = distribution.make_mixture(log_probs, parameters)
        unpacked = new_distribution.unpack_parameters(new_parameters)
        log_probs -= log_probs.logsumexp(dim=0)
        self.assertTrue(torch.allclose(unpacked["log_probs"], (log_probs[:, None] + components_log_probs).flatten()))
        self.assertTrue(torch.allclose(unpacked["mean"], means.reshape(-1, 2)))

    def test_spherical(self):
        dim = 16
        parameters = torch.randn((10, 1, dim))

        distribution = DiracDistribution(config={"dim": dim})
        norms = torch.linalg.norm(distribution.mean(parameters), dim=-1)
        self.assertFalse(norms.allclose(torch.ones_like(norms)))

        distribution = DiracDistribution(config={"dim": dim, "spherical": True})
        norms = torch.linalg.norm(distribution.mean(parameters), dim=-1)
        self.assertTrue(norms.allclose(torch.ones_like(norms)))

        distribution = DiracDistribution(config={"dim": dim, "spherical": True, "scale": 3.21})
        norms = torch.linalg.norm(distribution.mean(parameters), dim=-1)
        self.assertTrue(norms.allclose(torch.full_like(norms, 3.21)))

    def test_estimate(self):
        distribution = DiracDistribution(config={"dim": 1})
        sample = torch.tensor([
            [-2],
            [1]
        ]).float() # (2, 1).
        parameters_gt = torch.tensor([-0.5])
        parameters = distribution.estimate(sample)  # (1)
        mean = distribution.unpack_parameters(parameters)["mean"]  # (1).
        self.assertTrue(parameters.allclose(parameters_gt))


if __name__ == "__main__":
    main()
