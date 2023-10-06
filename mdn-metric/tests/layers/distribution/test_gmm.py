import itertools
import math
from unittest import TestCase, main

import numpy as np
import torch

from mdn_metric.layers.distribution import GMMDistribution


class TestGMMDistribution(TestCase):
    def test_split_join(self):
        """Test split is inverse of join."""
        dims = [2, 3, 5, 1024, 4086]
        components = [1, 2, 5]
        for dim, c in itertools.product(dims, components):
            distribution = GMMDistribution(config={"dim": dim, "components": c, "covariance": "spherical", "max_logivar": 5})
            with torch.no_grad():
                parameters = torch.randn(2, distribution.num_parameters)
                normalized = distribution.join_parameters(*distribution.split_parameters(parameters))
                splitted = distribution.split_parameters(normalized)
                joined = distribution.join_parameters(*splitted)
            self.assertTrue(np.allclose(joined.detach().numpy(), normalized.numpy()))

    def test_normalizer(self):
        """Test batch norm."""
        batch_size = 5
        dims = [2, 3, 5, 1024, 4086]
        components = [1, 2, 5]
        for dim, c in itertools.product(dims, components):
            with torch.no_grad():
                distribution = GMMDistribution(config={"dim": dim, "components": c, "covariance": "spherical", "max_logivar": 1e3})
                normalizer = distribution.make_normalizer().train()
                log_probs_gt = torch.randn(batch_size, c)
                means_gt = torch.randn(batch_size, c, dim)
                hidden_vars_gt = torch.randn(batch_size, c, 1)
                parameters = distribution.join_parameters(
                    log_probs=log_probs_gt,
                    means=means_gt,
                    hidden_vars=hidden_vars_gt
                )
                log_probs, means, hidden_vars = distribution.split_parameters(normalizer(parameters))
                self.assertTrue(np.allclose(log_probs, log_probs_gt - torch.logsumexp(log_probs_gt, dim=-1, keepdim=True), atol=1e-6))
                self.assertTrue(np.allclose(hidden_vars, hidden_vars_gt, atol=1e-6))
                self.assertTrue(np.allclose(means.mean((0, 1)), 0, atol=1e-5))
                self.assertTrue(np.allclose(means.std((0, 1), unbiased=False), 1, atol=1e-2))  # Atol because of epsilon in batchnorm.

    def test_prior_kld(self):
        """Test KL-divergence with standard in simple cases."""
        # KLD between standards is zero.
        for covariance in ["spherical", "diagonal"]:
            for delta in [0, 0.1]:
                distribution = GMMDistribution(config={"dim": 2, "components": 1, "covariance": covariance})
                vars = torch.ones(2, 1, 1) if covariance == "spherical" else torch.ones(2, 1, 2)
                parameters = distribution.join_parameters(
                    log_probs=torch.tensor([[1.],
                                            [1.]]).log(),
                    means=torch.tensor([[[0., delta]], [[0., 0.]]]),
                    hidden_vars=distribution._parametrization.ipositive(vars)
                )
                with torch.no_grad():
                    kld = distribution.prior_kld(parameters).numpy()
                if delta == 0:
                    self.assertTrue(np.allclose(kld, 0, atol=1e-6))
                else:
                    self.assertFalse(np.allclose(kld, 0, atol=1e-6))

    def test_logpdf(self):
        """Test density estimation in simple cases."""
        distribution = GMMDistribution(config={"dim": 2, "components": 1, "covariance": "spherical"})
        parameters = distribution.join_parameters(
            log_probs=torch.tensor([[1.],
                                    [1.]]).log(),
            means=torch.tensor([[[0., 0.]], [[1., 0.]]]),
            hidden_vars=distribution._parametrization.ipositive(torch.tensor([[[1.]], [[0.5]]]))
        )
        points = torch.tensor([
            [0, 0],
            [0, 1]
        ]).float()
        with torch.no_grad():
            logp = distribution.logpdf(parameters, points).numpy()
        logp_gt = np.array([
            -math.log(2 * math.pi * 1),
            -math.log(2 * math.pi * math.sqrt(0.5 * 0.5)) - 2
        ])
        self.assertTrue(np.allclose(logp, logp_gt, atol=1e-6, rtol=0))

        distribution = GMMDistribution(config={"dim": 2, "components": 2, "covariance": "diagonal"})
        parameters = distribution.join_parameters(
            log_probs=torch.tensor([[0.8, 0.2],
                                    [0.5, 0.5]]).log(),
            means=torch.tensor([[[0., 1.], [-1., 0.]],
                                [[1., 1.], [0., 0.]]]),
            hidden_vars=distribution._parametrization.ipositive(
                torch.tensor([[[1., 2.], [0.5, 0.5]],
                              [[1., 0.5], [0.5, 2.]]])
            )
        )
        points = torch.tensor([
            [0, 0],
            [1, 1]
        ]).float()
        with torch.no_grad():
            logp = distribution.logpdf(parameters, points).numpy()
        logp_gt = np.array([
            math.log(0.8 / 2 / math.pi / math.sqrt(1. * 2.) * math.exp(-0.5 * 0.5) +
                     0.2 / 2 / math.pi / math.sqrt(0.5 * 0.5) * math.exp(-0.5 * 2)),
            math.log(0.5 / 2 / math.pi / math.sqrt(1. * 0.5) * math.exp(0) +
                     0.5 / 2 / math.pi / math.sqrt(0.5 * 2.) * math.exp(-0.5 * 2.5))
        ])
        self.assertTrue(np.allclose(logp, logp_gt, atol=1e-6, rtol=0))

    def test_logpdf_integral(self):
        """Test integral of GMM is equal to 1."""
        dims = [2, 3, 5]
        components = [1, 2, 5]
        for dim, c in itertools.product(dims, components):
            distribution = GMMDistribution(config={"dim": dim, "components": c, "covariance": "diagonal"})
            parameters = torch.randn(1, distribution.num_parameters).double()
            scale = 10
            sample = scale * torch.rand(10000, dim).double() - scale / 2
            with torch.no_grad():
                pdfs = distribution.logpdf(parameters, sample).exp()
            volume = scale ** dim
            integral = pdfs.sum().item() / len(sample) * volume
            self.assertAlmostEqual(integral, 1, delta=0.5)

    def test_mls_shape(self):
        """Test broadcasting."""
        distribution = GMMDistribution(config={"dim": 2, "components": 2})
        # Test MLS shape.
        parameters1 = torch.randn(5, 1, 3, distribution.num_parameters)  # (5, 1, 3, P).
        parameters2 = torch.randn(1, 7, 3, distribution.num_parameters)  # (1, 7, 3, P).
        with torch.no_grad():
            result_shape = distribution.logmls(parameters1, parameters2).shape
        self.assertEqual(result_shape, (5, 7, 3))

    def test_mls_same(self):
        """Test MLS for GMM comparison with identical GMM."""
        distribution = GMMDistribution(config={"dim": 2, "components": 1, "covariance": "spherical"})
        parameters = distribution.join_parameters(
            log_probs=torch.tensor([[1.],
                                    [1.]]).log(),
            means=torch.tensor([[[0., 0.]], [[1., 0.]]]),
            hidden_vars=distribution._parametrization.ipositive(torch.tensor([[[1.]], [[0.5]]]))
        )
        with torch.no_grad():
            logmls = distribution.logmls(parameters, parameters).numpy()
        logmls_gt = np.array([
            -math.log(2 * math.pi) - math.log(2 * 1),
            -math.log(2 * math.pi) - math.log(2 * 0.5)
        ])
        self.assertTrue(np.allclose(logmls, logmls_gt, atol=1e-6, rtol=0))

        distribution = GMMDistribution(config={"dim": 2, "components": 1, "covariance": "diagonal"})
        parameters = distribution.join_parameters(
            log_probs=torch.tensor([[1]]).log(),
            means=torch.tensor([[[0., 1.]]]),
            hidden_vars=distribution._parametrization.ipositive(torch.tensor([[[1., 2.]]]))
        )
        with torch.no_grad():
            logmls = distribution.logmls(parameters, parameters).numpy()
        logmls_gt = np.array([
            -math.log(2 * math.pi) - 0.5 * math.log(2 * 1.0) - 0.5 * math.log(2 * 2.0)
        ])
        self.assertTrue(np.allclose(logmls, logmls_gt, atol=1e-6, rtol=0))

        distribution = GMMDistribution(config={"dim": 2, "components": 2, "covariance": "diagonal"})
        parameters = distribution.join_parameters(
            log_probs=torch.tensor([[0.8, 0.2]]).log(),
            means=torch.tensor([[[0., 1.], [-1., 0.]]]),
            hidden_vars=distribution._parametrization.ipositive(torch.tensor([[[1., 2.], [0.5, 0.5]]]))
        )
        with torch.no_grad():
            logmls = distribution.logmls(parameters, parameters).numpy()
        gt11 = -math.log(2 * math.pi) - 0.5 * math.log(2 * 1.0) - 0.5 * math.log(2 * 2.0)
        gt22 = -math.log(2 * math.pi) - 0.5 * math.log(2 * 0.5) - 0.5 * math.log(2 * 0.5)
        gt12 = -math.log(2 * math.pi) - 0.5 / 1.5 - 0.5 / 2.5 - 0.5 * math.log(1.5) - 0.5 * math.log(2.5)
        mls_gt = 0.64 * math.exp(gt11) + 0.32 * math.exp(gt12) + 0.04 * math.exp(gt22)
        logmls_gt = np.array([math.log(mls_gt)])
        self.assertTrue(np.allclose(logmls, logmls_gt, atol=1e-6, rtol=0))

    def test_mls_delta(self):
        """Test MLS for GMM comparison with different GMMs."""
        distribution = GMMDistribution(config={"dim": 4, "components": 2, "covariance": "diagonal", "max_logivar": None})
        log_probs = torch.randn(1, 2)
        means = torch.randn(1, 2, 4)
        hidden_vars = torch.randn(1, 2, 4)
        parameters1 = distribution.join_parameters(log_probs, means, hidden_vars)
        logmls_same = distribution.logmls(parameters1, parameters1).item()
        deltas = torch.arange(-5, 5, 0.1).numpy()
        for delta in deltas:
            parameters2 = distribution.join_parameters(log_probs + delta, means, hidden_vars)
            logmls = distribution.logmls(parameters1, parameters2)[0].item()
            self.assertAlmostEqual(logmls, logmls_same, places=6)

            parameters2 = distribution.join_parameters(log_probs, means + delta, hidden_vars)
            logmls = distribution.logmls(parameters1, parameters2)[0].item()
            if abs(delta) < 1e-6:
                self.assertAlmostEqual(logmls, logmls_same)
            else:
                self.assertLess(logmls, logmls_same)

            parameters2 = distribution.join_parameters(log_probs, means, hidden_vars + delta)
            logmls = distribution.logmls(parameters1, parameters2)[0].item()
            if abs(delta) < 1e-6:
                self.assertAlmostEqual(logmls, logmls_same)
            elif delta > 0:
                self.assertLess(logmls, logmls_same)
            else:
                self.assertGreater(logmls, logmls_same)

    def test_pdf_product(self):
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
        _, parameters = distribution.pdf_product(parameters1, parameters2)
        parameters_gt = distribution.join_parameters(
            torch.tensor([0]),
            torch.tensor([[5 / 3]]),
            distribution._parametrization.ipositive(torch.tensor([[2 / 3]]))
        )
        self.assertTrue(parameters.allclose(parameters_gt))
        for d in [1, 2]:
            for c in [1, 2]:
                for covariance in ["diagonal", "spherical"]:
                    distribution = GMMDistribution(config={"components": c, "dim": d, "covariance": covariance})
                    parameters1 = torch.randn(1, 3, distribution.num_parameters)
                    parameters2 = torch.randn(1, 3, distribution.num_parameters)
                    prod_distribution, prod_parameters = distribution.pdf_product(parameters1, parameters2)
                    points = torch.randn(2, 3, distribution.dim)
                    logpdf_gt = distribution.logpdf(parameters1, points) + distribution.logpdf(parameters2, points)
                    logpdf = prod_distribution.logpdf(prod_parameters, points)
                    # Product is equal up to normalization constant, difference removes normalization.
                    points0 = torch.zeros(distribution.dim)
                    logpdf0_gt = distribution.logpdf(parameters1, points0) + distribution.logpdf(parameters2, points0)
                    logpdf_gt = logpdf_gt - logpdf0_gt
                    logpdf0 = prod_distribution.logpdf(prod_parameters, points0)
                    logpdf = logpdf - logpdf0
                    self.assertTrue(logpdf.allclose(logpdf_gt, atol=1e-5))

    def test_estimate(self):
        for d in [1, 2]:
            for covariance in ["diagonal", "spherical"]:
                distribution = GMMDistribution(config={"dim": d, "covariance": covariance})
                parameters_gt = torch.randn(distribution.num_parameters)  # (P).
                sample, _ = distribution.sample(parameters_gt, [100000])  # (100000, D).
                parameters = distribution.estimate(sample)  # (P).
                self.assertTrue(parameters.allclose(parameters_gt, atol=1e-1))

    def test_sampling(self):
        """Test MLS is equal to estimation by sampling."""
        distribution = GMMDistribution(config={"components": 2, "dim": 2, "covariance": "diagonal"})
        parameters = distribution.join_parameters(
            log_probs=torch.tensor([[0.25, 0.75]]).log(),
            means=torch.tensor([[[-2, 0], [2, 0]]]),
            hidden_vars=distribution._parametrization.ipositive(torch.tensor([[[0.5, 1], [1, 0.5]]]).square())
        )
        with torch.no_grad():
            mls_gt = distribution.logmls(parameters, parameters).exp().item()
            sample, _ = distribution.sample(parameters, [100000])
            mls = distribution.logpdf(parameters, sample).exp().mean().item()
        self.assertAlmostEqual(mls, mls_gt, places=3)

    def test_make_mixture(self):
        distribution = GMMDistribution(config={"dim": 2, "components": 3})
        log_probs = torch.randn(5)
        components_log_probs = torch.randn(5, 3)
        components_log_probs -= components_log_probs.logsumexp(dim=-1, keepdim=True)
        means = torch.randn(5, 3, 2)
        covariance = torch.rand(5, 3, 1)
        parameters = distribution.pack_parameters({
            "log_probs": components_log_probs,
            "mean": means,
            "covariance": covariance
        })
        new_distribution, new_parameters = distribution.make_mixture(log_probs, parameters)
        unpacked = new_distribution.unpack_parameters(new_parameters)
        log_probs -= log_probs.logsumexp(dim=0)
        self.assertTrue(torch.allclose(unpacked["log_probs"], (log_probs[:, None] + components_log_probs).flatten()))
        self.assertTrue(torch.allclose(unpacked["mean"], means.reshape(-1, 2)))
        self.assertTrue(torch.allclose(unpacked["covariance"], covariance.reshape(-1, 1)))


if __name__ == "__main__":
    main()
