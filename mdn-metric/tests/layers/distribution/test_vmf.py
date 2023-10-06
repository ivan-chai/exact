import itertools
import math
from unittest import TestCase, main

import random
import numpy as np
import scipy.special
import torch
from numbers import Number

from mdn_metric.layers.distribution.vmf import VMFDistribution, logiv


class TestVMFDistribution(TestCase):
    def test_logiv(self):
        """Test Logarithm of modified Bessel Function of the first kind."""
        logiv = VMFDistribution.LOGIV["default"]
        for order in [0, 0.2, 0.5, 0.8, 1, 1.2, math.pi, 2, 2.5, 10]:
            xs = np.linspace(1e-3, 3, 100)
            ys_gt = scipy.special.iv(order, xs)

            # Test values.
            ys = logiv(order, torch.tensor(xs)).exp().numpy()
            self.assertTrue(np.allclose(ys, ys_gt, atol=1e-6, rtol=0))

            # Test derivative.
            delta = 1e-10
            d_gt = (np.log(scipy.special.iv(order, xs + delta)) - np.log(scipy.special.iv(order, xs))) / delta
            xs_tensor = torch.from_numpy(xs)
            xs_tensor.requires_grad = True
            ys_tensor = logiv(order, xs_tensor)
            ys_tensor.backward(gradient=torch.ones_like(ys_tensor))
            d = xs_tensor.grad.numpy()
            self.assertTrue(np.allclose(d_gt, d, atol=1e-4))

    def test_logiv_scl(self):
        logiv = VMFDistribution.LOGIV["scl"]
        for order in [1, 1.2, math.pi, 2, 2.5, 10]:
            xs = np.linspace(1e-3, 3, 100)
            ys_gt = scipy.special.iv(order, xs)

            # Test values.
            ys = logiv(order, torch.tensor(xs), eps=0).exp().numpy()
            if not np.allclose(ys, ys_gt, atol=1e-4, rtol=0):
                print("SCL logiv mismatch {} for order {}.".format(
                    np.max(np.abs(ys - ys_gt)), order))

            # Test derivative.
            delta = 1e-10
            d_gt = (np.log(scipy.special.iv(order, xs + delta)) - np.log(scipy.special.iv(order, xs))) / delta
            xs_tensor = torch.from_numpy(xs)
            xs_tensor.requires_grad = True
            ys_tensor = logiv(order, xs_tensor, eps=0)
            ys_tensor.backward(gradient=torch.ones_like(ys_tensor))
            d = xs_tensor.grad.numpy()
            if not np.allclose(d, d_gt, atol=1e-4):
                print("SCL logiv derivative mismatch {} for order {}".format(
                    np.max(np.abs(d - d_gt)), order))

    def test_split_join(self):
        """Test split is inverse of join."""
        dims = [2, 3, 5, 1024, 4086]
        components = [1, 2, 5]
        for dim, c, k in itertools.product(dims, components, ["separate", "norm", 1, 2]):
            distribution = VMFDistribution(config={"dim": dim, "components": c, "k": k})
            with torch.no_grad():
                parameters = torch.randn(2, distribution.num_parameters)
                normalized = distribution.join_parameters(*distribution.split_parameters(parameters))
                splitted = distribution.split_parameters(normalized)
                joined = distribution.join_parameters(*splitted)
            self.assertTrue(np.allclose(joined.detach().numpy(), normalized.numpy()))

    def test_mean(self):
        """Test distribution mean."""
        # Test single component.
        distribution = VMFDistribution(config={"dim": 5, "components": 1})
        directions = torch.randn(3, 1, 5)
        k = 0.5 + torch.rand(3, 1, 1)
        norms_gt = (logiv(2.5, k).exp() / logiv(1.5, k).exp()).squeeze(-1).mean(1)  # (3).
        parameters = distribution.join_parameters(
            log_probs=torch.ones((3, 1)),
            means=directions,
            hidden_ik=distribution._parametrization.ipositive(1 / k)
        )
        means = distribution.mean(parameters)  # (3, 5).
        normalized_directions = torch.nn.functional.normalize(directions.squeeze(1), dim=1)
        normalized_means = torch.nn.functional.normalize(means, dim=1)
        self.assertTrue(normalized_directions.allclose(normalized_means))
        self.assertTrue(torch.linalg.norm(means, dim=1).allclose(norms_gt))

        # Test two identical components.
        distribution = VMFDistribution(config={"dim": 5, "components": 2})
        parameters = distribution.join_parameters(
            log_probs=torch.randn(3, 2),
            means=directions.repeat(1, 2, 1),
            hidden_ik=distribution._parametrization.ipositive(1 / k.repeat(1, 2, 1))
        )
        means = distribution.mean(parameters)  # (3, 5).
        normalized_directions = torch.nn.functional.normalize(directions.squeeze(1), dim=1)
        normalized_means = torch.nn.functional.normalize(means, dim=1)
        self.assertTrue(normalized_directions.allclose(normalized_means))
        self.assertTrue(torch.linalg.norm(means, dim=1).allclose(norms_gt))

    def test_normalizer(self):
        """Test batch norm."""
        batch_size = 5
        dims = [2, 3, 5, 1024, 4086]
        components = [1, 2, 5]
        for dim, c, k in itertools.product(dims, components, ["separate", "norm", 1, 2]):
            with torch.no_grad():
                distribution = VMFDistribution(config={"dim": dim, "components": c, "k": k})
                normalizer = distribution.make_normalizer()
                log_probs_gt = torch.randn(batch_size, c)
                means_gt = torch.randn(batch_size, c, dim)
                hidden_ik_gt = (torch.randn(batch_size, c, 1) if not isinstance(k, Number)
                                else distribution._parametrization.ipositive(torch.full((batch_size, c, 1), 1 / float(k))))
                parameters = distribution.join_parameters(
                    log_probs=log_probs_gt,
                    means=means_gt,
                    hidden_ik=hidden_ik_gt
                )
                normalized = normalizer(parameters) if normalizer is not None else parameters
                log_probs, means, hidden_ik = distribution.split_parameters(normalized)
                self.assertTrue(np.allclose(log_probs, log_probs_gt - torch.logsumexp(log_probs_gt, dim=-1, keepdim=True), atol=1e-6))
                self.assertTrue(np.allclose(np.linalg.norm(means, axis=-1), 1, atol=1e-6))
                self.assertTrue(np.allclose(hidden_ik, hidden_ik_gt, atol=1e-5))

    def test_prior_kld(self):
        """Test KL-divergence with uniform in simple cases."""
        distribution = VMFDistribution(config={"dim": 2, "max_logk": None})
        ik = torch.ones(2, 1, 1) * 1e6  # Low concentration.
        parameters = distribution.join_parameters(
            log_probs=torch.tensor([[1.],
                                    [1.]]).log(),
            means=torch.tensor([[[0., 1]], [[1., 0.]]]),
            hidden_ik=distribution._parametrization.ipositive(ik)
        )
        with torch.no_grad():
            kld = distribution.prior_kld(parameters).numpy()
        self.assertTrue(np.allclose(kld, 0, atol=1e-6))

        k = 1e6
        ik = torch.ones(2, 1, 1) / k  # High concentration.
        parameters = distribution.join_parameters(
            log_probs=torch.tensor([[1.],
                                    [1.]]).log(),
            means=torch.tensor([[[0., 1]], [[1., 0.]]]),
            hidden_ik=distribution._parametrization.ipositive(ik)
        )
        with torch.no_grad():
            kld = distribution.prior_kld(parameters).numpy()
        kld_gt = distribution._vmf_logc(k) + distribution._log_unit_area() + k
        self.assertTrue(np.allclose(kld, kld_gt, atol=1e-6))

    def test_logpdf(self):
        """Compare PDF values to ground truth."""
        for k in ["separate", "norm"]:
            distribution = VMFDistribution(config={"dim": 2, "components": 2, "k": k})
            parameters = distribution.join_parameters(
                log_probs=torch.tensor([[0.2, 0.8]]).log(),
                means=torch.tensor([[[1, 0], [0, 1]]]).float(),
                hidden_ik=distribution._parametrization.ipositive(1 / torch.tensor([[[1], [4]]]))
            )
            points = torch.tensor([[1.0, 1.0]])
            cosine = np.cos(np.pi / 4)
            pdf1 = 1 / 2 / np.pi / scipy.special.iv(0, 1) * np.exp(1 * cosine)
            pdf2 = 1 / 2 / np.pi / scipy.special.iv(0, 4) * np.exp(4 * cosine)
            logp_gt = np.log(0.2 * pdf1 + 0.8 * pdf2)
            with torch.no_grad():
                logp = distribution.logpdf(parameters, points).item()
            self.assertAlmostEqual(logp, logp_gt, places=4)

    def test_logpdf_extremal_k(self):
        """Test PDF for small and large k values.

        Double precision is used.
        """
        dims = [2, 3, 5, 1024, 4086]
        components = [1, 2, 5]

        # Test k -> inf.
        k = 1e8
        for dim, c in itertools.product(dims, components):
            distribution = VMFDistribution(config={"dim": dim, "components": c, "max_logk": 1000})
            means = torch.randn(2, c, dim)
            parameters = distribution.join_parameters(
                log_probs=(torch.rand(2, c) + 0.1).log(),  # Add 0.1, because zero probability can disable some spike.
                means=means,
                hidden_ik=distribution._parametrization.ipositive(1 / torch.full((2, c, 1), k))
            )
            points = torch.stack([
                means[0, 0] * np.random.random(),  # Collinear with mean.
                torch.randn(dim) - 0.5 * means[1, 0]  # Not collinear with mean.
            ])
            with torch.no_grad():
                collinear_logp, uncollinear_logp = distribution.logpdf(parameters.double(), points.double()).numpy()
            self.assertGreater(collinear_logp, 4)
            self.assertLess(uncollinear_logp, -10)  # Can fail sometimes for small dimensions.

        # Test k -> 0.
        k = 1e-8
        for dim, c in itertools.product(dims, components):
            distribution = VMFDistribution(config={"dim": dim, "components": c})
            means = torch.randn(2, c, dim)
            parameters = distribution.join_parameters(
                log_probs=(torch.rand(2, c) + 0.1).log(),  # Add 0.1, because zero probability can disable some spike.
                means=means,
                hidden_ik=distribution._parametrization.ipositive(1 / torch.full((2, c, 1), k))
            )
            points = torch.stack([
                means[0, 0] * np.random.random(),  # Collinear with mean.
                torch.randn(dim) - 0.5 * means[1, 0]  # Not collinear with mean.
            ])
            with torch.no_grad():
                collinear_logp, uncollinear_logp = distribution.logpdf(parameters, points).numpy()
            self.assertAlmostEqual(collinear_logp, uncollinear_logp, places=6)

    def test_logpdf_mixture_weights(self):
        """Check output for equal weights and one-hot weights.

        Double precision is used.
        """
        dims = [2, 3, 5, 1024, 4086]
        components = [1, 2, 5]

        # Test mixture equal weights.
        k = 1e8
        for dim, c in itertools.product(dims, components):
            distribution = VMFDistribution(config={"dim": dim, "components": c, "max_logk": None})
            # All distributions all the same.
            means = torch.randn(c, dim)
            means /= torch.linalg.norm(means, dim=-1, keepdim=True)
            parameters = distribution.join_parameters(
                log_probs=torch.full((c, c), np.log(1 / c)),
                means=torch.stack([means] * c),
                hidden_ik=distribution._parametrization.ipositive(1 / torch.full((c, c, 1), k))
            )
            points = means * (torch.rand(c, 1) + 0.1)  # Each point is collinear with corresponding mean.
            with torch.no_grad():
                logp = distribution.logpdf(parameters.double(), points.double()).numpy()
            # Assert all PDFs are the same (actually they approach infinity).
            self.assertTrue(np.allclose(logp, logp[0], atol=1e-2))

        # Test mixture one-hot weights.
        for dim, c in itertools.product(dims, components):
            k = np.random.random() * 2
            non_zero = random.randint(0, c - 1)
            means = torch.randn(1, c, dim)
            points = torch.randn(10, dim)

            # Multimodel distribution with one-hot weights.
            distribution = VMFDistribution(config={"dim": dim, "components": c})
            priors = torch.zeros(1, c) + 1e-10
            priors[0, non_zero] = 1
            parameters = distribution.join_parameters(
                log_probs=priors.log(),
                means=means,
                hidden_ik=distribution._parametrization.ipositive(1 / torch.full((1, c, 1), k))
            )
            with torch.no_grad():
                logp1 = distribution.logpdf(parameters, points).numpy()

            # Unimodal distribution equal to multimodal with one-hot weights.
            distribution = VMFDistribution(config={"dim": dim, "components": 1})
            parameters = distribution.join_parameters(
                log_probs=torch.zeros(1, 1),
                means=means[:, non_zero:non_zero + 1],
                hidden_ik=distribution._parametrization.ipositive(1 / torch.full((1, 1, 1), k))
            )
            with torch.no_grad():
                logp2 = distribution.logpdf(parameters, points).numpy()

            self.assertTrue(np.allclose(logp1, logp2, atol=1e-6))

    def test_logpdf_integral(self):
        """Test integral of vMF is equal to 1."""
        dims = [2, 3, 5]
        components = [1, 2, 5]
        for dim, c in itertools.product(dims, components):
            distribution = VMFDistribution(config={"dim": dim, "components": c})
            parameters = distribution.join_parameters(
                log_probs=torch.randn(1, c),
                means=torch.randn(1, c, dim),
                hidden_ik=distribution._parametrization.ipositive(1 / torch.rand(1, c, 1))
            )
            sample = torch.randn(1000, dim)
            with torch.no_grad():
                pdfs = distribution.logpdf(parameters, sample).exp()
            surface = 2 * np.pi ** (dim / 2) / scipy.special.gamma(dim / 2)
            integral = pdfs.mean().item() * surface
            self.assertAlmostEqual(integral, 1, delta=0.2)

    def test_mls(self):
        """Test MLS for simple case."""
        distribution = VMFDistribution(config={"dim": 2, "components": 2})
        # Test MLS shape.
        parameters1 = torch.randn(5, 1, 3, distribution.num_parameters)  # (5, 1, 3, P).
        parameters2 = torch.randn(1, 7, 3, distribution.num_parameters)  # (1, 7, 3, P).
        with torch.no_grad():
            result_shape = distribution.logmls(parameters1, parameters2).shape
        self.assertEqual(result_shape, (5, 7, 3))

        # Test MLS between similar distributions.
        parameters = distribution.join_parameters(
            log_probs=torch.tensor([[0.2, 0.8]]).log(),
            means=torch.tensor([[[1, 0], [0, 1]]]),
            hidden_ik=distribution._parametrization.ipositive(1 / torch.tensor([[[1], [4]]]))
        )
        mls11 = scipy.special.iv(0, 2 * 1) / 2 / np.pi / scipy.special.iv(0, 1) ** 2
        mls22 = scipy.special.iv(0, 2 * 4) / 2 / np.pi / scipy.special.iv(0, 4) ** 2
        mls12 = scipy.special.iv(0, np.sqrt(1 ** 2 + 4 ** 2)) / 2 / np.pi / scipy.special.iv(0, 1) / scipy.special.iv(0, 4)
        logmls_gt = np.log(0.04 * mls11 + 0.64 * mls22 + 0.32 * mls12)
        logmls = distribution.logmls(parameters, parameters).item()
        self.assertAlmostEqual(logmls, logmls_gt, places=4)

    def test_pdf_product(self):
        for d in [2, 3]:
            for c in [1, 2]:
                distribution = VMFDistribution(config={"components": c, "dim": d})
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
                self.assertTrue(logpdf.allclose(logpdf_gt, atol=1e-6))

    def test_estimate(self):
        for d in [2, 3, 4, 5]:
            distribution = VMFDistribution(config={"dim": d})
            mean_gt = torch.randn(distribution.dim)
            k_gt = torch.tensor([100.])
            parameters_gt = distribution.pack_parameters({"log_probs": torch.zeros(1), "mean": mean_gt, "k": k_gt})
            sample, _ = distribution.sample(parameters_gt, [100000])  # (100000, D).
            parameters = distribution.estimate(sample)  # (P).
            self.assertTrue(distribution.unpack_parameters(parameters)["mean"].allclose(
                            distribution.unpack_parameters(parameters_gt)["mean"], atol=1e-2))
            self.assertTrue(distribution.unpack_parameters(parameters)["k"].allclose(
                            distribution.unpack_parameters(parameters_gt)["k"], atol=5))

    def test_sampling(self):
        """Test MLS is equal to estimation by sampling."""
        distribution = VMFDistribution(config={"components": 2, "dim": 2})
        parameters = distribution.join_parameters(
            log_probs=torch.tensor([[0.25, 0.75]]).log(),
            means=torch.tensor([[[-2, 0], [2, 0]]]).float(),
            hidden_ik=distribution._parametrization.ipositive(torch.tensor([[[0.5], [1]]]))
        )
        with torch.no_grad():
            mls_gt = distribution.logmls(parameters, parameters).exp().item()
            sample, _ = distribution.sample(parameters, [100000])
            mls = distribution.logpdf(parameters, sample).exp().mean().item()
        self.assertAlmostEqual(mls, mls_gt, places=3)

    def test_make_mixture(self):
        distribution = VMFDistribution(config={"dim": 2, "components": 3})
        log_probs = torch.randn(5)
        components_log_probs = torch.randn(5, 3)
        components_log_probs -= components_log_probs.logsumexp(dim=-1, keepdim=True)
        means = torch.randn(5, 3, 2)
        k = torch.rand(5, 3, 1)
        parameters = distribution.pack_parameters({
            "log_probs": components_log_probs,
            "mean": means,
            "k": k
        })
        new_distribution, new_parameters = distribution.make_mixture(log_probs, parameters)
        unpacked = new_distribution.unpack_parameters(new_parameters)
        log_probs -= log_probs.logsumexp(dim=0)
        means = torch.nn.functional.normalize(means, dim=-1)
        self.assertTrue(torch.allclose(unpacked["log_probs"], (log_probs[:, None] + components_log_probs).flatten()))
        self.assertTrue(torch.allclose(unpacked["mean"], means.reshape(-1, 2)))
        self.assertTrue(torch.allclose(unpacked["k"], k.reshape(-1, 1)))


if __name__ == "__main__":
    main()
