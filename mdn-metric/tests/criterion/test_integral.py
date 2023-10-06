#!/usr/bin/env python3
import time
from unittest import TestCase, main

import numpy as np
import torch

from mdn_metric.criterion.integral import genz_integral, mc_integral
from mdn_metric.criterion.integral_pp import get_cholesky, conditional_distribution_pp, conditional_distribution_grads_pp
from mdn_metric.criterion.integral_pp import integral_pp, genz_integral_pp_impl, mc_integral_pp_impl


class TestIntegral(TestCase):
    def test_cholesky(self):
        """Test Cholesky decomposition of EXACT point prediction (PP) matrix."""
        dim = 4
        m = torch.eye(dim) + 1
        diag, alt = get_cholesky(dim)
        diag = diag.numpy()
        alt = alt.numpy()
        L = torch.linalg.cholesky(m).numpy()
        for i in range(dim):
            for j in range(i):
                self.assertAlmostEqual(L[i, j], alt[j])
            self.assertAlmostEqual(L[i, i], diag[i])

        a = 3.1
        b = 0.2
        m = (a - b) * torch.eye(dim) + b
        diag, alt = get_cholesky(dim, a, b)
        diag = diag.numpy()
        alt = alt.numpy()
        L = torch.linalg.cholesky(m).numpy()
        for i in range(dim):
            for j in range(i):
                self.assertAlmostEqual(L[i, j], alt[j], places=5)
            self.assertAlmostEqual(L[i, i], diag[i], places=5)

    def test_conditional_distribution_pp(self):
        mean = torch.tensor([4, 2, 0, 3]).double()  # (D).
        points = torch.tensor([0, 1]).double()  # (CD).

        cond_mean, cond_diag, cond_alt = conditional_distribution_pp(mean[:2], mean[2:], points, 1, 0)
        self.assertAlmostEqual(cond_mean.shape, (2,))
        self.assertAlmostEqual(cond_mean[0].item(), 4)
        self.assertAlmostEqual(cond_mean[1].item(), 2)
        self.assertAlmostEqual(cond_diag, 1)
        self.assertAlmostEqual(cond_alt, 0)

        cond_mean, cond_diag, cond_alt = conditional_distribution_pp(mean[:2], mean[2:], points)
        self.assertAlmostEqual(cond_mean.shape, (2,))
        self.assertAlmostEqual(cond_mean[0].item(), 4 + 1 / 3 * (0 - 2))
        self.assertAlmostEqual(cond_mean[1].item(), 2 + 1 / 3 * (0 - 2))
        self.assertAlmostEqual(cond_diag, 2 - 2 / 3)
        self.assertAlmostEqual(cond_alt, 1 - 2 / 3)

        cond_mean_grad, cond_diag_grad, cond_alt_grad = conditional_distribution_grads_pp(mean)  # (D, D - 1), scalar, scalar.
        cond_mean_grad = cond_mean_grad[-1]  # (D - 1).
        cond_mean, cond_diag, cond_alt = conditional_distribution_pp(mean[:3], mean[3:], torch.zeros_like(mean[3:]))
        self.assertTrue(cond_mean_grad.allclose(cond_mean))

        cond_mean_grad, cond_diag_grad, cond_alt_grad = conditional_distribution_grads_pp(mean, 5, 3)  # (D, D - 1), scalar, scalar.
        cond_mean_grad = cond_mean_grad[-1]  # (D - 1).
        cond_mean, cond_diag, cond_alt = conditional_distribution_pp(mean[:3], mean[3:], torch.zeros_like(mean[3:]), 5, 3)
        self.assertTrue(cond_mean_grad.allclose(cond_mean))

        cond_mean_grad, cond_diag_grad, cond_alt_grad = conditional_distribution_grads_pp(mean, 5, 3)  # (D, D - 1), scalar, scalar.
        cond_mean_grad = cond_mean_grad[0]  # (D - 1).
        cond_mean, cond_diag, cond_alt = conditional_distribution_pp(mean.flip(0)[:3], mean.flip(0)[3:], torch.zeros_like(mean.flip(0)[3:]), 5, 3)
        self.assertTrue(cond_mean_grad.allclose(cond_mean.flip(0)))

    def test_genz(self):
        x = torch.tensor([1, 4, 2], dtype=torch.float, requires_grad=True)
        a = torch.tensor([
            [1, 3 / 5, 1 / 3],
            [3 / 5, 1, 11 / 15],
            [1 / 3, 11 / 15, 1]
        ], dtype=torch.float, requires_grad=True)
        values = [genz_integral(x, a, n=1).item() for _ in range(1000)]
        self.assertAlmostEqual(np.mean(values), 0.82798, delta=1e-2)
        self.assertAlmostEqual(np.var(values), 0.000066, delta=1e-5)

        a = torch.eye(len(x)) + 1
        values = [genz_integral(x, a, n=100).item() for _ in range(10)]
        values_pp = [genz_integral_pp_impl(x[None], n=100)[0].item() for _ in range(10)]
        self.assertAlmostEqual(np.mean(values), np.mean(values_pp), delta=1e-2)

        values = genz_integral(x, a, n=1000, mean_grads=True).numpy()
        values_pp = genz_integral_pp_impl(x[None], n=1000)[1][0].detach().numpy()
        self.assertTrue(np.allclose(values, values_pp, rtol=0.2, atol=1e-4))

    def test_mc(self):
        x = torch.tensor([1, 4, 2], dtype=torch.float, requires_grad=True)
        a = torch.tensor([
            [1, 3 / 5, 1 / 3],
            [3 / 5, 1, 11 / 15],
            [1 / 3, 11 / 15, 1]
        ], dtype=torch.float, requires_grad=True)
        values = [mc_integral(x, a, n=1).item() for _ in range(10000)]
        self.assertAlmostEqual(np.mean(values), 0.82798, delta=1e-2)
        self.assertAlmostEqual(np.var(values), 0.142, delta=1e-2)

        a = torch.eye(len(x)) + 1
        values = genz_integral(x, a, n=1000).item()
        values_pp = mc_integral_pp_impl(x[None], n=10000)[0].item()
        self.assertAlmostEqual(values, values_pp, delta=0.02)

        values = genz_integral(x, a, n=1000, mean_grads=True)
        values_pp = mc_integral_pp_impl(x[None], n=100000)[1][0]
        self.assertLess(self._delta_norm(values_pp, values), 0.05)

        x = torch.tensor([1, 4, 2, 5, -3], dtype=torch.float, requires_grad=True)
        a = torch.eye(len(x)) + 1
        values = [genz_integral(x, a, n=100).item() for _ in range(10)]
        values_pp = [mc_integral_pp_impl(x[None], n=100)[0].item() for _ in range(100)]
        self.assertAlmostEqual(np.mean(values), np.mean(values_pp), delta=1e-2)

        values = genz_integral(x, a, n=1000, mean_grads=True)
        values_pp = mc_integral_pp_impl(x[None], n=100000)[1][0]
        self.assertLess(self._delta_norm(values_pp, values), 0.05)

    def test_truncated(self):
        x = torch.tensor([[0.5, 1], [-1, 0.2]], dtype=torch.float, requires_grad=True)  # (2, 2).
        value_gt, grads_gt = integral_pp(x, n=1000)
        value, grads = integral_pp(x, n=1000, robust_dims=1, truncated=True)
        self.assertFalse(np.allclose(value, value_gt, atol=1e-2))
        x = torch.tensor([[0.5], [0.2]], dtype=torch.float, requires_grad=True)  # (2, 1).
        value_gt, grads_gt_raw = integral_pp(x, n=1000)
        grads_gt = torch.zeros(2, 2)
        grads_gt[0, 0] = grads_gt_raw[0, 0]
        grads_gt[1, 1] = grads_gt_raw[1, 0]
        self.assertTrue(np.allclose(value, value_gt, atol=1e-4))
        self.assertTrue(np.allclose(grads, grads_gt, rtol=0.2, atol=1e-4))

    def test_hybrid(self):
        x = torch.tensor([1, 2], dtype=torch.float, requires_grad=True)
        value_gt, grads_gt = integral_pp(x, n=1000)
        value, grads = integral_pp(x, n=100000, robust_dims=1)
        self.assertAlmostEqual(value.item(), value_gt.item(), delta=1e-2)
        self.assertTrue(np.allclose(grads, grads_gt, rtol=0.2, atol=1e-4))

        x = torch.tensor([1, 2, 3, 4], dtype=torch.float, requires_grad=True)
        value_gt, grads_gt = integral_pp(x, n=1000)
        for robust_dims in range(len(x) + 1):
            value, grads = integral_pp(x, n=100000, robust_dims=robust_dims)
            self.assertAlmostEqual(value.item(), value_gt.item(), delta=1e-2)
            self.assertTrue(np.allclose(grads, grads_gt, rtol=0.2, atol=0.05))

        torch.manual_seed(0)
        xs = torch.randn(5, 30).float()
        for robust_dims in range(0, 31, 5):
            print("Robust", robust_dims)
            integral_deltas = []
            gradient_deltas = []
            for x in xs:
                value_gt, grads_gt = integral_pp(x, n=512)
                value, grads = integral_pp(x, n=512, robust_dims=robust_dims, reminder_n=1000)
                integral_deltas.append(((value - value_gt).abs() / value_gt.abs()).item())
                gradient_deltas.append(self._delta_norm(grads, grads_gt))
            print("Hybrid integral delta", np.median(integral_deltas))
            print("Hybrid gradients delta", np.median(gradient_deltas))
            self.assertLess(np.median(integral_deltas), 0.1)
            self.assertLess(np.median(gradient_deltas), 0.5)

        start = time.time()
        n = 10
        b = 32
        for _ in range(n):
            value, grads = integral_pp(x[:1].tile(b, 1), n=512, robust_dims=10)
        print("Hybrid RPS", b * n / (time.time() - start))

    def _delta_norm(self, y, y_true):
        return (torch.linalg.norm(y - y_true) / torch.linalg.norm(y_true)).item()


if __name__ == "__main__":
    main()
