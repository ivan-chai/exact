import math
import numpy as np
import torch


def genz_integral_impl(mean, cov, n : int = 10, mean_grads : bool = False):
    SQ2 = math.sqrt(2)
    NORMAL_NORM = math.sqrt(2 / math.pi)

    b, dim = mean.shape
    if cov.shape != (b, dim, dim):
        raise ValueError("Dimension mismatch between covariance and mean.")
    sample = torch.rand(dim - 1, n, b, dtype=mean.dtype, device=mean.device)  # (D - 1, n, ...).
    mean = mean[None].movedim(-1, 0)  # (D, n, B).

    mean = -mean
    l = torch.linalg.cholesky(cov)[None].movedim((-2, -1), (0, 1))  # (n, ..., D, D).

    # Create buffers.
    if mean_grads:
        mean = mean.unsqueeze(-1)  # (D, n, ..., 1).
        l = l.unsqueeze(-1)  # (D, D, n, ..., 1).
        sample = sample.unsqueeze(-1)  # (D - 1, n, ..., 1).
        ds_arg = torch.zeros(dim, n, b, dim, dtype=mean.dtype, device=mean.device)  # (D, n, ...).
    else:
        ds_arg = torch.zeros(dim, n, b, dtype=mean.dtype, device=mean.device)  # (D, n, ...).
    ds = torch.empty_like(ds_arg)  # (D, n, ...).
    es = torch.ones(1, dtype=mean.dtype, device=mean.device)  # (1).
    ys = torch.empty_like(ds[:-1, ...])  # (D - 1, n, ...).

    ds_arg[0] = mean[0] / (SQ2 * l[0, 0])
    ds[0] = torch.erf(ds_arg[0])

    for i in range(1, dim):
        interp = torch.lerp(ds[i - 1], es, sample[i - 1])
        if mean_grads:
            interp[..., i - 1] = ds[i - 1, ..., i - 1]
        ys[i - 1] = (torch.erfinv(interp) * SQ2).clip(-5, 5)
        s = (l[i, :i] * ys[:i]).sum(0)  # (...).
        ds_arg[i] = (mean[i] - s) / (l[i, i] * SQ2)  # (...).
        ds[i] = torch.erf(ds_arg[i])  # (...).

    deltas = 1 - ds
    l = l.squeeze(-1)
    if mean_grads:
        for i in range(dim):
            deltas[i, ..., i] = NORMAL_NORM / l[i, i] * (-(ds_arg[i, ..., i]).square()).exp()  # (...).
    deltas /= 2
    sums = deltas.prod(dim=0).mean(0)  # (...).
    return sums


def genz_integral(mean, cov, n=10, reorder=True, mean_grads=False):
    """Compute multivariate density in the orthant > 0.

    Args:
        mean: Mean tensors with shape (..., D).
        cov: Covariance matrices with shape (..., D, D).
        n: Sample size.
        reorder: Whether to use reordered integration or not.
        mean_grads: If true, compute gradients with respect to mean instead of integral value.

    Returns:
        Integral values with shape (...) if mean_grad is False and gradient values with shape (..., D) otherwise.
    """
    mean = mean.detach()
    cov = cov.detach()
    if mean.shape[:-1] != cov.shape[:-2]:
        raise ValueError("Mean and cov shape mismatch.")
    prefix = list(mean.shape[:-1])
    b = int(np.prod(prefix))
    dim = mean.shape[-1]

    mean = mean.reshape(b, dim)
    cov = cov.reshape(b, dim, dim)

    if reorder:
        mean, order = mean.sort(dim=-1)  # (B, D), (B, D).
        cov = cov.take_along_dim(order.unsqueeze(-1), -2).take_along_dim(order.unsqueeze(-2), -1)  # (B, D, D).
    else:
        order = torch.arange(dim, dtype=mean.dtype, device=mean.device).reshape([1, dim])  # (B, D).

    sums = genz_integral_impl(mean, cov, n, mean_grads)

    if mean_grads and reorder:
        # Sums: (B, D).
        iorder = order.argsort(dim=-1)  # (B, D).
        sums = sums.take_along_dim(iorder, -1)
    if mean_grads:
        prefix = prefix + [dim]
    return sums.reshape(*(prefix or [[]]))


def mc_integral(mean, cov, f=None, n=10):
    """Compute expectation of function f with multivariate density.

    Args:
        mean: Mean tensors with shape (..., D).
        cov: Covariance matrices with shape (..., D, D).
        f: Batched function from vectors to scalars (..., D) -> (...).
        n: Sample size.

    Returns:
        Integral values with shape (...).
    """
    with torch.no_grad():
        dim = mean.shape[-1]
        prefix = list(mean.shape[:-1])
        if cov.shape[-2:] != (dim, dim):
            raise ValueError("Dimension mismatch between covariance and mean.")

        l = torch.linalg.cholesky(cov).transpose(-2, -1).unsqueeze(-3)  # (..., 1, D, D).
        sample_uni = torch.randn(*(prefix + [n, dim]), dtype=mean.dtype, device=mean.device)  # (..., N, D).
        sample = torch.matmul(sample_uni.unsqueeze(-2), l).squeeze(-2) + mean.unsqueeze(-2)  # (..., N, D).
        mask = (sample > 0).all(dim=-1)  # (..., N).
    values = f(sample) if f is not None else torch.ones_like(sample[..., 0])  # (..., N).
    return (values * mask).mean(-1)


class PositiveNormalProb(torch.autograd.Function):
    @staticmethod
    def forward(self, mean, cov, n=10):
        self.save_for_backward(mean, cov)
        self.n = n
        with torch.no_grad():
            return genz_integral(mean, cov, n=n)  # Like in inference, don't add t_accuracy

    @staticmethod
    def backward(self, grad_output):
        mean, cov = self.saved_tensors
        n = self.n
        mean_grad = genz_integral(mean, cov, n=n, mean_grads=True)
        return mean_grad * grad_output.unsqueeze(-1), None, None, None
