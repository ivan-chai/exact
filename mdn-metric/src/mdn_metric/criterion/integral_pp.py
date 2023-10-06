import math
import numpy as np
import torch

from .common import non_diag


def get_cholesky(dim, diag=2, alt=1, device=None, dtype=None):
    """Get diagonal and non-diagonal elements of Cholesky decomposition
    of the matrix with given diagonal and non-diagonal elements.

    Cholesky matrix has the form
    D_1  0  ...  0
    A_1 D_2 ...  0
    A_1 A_2 ...  0
    ...
    A_1 A_2 ... D_n
    """
    delta = diag - alt
    l_diag = np.empty(dim)
    l_alt = np.empty(dim)
    s = 0.0
    for i in range(dim):
        l_diag[i] = math.sqrt(diag - s)
        l_alt[i] = l_diag[i] - delta / l_diag[i]
        s += l_alt[i] ** 2
    l_diag = torch.tensor(l_diag, dtype=dtype, device=device)
    l_alt = torch.tensor(l_alt, dtype=dtype, device=device)
    return l_diag, l_alt


def cholesky_matrix(l_diag, l_alt):
    """Create Cholesky decomposition matrix from diagonal and non-diagonal
    elements of this matrix."""
    l = torch.diag_embed(l_diag)  # (D, D).
    for i in range(len(l_diag) - 1):
        l[i + 1:, i] = l_alt[i]
    return l


def cov_inverse(dim, cov_diag=2, cov_alt=1):
    denum = (dim - 1) * cov_alt ** 2 - cov_diag ** 2 - max(dim - 2, 0) * cov_diag * cov_alt
    inv_cov_diag = - (cov_diag + max(dim - 2, 0) * cov_alt) / denum
    inv_cov_alt = cov_alt / denum
    return inv_cov_diag, inv_cov_alt


def cond_scales(cond_dim, cov_diag=2, cov_alt=1):
    inv_cov_diag, inv_cov_alt = cov_inverse(cond_dim, cov_diag, cov_alt)
    mean_scale = cov_alt * (inv_cov_diag + max(cond_dim - 1, 0) * inv_cov_alt)
    cov_scale = mean_scale * cond_dim * cov_alt
    return mean_scale, cov_scale


@torch.jit.script
def sample_tun(mean, n : int, cov_diag: float = 2):
    """Sample truncated univariate normal: (B) -> (B, N)."""
    b, = mean.shape
    sigma_sq2 = math.sqrt(2) * math.sqrt(cov_diag)
    lower = -torch.ones(1, dtype=mean.dtype, device=mean.device)
    sample = torch.rand(b, n, dtype=mean.dtype, device=mean.device)
    fb = torch.erf(mean / sigma_sq2).unsqueeze(-1)  # (B, N).
    sample = torch.erfinv(torch.lerp(lower, fb, sample)).clip(-5, 5) * (-sigma_sq2) + mean.unsqueeze(-1)
    return sample


def sample_tmn_pp(mean, n : int, cov_diag : float = 2, cov_alt : float = 1, n_iter : int = 2, initial_sample=None):
    """Sample truncated multivariate normal (B, D) -> (B, N, D)."""
    b, dim = mean.shape
    if dim == 0:
        return torch.zeros(b, n, 0, dtype=mean.dtype, device=mean.device)
    if dim == 1:
        return sample_tun(mean.squeeze(1), n=n, cov_diag=cov_diag).unsqueeze(-1)  # (B, N, 1).

    mean = mean.T  # (D, B).

    inv_cov_diag, inv_cov_alt = cov_inverse(dim - 1, cov_diag, cov_alt)
    mean_scale, cov_scale = cond_scales(dim - 1, cov_diag, cov_alt)
    cond_var = cov_diag - cov_scale
    cond_mean_part = ((1 + mean_scale) * mean - mean_scale * mean.sum(0, keepdim=True)).unsqueeze(-1)  # (D, B, 1).
    if initial_sample is None:
        sample = mean.clip(min=0).unsqueeze(2).tile(1, 1, n)  # (D, B, N).
    else:
        if initial_sample.shape[1] != n:
            raise ValueError("Initial sample size mismatch")
        sample = initial_sample.clone().permute(2, 0, 1)  # (D, B, N).
    sample_sum = sample.sum(0)  # (B, N).
    for _ in range(n_iter):
        for i in range(dim):
            sample_sum -= sample[i]
            cond_mean = cond_mean_part[i] + mean_scale * sample_sum  # (B, N)
            new_sample = sample_tun(cond_mean.reshape(b * n), 1, cond_var).reshape(b, n)
            sample_sum += new_sample
            sample[i] = new_sample
    return sample.permute(1, 2, 0)


def conditional_distribution_pp(mean, condition_mean, condition, cov_diag=2, cov_alt=1):
    """Get conditional distributions on last dimensions.

    Args:
        mean: Mean of free variables with shape (..., D).
        condition_mean: Mean of the condition with shape (..., CD).
        condition: Points to condition on with shape (..., CD).
        cov_diag: Diagonal value of the covariance matrix.
        cov_alt: Non-diagonal value of the covariance matrix.

    Returns:
        Tuple of (cond_mean, cond_cov_diag, cond_cov_alt) with conditional distribution parameters.
        Conditional mean will have the shape (..., D - CD).
    """
    dim = mean.shape[-1]
    condition_dim = condition_mean.shape[-1]
    if condition_dim != condition.shape[-1]:
        raise ValueError("Condition mean and condition dimensions mismatch.")
    if dim == 0:
        raise ValueError("Zero dimension.")
    mean_scale, cov_scale = cond_scales(condition_dim, cov_diag, cov_alt)
    cond_mean = mean + mean_scale * (condition - condition_mean).sum(-1, keepdim=True)
    cond_diag = cov_diag - cov_scale
    cond_alt = cov_alt - cov_scale
    return cond_mean, cond_diag, cond_alt


def conditional_distribution_grads_pp(mean, cov_diag=2, cov_alt=1):
    """Get conditional distributions for gradient computation.

    Method implements conditionining on negative mean value for all dimensions.

    Args:
        mean: Distribution mean with shape (..., D).
        diag: Diagonal value of the covariance matrix.
        alt: Non-diagonal value of the covariance matrix.

    Returns:
        Tuple of (cond_mean, cond_l_diag, cond_l_alt) with conditional distribution parameters.
        Conditional mean will have the shape (..., D, D - 1).
    """
    dim = mean.shape[-1]
    delta = cov_alt / cov_diag
    mean_scale, cov_scale = cond_scales(1, cov_diag, cov_alt)
    tile_shape = [1] * (mean.ndim + 1)
    tile_shape[-2] = dim
    cond_mean = non_diag(mean.unsqueeze(-2).tile(*tile_shape)) - mean_scale * mean.unsqueeze(-1)  # (B, D, D - 1).
    cond_diag = cov_diag - cov_scale
    cond_alt = cov_alt - cov_scale
    return cond_mean, cond_diag, cond_alt


def mc_integral_pp_impl(mean, cov_diag : int = 2, cov_alt : int = 1, n : int = 10):
    """Compute multivariate density in the orthant > 0 with covariance matrix equal to I + 1.

    Args:
        mean: Mean tensors with shape (B, D).
        cov_diag: Diagonal elements of the covariance matrix.
        cov_alt: Non-diagonal elements of the covariance matrix.
        n: Sample size.

    Returns:
        Tuple of integral values with shape (B) and gradients with shape (B, D).
    """
    b, dim = mean.shape
    sample_std = torch.randn([b, n, dim], dtype=mean.dtype, device=mean.device)  # (B, N, D).

    # Integral.
    l_diag, l_alt = get_cholesky(dim, diag=cov_diag, alt=cov_alt, dtype=mean.dtype, device=mean.device)
    l = cholesky_matrix(l_diag, l_alt)  # (D, D).
    sample = torch.nn.functional.linear(sample_std, l)  # (B, N, D).
    mask = (sample > -mean.unsqueeze(1)).all(dim=-1)  # (B, N).
    integrals = mask.float().mean(1)  # (B).

    inv_diag, inv_alt = cov_inverse(dim, cov_diag, cov_alt)
    gradients = inv_alt * sample.sum(-1, keepdim=True) + (inv_diag - inv_alt) * sample  # (B, N, D).
    gradients = (gradients * mask.unsqueeze(-1)).mean(1)  # (B, D).

    return integrals, gradients


def genz_integral_pp_impl(mean, cov_diag : int = 2, cov_alt : int = 1, n : int = 10, get_points : bool = False):
    """Compute multivariate density in the orthant > 0 with covariance matrix equal to I + 1.

    Args:
        mean: Distribution mean with shape (B, D).
        get_points: Return tuple (sums, points) with points being conditional sample with shape (B, N, D).

    Returns:
        Tuple of integral values with shape (B) and gradients with shape (B, D).
        If get_point is True, points with shape (B, N, D) are attached to the tuple.
    """
    SQ2 = math.sqrt(2)
    NORMAL_NORM = math.sqrt(2 / math.pi)

    b, dim = mean.shape
    sample = torch.rand(dim, n, b, 1, dtype=mean.dtype, device=mean.device)  # (D, N, B, D).
    mean = -mean.T.reshape(dim, 1, b, 1)  # (D, N, B, D).

    l_diag, l_alt = get_cholesky(dim, diag=cov_diag, alt=cov_alt, dtype=mean.dtype, device=mean.device)
    l_diag_sq = l_diag * SQ2

    # Create buffers.
    ds_arg = torch.zeros(dim, n, b, dim + 1, dtype=mean.dtype, device=mean.device)  # (D, N, B, D + 1).
    ds = torch.empty_like(ds_arg)  # (D, N, B, D + 1).
    es = torch.ones(1, dtype=mean.dtype, device=mean.device)  # (1).
    if get_points:
        ys = torch.empty(dim, n, b, dtype=mean.dtype, device=mean.device)  # (D, N, B).
    y_sums = torch.zeros_like(ds[0])  # (N, B, D + 1).

    ds_arg[0] = mean[0] / l_diag_sq[0]
    ds[0] = torch.erf(ds_arg[0])

    for i in range(1, dim):
        interp = torch.lerp(ds[i - 1], es, sample[i - 1])
        interp[..., i - 1] = ds[i - 1, ..., i - 1]
        y = (torch.erfinv(interp) * SQ2).clip(-5, 5)  # (N, B, D + 1).
        if get_points:
            ys[i - 1] = y[..., -1]
        y_sums += y * l_alt[i - 1]
        ds_arg[i] = (mean[i] - y_sums) / l_diag_sq[i]  # (...).
        ds[i] = torch.erf(ds_arg[i])  # (...).
    if get_points:
        interp = torch.lerp(ds[dim - 1, ..., -1], es, sample[dim - 1, ..., -1])
        ys[dim - 1] = (torch.erfinv(interp) * SQ2).clip(-5, 5)
        points = ys.transpose(0, -1)  # (B, N, D).
        l = cholesky_matrix(l_diag, l_alt)  # (D, D).
        points_mean = mean.squeeze(-1)
        points = (l[None, None] @ points[..., None]).squeeze(-1) - points_mean.permute(2, 1, 0)
    deltas = 1 - ds
    for i in range(dim):
        deltas[i, ..., i] = NORMAL_NORM / l_diag[i] * (-(ds_arg[i, ..., i]).square()).exp()
    deltas /= 2
    sums = deltas.prod(dim=0).mean(0)  # (B, D + 1).
    integrals = sums[:, -1]
    gradients = sums[:, :-1]
    if not get_points:
        return integrals, gradients
    else:
        return integrals, gradients, points


def select_truncated_dims(mean, n):
    """Returns indices of mean components used for optimization.

    Args:
        mean: Normal distribution means with shape (B, D).
        n: Number of output dimensions.

    Returns:
        indices: Indices of mean components used for optimization with shape (B, N).
    """
    indices = mean.abs().argsort(dim=-1)[:, :n]  # (B, D).
    return indices


def integral_pp(mean, n=10, reorder=True, robust_dims=None, robust_sampling_iters=5, reminder_n=1, truncated=False):
    """Compute multivariate density in the orthant > 0 with covariance matrix equal to I + 1.

    If robust_dims is 0, use Monte-Carlo with log-derivative trick. If robust_dims is None or equal to dim, use Genz algorithm.
    Use hybrid scheme in other cases with one part of the integral computed with Genz and other with MC.

    Args:
        mean: Mean tensors with shape (..., D).
        cov_diag: Diagonal elements of the covariance matrix.
        cov_alt: Non-diagonal elements of the covariance matrix.
        n: Sample size.
        reorder: Whether to use reordered integration or not.
        robust_dims: Maximum number of dimensions to compute with Genz algorithm.
            Compute the remainder with Monte-Carlo. Default is to use Genz for all dimensions.
        robust_sampling_iters: Number of iteration in Gibbs sampling. More iterations lead to better estimation.
        reminder_n: Sample size of the reminder MC integral.
        truncated: Don't compute reminder gradients.

    Returns:
        Integral values with shape (...) if mean_grad is False and gradient values with shape (..., D) otherwise.
    """
    mean = mean.detach()
    prefix = list(mean.shape[:-1])
    b = int(np.prod(prefix))
    dim = mean.shape[-1]
    robust_dims = min(robust_dims, dim) if robust_dims is not None else dim

    mean = mean.reshape(b, dim)  # (B, D).

    if reorder:
        mean, order = mean.sort(dim=-1)  # (B, D), (B, D).
    else:
        order = torch.arange(dim, dtype=mean.dtype, device=mean.device).reshape([1, dim])  # (B, D).

    if robust_dims == 0:
        integrals, gradients = mc_integral_pp_impl(mean, n=n * reminder_n)
    elif robust_dims == dim:
        integrals, gradients = genz_integral_pp_impl(mean, n=n)
    elif truncated:
        reminder_dims = dim - robust_dims
        robust_indices = select_truncated_dims(mean, robust_dims)  # (B, N).
        robust_mean = mean.take_along_dim(robust_indices, dim=1)  # (B, N).
        integrals, robust_gradients = integral_pp(robust_mean, n=n, reorder=reorder)  # (B), (B, DR).
        zero_gradients = torch.zeros(b, dim, dtype=mean.dtype, device=mean.device)
        gradients = torch.scatter(zero_gradients, 1, robust_indices, robust_gradients)
    else:
        reminder_dims = dim - robust_dims
        robust_mean = mean[..., :robust_dims]
        reminder_mean = mean[..., robust_dims:]

        robust_integrals, robust_gradients, robust_points = genz_integral_pp_impl(robust_mean, n=n, get_points=True)  # (B), (B, DR).
        robust_points = sample_tmn_pp(robust_mean, n=n, initial_sample=robust_points, n_iter=robust_sampling_iters)  # (B, N, DR).
        cond_mean, cond_cov_diag, cond_cov_alt = conditional_distribution_pp(reminder_mean.unsqueeze(1),
                                                                             robust_mean.unsqueeze(1),
                                                                             robust_points)  # (B, N, DRM), scalar, scalar.
        reminder_integrals, reminder_gradients = mc_integral_pp_impl(cond_mean.reshape(b * n, reminder_dims),
                                                                     cov_diag=cond_cov_diag, cov_alt=cond_cov_alt,
                                                                     n=reminder_n)  # (B * N), (B * N, DRM).
        reminder_integrals = reminder_integrals.reshape(b, n)  # (B, N).
        reminder_gradients = reminder_gradients.reshape(b, n, reminder_dims)  # (B, N, DRM).
        reminder_integrals_mean = reminder_integrals.mean(1)  # (B).
        integrals = robust_integrals * reminder_integrals_mean  # (B).
        reminder_gradients_full = reminder_gradients.mean(1) * robust_integrals.unsqueeze(-1)  # (B, DRM).

        cov_inv_diag, cov_inv_alt = cov_inverse(robust_dims)
        mean_scale, cov_scale = cond_scales(robust_dims)
        deltas = robust_mean.unsqueeze(1) - robust_points  # (B, N, DR).
        robust_points_logprob_grad = (cov_inv_alt - cov_inv_diag) * deltas - cov_inv_alt * deltas.sum(-1, keepdim=True)  # (B, N, DR).
        robust_gradients_alt = (- mean_scale * reminder_gradients.mean(1).sum(-1, keepdim=True)  # (B, N, 1).
                                + (reminder_integrals.unsqueeze(-1) * robust_points_logprob_grad).mean(1)  # (B, DR).
                                - reminder_integrals_mean.unsqueeze(-1) / robust_integrals[:, None].clip(min=1e-4) * robust_gradients  # (B, DR).
                                )  # (B, DR).
        robust_gradients_full = robust_gradients_alt * robust_integrals[:, None] + robust_gradients * reminder_integrals_mean.unsqueeze(-1)  # (B, DR).
        gradients = torch.cat([robust_gradients_full,
                               reminder_gradients_full],
                              dim=-1)  # (B, D).

    if reorder:
        # Sums: (B, D).
        iorder = order.argsort(dim=-1)  # (B, D).
        gradients = gradients.take_along_dim(iorder, -1)

    integrals = integrals.reshape(*(prefix or [[]]))
    gradients = gradients.reshape(*(prefix + [dim]))
    return integrals, gradients


def balance_gradients_inplace(mask, gradients):
    b, dim = gradients.shape
    if mask.shape != (b,):
        raise ValueError("Wrong mask shape")
    norms = torch.linalg.norm(gradients, dim=-1)  # (B).
    num_negative = mask.sum().item()
    if num_negative > 0:
        norm_negative = norms[mask].square().sum().sqrt().item() / math.sqrt(num_negative)
        gradients[mask] *= norm_negative / norms[mask].unsqueeze(-1).clip(min=1e-6)


class PositiveNormalProbPP(torch.autograd.Function):
    @staticmethod
    def forward(self, mean, n=10, robust_dims=None, truncated=False):
        with torch.no_grad():
            integrals, gradients = integral_pp(mean,
                                               n=n,
                                               robust_dims=robust_dims,
                                               truncated=truncated)
        self.save_for_backward(integrals, gradients)
        return integrals

    @staticmethod
    def backward(self, grad_output):
        integrals, gradients = self.saved_tensors
        return gradients * grad_output.unsqueeze(-1), None, None, None, None
