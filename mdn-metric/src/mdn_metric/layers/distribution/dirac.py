from collections import OrderedDict

import torch

from mdn_metric.config import prepare_config

from .common import DistributionBase, BatchNormNormalizer


class DiracDistribution(DistributionBase):
    """Single-point distribution with infinity density in one point and zero in others."""

    @staticmethod
    def get_default_config(dim=512, spherical=False, components=1, scale=1.0):
        """Get distribution parameters.

        Args:
            dim: Point dimension.
            spherical: Whether distribution is on sphere or R^n.
            components: Number of distribution components (modes).
            scale: Normalization scale when spherical embeddings are used.
        """
        return OrderedDict([
            ("dim", dim),
            ("spherical", spherical),
            ("components", components),
            ("scale", scale)
        ])

    def __init__(self, config=None):
        self._config = prepare_config(self, config)

    @property
    def dim(self):
        """Point dimension."""
        return self._config["dim"]

    @property
    def num_components(self):
        """Number of components in the mixture."""
        return self._config["components"]

    @property
    def is_spherical(self):
        """Whether distribution is on sphere or R^n."""
        return self._config["spherical"]

    @property
    def has_confidences(self):
        """Whether distribution has builtin confidence estimation or not."""
        return False

    @property
    def num_parameters(self):
        """Number of distribution parameters."""
        prob_parameters = self._config["components"] if self._config["components"] > 1 else 0
        mean_parameters = self._config["dim"]
        return prob_parameters + self._config["components"] * mean_parameters

    def unpack_parameters(self, parameters):
        """Returns dict with distribution parameters."""
        if parameters.shape[-1] != self.num_parameters:
            raise ValueError("Wrong number of parameters: {} != {}.".format(
                parameters.shape[-1], self.num_parameters))
        dim = self._config["dim"]
        c = self._config["components"]
        dim_prefix = list(parameters.shape)[:-1]
        if c > 1:
            scaled_log_probs = parameters[..., :c]
            means_offset = c
        else:
            scaled_log_probs = torch.zeros(*(dim_prefix + [c]), dtype=parameters.dtype, device=parameters.device)
            means_offset = 0
        log_probs = scaled_log_probs - torch.logsumexp(scaled_log_probs, dim=-1, keepdim=True)
        means = parameters[..., means_offset:].reshape(*(dim_prefix + [c, dim]))
        means = self._normalize(means)
        return {
            "log_probs": log_probs,
            "mean": means
        }

    def pack_parameters(self, parameters):
        """Returns vector from parameters dict."""
        keys = {"log_probs", "mean"}
        if set(parameters) != keys:
            raise ValueError("Expected dict with keys {}.".format(keys))
        if parameters["mean"].shape[-1] != self.dim:
            raise ValueError("Parameters dim mismatch.")
        log_probs = parameters["log_probs"]
        means = parameters["mean"]

        dim_prefix = list(torch.broadcast_shapes(
            log_probs.shape[:-1],
            means.shape[:-2]
        ))
        log_probs = log_probs.broadcast_to(*(dim_prefix + list(log_probs.shape[-1:])))
        means = means.broadcast_to(*(dim_prefix + list(means.shape[-2:])))
        flat_parts = []
        if self._config["components"] > 1:
            flat_parts.append(log_probs)
        flat_parts.extend([means.reshape(*(dim_prefix + [-1]))])
        return torch.cat(flat_parts, dim=-1)

    def make_normalizer(self):
        """Create and return normalization layer."""
        dim = self._config["dim"]
        c = self._config["components"]
        means_offset = c if c > 1 else 0
        return BatchNormNormalizer(self.num_parameters, begin=means_offset)

    def estimate(self, sample):
        """Estimate distribution parameters from sample.

        Args:
            sample: Tensor with shape (*, S, D).

        Returns:
            Distribution parameters with shape (*, P).
        """
        if self._config["components"] != 1:
            raise NotImplementedError("Can't fit mixtures yet.")
        sample = self._normalize(sample)  # (*, S, D).
        means = sample.mean(dim=-2, keepdim=True)  # (*, 1, D).
        return self.pack_parameters({"log_probs": torch.zeros_like(means[..., 0]), "mean": means})

    def sample(self, parameters, size=None):
        """Sample from distributions.

        Args:
            parameters: Distribution parameters with shape (..., K).
            size: Sample size (output shape without dimension). Parameters must be broadcastable to the given size.
              If not provided, output shape will be consistent with parameters.

        Returns:
            Tuple of:
                - Samples with shape (..., D).
                - Choosen components with shape (...).
        """
        if size is None:
            size = parameters.shape[:-1]
        parameters = parameters.reshape(list(parameters.shape[:-1]) + [1] * (len(size) - len(parameters.shape[:-1])) + [parameters.shape[-1]])
        c = self._config["components"]
        unpacked = self.unpack_parameters(parameters)
        log_probs, means = unpacked["log_probs"], unpacked["mean"]  # (..., C), (..., C, D).

        # Sample components.
        probs = log_probs.exp().broadcast_to(list(size) + [c])  # (..., C).
        components = torch.multinomial(probs.reshape(-1, c), 1).reshape(*size)  # (...).
        broad_components = components.unsqueeze(-1).unsqueeze(-1).broadcast_to(list(size) + [1, self.dim])  # (..., 1, D).
        means = means.broadcast_to(list(size) + [c, self.dim])
        means = torch.gather(means, -2, broad_components).squeeze(-2)  # (..., D).
        return means, components

    def mean(self, parameters):
        """Extract mean for each distribution.

        Args:
            parameters: Distribution parameters with shape (..., K).

        Returns:
            Distribution means with shape (..., D).
        """
        if parameters.shape[-1] != self.num_parameters:
            raise ValueError("Unexpected number of parameters: {} != {}.".format(parameters.shape[-1], self.num_parameters))
        unpacked = self.unpack_parameters(parameters)
        log_probs, means = unpacked["log_probs"], unpacked["mean"]  # (..., C), (..., C, D).
        if self._config["components"] == 1:
            means = means.squeeze(-2)
        else:
            means = (means * log_probs[..., None].exp()).sum(-2)  # (..., D).
        return means

    def modes(self, parameters):
        """Get modes of distributions.

        Args:
            parameters: Distribution parameters with shape (..., K).

        Returns:
            Tuple of mode log probabilities with shape (..., C) and modes with shape (..., C, D).
        """
        unpacked = self.unpack_parameters(parameters)
        log_probs, means = unpacked["log_probs"], unpacked["mean"]  # (..., C), (..., C, D).
        return log_probs, means

    def confidences(self, parameters):
        """Get confidence score for each element of the batch.

        Args:
            parameters: Distribution parameters with shape (..., K).

        Returns:
            Confidences with shape (...).
        """
        raise RuntimeError("Dirac distribution doesn't have confidence.")

    def prior_kld(self, parameters):
        """Get KL-divergence between distributions and prior.

        Is not defined for dirac.
        """
        raise RuntimeError("KLD is meaningless for dirac distribution.")

    def logpdf(self, parameters, x):
        """Compute log density for all points.

        Args:
            parameters: Distribution parameters with shape (..., K).
            points: Points for density evaluation with shape (..., D).

        Returns:
            Log probabilities with shape (...).
        """
        raise RuntimeError("Logpdf can't be estimated for Dirac density since it can be infinity.")

    def logmls(self, parameters1, parameters2):
        """Compute Log Mutual Likelihood Score (MLS) for pairs of distributions.

        Args:
            parameters1: Distribution parameters with shape (..., K).
            parameters2: Distribution parameters with shape (..., K).

        Returns:
            MLS scores with shape (...).
        """
        raise RuntimeError("MLS can't be estimated for Dirac density since it can be infinity.")

    def pdf_product(self, parameters1, paramaters2):
        """Compute product of two densities.

        Returns:
            Tuple of new distribution class and it's parameters.
        """
        raise RuntimeError("PDF product can't be estimated for Dirac density since it is unstable.")

    def make_mixture(self, log_probs, parameters):
        """Convert a set of distributions into mixture.

        Args:
            log_probs: Log probabilities with shape (N).
            parameters: Component parameters with shape (N, P).

        Returns:
            Tuple of new distribution class and it's parameters.
        """
        if log_probs.ndim != 1:
            raise ValueError("Unexpected log_probs shape: {}.".format(log_probs.shape))
        if parameters.ndim != 2:
            raise ValueError("Unexpected parameters shape: {}.".format(parameters.shape))
        if len(log_probs) != len(parameters):
            raise ValueError("Log probs and parameters shape mismatch: {}, {}.".format(log_probs.shape, parameters.shape))

        c = self._config["components"]
        n = len(log_probs)
        new_config = self._config.copy()
        new_config["components"] = c * n
        new_distribution = DiracDistribution(new_config)

        log_probs = log_probs - log_probs.logsumexp(dim=-1, keepdim=True)
        unpacked = self.unpack_parameters(parameters)
        new_parameters = new_distribution.pack_parameters({
            "log_probs": (log_probs[:, None] + unpacked["log_probs"]).flatten(),  # (N * C).
            "mean": unpacked["mean"].reshape(n * c, -1)  # (N * C, D).
        })
        return new_distribution, new_parameters

    def statistics(self, parameters):
        """Compute useful statistics for logging.

        Args:
            parameters: Distribution parameters with shape (..., K).

        Returns:
            Dictionary with floating-point statistics values.
        """
        return {}

    def _normalize(self, points):
        return torch.nn.functional.normalize(points, dim=-1) * self._config["scale"] if self.is_spherical else points
