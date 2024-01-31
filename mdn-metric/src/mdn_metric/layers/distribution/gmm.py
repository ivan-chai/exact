import math
from collections import OrderedDict
from numbers import Number

import torch

from mdn_metric.config import prepare_config, update_config, ConfigError

from ..parametrization import Parametrization
from .common import DistributionBase, BatchNormNormalizer
from .common import auto_matmul


class GMMDistribution(DistributionBase):
    """Gaussian Mixture Model.

    Variances are parametrized as input of :meth:`positive` function.
    """

    @staticmethod
    def get_default_config(dim=512, spherical=False, components=1, covariance="spherical",
                           parametrization_params=None, min_logivar=None, max_logivar=10):
        """Get GMM parameters.

        Args:
            dim: Point dimension.
            spherical: Whether distribution is on sphere or R^n.
            components: Number of GMM components.
            covariance: Type of covariance matrix (`diagonal`, `spherical` or number).
            parameterization_params: Variance parametrization parameters.
            min_logivar: Minimum value of log inverse variance (log concentration).
            max_logivar: Maximum value of log inverse variance (log concentration).
        """
        return OrderedDict([
            ("dim", dim),
            ("spherical", spherical),
            ("components", components),
            ("covariance", covariance),
            ("parametrization_params", parametrization_params),
            ("min_logivar", min_logivar),
            ("max_logivar", max_logivar)
        ])

    def __init__(self, config=None):
        self._config = prepare_config(self, config)
        if ((self._config["covariance"] not in ["diagonal", "spherical"]) and
            (not isinstance(self._config["covariance"], Number))):
            raise ConfigError("Unknown covariance type: {}".format(self._config["covariance"]))
        if self._config["max_logivar"] is None:
            min_var = 0
        else:
            min_var = math.exp(-self._config["max_logivar"])
        if self._config["min_logivar"] is None:
            max_var = None
        else:
            max_var = math.exp(-self._config["min_logivar"])
        self._parametrization = Parametrization(
            config=update_config(self._config["parametrization_params"], {"min": min_var, "max": max_var}, no_overwrite=True))

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
        return True

    @property
    def num_parameters(self):
        """Number of distribution parameters."""
        prob_parameters = self._config["components"] if self._config["components"] > 1 else 0
        mean_parameters = self._config["dim"]
        if isinstance(self._config["covariance"], Number):
            cov_parameters = 0
        elif self._config["covariance"] == "spherical":
            cov_parameters = 1
        elif self._config["covariance"] == "diagonal":
            cov_parameters = self._config["dim"]
        else:
            assert False
        return prob_parameters + self._config["components"] * (mean_parameters + cov_parameters)

    def unpack_parameters(self, parameters):
        """Returns dict with distribution parameters."""
        log_probs, means, hidden_vars = self.split_parameters(parameters)
        return {
            "log_probs": log_probs,
            "mean": means,
            "covariance": self._parametrization.positive(hidden_vars)
        }

    def pack_parameters(self, parameters):
        """Returns vector from parameters dict."""
        keys = {"log_probs", "mean", "covariance"}
        if set(parameters) != keys:
            raise ValueError("Expected dict with keys {}.".format(keys))
        hidden_vars = self._parametrization.ipositive(parameters["covariance"])
        return self.join_parameters(parameters["log_probs"], parameters["mean"], hidden_vars)

    def make_normalizer(self):
        """Create and return normalization layer."""
        dim = self._config["dim"]
        c = self._config["components"]
        means_offset = c if c > 1 else 0
        return BatchNormNormalizer(self.num_parameters, begin=means_offset, end=means_offset + c * dim)

    def split_parameters(self, parameters, normalize=True):
        """Extract component log probs, means and hidden variances from parameters."""
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
        means = parameters[..., means_offset:means_offset + c * dim].reshape(*(dim_prefix + [c, dim]))
        if isinstance(self._config["covariance"], Number):
            with torch.no_grad():
                hidden_covariance = self._parametrization.ipositive(torch.tensor([self._config["covariance"]])).item()
            hidden_vars = torch.full((dim_prefix + [c, 1]), hidden_covariance, dtype=parameters.dtype, device=parameters.device)
        else:
            hidden_vars = parameters[..., means_offset + c * dim:].reshape(*(dim_prefix + [c, -1]))

        if normalize:
            log_probs = scaled_log_probs - torch.logsumexp(scaled_log_probs, dim=-1, keepdim=True)
            means = self._normalize(means)
            return log_probs, means, hidden_vars
        else:
            return scaled_log_probs, means, hidden_vars

    def join_parameters(self, log_probs, means, hidden_vars):
        """Join different GMM parameters into vectors."""
        dim_prefix = list(torch.broadcast_shapes(
            log_probs.shape[:-1],
            means.shape[:-2],
            hidden_vars.shape[:-2]
        ))
        log_probs = log_probs.broadcast_to(*(dim_prefix + list(log_probs.shape[-1:])))
        means = means.broadcast_to(*(dim_prefix + list(means.shape[-2:])))
        flat_parts = []
        if self._config["components"] > 1:
            flat_parts.append(log_probs)
        flat_parts.extend([means.reshape(*(dim_prefix + [-1]))])
        if isinstance(self._config["covariance"], Number):
            with torch.no_grad():
                hidden_covariance = self._parametrization.ipositive(torch.tensor([self._config["covariance"]],
                                                                                 dtype=hidden_vars.dtype,
                                                                                 device=hidden_vars.device))
            if not torch.allclose(hidden_vars, hidden_covariance):
                raise ValueError("Covariance value changed: {} != {}.".format(
                    self._parametrization.positive(hidden_vars),
                    self._parametrization.positive(hidden_covariance)
                ))
        else:
            hidden_vars = hidden_vars.broadcast_to(*(dim_prefix + list(hidden_vars.shape[-2:])))
            flat_parts.extend([hidden_vars.reshape(*(dim_prefix + [-1]))])
        return torch.cat(flat_parts, dim=-1)

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
        mean = sample.mean(dim=-2, keepdim=True)  # (*, 1, D).
        if isinstance(self._config["covariance"], Number):
            covariance = torch.full_like(mean[..., :1], self._config["covariance"])
        elif self._config["covariance"] == "diagonal":
            covariance = sample.var(dim=-2, keepdim=True, unbiased=False)  # (*, 1, D).
        else:
            assert self._config["covariance"] == "spherical"
            covariance = sample.var(dim=-2, keepdim=True, unbiased=False).mean(dim=-1, keepdim=True)  # (*, 1, 1).
        log_probs = torch.zeros_like(mean[..., 0])
        return self.pack_parameters({"log_probs": log_probs, "mean": mean, "covariance": covariance})

    def sample(self, parameters, size=None, temperature=1):
        """Sample from distributions.

        Args:
            parameters: Distribution parameters with shape (..., K).
            size: Sample size (output shape without dimension). Parameters must be broadcastable to the given size.
              If not provided, output shape will be consistent with parameters.
            temperature: Parameter controlling diversity.

        Returns:
            Tuple of:
                - Samples with shape (..., D).
                - Choosen components with shape (...).
        """
        if size is None:
            size = parameters.shape[:-1]
        parameters = parameters.reshape(list(parameters.shape[:-1]) + [1] * (len(size) - len(parameters.shape[:-1])) + [parameters.shape[-1]])
        c = self._config["components"]
        log_probs, means, hidden_vars = self.split_parameters(parameters)  # (..., C), (..., C, D), (..., C, D).
        if temperature != 1:
            log_probs = log_probs / temperature
            log_probs = log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True)

        # Sample components.
        probs = log_probs.exp().broadcast_to(list(size) + [c])  # (..., C).
        components = torch.multinomial(probs.reshape(-1, c), 1).reshape(*size)  # (...).
        broad_components = components.unsqueeze(-1).unsqueeze(-1).broadcast_to(list(size) + [1, self.dim])  # (..., 1, D).
        means = means.broadcast_to(list(size) + [c, self.dim])
        means = torch.gather(means, -2, broad_components).squeeze(-2)  # (..., D).
        hidden_vars = hidden_vars.broadcast_to(list(size) + [c, self.dim])
        hidden_vars = torch.gather(hidden_vars, -2, broad_components).squeeze(-2)  # (..., D).

        # Sample from components.
        normal = torch.randn(*(list(size) + [self.dim]), dtype=parameters.dtype, device=parameters.device)  # (..., D).
        stds = self._parametrization.positive(hidden_vars).sqrt()  # (..., D).
        if temperature != 1:
            stds = stds * math.sqrt(temperature)
        samples = normal * stds + means  # (..., D).
        return samples, components

    def mean(self, parameters):
        """Extract mean for each distribution.

        Args:
            parameters: Distribution parameters with shape (..., K).

        Returns:
            Distribution means with shape (..., D).
        """
        log_probs, means, _ = self.split_parameters(parameters)  # (..., C), (..., C, D).
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
        log_probs, means, _ = self.split_parameters(parameters)  # (..., C), (..., C, D).
        return log_probs, means

    def confidences(self, parameters):
        """Get confidence score for each element of the batch.

        Args:
            parameters: Distribution parameters with shape (..., K).

        Returns:
            Confidences with shape (...).
        """
        log_probs, means, hidden_vars = self.split_parameters(parameters)  # (..., 1), (..., 1, D), (..., 1, D).
        logvars = self._parametrization.log_positive(hidden_vars)  # (..., 1, D).
        # Proportional to log entropy.
        return -logvars.mean((-1, -2))  #  (...).

    def prior_kld(self, parameters):
        """Get KL-divergence between distributions and standard normal distribution.

        Args:
            parameters: Distribution parameters with shape (..., K).

        Returns:
            KL-divergence of each distribution with shape (...).
        """
        if self._config["components"] != 1:
            raise NotImplementedError("KL divergence with standard normal distribution for the mixture.")
        log_probs, means, hidden_vars = self.split_parameters(parameters)  # (..., 1), (..., 1, D), (..., 1, D).
        vars = self._parametrization.positive(hidden_vars)  # (..., 1, D).
        logvars = self._parametrization.log_positive(hidden_vars)  # (..., 1, D).
        # There is error in original DUL formula for KLD. Below is true KLD.
        if self._config["covariance"] == "spherical":
            assert logvars.shape[-1] == 1
            logdet = logvars[..., 0] * self.dim  # (..., 1).
            trace = vars[..., 0] * self.dim  # (..., 1).
        else:
            assert self._config["covariance"] == "diagonal"
            assert logvars.shape[-1] == self.dim
            logdet = logvars.sum(dim=-1)  # (..., 1).
            trace = vars.sum(dim=-1)  # (..., 1).
        means_sqnorm = means.square().sum(dim=-1)  # (..., 1).
        kld = 0.5 * (-logdet - self.dim + trace + means_sqnorm)  # (..., 1).
        return kld.squeeze(-1)  # (...).

    def logpdf(self, parameters, points):
        """Compute log density for all points.

        Args:
            parameters: Distribution parameters with shape (..., K).
            points: Points for density evaluation with shape (..., D).

        Returns:
            Log probabilities with shape (...).
        """
        log_probs, means, hidden_vars = self.split_parameters(parameters)  # (..., C), (..., C, D), (..., C, D).
        vars = self._parametrization.positive(hidden_vars)
        logivars = -self._parametrization.log_positive(hidden_vars)
        c = -self._config["dim"] / 2 * math.log(2 * math.pi)

        points = self._normalize(points)

        # Compute L2 using dot product and torch.nn.linear for better memory usage
        # during broadcasting.
        means_sq_norms = (means.square() / vars).sum(-1)  # (..., C).
        products = auto_matmul(means / vars, points.unsqueeze(-1)).squeeze(-1)  # (..., C).
        if ((self._config["covariance"] == "spherical") or isinstance(self._config["covariance"], Number)):
            assert logivars.shape[-1] == 1
            logidet = logivars[..., 0] * self.dim  # (..., C).
            points_sq_norms = points.unsqueeze(-2).square().sum(-1) / vars.squeeze(-1)  # (..., C).
        else:
            assert self._config["covariance"] == "diagonal"
            assert logivars.shape[-1] == self.dim
            logidet = logivars.sum(dim=-1)  # (..., C).
            points_sq_norms = auto_matmul(1 / vars, points.square().unsqueeze(-1)).squeeze(-1)  # (..., C).
        logexp = products - 0.5 * (means_sq_norms + points_sq_norms)  # (..., C).
        return torch.logsumexp(log_probs + c + 0.5 * logidet + logexp, dim=-1)  # (...).

    def logmls(self, parameters1, parameters2):
        """Compute Log Mutual Likelihood Score (MLS) for pairs of distributions.


        Args:
            parameters1: Distribution parameters with shape (..., K).
            parameters2: Distribution parameters with shape (..., K).

        Returns:
            MLS scores with shape (...).
        """
        log_probs1, means1, hidden_vars1 = self.split_parameters(parameters1)  # (..., C), (..., C, D), (..., C, D).
        log_probs2, means2, hidden_vars2 = self.split_parameters(parameters2)  # (..., C), (..., C, D), (..., C, D).
        logvars1 = self._parametrization.log_positive(hidden_vars1)
        logvars2 = self._parametrization.log_positive(hidden_vars2)
        pairwise_logmls = self._normal_logmls(
            means1=means1[..., :, None, :],  # (..., C, 1, D).
            logvars1=logvars1[..., :, None, :],  # (..., C, 1, D).
            means2=means2[..., None, :, :],  # (..., 1, C, D).
            logvars2=logvars2[..., None, :, :]  # (..., 1, C, D).
        )  # (..., C, C).
        pairwise_logprobs = log_probs1[..., :, None] + log_probs2[..., None, :]  # (..., C, C).
        dim_prefix = list(pairwise_logmls.shape)[:-2]
        logmls = torch.logsumexp((pairwise_logprobs + pairwise_logmls).reshape(*(dim_prefix + [-1])), dim=-1)  # (...).
        return logmls

    def pdf_product(self, parameters1, parameters2):
        """Compute product of two densities.

        Returns:
            Tuple of new distribution class and it's parameters.
        """
        c = self._config["components"]
        c2 = c * c

        # Init output distribution type.
        new_config = self._config.copy()
        new_config["components"] = c2
        if isinstance(self._config["covariance"], Number):
            new_config["covariance"] = "spherical"
        new_distribution = GMMDistribution(new_config)

        # Parse inputs.
        log_probs1, means1, hidden_vars1 = self.split_parameters(parameters1)  # (..., C), (..., C, D), (..., C, D).
        log_probs2, means2, hidden_vars2 = self.split_parameters(parameters2)  # (..., C), (..., C, D), (..., C, D).
        log_probs1 = log_probs1.unsqueeze(-1)  # (..., C, 1).
        log_probs2 = log_probs2.unsqueeze(-2)  # (..., 1, C).
        means1 = means1.unsqueeze(-2)  # (..., C, 1, D).
        means2 = means2.unsqueeze(-3)  # (..., 1, C, D).
        vars1 = self._parametrization.positive(hidden_vars1).unsqueeze(-2)  # (..., C, 1, D).
        vars2 = self._parametrization.positive(hidden_vars2).unsqueeze(-3)  # (..., 1, C, D).

        # Compute distribution parameters.
        vars_sum = vars1 + vars2  # (..., C, C, D)
        norm_config = self._config.copy()
        norm_config["components"] = 1
        if isinstance(self._config["covariance"], Number):
            norm_config["covariance"] = "spherical"
        norm_distribution = GMMDistribution(norm_config)
        norm_means = means1 - means2  # (..., C, C, D).
        norm_parameters = norm_distribution.join_parameters(
            torch.zeros_like(vars_sum[..., :1]),  # (..., C, C).
            norm_means.unsqueeze(-2),  # (..., C, C, 1, D).
            self._parametrization.ipositive(vars_sum).unsqueeze(-2)  # (..., C, C, 1, D).
        )  # (..., C, C).
        new_log_probs = (log_probs1 + log_probs2) + norm_distribution.logpdf(norm_parameters, torch.zeros_like(norm_means))  # (..., C, C).
        new_vars = vars1 / vars_sum * vars2  # (..., C, C, D).
        new_hidden_vars = self._parametrization.ipositive(new_vars)  # (..., C, C, D).
        new_means = vars2 / vars_sum * means1 + vars1 / vars_sum * means2  # (..., C, C, D).
        prefix = tuple(new_means.shape[:-3])
        new_parameters = new_distribution.join_parameters(
            new_log_probs.reshape(*(prefix + (c2,))),
            new_means.reshape(*(prefix + (c2, -1))),
            new_hidden_vars.reshape(*(prefix + (c2, -1)))
        )  # (..., P).
        return new_distribution, new_parameters

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
        new_distribution = GMMDistribution(new_config)

        log_probs = log_probs - log_probs.logsumexp(dim=-1, keepdim=True)
        components_log_probs, means, hidden_vars = self.split_parameters(parameters)  # (N, C), (N, D), (N, D).
        new_parameters = new_distribution.join_parameters(
            (log_probs[:, None] + components_log_probs).flatten(),  # (N * C).
            means.reshape(n * c, -1),  # (N * C, D).
            hidden_vars.reshape(n * c, -1)  # (N * C, D).
        )
        return new_distribution, new_parameters

    def statistics(self, parameters):
        """Compute useful statistics for logging.

        Args:
            parameters: Distribution parameters with shape (..., K).

        Returns:
            Dictionary with floating-point statistics values.
        """
        parameters = parameters.reshape(-1, parameters.shape[-1])
        log_probs, means, hidden_vars = self.split_parameters(parameters)  # (N, C), (N, D), (N, D).
        stds = self._parametrization.positive(hidden_vars).sqrt()
        return {
            "gmm_std/mean": stds.mean(),
            "gmm_std/std": stds.std()
        }

    def _normal_logmls(self, means1, logvars1, means2, logvars2):
        """Compute Log MLS for unimodal distributions.

        For implementation details see "Probabilistic Face Embeddings":
        https://openaccess.thecvf.com/content_ICCV_2019/papers/Shi_Probabilistic_Face_Embeddings_ICCV_2019_paper.pdf
        """
        c = -0.5 * self._config["dim"] * math.log(2 * math.pi)
        delta2 = torch.square(means1 - means2)  # (..., D).
        covsum = logvars1.exp() + logvars2.exp()   # (..., D).
        logcovsum = torch.logaddexp(logvars1, logvars2)  # (..., D).
        mls = c - 0.5 * (delta2 / covsum + logcovsum).sum(-1)  # (...).
        return mls

    def _normalize(self, points):
        return torch.nn.functional.normalize(points, dim=-1) if self.is_spherical else points
