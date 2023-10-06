import math
from collections import OrderedDict
from contextlib import contextmanager

import torch

from ..config import prepare_config, ConfigError
from ..third_party.hyperbolic.hypnn import HyperbolicDistanceLayer, ToPoincare


class DotProductScorer(torch.nn.Module):
    """Compare two embeddings using dot product.

    Args:
        distribution: Distribution used in the model.

    Inputs:
        - parameters1: First group of distributions with shape (..., K).
        - parameters2: Second group of distributions with shape (..., K).

    Outputs:
        - scores: Similarities with shape (...).
    """

    def __init__(self, distribution, *, config=None):
        if config:
            raise ConfigError("Scorer doesn't have parameters.")
        super().__init__()
        self._distribution = distribution

    def __call__(self, parameters1, parameters2):
        if self.training:
            embeddings1, _ = self._distribution.sample(parameters1)
            embeddings2, _ = self._distribution.sample(parameters2)
        else:
            embeddings1 = self._distribution.mean(parameters1)
            embeddings2 = self._distribution.mean(parameters2)
        products = (embeddings1 * embeddings2).sum(-1)
        return products

    def statistics(self):
        """Compute useful statistics for logging.

        Returns:
            Dictionary with floating-point statistics values.
        """
        return {}


class CosineScorer(torch.nn.Module):
    """Compare two embeddings using cosine similarity.

    Args:
        distribution: Distribution used in the model.

    Inputs:
        - parameters1: First group of distributions with shape (..., K).
        - parameters2: Second group of distributions with shape (..., K).

    Outputs:
        - scores: Similarities with shape (...).
    """

    def __init__(self, distribution, *, config=None):
        if config:
            raise ConfigError("Scorer doesn't have parameters.")
        super().__init__()
        self._distribution = distribution

    def __call__(self, parameters1, parameters2):
        if self.training:
            embeddings1, _ = self._distribution.sample(parameters1)
            embeddings2, _ = self._distribution.sample(parameters2)
        else:
            embeddings1 = self._distribution.mean(parameters1)
            embeddings2 = self._distribution.mean(parameters2)
        embeddings1 = torch.nn.functional.normalize(embeddings1, dim=-1)
        embeddings2 = torch.nn.functional.normalize(embeddings2, dim=-1)
        cosines = (embeddings1 * embeddings2).sum(-1)
        return cosines

    def statistics(self):
        """Compute useful statistics for logging.

        Returns:
            Dictionary with floating-point statistics values.
        """
        return {}


class ExpectedCosineScorer(torch.nn.Module):
    """Compare two embeddings using expected cosine similarity.

    Args:
        distribution: Distribution used in the model.

    Inputs:
        - parameters1: First group of distributions with shape (..., K).
        - parameters2: Second group of distributions with shape (..., K).

    Outputs:
        - scores: Similarities with shape (...).
    """

    SAMPLE_SIZE = 10
    BATCH_SIZE = 128

    def __init__(self, distribution, *, config=None):
        if config:
            raise ConfigError("Scorer doesn't have parameters.")
        super().__init__()
        self._distribution = distribution

    def __call__(self, parameters1, parameters2):
        if (len(parameters1) == len(parameters2)) and (len(parameters1) > self.BATCH_SIZE):
            batch_size = len(parameters1)
            scores = []
            for i in range(0, batch_size, self.BATCH_SIZE):
                scores.append(self(parameters1[i:i + self.BATCH_SIZE], parameters2[i:i + self.BATCH_SIZE]))
            return torch.cat(scores)
        shape1 = list(parameters1.shape[:-1]) + [self.SAMPLE_SIZE, 1]
        shape2 = list(parameters2.shape[:-1]) + [1, self.SAMPLE_SIZE]
        embeddings1 = torch.nn.functional.normalize(self._distribution.sample(parameters1, shape1)[0], dim=-1)  # (..., K, 1, D).
        embeddings2 = torch.nn.functional.normalize(self._distribution.sample(parameters2, shape2)[0], dim=-1)  # (..., 1, K, D).
        cosines = (embeddings1 * embeddings2).sum(-1)  # (..., K, K).
        return cosines.mean((-1, -2))  # (...).

    def statistics(self):
        """Compute useful statistics for logging.

        Returns:
            Dictionary with floating-point statistics values.
        """
        return {}


class NegativeL2Scorer(torch.nn.Module):
    """Compare two embeddings using similarity based on euclidean distance.

    Args:
        distribution: Distribution used in the model.

    Inputs:
        - parameters1: First group of distributions with shape (..., K).
        - parameters2: Second group of distributions with shape (..., K).

    Outputs:
        - scores: Similarities with shape (...).
    """

    def __init__(self, distribution, *, config=None):
        if config:
            raise ConfigError("Scorer doesn't have parameters.")
        super().__init__()
        self._distribution = distribution

    def __call__(self, parameters1, parameters2):
        if self.training:
            embeddings1, _ = self._distribution.sample(parameters1)
            embeddings2, _ = self._distribution.sample(parameters2)
        else:
            embeddings1 = self._distribution.mean(parameters1)
            embeddings2 = self._distribution.mean(parameters2)
        distances = torch.square(embeddings1 - embeddings2).sum(-1)
        return -distances

    def statistics(self):
        """Compute useful statistics for logging.

        Returns:
            Dictionary with floating-point statistics values.
        """
        return {}


class MutualLikelihoodScorer(torch.nn.Module):
    """Compare two embeddings using MLS.

    Args:
        distribution: Distribution used in the model.

    Inputs:
        - parameters1: First group of distributions with shape (..., K).
        - parameters2: Second group of distributions with shape (..., K).

    Outputs:
        - scores: Similarities with shape (...).
    """

    def __init__(self, distribution, *, config=None):
        if config:
            raise ConfigError("Scorer doesn't have parameters.")
        super().__init__()
        self._distribution = distribution

    def __call__(self, parameters1, parameters2):
        return self._distribution.logmls(parameters1, parameters2)

    def statistics(self):
        """Compute useful statistics for logging.

        Returns:
            Dictionary with floating-point statistics values.
        """
        return {}


class FeatureMatchingScorer(torch.nn.Module):
    """Compare two embeddings using FMS.

    Args:
        distribution: Distribution used in the model.

    Inputs:
        - parameters1: First group of distributions with shape (..., K).
        - parameters2: Second group of distributions with shape (..., K).

    Outputs:
        - scores: Similarities with shape (...).
    """

    @staticmethod
    def get_default_config(method="intersection-sampling", sample_size=16):
        """Get classifier config.

        Args:
            method: FMS estimation method (`intersection-sampling` or `point`).
            sample_size: Sample size used in sampling-based estimation methods.
        """
        return OrderedDict([
            ("method", method),
            ("sample_size", sample_size)
        ])

    def __init__(self, distribution, *, config=None):
        super().__init__()
        self._config = prepare_config(self, config)
        self._distribution = distribution
        self._ubm = None

    @contextmanager
    def set_ubm(self, ubm_distribution, ubm_parameters):
        """Temporary set universal background model (UBM).

        NOTE: UBM is not detached.
        """
        if ubm_parameters.ndim != 1:
            raise ValueError("Expected UBM distribution parameters vector.")
        self._ubm = (ubm_distribution, ubm_parameters)
        try:
            yield self
        finally:
            self._ubm = None

    def __call__(self, parameters1, parameters2):
        prefix = torch.broadcast_shapes(parameters1.shape[:-1], parameters2.shape[:-1])
        ubm_distribution, ubm_parameters = self._get_fms_ubm(parameters1, parameters2)  # (UP).
        ubm_parameters = ubm_parameters.reshape(*([1] * len(prefix) + [ubm_distribution.num_parameters]))  # (..., UP).

        if self._config["method"] == "point":
            algorithm = self._point_fms
        elif self._config["method"] == "intersection-sampling":
            algorithm = self._intersection_sampling_fms
        else:
            raise ConfigError("Unknown FMS computation method: {}.".format(self._config["method"]))
        log_fms = algorithm(parameters1, parameters2, ubm_distribution, ubm_parameters)  # (...).
        return log_fms

    def _point_fms(self, parameters1, parameters2, ubm_distribution, ubm_parameters):
        log_mls = self._distribution.logmls(parameters1, parameters2)  # (...).
        product_distribution, product_parameters = self._distribution.pdf_product(parameters1, parameters2)
        modes = product_distribution.mode(product_parameters)  # (..., D).
        log_prior = ubm_distribution.logpdf(ubm_parameters, modes)  # (...).
        return log_mls - log_prior

    def _intersection_sampling_fms(self, parameters1, parameters2, ubm_distribution, ubm_parameters):
        log_mls = self._distribution.logmls(parameters1, parameters2)  # (...).
        prefix = tuple(log_mls.shape)  # (...).
        product_distribution, product_parameters = self._distribution.pdf_product(parameters1, parameters2)
        sample, _ = product_distribution.sample(product_parameters, prefix + (self._config["sample_size"],))  # (..., S, D).
        log_priors = ubm_distribution.logpdf(ubm_parameters.unsqueeze(-2), sample)  # (..., S).
        log_inv_prior = (-log_priors).logsumexp(-1) - math.log(self._config["sample_size"])  # (...).
        log_fms = log_mls + log_inv_prior  # (...).
        return log_fms

    def _get_fms_ubm(self, parameters1, parameters2):
        if self._ubm is not None:
            ubm_distribution, ubm_parameters = self._ubm
        else:
            log_probs1 = torch.zeros_like(parameters1[..., 0]).flatten()  # (N1).
            parameters1 = parameters1.reshape(-1, parameters1.shape[-1])  # (N1, P).
            log_probs2 = torch.zeros_like(parameters2[..., 0]).flatten()  # (N2).
            parameters2 = parameters2.reshape(-1, parameters2.shape[-1])  # (N2, P).
            log_probs = torch.cat([log_probs1, log_probs2])  # (N).
            parameters = torch.cat([parameters1, parameters2])  # (N, P).
            ubm_distribution, ubm_parameters = self._distribution.make_mixture(log_probs, parameters)  # (UP).
        return ubm_distribution, ubm_parameters

    def statistics(self):
        """Compute useful statistics for logging.

        Returns:
            Dictionary with floating-point statistics values.
        """
        return {}


class HIBScorer(torch.nn.Module):
    """Compare two embeddings using expectation of L2 sigmoid with trainable scale and bias.

    Scorer is used by HIB: https://arxiv.org/pdf/1810.00319.pdf

    Args:
        distribution: Distribution used in the model.

    Inputs:
        - parameters1: First group of distributions with shape (..., K).
        - parameters2: Second group of distributions with shape (..., K).

    Outputs:
        - scores: Similarities with shape (...).
    """

    NUM_SAMPLES = 8
    BATCH_SIZE = 128

    def __init__(self, distribution, *, config=None):
        if config:
            raise ConfigError("Scorer doesn't have parameters.")
        super().__init__()
        self._distribution = distribution
        self.scale = torch.nn.Parameter(torch.ones([]))
        self.bias = torch.nn.Parameter(torch.zeros([]))

    def __call__(self, parameters1, parameters2):
        if (len(parameters1) == len(parameters2)) and (len(parameters1) > self.BATCH_SIZE):
            batch_size = len(parameters1)
            scores = []
            for i in range(0, batch_size, self.BATCH_SIZE):
                scores.append(self(parameters1[i:i + self.BATCH_SIZE], parameters2[i:i + self.BATCH_SIZE]))
            return torch.cat(scores)
        samples1 = self._distribution.sample(parameters1, list(parameters1.shape)[:-1] + [self.NUM_SAMPLES])[0]  # (..., K, D).
        samples2 = self._distribution.sample(parameters2, list(parameters2.shape)[:-1] + [self.NUM_SAMPLES])[0]  # (..., K, D).
        # ||a - b|| = sqrt(||a||^2 + ||b|| ^ 2 - 2(a, b)).
        norm1sq = (samples1 ** 2).sum(-1)  # (..., K).
        norm2sq = (samples2 ** 2).sum(-1)  # (..., K).
        dot = torch.matmul(samples1, samples2.transpose(-1, -2))  # (..., K, K).
        distances = (norm1sq.unsqueeze(-1) + norm2sq.unsqueeze(-2) - 2 * dot).sqrt()
        scores = torch.sigmoid(-self.scale * distances + self.bias).mean(dim=(-1, -2))  # (...).
        return scores

    def statistics(self):
        """Compute useful statistics for logging.

        Returns:
            Dictionary with floating-point statistics values.
        """
        return {
            "scorer_scale": self.scale.item(),
            "scorer_bias": self.bias.item()
        }


class HyperbolicScorer(torch.nn.Module):
    """Compare two embeddings using similarity based on hyperbolic distance. https://arxiv.org/pdf/1805.09112.pdf

    Args:
        distribution: Distribution used in the model.

    Inputs:
        - parameters1: First group of distributions with shape (..., K).
        - parameters2: Second group of distributions with shape (..., K).

    Outputs:
        - scores: Similarities with shape (...).
    """

    @staticmethod
    def get_default_config(curvature=0.01):
        """Get classifier config.

        Args:
            curvature: Curvature of the Poincare ball.
        """
        return OrderedDict([
            ("curvature", curvature)
        ])

    def __init__(self, distribution, *, config=None):
        super().__init__()
        self._config = prepare_config(self, config)

        self._distribution = distribution
        self.expmap = ToPoincare(self._config["curvature"])
        self.dist = HyperbolicDistanceLayer(self._config["curvature"])

    def __call__(self, parameters1, parameters2):
        if self.training:
            embeddings1, _ = self._distribution.sample(parameters1)
            embeddings2, _ = self._distribution.sample(parameters2)
        else:
            embeddings1 = self._distribution.mean(parameters1)
            embeddings2 = self._distribution.mean(parameters2)
        distances = torch.squeeze(self.dist(self.expmap(embeddings1), self.expmap(embeddings2)))
        return -distances

    def statistics(self):
        """Compute useful statistics for logging.

        Returns:
            Dictionary with floating-point statistics values.
        """
        return {}
