from collections import OrderedDict

import torch
from catalyst import dl
from catalyst.utils.misc import get_attr

from ..config import prepare_config, ConfigError
from ..layers import DiracDistribution, GMMDistribution
from ..torch import get_base_module, disable_amp
from .common import logits_deltas
from .integral_pp import PositiveNormalProbPP
from .multisim import MultiSimilarityLoss
from .proxynca import ProxyNCALoss
from .relaxation import Relaxed01Loss


class Criterion(torch.nn.Module):
    """Combination of crossentropy and KL-divergence regularization.

    PFE loss is described in Probabilistic Face Embeddings:
      https://openaccess.thecvf.com/content_ICCV_2019/papers/Shi_Probabilistic_Face_Embeddings_ICCV_2019_paper.pdf

    HIB loss is described in Modeling Uncertainty with Hedged Instance Embedding:
      https://arxiv.org/pdf/1810.00319.pdf

    Inputs:
        embeddings: Tensor with shape (B, D) or (B, N, D), where N is an ensemble size.
        labels: Tensor with shape (B) containing labels or one-hot labels with shape (B, C).
        logits: Tensor with classification logits (B, C) or (B, N, C), matching embeddings shape.
    """

    @staticmethod
    def get_default_config(use_softmax=True, logits_batchnorm=False, xent_weight=1.0, xent_smoothing=0.0,
                           hinge_weight=0.0, hinge_margin=1.0,
                           bce_weight=0.0, bce_log_prob=True,
                           magnetic_weight=0.0,
                           proxy_archor_weight=0.0, proxy_nca_weight=0.0,
                           multi_similarity_weight=0.0, multi_similarity_params=None,
                           prior_kld_weight=0.0, weights_prior_kld_weight=0.0,
                           pfe_weight=0.0, pfe_match_self=True, hib_weight=0.0,
                           relaxed01_weight=0.0, relaxed01_type="sigmoid",
                           exact_weight=0.0, exact_sample_size=64,
                           exact_robust_dims=None, exact_truncated=False,
                           exact_margin=None, exact_aggregation="mean",
                           reloss_weight=0.0):
        """Get optimizer parameters.

        Args:
            magnetic_weight: Weight of the magnetic loss, which is squared L2 between prediction and class centroid.
        """
        return OrderedDict([
            ("use_softmax", use_softmax),
            ("logits_batchnorm", logits_batchnorm),
            ("xent_weight", xent_weight),
            ("xent_smoothing", xent_smoothing),
            ("hinge_weight", hinge_weight),
            ("hinge_margin", hinge_margin),
            ("bce_weight", bce_weight),
            ("bce_log_prob", bce_log_prob),
            ("magnetic_weight", magnetic_weight),
            ("proxy_anchor_weight", proxy_archor_weight),
            ("proxy_nca_weight", proxy_nca_weight),
            ("multi_similarity_weight", multi_similarity_weight),
            ("multi_similarity_params", multi_similarity_params),
            ("prior_kld_weight", prior_kld_weight),
            ("weights_prior_kld_weight", weights_prior_kld_weight),
            ("pfe_weight", pfe_weight),
            ("pfe_match_self", pfe_match_self),
            ("hib_weight", hib_weight),
            ("relaxed01_weight", relaxed01_weight),
            ("relaxed01_type", relaxed01_type),
            ("exact_weight", exact_weight),
            ("exact_sample_size", exact_sample_size),
            ("exact_robust_dims", exact_robust_dims),
            ("exact_truncated", exact_truncated),
            ("exact_margin", exact_margin),
            ("exact_aggregation", exact_aggregation),
            ("reloss_weight", reloss_weight)
        ])

    def __init__(self, *, config=None):
        super().__init__()
        self._config = prepare_config(self, config)
        if self._config["multi_similarity_weight"] > 0:
            self._multi_similarity_loss = MultiSimilarityLoss(config=self._config["multi_similarity_params"])
        if self._config["proxy_nca_weight"] > 0:
            self._proxy_nca_loss = ProxyNCALoss()
        if self._config["relaxed01_weight"] > 0:
            self._relaxed01 = Relaxed01Loss(self._config["relaxed01_type"])
        self.distribution = None
        self.scorer = None

    def __call__(self, embeddings, labels, logits=None,
                 weights_prior_kld=None,
                 final_weights=None, final_bias=None, final_variance=None):
        if labels.ndim not in {1, 2}:
            raise ValueError("Expected labels with shape (B) or one-hot labels with shape (B, C).")
        if (logits is not None) and (logits.ndim != embeddings.ndim):
            raise ValueError("Expected logits with shape (B, C) or (B, N, C) matching embeddings, got {}.".format(logits.shape))
        if embeddings.ndim == 3:
            # Handle ensembles: average losses for all models.
            b, n, d = embeddings.shape
            embeddings = embeddings.reshape(b * n, d)
            if n > 1:
                labels = labels.repeat_interleave(n, dim=0)
            if logits is not None:
                logits = logits.reshape(b * n, logits.shape[-1])
        elif embeddings.ndim != 2:
            raise ValueError("Expected embeddings with shape (B, D) or (B, N, D), where N is ensemble size.")
        loss = 0
        if (logits is not None) and self._config["logits_batchnorm"]:
            logits = self._batchnorm0d(logits)
        if self._config["xent_weight"] != 0:
            if logits is None:
                raise ValueError("Need logits for Xent loss.")
            loss = loss + self._config["xent_weight"] * self._xent_loss(logits, labels)
        if self._config["hinge_weight"] != 0:
            if logits is None:
                raise ValueError("Need logits for Hinge loss.")
            loss = loss + self._config["hinge_weight"] * self._hinge_loss(logits, labels)
        if self._config["bce_weight"] != 0:
            if logits is None:
                raise ValueError("Need logits for BCE loss.")
            loss = loss + self._config["bce_weight"] * self._bce_loss(logits, labels)
        if self._config["magnetic_weight"] != 0:
            if final_weights is None:
                raise ValueError("Need final weights for magnetic loss.")
            loss = loss + self._config["magnetic_weight"] * self._magnetic_loss(embeddings, labels, final_weights)
        if self._config["proxy_anchor_weight"] != 0:
            if logits is None:
                raise ValueError("Need logits for Proxy-Anchor loss.")
            loss = loss + self._config["proxy_anchor_weight"] * self._proxy_anchor_loss(logits, labels)
        if self._config["proxy_nca_weight"] != 0:
            if self.scorer is None:
                raise ValueError("Need scorer for Proxy-NCA loss.")
            if final_weights is None:
                raise ValueError("Need final weights for Proxy-NCA loss.")
            if final_bias is not None:
                raise ValueError("Final bias is redundant for Proxy-NCA loss.")
            loss = loss + self._config["proxy_nca_weight"] * self._proxy_nca_loss(embeddings, labels, final_weights, self.scorer)
        if self._config["multi_similarity_weight"] > 0:
            if self.scorer is None:
                raise ValueError("Need scorer for Multi-similarity loss.")
            loss = loss + self._config["multi_similarity_weight"] * self._multi_similarity_loss(embeddings, labels, self.scorer)
        if self._config["prior_kld_weight"] != 0:
            loss = loss + self._config["prior_kld_weight"] * self._prior_kld_loss(embeddings)
        if self._config["weights_prior_kld_weight"] != 0:
            if weights_prior_kld is None:
                raise ValueError("Need Bayesian network to apply KLD loss.")
            loss = loss + self._config["weights_prior_kld_weight"] * weights_prior_kld
        if self._config["pfe_weight"] != 0:
            loss = loss + self._config["pfe_weight"] * self._pfe_loss(embeddings, labels)
        if self._config["hib_weight"] != 0:
            loss = loss + self._config["hib_weight"] * self._hib_loss(embeddings, labels)
        if self._config["relaxed01_weight"] > 0:
            if logits is None:
                raise ValueError("Need logits for relaxed 01 loss.")
            temperature = 1 / final_variance.sqrt() if final_variance is not None else None
            loss = loss + self._config["relaxed01_weight"] * self._relaxed01(logits, labels, temperature=temperature)
        if self._config["exact_weight"] != 0:
            if logits is None:
                raise ValueError("Need logits for EXACT loss.")
            loss = loss + self._config["exact_weight"] * self._exact_loss(embeddings, labels, logits,
                                                                          target_variance=final_variance)
        if self._config["reloss_weight"] != 0:
            if logits is None:
                raise ValueError("Need logits for ReLoss.")
            loss = loss + self._config["reloss_weight"] * self._reloss(logits, labels)
        return loss

    def _xent_loss(self, logits, labels):
        if self._config["use_softmax"]:
            kwargs = {}
            if self._config["xent_smoothing"] > 0:
                # Old PyTorch (below 1.10) doesn't support label_smoothing.
                kwargs["label_smoothing"] = self._config["xent_smoothing"]
            return torch.nn.functional.cross_entropy(logits, labels, **kwargs)
        else:
            return torch.nn.functional.nll_loss(logits, labels)

    def _hinge_loss(self, logits, labels):
        """Compute Hinge loss.

        Args:
            logits: Logits tensor with shape (*, N).
            labels: Integer labels with shape (*).

        Returns:
            Loss value.
        """
        n = logits.shape[-1]
        gt_logits = logits.take_along_dim(labels.unsqueeze(-1), -1)  # (*, 1).
        alt_mask = labels.unsqueeze(-1) != torch.arange(n, device=logits.device)  # (*, N).
        loss = (self._config["hinge_margin"] - gt_logits + logits).clip(min=0)[alt_mask].mean()
        return loss

    def _bce_loss(self, logits, labels):
        if labels.ndim == logits.ndim - 1:
            labels = torch.nn.functional.one_hot(labels, logits.shape[-1])
        if not self._config["bce_log_prob"]:
            return torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), labels.float())
        else:
            probs = torch.exp(logits)
            return torch.nn.functional.binary_cross_entropy(probs.float(), labels.float())

    def _magnetic_loss(self, embeddings, labels, final_weights):
        embeddings, _ = self.distribution.sample(embeddings)  # (B, D).
        centroids = final_weights[labels]  # (B, D).
        return (embeddings - centroids).square().mean()

    def _proxy_anchor_loss(self, logits, labels):
        """See Proxy Anchor Loss for Deep Metric Learning (2020):
        https://arxiv.org/pdf/2003.13911.pdf
        """
        b, c = logits.shape
        one_hot = torch.zeros_like(logits)  # (B, C).
        one_hot.scatter_(-1, labels.unsqueeze(-1).long(), 1)  # (B, C).
        num_positives = one_hot.sum(0)  # (C).
        ninf = -1.0e10
        positive = (-logits + (1 - one_hot) * ninf)[:, num_positives > 0].logsumexp(0)  # (P).
        positive = torch.nn.functional.softplus(positive).mean()
        negative = (logits + one_hot * ninf)[:, num_positives < b].logsumexp(0)  # (N).
        negative = torch.nn.functional.softplus(negative).mean()
        return positive + negative

    def _prior_kld_loss(self, distributions):
        return self.distribution.prior_kld(distributions).mean()

    def _pfe_loss(self, distributions, labels):
        pair_mls = self.distribution.logmls(distributions[None], distributions[:, None])
        same_mask = labels[None] == labels[:, None]  # (B, B).
        if not self._config["pfe_match_self"]:
            same_mask.fill_diagonal_(False)
        # TODO: check there is at least one positive.
        same_mls = pair_mls[same_mask]
        return -same_mls.mean()

    def _hib_loss(self, distributions, labels):
        same_probs = self.scorer(distributions[None], distributions[:, None])  # (B, B).
        same_mask = labels[None] == labels[:, None]  # (B, B).
        positive_probs = same_probs[same_mask]
        negative_probs = same_probs[~same_mask]
        positive_xent = torch.nn.functional.binary_cross_entropy(positive_probs, torch.ones_like(positive_probs))
        negative_xent = torch.nn.functional.binary_cross_entropy(negative_probs, torch.zeros_like(negative_probs))
        return 0.5 * (positive_xent + negative_xent)

    def _exact_loss(self, distributions, labels, logits, target_variance=None):
        dim = self.distribution.dim
        prefix = list(distributions.shape[:-1])
        ndim = len(prefix)
        c = logits.shape[-1]

        if isinstance(self.distribution, GMMDistribution):
            _, mean, hidden_vars = self.distribution.split_parameters(distributions)
            mean = mean.squeeze(-2)  # (..., D).
            std = self.distribution._parametrization.positive(hidden_vars).sqrt().squeeze(-2)  # (..., 1).
        elif isinstance(self.distribution, DiracDistribution):
            mean = self.distribution.mean(distributions)  # (..., D).
            std = None
        else:
            raise NotImplementedError("Only Dirac and GMM distributions are supported in EXACT.")

        if target_variance is not None:
            target_std = target_variance.sqrt()
        else:
            target_std = None

        if std is None:
            std = torch.ones([], device=mean.device, dtype=mean.dtype)
        if target_std is not None:
            std = std * target_std
        loss = self._fast_exact_logits_uncertainty(labels, logits, std)
        return loss

    def _fast_exact_logits_uncertainty(self, labels, logits, std):
        """Fast implementation of the EXACT with point prediction."""
        dim = self.distribution.dim
        prefix = list(labels.shape)
        ndim = len(prefix)
        c = logits.shape[-1]

        deltas = logits_deltas(logits, labels)

        # Apply margin and compute ratio.
        if self._config["exact_margin"] is not None:
            deltas = torch.clip(deltas, max=self._config["exact_margin"])
        target_mean = deltas / std

        prob_t = PositiveNormalProbPP.apply(target_mean, self._config["exact_sample_size"],
                                            self._config["exact_robust_dims"], self._config["exact_truncated"])  # (...).
        loss = self._exact_aggregate(prob_t)
        return loss

    @staticmethod
    def _batchnorm0d(logits):
        scale = logits.std()
        return (logits - logits.mean()) / (1e-6 + scale)

    def _exact_aggregate(self, probs):
        if self._config["exact_aggregation"] == "mean":
            return 1 - probs.clip(min=0).mean()
        elif self._config["exact_aggregation"] == "meanlog":
            return - (1e-6 + probs.clip(min=0)).log().mean()
        else:
            raise ConfigError("Unknown EXACT aggregation: {}".format(self._config["exact_aggregation"]))

    def _reloss(self, logits, labels):
        """Approximation of the standard checkpoint from the ReLoss paper.

        See https://github.com/hunto/ReLoss/blob/main/reloss/cls.py for more details.
        Approximation was evaluated for the pretrained model.
        """
        probs = torch.softmax(logits, 1)
        probs_gt = probs.gather(1, labels.unsqueeze(1))[:, 0]
        losses = (probs_gt - 0.8138719201087952).abs() * 128.1629180908203
        return losses.mean()


class CriterionCallback(dl.CriterionCallback):
    """Compute criterion in FP32 and pass distribution and scorer to criterion."""
    def __init__(self, *args, **kwargs):
        amp = kwargs.pop("amp", False)
        super().__init__(*args, **kwargs)
        self._amp = amp

    def _metric_fn(self, *args, **kwargs):
        with disable_amp(not self._amp):
            return self.criterion(*args, **kwargs)

    def on_stage_start(self, runner: "IRunner"):
        super().on_stage_start(runner)
        model = get_attr(runner, key="model", inner_key="model")
        scorer = get_attr(runner, key="model", inner_key="scorer")
        assert scorer is not None
        self.criterion.scorer = scorer
        distribution = get_base_module(model).distribution
        assert distribution is not None
        self.criterion.distribution = distribution

    def on_stage_end(self, runner: "IRunner"):
        super().on_stage_end(runner)
        self.criterion.scorer = None
        self.criterion.distribution = None
