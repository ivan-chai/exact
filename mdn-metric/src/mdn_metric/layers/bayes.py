import math
from collections import OrderedDict

import torch

from ..config import prepare_config, ConfigError
from .distribution import GMMDistribution


class LinearBayes(torch.nn.Linear):
    """Linear layer with Bayesian extensions.

    Variational dropout extension is based on "Variational dropout sparsifies deep neural networks", 2017.

    Inputs:
      - input: Input tensor with shape (*, D).

    Outputs:
      Tensor with shape (*, S, D_out), with S equal to sample size.
    """

    @staticmethod
    def get_default_config(variational_weight=True, variational_bias=True,
                           local_reparametrization=True, init_sigma=0.006738,
                           kld_reduction="sum", dropout=False, dropout_max_prior=10.0,
                           inference_sample_size=None, inference_sparsify_ratio=None):
        """Get default parameters.

        Args:
            variational_weight: Apply dropout to weight matrix.
            variational_bias: Apply dropout to bias vector.
            local_reparametrization: Apply independent parametrization for each batch element.
            init_sigma: Initial value of Gaussian STD.
            kld_reduction: Type of KLD reduction (`mean` or `sum`).
            dropout: Apply VD prior to weights.
                You may also need to set variational_bias to False to emulate original VD.
            dropout_max_prior: Maximum density of the VD prior distribution (log uniform has infinity at zero).
            inference_sample_size: Sample size used during inference. By default disable sampling and use mean weights.
            inference_sparsify_ratio: Disable weights with log (variance / weight^2) exceeding threshold. By default turned off.
        """
        return OrderedDict([
            ("variational_weight", variational_weight),
            ("variational_bias", variational_bias),
            ("local_reparametrization", local_reparametrization),
            ("init_sigma", init_sigma),
            ("kld_reduction", kld_reduction),
            ("dropout", dropout),
            ("dropout_max_prior", dropout_max_prior),
            ("inference_sample_size", inference_sample_size),
            ("inference_sparsify_ratio", inference_sparsify_ratio)
        ])

    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None, *, config=None):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self._config = prepare_config(self, config)
        init_log_sigma2 = 2 * math.log(self._config["init_sigma"])
        if self._config["variational_weight"]:
            self.weight_log_sigma2 = torch.nn.Parameter(torch.full_like(self.weight, init_log_sigma2))
        if (self.bias is not None) and self._config["variational_bias"]:
            self.bias_log_sigma2 = torch.nn.Parameter(torch.full_like(self.bias, init_log_sigma2))
        self._weight_distribution = GMMDistribution(config={"dim": 1, "max_logivar": None, "parametrization_params": {"type": "exp"}})

    @property
    def prior_kld(self):
        klds = []
        if self._config["variational_weight"]:
            if self._config["dropout"]:
                klds.append(self._dropout_prior_kld(self.weight, self.weight_log_sigma2))
            else:
                klds.append(self._prior_kld(self.weight, self.weight_log_sigma2))
        if (self.bias is not None) and self._config["variational_bias"]:
            klds.append(self._prior_kld(self.bias, self.bias_log_sigma2))
        return sum(klds[1:], klds[0]) if klds else None

    def forward(self, input):
        prefix = tuple(input.shape[:-1])
        in_channels = input.shape[-1]
        out_channels = self.weight.shape[0]
        flattened = input.reshape(-1, in_channels)  # (B, D).
        local_size = len(flattened) if self._config["local_reparametrization"] else 1
        sample_size = 1 if self.training else self._config["inference_sample_size"]
        # Reparametrize.
        weight = self.weight[None, None]  # (1, 1, C, D).
        if self._config["variational_weight"]:
            if sample_size is not None:
                weight = self._reparametrize(self.weight, self.weight_log_sigma2, local_size * sample_size)  # (B * S, C, D).
                weight = weight.reshape(local_size, sample_size, out_channels, in_channels)  # (B, S, C, D).
            weight = self._sparsify_on_inference(weight, self.weight, self.weight_log_sigma2)  # (B, S, C, D).
        bias = None if self.bias is None else self.bias[None, None]  # (1, 1, C).
        if self._config["variational_bias"] and (self.bias is not None):
            if (sample_size is not None):
                bias = self._reparametrize(self.bias, self.bias_log_sigma2, local_size * sample_size)  # (B * S, C).
                bias = bias.reshape(local_size, sample_size, out_channels)  # (B, S, C).
            bias = self._sparsify_on_inference(bias, self.bias, self.bias_log_sigma2)  # (B, S, C).
        # Apply batch linear transform.
        result = torch.matmul(flattened[:, None, None, :], weight.permute(0, 1, 3, 2)).squeeze(2)  # (B, S, C).
        if bias is not None:
            result = result + bias
        return result.reshape(*(prefix + (result.shape[1], out_channels)))

    def _reduce_kld(self, klds):
        if self._config["kld_reduction"] == "sum":
            return klds.sum()
        elif self._config["kld_reduction"] == "mean":
            return klds.mean()
        else:
            raise ConfigError("Unknown reduction type: {}".format(self._config["kld_reduction"]))

    def _prior_kld(self, theta, log_sigma2):
        parameters = self._weight_distribution.join_parameters(
            torch.zeros_like(theta[..., None]),  # (..., C).
            theta[..., None, None],  # (..., C, D).
            log_sigma2[..., None, None]  # (..., C, D).
        )  # (..., P).
        return self._reduce_kld(self._weight_distribution.prior_kld(parameters))

    def _dropout_prior_kld(self, theta, log_sigma2):
        k1, k2, k3 = 0.63576, 1.87320, 1.48695
        log_alpha = log_sigma2 - 2 * theta.abs().clip(min=1.0 / self._config["dropout_max_prior"]).log()
        kld = k1 - k1 * torch.sigmoid(k2 + k3 * log_alpha) - 0.5 * torch.nn.functional.logsigmoid(log_alpha)
        return self._reduce_kld(kld)

    def _sparsify_on_inference(self, sample, theta, log_sigma2):
        if self.training or (self._config["inference_sparsify_ratio"] is None):
            return sample
        assert theta.shape == log_sigma2.shape
        log_alpha = log_sigma2 - (theta.square() + 1e-6).log()
        mask = log_alpha < self._config["inference_sparsify_ratio"]
        return sample * mask

    @staticmethod
    def _reparametrize(theta, log_sigma2, sample_size):
        assert theta.shape == log_sigma2.shape
        output_shape = (sample_size,) + tuple(theta.shape)
        sigma = (0.5 * log_sigma2).exp()  # (*).
        noise = torch.randn(*output_shape, dtype=theta.dtype, device=theta.device)  # (S, *).
        return theta[None] + sigma[None] * noise  # (S, *).
