from collections import OrderedDict

import torch

from mdn_metric.config import prepare_config, ConfigError


class Parametrization:
    """Mapping from real numbers to non-negative ones and vise-versa."""

    @staticmethod
    def get_default_config(type="invlin", min=0, max=None, center=0, scale=1):
        """Get default parametrization parameters.

        Args:
            type: Type of parametrization (`exp`, `softplus`, `invlin`, `abs` or `sigmoid`).
            min: Minimum positive value.
            max: Maximum value for sigmoid parametrization.
            center: Liner offset of the hidden value.
            scale: Liner scale of the hidden value.
        """
        return OrderedDict([
            ("type", type),
            ("min", min),
            ("max", max),
            ("center", center),
            ("scale", scale)
        ])

    def __init__(self, *, config=None):
        config = prepare_config(self, config)
        if config["type"] not in {"exp", "softplus", "invlin", "abs", "sigmoid"}:
            raise ValueError("Unknown parametrization: {}.".format(type))
        if (config["max"] is not None) and (config["type"] != "sigmoid"):
            raise ValueError("Maximum is supported for sigmoid parametrization only.")
        if (config["max"] is None) and (config["type"] == "sigmoid"):
            raise ValueError("Maximum value must be provided for sigmoid parametrization.")
        self._type = config["type"]
        self._min = config["min"]
        self._max = config["max"]
        self._center = config["center"]
        self._scale = config["scale"]

        if self._min < 0:
            raise ConfigError("Min must be non-negative.")
        if (self._max is not None) and (self._max < self._min):
            raise ConfigError("Maximum is less that minimum.")

    def positive(self, x):
        """Smooth mapping from real to positive numbers."""
        x = self._linear(x)
        if self._type == "exp":
            return self._exp(x, min=self._min)
        elif self._type == "softplus":
            return self._softplus(x, min=self._min)
        elif self._type == "invlin":
            return self._invlin(x, min=self._min)
        elif self._type == "sigmoid":
            return self._sigmoid(x, min=self._min, max=self._max)
        elif self._type == "abs":
            return self._abs(x, min=self._min)
        else:
            assert False

    def log_positive(self, x):
        """Logarithm of positive function."""
        x = self._linear(x)
        if self._type == "exp":
            return self._log_exp(x, min=self._min)
        elif self._type == "softplus":
            return self._log_softplus(x, min=self._min)
        elif self._type == "invlin":
            return self._log_invlin(x, min=self._min)
        elif self._type == "sigmoid":
            return self._log_sigmoid(x, min=self._min, max=self._max)
        elif self._type == "abs":
            return self._log_abs(x, min=self._min)
        else:
            assert False

    def ipositive(self, x):
        """Inverse of positive function."""
        if self._type == "exp":
            x = self._iexp(x, min=self._min)
        elif self._type == "softplus":
            x=  self._isoftplus(x, min=self._min)
        elif self._type == "invlin":
            x=  self._iinvlin(x, min=self._min)
        elif self._type == "sigmoid":
            x = self._isigmoid(x, min=self._min, max=self._max)
        elif self._type == "abs":
            x=  self._iabs(x, min=self._min)
        else:
            assert False
        x = self._ilinear(x)
        return x

    def _linear(self, x):
        if self._scale != 1:
            x = x / self._scale
        if self._center != 0:
            x = x - self._center
        return x

    def _ilinear(self, x):
        if self._center != 0:
            x = x + self._center
        if self._scale != 1:
            x = x * self._scale
        return x

    @staticmethod
    def _exp(x, min=0):
        """Smooth mapping from real to positive numbers."""
        result = x.exp()
        if min > 0:
            result = result + min
        return result

    @staticmethod
    def _log_exp(x, min=0):
        """Logarithm of exponential function with min."""
        result = x
        if min > 0:
            min = torch.tensor(min, dtype=x.dtype, device=x.device)
            result = torch.logaddexp(result, min.log())
        return result

    @staticmethod
    def _iexp(x, min=0):
        """Inverse of exp function with min."""
        if min > 0:
            x = x - min
        return x.log()

    @staticmethod
    def _softplus(x, min=0):
        """Smooth mapping from real to positive numbers."""
        result = torch.nn.functional.softplus(x)
        if min > 0:
            result = result + min
        return result

    @staticmethod
    def _log_softplus(x, min=0):
        """Logarithm of invlin function."""
        result = torch.where(x < -10, x, torch.nn.functional.softplus(x).log())
        if min > 0:
            min = torch.tensor(min, dtype=x.dtype, device=x.device)
            result = torch.logaddexp(result, min.log())
        return result

    @staticmethod
    def _isoftplus(x, min=0):
        """Inverse of invlin."""
        if min > 0:
            x = x - min
        return x + (-torch.expm1(-x)).log()

    @staticmethod
    def _invlin(x, min=0):
        """Smooth mapping from real to positive numbers.

        Inverse function for x < 0 and linear for x > 0.
        """
        result = torch.where(x < 0, 1 / (1 - x.clip(max=0)), 1 + x)  # Clip max to prevent NaN gradient for x = 1.
        if min > 0:
            result = result + min
        return result

    @staticmethod
    def _log_invlin(x, min=0):
        """Logarithm of invlin function."""
        is_negative = x < 0
        nxp1 = 1 - x
        xp1 = 1 + x
        if min > 0:
            xp1 = xp1 + min
        result = torch.where(is_negative, -nxp1.log(), xp1.log())
        if min > 0:
            nxp1ge1 = torch.clip(nxp1, min=1)
            result = result + is_negative * (1 + min * nxp1ge1).log()
        return result

    @staticmethod
    def _iinvlin(x, min=0):
        """Inverse of invlin."""
        if min > 0:
            x = x - min
        return torch.where(x < 1, 1 - 1 / x, x - 1)

    @staticmethod
    def _abs(x, min=0):
        """Mapping from real to positive numbers."""
        result = x.abs()
        if min > 0:
            result = result + min
        return result

    @staticmethod
    def _log_abs(x, min=0):
        """Logarithm of abs function."""
        return Parametrization._abs(x, min=min).log()

    @staticmethod
    def _iabs(x, min=0):
        """Inverse of abs (true inverse for positives only)."""
        if min > 0:
            x = x - min
        return x

    @staticmethod
    def _sigmoid(x, min=0, max=1):
        """Smooth mapping from real to positive numbers."""
        result = torch.sigmoid(x) * (max - min) + min
        return result

    @staticmethod
    def _log_sigmoid(x, min=0, max=1):
        """Logarithm of sigmoid function."""
        result = torch.log(torch.sigmoid(x) * (max - min) + min)
        return result

    @staticmethod
    def _isigmoid(x, min=0, max=1):
        """Inverse sigmoid."""
        result = torch.logit((x - min) / (max - min), eps=1-6)
        return result
