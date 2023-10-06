import numbers
import torch


class PowerPooling2d(torch.nn.Module):
    def __init__(self, power):
        super().__init__()
        self._power = power

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError("Expected tensor with shape (b, c, h, w).")
        x.pow(self._power).sum(dim=(2, 3), keepdim=True).pow(1 / self._power)
        return x


class MultiPool2d(torch.nn.Module):
    """Combines average, power and max poolings.

    Args:
        mode: Combination of "a", "m", and digits to describe poolings used.
            For example "am3" means average, maximum and power-3 poolings.
        aggregate: Either "sum" or "cat".
    """
    def __init__(self, in_channels, mode="am", aggregate="sum"):
        super().__init__()
        self._in_channels = in_channels
        if aggregate not in ["sum", "cat"]:
            raise ValueError("Unknown aggrageation: {}.".format(aggregate))
        self._aggregate = aggregate
        self._poolings = []
        for m in mode:
            if m == "a":
                self._poolings.append(torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)))
            elif m == "m":
                self._poolings.append(torch.nn.AdaptiveMaxPool2d(output_size=(1, 1)))
            else:
                try:
                    power = int(m)
                except Exception:
                    raise ValueError("Unknown pooling: {}.".format(m))
                self._poolings.append(PowerPooling2d(power))
        for i, module in enumerate(self._poolings):
            setattr(self, "pool{}".format(i), module)

    @property
    def channels(self):
        multiplier = len(self._poolings) if self._aggregate == "cat" else 1
        return self._in_channels * multiplier

    def forward(self, x):
        results = [pooling(x) for pooling in self._poolings]
        if self._aggregate == "sum":
            result = torch.stack(results).sum(dim=0)
        else:
            assert self._aggregate == "cat"
            result = torch.cat(results, dim=-1)
        return result


class FlattenPooling(torch.nn.Module):
    """Flattens HW embeddings without aggregation."""
    def __init__(self, in_channels, in_size):
        super().__init__()
        self._in_channels = in_channels
        if isinstance(in_size, numbers.Number):
            in_size = (in_size, in_size)
        self._in_size = tuple(in_size)

    @property
    def channels(self):
        return self._in_size[0] * self._in_size[1] * self._in_channels

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError("Expected tensor with shape (b, c, h, w).")
        c, h, w = x.shape[1:]
        if (w, h) != self._in_size:
            raise ValueError("Expected {}x{} tensor, got {}x{}".format(self._in_size[0], self._in_size[1], w, h))
        return x.permute(0, 2, 3, 1).reshape(len(x), -1)  # (B, H * W * C).


class DistributionPooling(torch.nn.Module):
    """Estimate distribution parameters from embeddings."""
    def __init__(self, in_channels, distribution):
        super().__init__()
        self._distribution = distribution

    @property
    def channels(self):
        return self._distribution.num_parameters

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError("Expected tensor with shape (b, c, h, w).")
        c, h, w = x.shape[1:]
        sample = x.permute(0, 2, 3, 1).reshape(len(x), h * w, c)  # (B, HW, C).
        parameters = self._distribution.estimate(sample)  # (B, P).
        return parameters


class EnsemblePooling(torch.nn.Module):
    """Gathers embeddings into (B, HW, D) tensors, simulating ensemble of HW models."""
    def __init__(self, in_channels):
        super().__init__()
        self._in_channels = in_channels

    @property
    def channels(self):
        return self._in_channels

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError("Expected tensor with shape (b, c, h, w).")
        b, c, h, w = x.shape
        return x.permute(0, 2, 3, 1).reshape(b, h * w, c)  # (B, H * W, C).
