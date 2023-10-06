import torch
from .common import logits_deltas


class BetaBernoulliLoss(torch.nn.Module):
    """Implementation of the Beta-Bernoully loss function.

    See "A New Smooth Approximation to the Zero One Loss with a Probabilistic Interpretation" (2020).
    """
    def __init__(self):
        super().__init__()
        self.hidden_weight = torch.nn.Parameter(torch.zeros([]))  # torch.full([], math.log(0.01)))
        self.hidden_theta = torch.nn.Parameter(torch.zeros([]))

    def forward(self, delta_logits):
        prior_weight = torch.sigmoid(self.hidden_weight)
        theta = torch.sigmoid(self.hidden_theta)
        probs = prior_weight * theta + (1 - prior_weight) * torch.sigmoid(delta_logits)
        return -probs.log()


class SigmoidLoss(torch.nn.Module):
    """Implementation of the smooth 0-1 loss function.

    See "Algorithms for Direct 0–1 Loss Optimization in Binary Classification" (2013).
    """
    def forward(self, delta_logits):
        probs = torch.sigmoid(delta_logits)
        return 1 - probs


class PolyLoss(torch.nn.Module):
    """Implementation of the polynomial 0-1 relaxation loss.

    See "A Robust Loss Function for Multiclass Classification" (2013).
    """
    def forward(self, delta_logits):
        delta_logits = delta_logits.clip(-1, 1)
        return 0.25 * delta_logits ** 3 - 0.75 * delta_logits + 0.5


class Relaxed01Loss(torch.nn.Module):
    TYPES = {
        "beta-bernoulli": BetaBernoulliLoss,
        "sigmoid": SigmoidLoss,
        "poly": PolyLoss
    }
    def __init__(self, type="sigmoid", reduction="mean"):
        super().__init__()
        self._loss = self.TYPES[type]()
        self._reduction = reduction

    def forward(self, logits, labels, temperature=None):
        deltas = logits_deltas(logits, labels)
        if temperature is not None:
            deltas = deltas * temperature
        losses = self._loss(deltas).sum(-1)
        if self._reduction == "none":
            loss = losses
        elif self._reduction == "mean":
            loss = losses.mean()
        elif self._reduction == "sum":
            loss = losses.sum()
        return loss
