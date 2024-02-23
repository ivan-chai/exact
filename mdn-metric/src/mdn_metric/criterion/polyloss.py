import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Poly1CrossEntropyLoss(nn.Module):
    def __init__(self,
                 epsilon: float = 1.0,
                 reduction: str = "none",
                 weight: Tensor = None):
        """
        Create instance of Poly1CrossEntropyLoss
        :param num_classes:
        :param epsilon:
        :param reduction: one of none|sum|mean, apply reduction to final loss tensor
        :param weight: manual rescaling weight for each class, passed to Cross-Entropy loss
        """
        super(Poly1CrossEntropyLoss, self).__init__()
        self.epsilon = epsilon
        self.reduction = reduction
        self.weight = weight

    def forward(self, logits, labels):
        """
        Forward pass
        :param logits: tensor of shape [N, num_classes]
        :param labels: tensor of shape [N]
        :return: poly cross-entropy loss
        """
        num_classes = logits.size(-1)
        labels_onehot = F.one_hot(labels, num_classes=num_classes).to(device=logits.device,
                                                                      dtype=logits.dtype)
        pt = torch.sum(labels_onehot * F.softmax(logits, dim=-1), dim=-1)
        ce = F.cross_entropy(input=logits,
                             target=labels,
                             reduction='none',
                             weight=self.weight)
        poly1 = ce + self.epsilon * (1 - pt)
        if self.reduction == "mean":
            poly1 = poly1.mean()
        elif self.reduction == "sum":
            poly1 = poly1.sum()
        else:
            assert self.reduction == "none"
        return poly1
