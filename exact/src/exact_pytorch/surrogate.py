import torch


class HingeLoss(torch.nn.Module):
    def __init__(self, margin=1):
        super().__init__()
        self._margin = margin

    def __call__(self, logits, labels):
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
        loss = (self._margin - gt_logits + logits).clip(min=0)[alt_mask].mean()
        return loss


class Poly1CrossEntropyLoss(torch.nn.Module):
    def __init__(self, epsilon=1.0):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, logits, labels):
        num_classes = logits.shape[-1]
        probs = torch.nn.functional.softmax(logits, dim=-1)
        probs = probs.take_along_dim(labels.unsqueeze(1), 1).squeeze(1)
        xent = torch.nn.functional.cross_entropy(logits, labels)
        loss = xent + self.epsilon * (1 - probs).mean()
        return loss
