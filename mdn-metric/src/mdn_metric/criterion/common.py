import torch


def non_diag(a):
    """Get non-diagonal elements of matrices.

    Args:
        a: Matrices tensor with shape (..., N, N).

    Returns:
        Non-diagonal elements with shape (..., N, N - 1).
    """
    n = a.shape[-1]
    prefix = list(a.shape)[:-2]
    return a.reshape(*(prefix + [n * n]))[..., :-1].reshape(*(prefix + [n - 1, n + 1]))[..., 1:].reshape(*(prefix + [n, n - 1]))


def logits_deltas(logits, labels):
    """Get deltas between positive and alternative classes.

    Args:
        logits: Logits tensor with shape (*, C).
        labels: Labels tensor with shape (*).

    Returns:
        Deltas tensor with shape (*, C - 1), containing lpos - lalt.
    """
    deltas = logits.take_along_dim(labels.unsqueeze(-1), -1) - logits  # (*, C).
    mask = torch.ones_like(deltas, dtype=torch.bool)  # (*, C).
    mask.scatter_(-1, labels.unsqueeze(-1), False)
    deltas = deltas[mask].reshape(*(list(deltas.shape[:-1]) + [deltas.shape[-1] - 1]))  # (*, C - 1).
    return deltas
