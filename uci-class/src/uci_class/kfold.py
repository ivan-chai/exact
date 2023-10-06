import torch
from sklearn.model_selection import KFold


class KFoldInterleave:
    def __init__(self, n_splits):
        self._n_splits = n_splits

    def split(self, classes):
        folds = [[] for _ in range(self._n_splits)]
        for i, c in enumerate(classes):
            folds[i % self._n_splits].append(c)
        sets = []
        for i in range(self._n_splits):
            train = sum([folds[j] for j in range(self._n_splits) if j != i], [])
            test = folds[i]
            sets.append((train, test))
        return sets


def split_crossval(X, y, i, k=5, interleave=False):
    """Get i-th training and validation sets using k element-based folds."""
    if i >= k:
        raise IndexError(i)
    indices = list(range(len(X)))
    if interleave:
        kfolder = KFoldInterleave(n_splits=k)
    else:
        kfolder = KFold(n_splits=k, shuffle=True, random_state=0)
    train_indices, val_indices = list(kfolder.split(indices))[i]
    return X[train_indices], X[val_indices], y[train_indices], y[val_indices]
