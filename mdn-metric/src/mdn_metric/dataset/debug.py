import numpy as np
from PIL import Image

from .common import Dataset


class DebugDataset(Dataset):
    """Simple dataset for debugging.

    Args:
        root: Dataset root.
        train: Whether to use train or test part of the dataset.
        multilabel: Generate multiple labels per element.

    """

    def __init__(self, root, *, train=True, multilabel=False):
        super().__init__()

        num_classes = 4 if train or multilabel else 2
        num_samples = 20
        if not multilabel:
            self._labels = np.concatenate([
                np.arange(num_classes),
                np.arange(num_classes),
                np.random.randint(0, num_classes, size=num_samples - 2 * num_classes)
            ])  # (B).
        else:
            self._labels = np.random.randint(0, 2, (num_samples, num_classes))  # (B, C).
        self._qualities = np.random.rand(len(self._labels)).astype(np.float32)

    @property
    def classification(self):
        """Whether dataset is classification or matching."""
        return True

    @property
    def multilabel(self):
        """Whether dataset is multilabel."""
        return self._labels.ndim == 2

    @property
    def openset(self):
        """Whether dataset is for open-set or closed-set classification."""
        return not self.multilabel

    @property
    def has_quality(self):
        return True

    @property
    def labels(self):
        """Get dataset labels array.

        Labels are integers in the range [0, N-1] or one-hot vectors in multilabel datasets.

        """
        return self._labels

    def __getitem__(self, index):
        """Get element of the dataset.

        Returns tuple (image, label, quality).

        """
        label = self._labels[index]
        image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        image = Image.fromarray(image)
        return image, label, self._qualities[index]
