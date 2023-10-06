import numpy as np

from ...torch import tmp_seed
from .base import DatasetWrapper


class UniformLabelNoiseDataset(DatasetWrapper):
    """
    This wrapper introduces uniform label noise into a given fraction of a dataset.
    It should be used on classification datasets.
    Note: number of noised labels can be less than the requested amount.

    Args:
        dataset: Dataset to wrap.
        noise_fraction: A proportion of dataset samples to be noised. The samples are selected randomly.
        seed: Random seed.
    """
    def __init__(self, dataset, noise_fraction: float, seed: int = 0):
        super().__init__(dataset)
        if not dataset.classification:
            raise ValueError("Label noise is now available only for classification datasets.")

        num_labels = dataset.num_classes
        num_noisy_samples = int(noise_fraction * len(dataset))

        with tmp_seed(seed):
            indices_to_noise = np.random.choice(np.arange(len(dataset)), num_noisy_samples, replace=False)
            self._labels = np.array(dataset.labels)
            self._labels[indices_to_noise] = np.random.randint(0, num_labels, num_noisy_samples)

    @property
    def labels(self):
        """Get dataset labels array.

        Labels are integers in the range [0, N-1].
        """
        return self._labels

    def __getitem__(self, index):
        """Get element of the dataset.

        Classification dataset returns tuple (image, label).
        Verification dataset returns ((image1, image2), label).

        """
        item = self.dataset[index]
        return (item[0], self._labels[index]) + item[2:]


class TailIdentityLabelNoiseDataset(DatasetWrapper):
    """
    This wrapper selects the smallest (largest) classes from the datasets and noises labels in them.
    The number of classes to noise is selected according to the noise_fraction.
    Note: number of noised labels can be less than the requested amount.

    Args:
        dataset: Dataset to wrap.
        noise_fraction: A proportion of dataset samples to be noised.
        seed: Random seed.
        noise_large_ids: Whether to noise large ids instead of small.
    """
    def __init__(self, dataset, noise_fraction: float, noise_large_ids=False,
                 preserve_non_tail_classes=False, seed: int = 13):
        super().__init__(dataset)
        if not dataset.classification:
            raise ValueError("Tail identity label noise is now available only for classification datasets.")

        num_labels = dataset.num_classes
        labels = np.array(dataset.labels)
        unique_labels, label_counts = np.unique(labels, return_counts=True)

        sorted_idx = np.argsort(label_counts)
        if noise_large_ids:
            sorted_idx = sorted_idx[::-1]
        unique_labels = unique_labels[sorted_idx]
        label_counts = label_counts[sorted_idx]

        label_cum_fraction = np.cumsum(label_counts / sum(label_counts))
        thresh_label_idx = np.argmin(np.abs(label_cum_fraction - noise_fraction))
        labels_to_noise = unique_labels[:thresh_label_idx + 1]
        indices = np.arange(len(dataset))
        indices_to_noise = np.hstack([indices[labels == label] for label in labels_to_noise])

        print(f"Actual noise fraction: {len(indices_to_noise) / len(dataset)}")
        print(f"Actual noisy identity fraction: {len(labels_to_noise) / len(unique_labels)}")

        with tmp_seed(seed):
            self._labels = np.array(dataset.labels)
            if preserve_non_tail_classes:
                self._labels[indices_to_noise] = np.random.choice(labels_to_noise, len(indices_to_noise))
            else:
                self._labels[indices_to_noise] = np.random.randint(0, num_labels, len(indices_to_noise))

    @property
    def labels(self):
        """Get dataset labels array.

        Labels are integers in the range [0, N-1].
        """
        return self._labels

    def __getitem__(self, index):
        """Get element of the dataset.

        Classification dataset returns tuple (image, label).
        Verification dataset returns ((image1, image2), label).

        """
        item = self.dataset[index]
        return (item[0], self._labels[index]) + item[2:]
