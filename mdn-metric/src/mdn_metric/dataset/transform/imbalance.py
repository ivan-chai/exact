import numpy as np

from ..common import DatasetWrapper, Dataset
from ...torch import tmp_seed


class ImbalancedDataset(DatasetWrapper):
    """
    Dataset wrapper for creating imbalanced datasets.
    Args:
        dataset: Dataset for applying imbalance.
        distribution_type: Class size distribution type. One of `exp` or `step`.
        imbalance_ratio: Measure of dataset imbalance: size(the smallest class) / size(the largest class).
        random_seed: Random seed.
        resampling_type: One of `legacy` or `max_entropy`. `legacy` is taken from https://github.com/kaidic/LDAM-DRW/blob/master/imbalance_cifar.py.
    """
    def __init__(self, dataset, distribution_type="exp",
                 resampling_type="legacy", imbalance_ratio=0.01, random_seed=0):
        if resampling_type not in ("legacy", "max_entropy"):
            raise ValueError("Unknown resampling type. Should be `legacy` or `max_entropy`.")
        # TODO implement `max_entropy` resampling. Sort classes by size and upsample each of then to a target dist.
        if resampling_type == "max_entropy":
            raise NotImplementedError("`max_entropy` mode is not implemented yet.")
        super().__init__(dataset)
        with tmp_seed(random_seed):
            self.class_sizes = self._get_class_sizes(dataset.num_classes, distribution_type, imbalance_ratio)
            self._indices, self._labels = self._get_indices_and_labels(self.class_sizes)

    def _get_class_sizes(self, num_classes, distribution_type, imbalance_ratio):
        max_class_size = len(self._dataset) / num_classes
        class_sizes = []
        if distribution_type == "exp":
            for cls_idx in range(num_classes):
                num = max_class_size * (imbalance_ratio ** (cls_idx / (num_classes - 1.0)))
                class_sizes.append(int(num))
        elif distribution_type == "step":
            for cls_idx in range(num_classes // 2):
                class_sizes.append(int(max_class_size))
            for cls_idx in range(num_classes - num_classes // 2):
                class_sizes.append(int(max_class_size * imbalance_ratio))
        else:
            raise ValueError("Unknown imbalanced class distribution type. Should be `exp` or `step`.")
        return class_sizes

    def _get_indices_and_labels(self, images_per_class):
        new_data_indices = []
        new_labels = []
        labels = np.array(self._dataset.labels, dtype=np.int64)
        unique_labels = np.unique(labels)
        for label, num_images in zip(unique_labels, images_per_class):
            same_class_indices = np.where(labels == label)[0]
            np.random.shuffle(same_class_indices)
            if num_images <= len(same_class_indices):
                selected_idx = same_class_indices[:num_images]
            else:
                selected_idx = np.hstack([same_class_indices,
                                          np.random.choice(same_class_indices, num_images - len(same_class_indices))])
            new_data_indices.append(selected_idx)
            new_labels.extend([label, ] * num_images)
        new_data_indices = np.hstack(new_data_indices)
        new_labels = np.array(new_labels)
        return new_data_indices, new_labels

    @property
    def labels(self):
        return self._labels

    def __getitem__(self, item):
        return self.dataset[self._indices[item]][0], self._labels[item]

    def __len__(self):
        return len(self._labels)
