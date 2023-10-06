import random
from collections import defaultdict

import numpy as np
import torch

from ..torch import tmp_seed


class UniformLabelsSampler:
    """Sample labels with equal probabilities."""
    def __init__(self, labels, labels_per_batch, num_batches):
        self._labels = set(labels)
        self._labels_per_batch = labels_per_batch
        self._num_batches = num_batches
        if len(self._labels) < labels_per_batch:
            raise ValueError("Can't sample equal number of labels. Batch is too large.")

    def __iter__(self):
        labels = list(self._labels)
        random.shuffle(labels)
        i = 0
        for _ in range(self._num_batches):
            if i + self._labels_per_batch > len(labels):
                random.shuffle(labels)
                i = 0
            yield list(labels[i:i + self._labels_per_batch])
            i += self._labels_per_batch


class BalancedLabelsSampler:
    """Sample labels with probabilities equal to labels frequency."""
    def __init__(self, labels, labels_per_batch, num_batches):
        counts = np.bincount(labels)
        self._probabilities = counts / np.sum(counts)
        self._labels_per_batch = labels_per_batch
        self._num_batches = num_batches

    def __iter__(self):
        for _ in range(self._num_batches):
            batch = np.random.choice(len(self._probabilities), self._labels_per_batch, p=self._probabilities, replace=False)
            yield list(batch)


class ShuffledClassBalancedBatchSampler(torch.utils.data.Sampler):
    """Sampler which extracts balanced number of samples for each class.

    Args:
        data_source: Source dataset. Labels field must be implemented.
        batch_size: Required batch size.
        samples_per_class: Number of samples for each class in the batch.
            Batch size must be a multiple of samples_per_class.
        uniform: If true, sample labels uniformly. If false, sample labels according to frequency.
        subset_classes: Use a fraction of classes during training. Must be in the range [0, 1].
    """

    def __init__(self, data_source, batch_size, samples_per_class, uniform=False, subset_classes=1):
        if batch_size > len(data_source):
            raise ValueError("Dataset size {} is too small for batch size {}.".format(
                len(data_source), batch_size))
        if batch_size % samples_per_class != 0:
            raise ValueError("Batch size must be a multiple of samples_per_class, but {} != K * {}.".format(
                batch_size, samples_per_class))

        self._source_len = len(data_source)
        self._batch_size = batch_size
        self._labels_per_batch = self._batch_size // samples_per_class
        self._samples_per_class = samples_per_class
        labels = np.asarray(data_source.labels)
        valid_labels = labels
        by_label = defaultdict(list)
        for i, label in enumerate(labels):
            by_label[label].append(i)
        if subset_classes != 1:
            all_labels = list(sorted(by_label))
            with tmp_seed(0):
                random.shuffle(all_labels)
            all_labels = set(all_labels[:int(len(all_labels) * subset_classes)])
            valid_labels = [label for label in labels if label in all_labels]
            by_label = {label: items for label, items in by_label.items() if label in all_labels}
        self._by_label = by_label
        if self._labels_per_batch > len(self._by_label):
            raise ValueError("Can't sample {} classes from dataset with {} classes.".format(
                self._labels_per_batch, len(self._by_label)))

        label_sampler_cls = UniformLabelsSampler if uniform else BalancedLabelsSampler
        self._label_sampler = label_sampler_cls(valid_labels, self._labels_per_batch,
                                                num_batches=len(self))


    @property
    def batch_size(self):
        return self._batch_size

    def __iter__(self):
        for labels in self._label_sampler:
            batch = []
            for label in labels:
                batch.extend(np.random.choice(self._by_label[label], size=self._samples_per_class, replace=True))
            yield batch

    def __len__(self):
        return self._source_len // self._batch_size


class SameClassMixupCollator:
    """Applies same-class mixup to a batch from base sampler."""

    def __call__(self, values):
        images, labels = torch.utils.data._utils.collate.default_collate(values)
        return self._mixup(images, labels)

    def _mixup(self, images, labels):
        if isinstance(images, (list, tuple)):
            raise ValueError("Expected classification dataset for mixup.")
        cpu_labels = labels.long().cpu().numpy()
        by_label = defaultdict(list)
        for i, label in enumerate(cpu_labels):
            by_label[label].append(i)
        alt_indices = [random.choice(by_label[label]) for label in cpu_labels]
        alt_indices = torch.tensor(alt_indices, dtype=torch.long, device=labels.device)
        alt_images = images[alt_indices]
        weights = torch.rand(len(labels)).reshape(-1, 1, 1, 1)
        new_images = images * weights + alt_images * (1 - weights)
        return new_images, labels
