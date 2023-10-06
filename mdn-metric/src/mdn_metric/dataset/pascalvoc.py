import numpy as np
from torchvision.datasets import VOCDetection
from .common import Dataset


class PASCALVOCDataset(Dataset):
    """PASCAL VOC 2007 dataset class: http://host.robots.ox.ac.uk/pascal/VOC/

    Args:
        root: Dataset root.
        split: Dataset split to use. Should be one of "train", "val", "test".
        download: Whether to download dataset from the Internet.
    """
    def __init__(self, root, split="train", download=False):
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown dataset split {split}. Should be in 'train', 'val', 'test'.")
        self.data = VOCDetection(root, year="2007", download=download, image_set=split)

        classnames = set()
        for image in self.data:
            for obj in image[1]["annotation"]["object"]:
                classnames.add(obj["name"])

        classnames = sorted(list(classnames))
        classname2label = dict(zip(classnames, list(range(len(classnames)))))
        self.classname2label = classname2label

        labels = np.zeros((len(self.data), len(self.classname2label)), dtype=np.int64)
        for i, image in enumerate(self.data):
            for obj in image[1]["annotation"]["object"]:
                labels[i][self.classname2label[obj["name"]]] = 1

        self.image_labels = labels

    def __getitem__(self, item):
        image, _ = self.data[item]
        label = self.labels[item]
        return image, label

    @property
    def classification(self):
        """Whether dataset is classification or verification."""
        return True

    @property
    def multilabel(self):
        """Whether dataset is multilabel."""
        return True

    @property
    def openset(self):
        """Whether dataset is for open-set or closed-set classification."""
        return False

    @property
    def labels(self):
        """Get one-hot labels matrix."""
        return self.image_labels

    @property
    def num_classes(self):
        """Get total number of classes."""
        return self.image_labels.shape[1]
