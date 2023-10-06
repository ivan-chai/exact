import os
import numpy as np
from tqdm import tqdm
from torchvision.datasets import CocoDetection
from .common import Dataset


class MSCOCODataset(Dataset):
    """MS COCO 2014 dataset class: https://cocodataset.org/

    Args:
        root: Dataset root.
        split: Dataset split to use. Should be one of "train", "val".
    """
    def __init__(self, root, split="train"):
        if split not in ("train", "val"):
            raise ValueError(f"Unknown dataset split {split}. Should be in 'train', 'val'.")

        images_path = os.path.join(root, f"{split}2014")
        annotation_path = os.path.join(root, f"annotations/instances_{split}2014.json")
        self.data = CocoDetection(images_path, annotation_path)

        labels_path = os.path.join(root, f"labels_{split}.npy")
        if os.path.exists(labels_path):
            labels = np.load(labels_path)
        else:
            classes = set()
            for image in tqdm(self.data):
                for obj in image[1]:
                    classes.add(obj["category_id"])

            classes = sorted(list(classes))
            class2label = dict(zip(classes, list(range(len(classes)))))
            self.class2label = class2label

            labels = np.zeros((len(self.data), len(self.class2label)), dtype=np.int64)
            for i, image in tqdm(enumerate(self.data)):
                for obj in image[1]:
                    labels[i][self.class2label[obj["category_id"]]] = 1
            if os.access(labels_path, os.W_OK):
                np.save(labels_path, labels)

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
