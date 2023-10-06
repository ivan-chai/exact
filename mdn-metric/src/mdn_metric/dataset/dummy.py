from .common import Dataset


class EmptyDataset(Dataset):
    def __init__(self, root=None, classification=True, openset=True, multilabel=False):
        super().__init__()
        self._classification = classification
        self._openset = openset
        self._multilabel = multilabel

    @property
    def classification(self):
        """Whether dataset is classification or matching."""
        return self._classification

    @property
    def multilabel(self):
        return self._multilabel

    @property
    def openset(self):
        """Whether dataset is for open-set or closed-set classification."""
        return self._openset

    @property
    def labels(self):
        return []

    def __getitem__(self, index):
        raise IndexError("No items in the dataset.")
