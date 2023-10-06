import os
import tempfile
import urllib.parse
import urllib.request
import zipfile
from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch
from scipy.io.arff import loadarff


@contextmanager
def download(url, root=None, prefix=None):
    if root is None:
        with tempfile.TemporaryDirectory() as root:
            with download(url, root) as path:
                yield path
        return
    os.path.basename(url)
    filename = os.path.basename(urllib.parse.urlparse(url).path)
    if prefix is not None:
        filename = prefix + "-" + filename
    path = os.path.join(root, filename)
    if not os.path.isfile(path):
        try:
            urllib.request.urlretrieve(url, path)
        except Exception:
            if os.path.isfile(path):
                os.remove(path)
            raise
    yield path


@contextmanager
def unzip(filename):
    with tempfile.TemporaryDirectory() as root:
        with zipfile.ZipFile(filename, "r") as zfp:
            zfp.extractall(root)
        yield root



class UCIDatasetBase(torch.utils.data.Dataset):
    """Base class for UCI datasets with helper tools and basic interface."""

    def __init__(self, X, y):
        if len(X) != len(y):
            raise ValueError("Data and labels size mismatch.")
        super().__init__()
        self.X, self.y = X, y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    @staticmethod
    def extract_features_labels(data, labels_column):
        data = data.loc[data[labels_column].notna()].copy()
        for c in data.columns:
            try:
                data[c] = pd.to_numeric(data[c])
            except ValueError:
                pass
        labels = data[labels_column]
        if not pd.api.types.is_numeric_dtype(labels):
            classes = set(labels)
            mapping = {c: i for i, c in enumerate(sorted(classes))}
            labels = labels.map(lambda c: mapping[c])
        labels = labels.to_numpy()
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("Need integer labels.")
        labels -= np.min(labels)
        features = UCIDatasetBase.prepare_features(data.loc[:, data.columns != labels_column])
        return features, labels

    @staticmethod
    def prepare_features(data):
        data = UCIDatasetBase.one_hot_categorical(data).astype(float)
        data = data.fillna(data.mean())
        data = data.dropna(axis=1)
        return data.to_numpy()

    @staticmethod
    def one_hot_categorical(data):
        for c in list(data.columns):
            column = data[c]
            if not pd.api.types.is_numeric_dtype(column):
                column = column.fillna("nan")
                if len(set(column)) > 50:
                    raise ValueError("Too many categorical values for the column {} (values: {}).".format(c, set(column)))
                data = data.drop(c, axis=1).join(pd.get_dummies(column).add_prefix("{}-".format(c)))
        return data


class BalanceScaleDataset(UCIDatasetBase):
    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data"

    def __init__(self, root=None, split="train"):
        if split != "train":
            raise ValueError("No {} split for {} dataset.".format(split, type(self).__name__))
        with download(self.URL, root, "balance-scale") as filename:
            data = pd.read_csv(filename, header=None)
        X, y = self.extract_features_labels(data, labels_column=0)
        super().__init__(X, y)


class GlassDataset(UCIDatasetBase):
    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data"

    def __init__(self, root=None, split="train"):
        if split != "train":
            raise ValueError("No {} split for {} dataset.".format(split, type(self).__name__))
        with download(self.URL, root, "glass") as filename:
            data = pd.read_csv(filename, header=None)
        X, y = self.extract_features_labels(data, labels_column=data.columns[-1])
        super().__init__(X, y)


class WineDataset(UCIDatasetBase):
    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"

    def __init__(self, root=None, split="train"):
        if split != "train":
            raise ValueError("No {} split for {} dataset.".format(split, type(self).__name__))
        with download(self.URL, root, "wine") as filename:
            data = pd.read_csv(filename, header=None)
        X, y = self.extract_features_labels(data, labels_column=0)
        super().__init__(X, y)


class MiceProteinDataset(UCIDatasetBase):
    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00342/Data_Cortex_Nuclear.xls"

    def __init__(self, root=None, split="train"):
        if split != "train":
            raise ValueError("No {} split for {} dataset.".format(split, type(self).__name__))
        with download(self.URL, root, "mice-protein") as filename:
            data = pd.read_excel(filename)
        data = data.drop(["MouseID", "Genotype", "Treatment", "Behavior"], axis=1)
        X, y = self.extract_features_labels(data, labels_column="class")
        super().__init__(X, y)


class AdultDataset(UCIDatasetBase):
    URL_TRAIN = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    URL_TEST = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

    def __init__(self, root=None, split="train"):
        if split not in ["train", "test"]:
            raise ValueError("No {} split for {} dataset.".format(split, type(self).__name__))

        with download(self.URL_TRAIN, root, "adult") as filename:
            trainset = pd.read_csv(filename, header=None)
        with download(self.URL_TEST, root, "adult") as filename:
            testset = pd.read_csv(filename, header=None, skiprows=1)
            testset[14] = testset[14].map(lambda v: v.strip("."))  # Fix train/test label mismatch.
        data = pd.concat([trainset, testset], ignore_index=True)
        for c in data.columns:
            data[c] = data[c].map(lambda v: (None if v == "?" else v))
        X, y = self.extract_features_labels(data, labels_column=14)
        trainsize = trainset.shape[0]
        if split == "train":
            X, y = X[:trainsize], y[:trainsize]
        else:
            X, y = X[trainsize:], y[trainsize:]
        super().__init__(X, y)


class AnnealingDataset(UCIDatasetBase):
    URL_TRAIN = "https://archive.ics.uci.edu/ml/machine-learning-databases/annealing/anneal.data"
    URL_TEST = "https://archive.ics.uci.edu/ml/machine-learning-databases/annealing/anneal.test"

    def __init__(self, root=None, split="train"):
        if split not in ["train", "test"]:
            raise ValueError("No {} split for {} dataset.".format(split, type(self).__name__))

        with download(self.URL_TRAIN, root, "annealing") as filename:
            trainset = pd.read_csv(filename, header=None, na_values=["?"])
        with download(self.URL_TEST, root, "annealing") as filename:
            testset = pd.read_csv(filename, header=None, na_values=["?"])
        data = pd.concat([trainset, testset], ignore_index=True)
        X, y = self.extract_features_labels(data, labels_column=data.columns[-1])
        trainsize = trainset.shape[0]
        if split == "train":
            X, y = X[:trainsize], y[:trainsize]
        else:
            X, y = X[trainsize:], y[trainsize:]
        super().__init__(X, y)


class BreastCancerWisconsinDataset(UCIDatasetBase):
    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"

    def __init__(self, root=None, split="train"):
        if split != "train":
            raise ValueError("No {} split for {} dataset.".format(split, type(self).__name__))
        with download(self.URL, root, "breastcancerwisconsindiag") as filename:
            data = pd.read_csv(filename, header=None)
        data = data.drop([0], axis=1)
        X, y = self.extract_features_labels(data, labels_column=1)
        super().__init__(X, y)


class CarDataset(UCIDatasetBase):
    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"

    def __init__(self, root=None, split="train"):
        if split != "train":
            raise ValueError("No {} split for {} dataset.".format(split, type(self).__name__))
        with download(self.URL, root, "car") as filename:
            data = pd.read_csv(filename, header=None)
        X, y = self.extract_features_labels(data, labels_column=data.columns[-1])
        super().__init__(X, y)


class DryBeanDataset(UCIDatasetBase):
    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00602/DryBeanDataset.zip"

    def __init__(self, root=None, split="train"):
        if split != "train":
            raise ValueError("No {} split for {} dataset.".format(split, type(self).__name__))
        with download(self.URL, root, "drybean") as filename:
            with unzip(filename) as zroot:
                data = pd.read_excel(os.path.join(zroot, "DryBeanDataset", "Dry_Bean_Dataset.xlsx"))
        X, y = self.extract_features_labels(data, labels_column="Class")
        super().__init__(X, y)


class AuditRiskDataset(UCIDatasetBase):
    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00475/audit_data.zip"

    def __init__(self, root=None, split="train"):
        if split != "train":
            raise ValueError("No {} split for {} dataset.".format(split, type(self).__name__))
        with download(self.URL, root, "auditrisk") as filename:
            with unzip(filename) as zroot:
                data = pd.read_csv(os.path.join(zroot, "audit_data", "audit_risk.csv"))
        X, y = self.extract_features_labels(data, labels_column="Risk")
        super().__init__(X, y)


class CylinderBands(UCIDatasetBase):
    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/cylinder-bands/bands.data"

    def __init__(self, root=None, split="train"):
        if split != "train":
            raise ValueError("No {} split for {} dataset.".format(split, type(self).__name__))
        with download(self.URL, root, "cylinder-bands") as filename:
            data = pd.read_csv(filename, header=None, na_values=["?"])
        data = data.drop([0, 1, 2, 3], axis=1)
        X, y = self.extract_features_labels(data, labels_column=data.columns[-1])
        super().__init__(X, y)


class HeartDiseaseClevelandDataset(UCIDatasetBase):
    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

    def __init__(self, root=None, split="train"):
        if split != "train":
            raise ValueError("No {} split for {} dataset.".format(split, type(self).__name__))
        with download(self.URL, root, "heart-disease-cleveland") as filename:
            data = pd.read_csv(filename, header=None)
        X, y = self.extract_features_labels(data, labels_column=data.columns[-1])
        super().__init__(X, y)


class IrisDataset(UCIDatasetBase):
    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

    def __init__(self, root=None, split="train"):
        if split != "train":
            raise ValueError("No {} split for {} dataset.".format(split, type(self).__name__))
        with download(self.URL, root, "iris") as filename:
            data = pd.read_csv(filename, header=None)
        X, y = self.extract_features_labels(data, labels_column=data.columns[-1])
        super().__init__(X, y)


class LungCancerDataset(UCIDatasetBase):
    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/lung-cancer/lung-cancer.data"

    def __init__(self, root=None, split="train"):
        if split != "train":
            raise ValueError("No {} split for {} dataset.".format(split, type(self).__name__))
        with download(self.URL, root, "lung-cancer") as filename:
            data = pd.read_csv(filename, header=None)
        X, y = self.extract_features_labels(data, labels_column=0)
        super().__init__(X, y)


class VotingDataset(UCIDatasetBase):
    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/voting-records/house-votes-84.data"

    def __init__(self, root=None, split="train"):
        if split != "train":
            raise ValueError("No {} split for {} dataset.".format(split, type(self).__name__))
        with download(self.URL, root, "voting") as filename:
            data = pd.read_csv(filename, header=None)
        X, y = self.extract_features_labels(data, labels_column=0)
        super().__init__(X, y)


class MushroomDataset(UCIDatasetBase):
    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"

    def __init__(self, root=None, split="train"):
        if split != "train":
            raise ValueError("No {} split for {} dataset.".format(split, type(self).__name__))
        with download(self.URL, root, "mushroom") as filename:
            data = pd.read_csv(filename, header=None)
        X, y = self.extract_features_labels(data, labels_column=0)
        super().__init__(X, y)


class AbaloneDataset(UCIDatasetBase):
    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"

    def __init__(self, root=None, split="train"):
        if split != "train":
            raise ValueError("No {} split for {} dataset.".format(split, type(self).__name__))
        with download(self.URL, root, "abalone") as filename:
            data = pd.read_csv(filename, header=None)
        X, y = self.extract_features_labels(data, labels_column=data.columns[-1])
        super().__init__(X, y)


class SpambaseDataset(UCIDatasetBase):
    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"

    def __init__(self, root=None, split="train"):
        if split != "train":
            raise ValueError("No {} split for {} dataset.".format(split, type(self).__name__))
        with download(self.URL, root, "spambase") as filename:
            data = pd.read_csv(filename, header=None)
        X, y = self.extract_features_labels(data, labels_column=data.columns[-1])
        super().__init__(X, y)


class StatLogDataset(UCIDatasetBase):
    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

    def __init__(self, root=None, split="train"):
        if split != "train":
            raise ValueError("No {} split for {} dataset.".format(split, type(self).__name__))
        with download(self.URL, root, "statlog") as filename:
            data = pd.read_csv(filename, header=None, sep="\s+")
        X, y = self.extract_features_labels(data, labels_column=data.columns[-1])
        super().__init__(X, y)


class YeastDataset(UCIDatasetBase):
    URL = "https://www.openml.org/data/download/4644190/file2754771351f4.arff"

    def __init__(self, root=None, split="train"):
        if split != "train":
            raise ValueError("No {} split for {} dataset.".format(split, type(self).__name__))
        with download(self.URL, root, "yeast") as filename:
            data = pd.DataFrame(loadarff(filename)[0])
        X = data.iloc[:, :-14].to_numpy().astype(np.float32)
        y = data.iloc[:, -14:].to_numpy() == b"TRUE"
        super().__init__(X, y)

    @property
    def num_classes(self):
        return 14


class ZooDataset(UCIDatasetBase):
    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/zoo/zoo.data"

    def __init__(self, root=None, split="train"):
        if split != "train":
            raise ValueError("No {} split for {} dataset.".format(split, type(self).__name__))
        with download(self.URL, root, "zoo") as filename:
            data = pd.read_csv(filename, header=None)
        data = data.drop([data.columns[0]], axis=1)
        X, y = self.extract_features_labels(data, labels_column=data.columns[-1])
        super().__init__(X, y)


UCI_DATASETS = {
    "abalone": AbaloneDataset,
    "adult": AdultDataset,
    "annealing": AnnealingDataset,
    "audit-risk": AuditRiskDataset,
    "balance-scale": BalanceScaleDataset,
    "breast-cancer-wisconsin": BreastCancerWisconsinDataset,
    "car": CarDataset,
    "cylinder-bands": CylinderBands,
    "dry-bean": DryBeanDataset,
    "iris": IrisDataset,
    "lung-cancer": LungCancerDataset,
    "glass": GlassDataset,
    "statlog": StatLogDataset,
    "mushroom": MushroomDataset,
    "heart-disease-cleveland": HeartDiseaseClevelandDataset,
    "mice-protein": MiceProteinDataset,
    "spambase": SpambaseDataset,
    "voting": VotingDataset,
    "wine": WineDataset,
    "zoo": ZooDataset
}

UCI_MULTILABEL_DATASETS = {
    "yeast": YeastDataset
}
