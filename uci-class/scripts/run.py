"""Train UCI dataset classifier."""
import argparse
import math
from collections import defaultdict
from copy import deepcopy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import numpy as np
import random
import torch

from exact_pytorch import EXACTLoss, Relaxed01Loss, GradientNormalizer
from uci_class import UCI_DATASETS, split_crossval


EPS = 1e-6


def parse_arguments():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("dataset", choices=list(UCI_DATASETS))
    parser.add_argument("--root", help="Datasets root.")
    parser.add_argument("-c", "--command", help="One of `train`, `hopt`, and `eval`", choices=["train", "train-val", "hopt", "eval"], default="train")
    parser.add_argument("--method", help="Type of the loss function.", default="exact",
                        choices=["exact", "xent", "hinge", "beta-bernoulli", "sigmoid", "poly", "sklearn"])
    parser.add_argument("--seed", help="Random seed.", type=int, default=0)
    parser.add_argument("--num-seeds", help="Number of random seeds for evaluation.", type=int, default=5)
    parser.add_argument("-r", "--regularization", help="Weights penalty.", type=float, default=0)
    parser.add_argument("--lr", help="Initial learning rate.", type=float, default=0.5)
    parser.add_argument("--clip", help="Gradient clipping.", type=float)
    parser.add_argument("--batch-size", help="PyTorch training batch size.", type=int, default=256)
    parser.add_argument("-m", "--margin", help="EXACT or Hinge margin.", type=float)
    parser.add_argument("-n", "--num-epochs", help="The number of training epochs.", type=int, default=16)
    parser.add_argument("--min-lr", help="The minimum value of LR at the last epoch.", type=float, default=0.0001)
    parser.add_argument("--init-std", help="Scale initial weights.", type=float, default=10)
    parser.add_argument("--min-std", help="Maximum EXACT scale.", type=float, default=0.01)
    parser.add_argument("--normalize-weights", help="Normalize weights.", action="store_true")
    parser.add_argument("--cross-validation", help="Use cross-validation scheme.", action="store_true")
    parser.add_argument("-v", "--verbose", help="Log more data on hopt", action="store_true")
    args = parser.parse_args()
    return args


def one_hot(labels):
    num_classes = np.max(labels) + 1
    result = np.zeros((len(labels), num_classes), dtype=np.bool)
    result[np.arange(len(labels)), labels] = 1
    return result


def compute_metrics(scores, labels, thresholds):
    binary_labels = one_hot(labels)
    preds = np.argmax(scores, axis=1)
    binary_preds = (scores > thresholds).astype(np.int)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(binary_labels, binary_preds, average="macro"),
        "macro_roc_auc": roc_auc_score(binary_labels, scores, average="macro"),
        "weighted_macro_roc_auc": roc_auc_score(binary_labels, scores, average="weighted"),
    }


def compute_metrics_pytorch(X, y, model, thresholds):
    with torch.no_grad():
        scores = model(torch.tensor(X, dtype=torch.float)).cpu().numpy()
    return compute_metrics(scores, y, thresholds)


def compute_accuracy_pytorch(X, y, model):
    return (model(torch.tensor(X, dtype=torch.float)).argmax(1) == torch.tensor(y)).float().mean().item()


def get_thresholds(scores, labels):
    num_classes = np.max(labels) + 1
    thresholds = np.zeros(num_classes)
    counts = np.arange(1, len(scores) + 1)  # (B).
    for i in range(num_classes):
        bin_labels = (labels == i).astype(np.int)
        total_positives = bin_labels.sum()
        if total_positives == 0:
            continue
        bin_scores = scores[:, i]
        order = np.argsort(bin_scores)
        bin_scores = bin_scores[order]  # (B).
        bin_labels = bin_labels[order]  # (B).
        n_positives = np.cumsum(bin_labels)  # (B).
        accuracies = (counts - n_positives + total_positives - n_positives) / len(scores)  # (B).
        thresholds[i] = bin_scores[np.argmax(accuracies)]
    return thresholds


class RepeatDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, n=1):
        super().__init__()
        self._dataset = dataset
        self._n = n

    def __len__(self):
        return len(self._dataset) * self._n

    def __getitem__(self, index):
        return self._dataset[index % len(self._dataset)]


class HingeLoss:
    def __init__(self, margin=1):
        self._margin = margin

    def __call__(self, logits, labels):
        """Compute Hinge loss.

        Args:
            logits: Logits tensor with shape (*, N).
            labels: Integer labels with shape (*).

        Returns:
            Loss value.
        """
        n = logits.shape[-1]
        gt_logits = logits.take_along_dim(labels.unsqueeze(-1), -1)  # (*, 1).
        alt_mask = labels.unsqueeze(-1) != torch.arange(n, device=logits.device)  # (*, N).
        loss = (self._margin - gt_logits + logits).clip(min=0)[alt_mask].mean()
        return loss


class Model(torch.nn.Linear):
    def __init__(self, num_features, num_classes, normalize_weights=False):
        super().__init__(num_features, num_classes - 1)
        self._normalize_weights = normalize_weights

    def forward(self, x):
        if self._normalize_weights:
            norm = torch.linalg.norm(torch.cat((self.weight.flatten(), self.bias.flatten())))
            weight = self.weight / norm
            bias = self.bias / norm
        else:
            weight, bias = self.weight, self.bias
        logits = torch.nn.functional.linear(x, weight, bias)
        logits = torch.cat([torch.zeros_like(logits[:, :1]), logits], dim=-1)
        return logits


def run_pytorch(X, y, X_val, y_val, X_test, y_test, args, verbose=True):
    model = Model(X.shape[-1], max(y) + 1, normalize_weights=args.normalize_weights)
    if verbose:
        for name, p in model.named_parameters():
            print(name, p.shape)
    dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float), torch.tensor(y))
    repeat = int(500 * args.batch_size / len(dataset))
    dataset = RepeatDataset(dataset, n=repeat)  # Increase epoch size.
    loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=args.batch_size, drop_last=True)
    assert len(list(model.parameters())) == 2
    optimizer = torch.optim.SGD([{"params": [model.weight], "weight_decay": args.regularization},
                                 {"params": [model.bias]}],
                                lr=args.lr, momentum=0.9)
    if args.min_lr < args.lr:
        lr_gamma = (args.min_lr / args.lr) ** (1 / (args.num_epochs - 1))
    else:
        lr_gamma = 1
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=lr_gamma)
    if args.method == "exact":
        criterion = EXACTLoss(margin=args.margin)
    elif args.method == "exact-softmax":
        criterion = EXACTLoss(margin=args.margin, shape="softmax")
    elif args.method == "xent":
        criterion = torch.nn.CrossEntropyLoss()
    elif args.method == "hinge":
        criterion = HingeLoss(margin=args.margin if args.margin is not None else 1)
    elif args.method == "beta-bernoulli":
        criterion = Relaxed01Loss("beta-bernoulli")
    elif args.method == "sigmoid":
        criterion = Relaxed01Loss("sigmoid")
    elif args.method == "poly":
        criterion = Relaxed01Loss("poly")
    else:
        raise ValueError(args.method)
    grad_normalizer = GradientNormalizer() if args.clip is None else (lambda p: torch.nn.utils.clip_grad_norm_(p, args.clip))
    epoch = 0
    for i in tqdm(range(args.num_epochs), disable=verbose):
        epoch += repeat
        model.train()
        criterion_kwargs = {}
        if args.method in ["exact", "exact-softmax", "beta-bernoulli", "sigmoid", "poly"]:
            criterion_kwargs["temperature"] = args.init_std * (args.min_std / args.init_std) ** (-i / args.num_epochs)
        losses = []
        accuracies = []
        logits_stds = []
        grad_norms = []
        for features, labels in loader:
            logits = model(features)
            loss = criterion(logits, labels, **criterion_kwargs)
            optimizer.zero_grad()
            loss.backward()
            grad_norm = grad_normalizer(model.parameters())
            optimizer.step()
            losses.append(loss.item())
            accuracies.append((logits.argmax(-1) == labels).float().sum().item())
            logits_stds.append(logits.std().item())
            grad_norms.append(grad_norm)
        train_accuracy = np.sum(accuracies) / len(loader) / args.batch_size
        train_loss = np.mean(losses)
        grad_norm = np.mean(grad_norms)
        model.eval()
        val_accuracy = compute_accuracy_pytorch(X_val, y_val, model)
        test_accuracy = compute_accuracy_pytorch(X_test, y_test, model)
        if verbose:
            print("{}\tTest acc {:.5f}\tVal acc {:.5f}\tTrain acc {:.5f}\tLoss {:.5f}\tSTD {:.3f}\tLogits STD {:.3f}\tLR {:.5f}\tGRAD {:.3f}".format(
                epoch, test_accuracy, val_accuracy, train_accuracy, train_loss,
                criterion_kwargs.get("std", -1),
                np.mean(logits_stds),
                optimizer.param_groups[0]["lr"],
                grad_norm))
        scheduler.step()
    if verbose:
        print("W", model.weight.data.squeeze().numpy())
        print("B", model.bias.data.numpy())
    model.eval()
    with torch.no_grad():
        thresholds = get_thresholds(model(torch.tensor(X_val, dtype=torch.float)).cpu().numpy(), y_val)
    metrics = {
        "train": compute_metrics_pytorch(X, y, model, thresholds),
        "val": compute_metrics_pytorch(X_val, y_val, model, thresholds),
        "test": compute_metrics_pytorch(X_test, y_test, model, thresholds)
    }
    if verbose:
        for split, split_metrics in metrics.items():
            for name, metric in split_metrics.items():
                print(split, name, metric)
    return metrics


def run_sklearn(X, y, X_val, y_val, X_test, y_test, args, verbose=True):
    model = LogisticRegression(C=1 / (args.regularization or 1),
                               random_state=args.seed,  # Practically useless.
                               max_iter=100000,
                               penalty="none" if args.regularization == 0 else "l2")
    model.fit(X, y)
    thresholds = get_thresholds(model.predict_log_proba(X_val), y_val)
    metrics = {
        "train": compute_metrics(model.predict_log_proba(X), y, thresholds),
        "val": compute_metrics(model.predict_log_proba(X_val), y_val, thresholds),
        "test": compute_metrics(model.predict_log_proba(X_test), y_test, thresholds)
    }
    if verbose:
        for split, split_metrics in metrics.items():
            for name, metric in split_metrics.items():
                print(split, name, metric)
    return metrics


def train(args, split_valset=False):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset = UCI_DATASETS[args.dataset](args.root, split="train")
    X, y = dataset.X, dataset.y
    C = max(y) + 1

    try:
        testset = UCI_DATASETS[args.dataset](args.root, split="test")
        X_t, y_t = testset.X, testset.y
    except ValueError:
        X, X_t, y, y_t = train_test_split(X, y, test_size=0.2, random_state=0)

    if split_valset:
        if args.cross_validation:
            X, X_v, y, y_v = split_crossval(X, y, args.seed % args.num_seeds, args.num_seeds)
        else:
            X, X_v, y, y_v = train_test_split(X, y, test_size=0.2, random_state=0)
    else:
        X_v, y_v = X, y  # Use train for validation.

    mean = np.mean(X, axis=0, keepdims=True)
    std = np.maximum(np.std(X, axis=0, keepdims=True), 1e-6)

    X = (X - mean) / std
    X_t = (X_t - mean) / std
    X_v = (X_v - mean) / std

    if args.verbose:
        print("CLASSES", C)
        print("TRAIN", X.shape, y.shape)
        print("TEST", X_t.shape, y_t.shape)

    if args.method == "sklearn":
        metrics = run_sklearn(X, y, X_v, y_v, X_t, y_t, args,
                              verbose=args.verbose)
    else:
        metrics = run_pytorch(X, y, X_v, y_v, X_t, y_t, args,
                              verbose=args.verbose)

    return metrics


def validate(args, split_valset=False):
    by_metric = {k: defaultdict(list) for k in ["train", "val", "test"]}
    for seed in range(args.num_seeds):
        trial_args = deepcopy(args)
        trial_args.seed = seed
        trial_args.verbose = False
        metrics = train(trial_args, split_valset=split_valset)
        for split, split_metrics in metrics.items():
            for name, metric in split_metrics.items():
                by_metric[split][name].append(metric)
    if args.verbose:
        print("Seeds:", args.num_seeds)
        for split, split_metrics in metrics.items():
            for name, values in split_metrics.items():
                print(split, name, np.mean(values), "+-", np.std(values))
    return {split: {k: np.mean(v) for k, v in split_metrics.items()} for split, split_metrics in by_metric.items()}


def hopt(args):
    random.seed(args.seed)
    params = {
        "seed": list(range(10)),
        "regularization": [0, 0.0001, 0.001, 0.01, 0.1, 1]
    }
    if args.method not in {"exact", "sklearn"}:
        params["clip"] = [0.01, 0.1, 1, 10]
    if args.method not in {"sklearn"}:
        params["lr"] = [0.01, 0.05, 0.1, 0.5, 1, 5]
    if args.method in {"exact", "hinge"}:
        params["margin"] = [None, 0, 0.1, 0.5, 1, 5, 10]
    trials = []
    for _ in range(20):
        trial_params = {}
        for name, values in params.items():
            value = random.choice(values)
            trial_params[name] = value
        trials.append(trial_params)
    best_params = None
    best_train_accuracy = -1
    best_val_accuracy = -1
    for trial, trial_params in enumerate(trials):
        trial_args = deepcopy(args)
        for name, value in trial_params.items():
            setattr(trial_args, name, value)
        method = validate if args.cross_validation else train
        metrics = method(trial_args, split_valset=True)
        train_accuracy = metrics["train"]["accuracy"]
        val_accuracy = metrics["val"]["accuracy"]
        test_accuracy = metrics["test"]["accuracy"]
        if ((val_accuracy > best_val_accuracy) or
            (abs(val_accuracy - best_val_accuracy) < EPS and (train_accuracy > best_train_accuracy))):
            best_train_accuracy = train_accuracy
            best_val_accuracy = val_accuracy
            best_params = trial_params
            print("Best parameters:", trial_params)
        print("Trial", trial, "Train", train_accuracy, "Test", test_accuracy, "Val", val_accuracy, "Best val accuracy", best_val_accuracy)
    print("Best val accuracy:", best_val_accuracy)
    print("Best parameters:")
    print(best_params)


if __name__ == "__main__":
    args = parse_arguments()
    if args.command == "train":
        args.verbose = True
        train(args)
    elif args.command == "train-val":
        args.verbose = True
        train(args, split_valset=True)
    elif args.command == "hopt":
        hopt(args)
    elif args.command == "eval":
        args.verbose = True
        validate(args)
    else:
        raise ValueError(args.command)
