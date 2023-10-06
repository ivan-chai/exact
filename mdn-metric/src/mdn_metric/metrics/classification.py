from collections import Counter, OrderedDict
from typing import Tuple, Dict

import numpy as np
import torch
from catalyst.core.runner import IRunner
from catalyst.metrics._additive import AdditiveMetric
from catalyst.metrics._metric import ICallbackBatchMetric, ICallbackLoaderMetric
from catalyst.callbacks.metric import BatchMetricCallback, LoaderMetricCallback
from catalyst.utils.distributed import all_gather, get_rank
from catalyst.utils.misc import get_attr
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

from ..config import prepare_config, ConfigError
from ..torch import get_base_module
from .nearest import NearestNeighboursMetrics, GroupedNearestNeighboursMetrics


class DummyMetric(ICallbackLoaderMetric):
    """No metric."""
    def __init__(self, compute_on_call: bool = True, prefix: str = None, suffix: str = None):
        super().__init__(compute_on_call=compute_on_call, prefix=prefix, suffix=suffix)

    def reset(self, num_batches, num_samples) -> None:
        pass

    def update(self) -> None:
        pass

    def compute(self):
        return tuple()

    def compute_key_value(self):
        return {}


class NearestMetric(ICallbackLoaderMetric):
    """Metric interface to Recall, MAP@R, and GAP@R."""

    def __init__(self, config: Dict = None,
                 compute_on_call: bool = True, prefix: str = None, suffix: str = None):
        super().__init__(compute_on_call=compute_on_call, prefix=prefix, suffix=suffix)
        self._config = dict(config)
        self._filter_out_bins = self._config.pop("num_filter_out_bins", 0)
        self._distribution = None
        self._scorer = None
        self.reset(None, None)

    @property
    def distribution(self):
        return self._distribution

    @distribution.setter
    def distribution(self, distribution):
        self._distribution = distribution

    @property
    def scorer(self):
        return self._scorer

    @scorer.setter
    def scorer(self, scorer):
        self._scorer = scorer

    def make_metric(self):
        return NearestNeighboursMetrics(self._distribution, self._scorer, config=self._config)

    def reset(self, num_batches, num_samples) -> None:
        """Resets all fields"""
        self._is_ddp = get_rank() > -1
        self._embeddings = []
        self._targets = []

    def update(self, embeddings: torch.Tensor, targets: torch.Tensor) -> None:
        """Updates metric value with statistics for new data.

        Args:
            embeddings: Tensor with embeddings distribution and shape (B, D).
            targets: Tensor with target labels.
        """
        if embeddings.ndim != 2:
            raise ValueError("Expected embeddings with shape (B, D).")
        self._embeddings.append(embeddings.detach())  # (B, D).
        self._targets.append(targets.detach())

    def compute(self) -> Tuple[torch.Tensor, float, float, float]:
        """Computes nearest neighbours metrics."""
        metrics = self.compute_key_value()
        return [v for k, v in sorted(metrics.items())]

    def compute_key_value(self) -> Dict[str, float]:
        """Computes nearest neighbours metrics."""
        if self.scorer is None:
            raise RuntimeError("Scorer must be set on stage_start.")
        if self.distribution is None:
            raise RuntimeError("Distribution must be set on stage_start.")

        metric = self.make_metric()

        embeddings = torch.cat(self._embeddings)
        targets = torch.cat(self._targets)

        if self._is_ddp:
            embeddings = torch.cat(all_gather(embeddings)).detach()
            targets = torch.cat(all_gather(targets)).detach()

        metrics = metric(embeddings, targets)

        if self.distribution.has_confidences:
            confidences = self.distribution.confidences(embeddings)
            values = torch.sort(confidences)[0]
            # Compute filter-out metrics.
            for fraction in np.linspace(0, 0.9, self._filter_out_bins):
                name = "{:.3f}".format(fraction)
                th = values[int(round((len(values) - 1) * fraction))]
                mask = confidences >= th
                partial_metrics = metric(embeddings[mask], targets[mask])
                for k, v in partial_metrics.items():
                    metrics["filter-out/{}/{}".format(name, k)] = v
        return {self.prefix + k + self.suffix: v for k, v in metrics.items()}


class NearestMetricCallback(LoaderMetricCallback):
    """Callback for Recall, MAP@R, and GAP@R computation.

    Args:
        scorer: Scorer object.
        input_key: Embeddings key.
        target_key: Labels key.
    """

    def __init__(
        self, input_key: str, target_key: str, prefix: str = None, suffix: str = None,
        config: Dict = None
    ):
        super().__init__(
            metric=NearestMetric(config=config, prefix=prefix, suffix=suffix),
            input_key=input_key,
            target_key=target_key
        )

    def on_stage_start(self, runner: "IRunner"):
        model = get_attr(runner, key="model", inner_key="model")
        scorer = get_attr(runner, key="model", inner_key="scorer")
        assert scorer is not None
        self.metric.scorer = scorer
        distribution = get_base_module(model).distribution
        assert distribution is not None
        self.metric.distribution = distribution

    def on_stage_end(self, runner: "IRunner"):
        self.metric.scorer = None
        self.metric.distribution = None


class NearestGroupedMetric(NearestMetric):
    """Metric interface to grouped Recall, MAP@R, and GAP@R."""
    def make_metric(self):
        return GroupedNearestNeighboursMetrics(self._distribution, self._scorer, config=self._config)


class NearestGroupedMetricCallback(NearestMetricCallback):
    """Callback for MAP@R or GAP@R computation.

    Args:
        scorer: Scorer object.
        input_key: Embeddings key.
        target_key: Labels key.
    """

    def __init__(
        self, input_key: str, target_key: str, prefix: str = None, suffix: str = None,
        config: Dict = None
    ):
        super(NearestMetricCallback, self).__init__(
            metric=NearestGroupedMetric(config=config, prefix=prefix, suffix=suffix),
            input_key=input_key,
            target_key=target_key
        )


class ScoresMetric(ICallbackLoaderMetric):
    """Positive scores statistics computation in classification pipeline."""

    def __init__(self, config: Dict = None,
                 compute_on_call: bool = True, prefix: str = None, suffix: str = None):
        super().__init__(compute_on_call=compute_on_call, prefix=prefix, suffix=suffix)
        self._config = config
        self._distribution = None
        self._scorer = None
        self.reset(None, None)

    @property
    def distribution(self):
        return self._distribution

    @distribution.setter
    def distribution(self, distribution):
        self._distribution = distribution

    @property
    def scorer(self):
        return self._scorer

    @scorer.setter
    def scorer(self, scorer):
        self._scorer = scorer

    def reset(self, num_batches, num_samples) -> None:
        """Resets all fields"""
        self._is_ddp = get_rank() > -1
        self._positive_scores = []

    def update(self, embeddings: torch.Tensor, labels: torch.Tensor, final_weights: torch.Tensor) -> None:
        """Updates metric value with statistics for new data.

        Args:
            embeddings: Tensor with embeddings distribution with shape (B, D).
            labels: Tensor with target labels.
            final_weights: Centroids of target classes.
        """
        if self.scorer is None:
            raise RuntimeError("Scorer must be set on stage_start.")
        if self.distribution is None:
            raise RuntimeError("Distribution must be set on stage_start.")
        if final_weights.shape[-1] != self.distribution.num_parameters:
            # Target embeddings are not distributions and can't be matched.
            return {}
        targets = final_weights.detach()[labels]
        assert embeddings.shape == targets.shape
        positive_scores = self.scorer(embeddings.detach(), targets).flatten()  # (B).
        self._positive_scores.append(positive_scores)

    def compute(self) -> Tuple[torch.Tensor, float, float, float]:
        """Computes nearest neighbours metrics."""
        metrics = self.compute_key_value()
        return [v for k, v in sorted(metrics.items())]

    def compute_key_value(self) -> Dict[str, float]:
        """Computes nearest neighbours metrics."""
        if len(self._positive_scores) == 0:
            return {}
        positive_scores = torch.cat(self._positive_scores)  # (B).
        if self._is_ddp:
            positive_scores = torch.cat(all_gather(positive_scores)).detach()
        values = {
            "positive_scores/mean": positive_scores.mean(),
            "positive_scores/std": positive_scores.std()
        }
        return {self.prefix + k + self.suffix: v for k, v in values.items()}


class ScoresMetricCallback(LoaderMetricCallback):
    """Callback for positive scores statistics computation in classification pipeline."""

    def __init__(
        self, input_key: str, target_key: str, final_weights_key: str, prefix: str = None, suffix: str = None,
        config: Dict = None
    ):
        super().__init__(
            metric=ScoresMetric(config=config, prefix=prefix, suffix=suffix),
            input_key={input_key: "embeddings", final_weights_key: "final_weights"},
            target_key={target_key: "labels"}
        )

    def on_stage_start(self, runner: "IRunner"):
        model = get_attr(runner, key="model", inner_key="model")
        scorer = get_attr(runner, key="model", inner_key="scorer")
        assert scorer is not None
        self.metric.scorer = scorer
        distribution = get_base_module(model).distribution
        assert distribution is not None
        self.metric.distribution = distribution

    def on_stage_end(self, runner: "IRunner"):
        self.metric.scorer = None
        self.metric.distribution = None


class QualityMetric(ICallbackLoaderMetric):
    """Compute sample quality estimation metrics."""

    def __init__(self, compute_on_call: bool = True, prefix: str = None, suffix: str = None):
        super().__init__(compute_on_call=compute_on_call, prefix=prefix, suffix=suffix)
        self.reset(None, None)

    def reset(self, num_batches, num_samples) -> None:
        """Resets all fields"""
        self._is_ddp = get_rank() > -1
        self._confidences = []
        self._quality = []

    def update(self, confidences: torch.Tensor, quality: torch.Tensor) -> None:
        """Updates metric value with statistics for new data."""
        self._confidences.append(confidences.detach())
        self._quality.append(quality.detach())

    def compute(self) -> Tuple[torch.Tensor, float, float, float]:
        """Computes nearest neighbours metrics."""
        metrics = self.compute_key_value()
        return [v for k, v in sorted(metrics.items())]

    def compute_key_value(self) -> Dict[str, float]:
        """Computes nearest neighbours metrics."""
        confidences = torch.cat(self._confidences)
        quality = torch.cat(self._quality)

        if self._is_ddp:
            confidences = torch.cat(all_gather(confidences)).detach()
            quality = torch.cat(all_gather(quality)).detach()

        values = {
            "quality_scc": spearmanr(quality.cpu().numpy(), confidences.cpu().numpy())[0]
        }
        return {self.prefix + k + self.suffix: v for k, v in values.items()}


class QualityMetricCallback(LoaderMetricCallback):
    """Callback for sample quality estimation metrics."""

    def __init__(
        self, input_key: str, target_key: str, prefix: str = None, suffix: str = None,
        config: Dict = None
    ):
        if config:
            raise ConfigError("No config available for quality metrics.")
        if (input_key is not None) and (target_key is not None):
            super().__init__(
                metric=QualityMetric(prefix=prefix, suffix=suffix),
                input_key=input_key,
                target_key=target_key
            )
        else:
            super().__init__(
                metric=DummyMetric(),
                input_key={},
                target_key={}
            )


class MAPMetric(ICallbackLoaderMetric):
    """
    Compute Mean Average Precision Metric for multi-label classification.
    """

    def __init__(self, compute_on_call: bool = True, prefix: str = None, suffix: str = None):
        super().__init__(compute_on_call=compute_on_call, prefix=prefix, suffix=suffix)
        self._is_ddp = get_rank() > -1
        self._logits = []
        self._targets = []

    def reset(self, num_batches, num_samples) -> None:
        """Resets all fields"""
        self._is_ddp = get_rank() > -1
        self._logits = []
        self._targets = []

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """Updates metric value with statistics for new data.

        Args:
            logits: Tensor with predicted logits.
            targets: Tensor with target labels.
        """
        self._logits.append(logits.detach().cpu())
        self._targets.append(targets.detach().cpu())

    def compute(self) -> Tuple[torch.Tensor, float, float, float]:
        """Computes nearest neighbours metrics."""
        metrics = self.compute_key_value()
        return [v for k, v in sorted(metrics.items())]

    def compute_key_value(self) -> Dict[str, float]:
        """Computes MAP metric based on it's accumulated state.

        Returns:
            Dict: computed value in key-value format: {"mAP": ...}.
        """
        logits = torch.cat(self._logits)
        targets = torch.cat(self._targets)

        if self._is_ddp:
            logits = torch.cat(all_gather(logits)).detach()
            targets = torch.cat(all_gather(targets)).detach()

        logits = np.asarray(logits)
        targets = np.asarray(targets).astype(np.int64)
        ap = np.zeros((logits.shape[1]))
        for i in range(logits.shape[1]):
            pred = logits[:, i]
            ground = targets[:, i]
            ap[i] = average_precision_score(ground, pred)
        values = {"mAP": ap.mean()}
        return {self.prefix + k + self.suffix: v for k, v in values.items()}


class MAPMetricCallback(LoaderMetricCallback):
    def __init__(
        self, input_key: str, target_key: str, prefix: str = None, suffix: str = None,
        config: Dict = None
    ):
        if config:
            raise ConfigError("No config available for MAP metrics.")
        if (input_key is not None) and (target_key is not None):
            super().__init__(
                metric=MAPMetric(prefix=prefix, suffix=suffix),
                input_key=input_key,
                target_key=target_key
            )
        else:
            super().__init__(
                metric=DummyMetric(),
                input_key={},
                target_key={}
            )


class InvertedAccuracyMetric(ICallbackBatchMetric):
    """
    Compute inverted accuracy (classification of batch elements based on labels).

    If K elements with same label are present in the batch, than recall@K is computed.
    The weight of each class is proportional to the number of elements in this class.
    """

    def __init__(self, compute_on_call: bool = True, prefix: str = None, suffix: str = None):
        super().__init__(compute_on_call=compute_on_call, prefix=prefix, suffix=suffix)
        self._metric = AdditiveMetric()

    def reset(self) -> None:
        """Reset all fields"""
        self._metric.reset()

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """Update metric value with statistics for new data.

        Args:
            logits: Tensor with predicted logits.
            targets: Tensor with target labels.
        """
        if logits.ndim != 2:
            raise NotImplementedError("Expected logits with shape (B, C), got {}.".format(logits.shape))
        if logits.shape[:-1] != targets.shape:
            raise ValueError("Logits and targets shape mismatch.")

        logits = logits.detach()
        class_sizes = Counter(targets.cpu().numpy())
        recall_sum = 0
        for c, k in class_sizes.items():
            _, top_indices = logits[:, c].topk(k)
            recall_sum += (targets[top_indices] == c).sum().item()
        value = recall_sum / len(targets)
        self._metric.update(value, len(targets))
        return value

    def update_key_value(self, logits: torch.Tensor, targets: torch.Tensor):
        value = self.update(logits, targets)
        return {
            self.prefix + "inverted_accuracy" + self.suffix: value
        }

    def compute(self) -> Tuple[torch.Tensor, float, float, float]:
        """Compute nearest neighbours metrics."""
        mean, std = self._metric.compute()
        return mean, std

    def compute_key_value(self) -> Dict[str, float]:
        """Compute inverted accuracy metric."""
        mean, std = self.compute()
        return {
            self.prefix + "inverted_accuracy" + self.suffix: mean,
            self.prefix + "inverted_accuracy" + self.suffix + "/std": std
        }


class InvertedAccuracyMetricCallback(BatchMetricCallback):
    def __init__(
        self, input_key: str, target_key: str, prefix: str = None, suffix: str = None, log_on_batch: bool = True,
        config: Dict = None
    ):
        if config:
            raise ConfigError("No config available for MAP metrics.")
        if (input_key is not None) and (target_key is not None):
            super().__init__(
                metric=InvertedAccuracyMetric(prefix=prefix, suffix=suffix),
                input_key=input_key,
                target_key=target_key,
                log_on_batch=log_on_batch
            )
        else:
            super().__init__(
                metric=DummyMetric(),
                input_key={},
                target_key={}
            )
