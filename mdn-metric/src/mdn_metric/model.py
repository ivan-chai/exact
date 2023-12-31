from collections import OrderedDict
import numpy as np
import torch

from mdn_metric.config import prepare_config
from .layers import IdentityEmbedder, CNNEmbedder
from .layers import DiracDistribution, GMMDistribution, VMFDistribution
from .layers import DotProductScorer, CosineScorer, ExpectedCosineScorer, NegativeL2Scorer
from .layers import MutualLikelihoodScorer, FeatureMatchingScorer, HIBScorer, HyperbolicScorer
from .layers import LinearClassifier, LinearBayesClassifier, ArcFaceClassifier, CosFaceClassifier, LogLikeClassifier
from .layers import VMFClassifier, SPEClassifier, ScorerClassifier
from .layers import BatchSoftmaxClassifier, CentroidFreeFMSClassifier, HyperbolicClassifier
from .torch import disable_amp, freeze, eval_bn


class Model(torch.nn.Module):
    """Embedder with optional classification head.

    Parts of the model:
    1. Stem (CNN model).
    2. Head (mapping from CNN output to embedding).
    3. Normalizer (batchnorm of embeddings for some models).
    4. Classifier (mapping from embedding to logits).

    Embedding pipeline just predicts distributions of embeddings.
    Classification pipeline matches samples from predicted distribution with target distribution.

    Stages of classification pipeline:
    1. Predict embeddings distribution.
    2. Evaluate negative log likelihood of the learnt target embedding.

    Args:
        num_classes: Number of output classes.
        priors: Precomputed class priors. Priors can be learned on-line or loaded from checkpoint.
        amp_classifier: Whether to use mixed precision for classifier or not.

    Inputs:
        - images: Batch of images with the shape (B, 3, S, S).
        - labels: Target labels used in some scorers during training.

    Outputs:
        Dictionary of:
            - distributions: Predicted distribution parameters.
            - logits (optional): Unnormalized log probabilities of each class.
    """

    DISTRIBUTIONS = {
        "dirac": DiracDistribution,
        "gmm": GMMDistribution,
        "vmf": VMFDistribution
    }

    EMBEDDERS = {
        "cnn": CNNEmbedder,
        "identity": IdentityEmbedder
    }

    SCORERS = {
        "dot": DotProductScorer,
        "cosine": CosineScorer,
        "ecs": ExpectedCosineScorer,
        "l2": NegativeL2Scorer,
        "mls": MutualLikelihoodScorer,
        "fms": FeatureMatchingScorer,
        "hib": HIBScorer,
        "hyperbolic": HyperbolicScorer
    }

    CLASSIFIERS = {
        "linear": LinearClassifier,
        "linear-bayes": LinearBayesClassifier,
        "arcface": ArcFaceClassifier,
        "cosface": CosFaceClassifier,
        "loglike": LogLikeClassifier,
        "vmf-loss": VMFClassifier,
        "spe": SPEClassifier,
        "scorer": ScorerClassifier,
        "batch-softmax": BatchSoftmaxClassifier,
        "centroid-free-fms": CentroidFreeFMSClassifier,
        "hyperbolic": HyperbolicClassifier
    }

    @staticmethod
    def get_default_config(distribution_type="dirac", distribution_params=None,
                           embedder_type="cnn", embedder_params=None,
                           scorer_type="dot", scorer_params=None,
                           classifier_type="linear", classifier_params=None,
                           freeze_classifier=False):
        """Get modle parameters.

        Args:
            distribution_type: Predicted emdedding distribution type ("dirac", "gmm" or "vmf").
            distribution_params: Predicted distribution hyperparameters.
            embedder_type: Type of the embedder network: "cnn" for cnn embedder or "identity"
                if embeddings are directly providided as a model's input.
            embedder_params: Parameters of the network for embeddings distribution estimation.
            scorer_type: Type of verification embeddings scorer ("l2" or "cosine").
            scorer_params: Parameters of the scorer.
            classifier_type: Type of classification embeddings scorer ("linear", "arcface", "cosface", "loglike", "vmf-loss" or "spe").
            classifier_params: Parameters of target distributions and scoring.
            freeze_classifier: If true, freeze classifier parameters (target classes embeddings).
        """
        return OrderedDict([
            ("distribution_type", distribution_type),
            ("distribution_params", distribution_params),
            ("embedder_type", embedder_type),
            ("embedder_params", embedder_params),
            ("scorer_type", scorer_type),
            ("scorer_params", scorer_params),
            ("classifier_type", classifier_type),
            ("classifier_params", classifier_params),
            ("freeze_classifier", freeze_classifier)
        ])

    def __init__(self, num_classes, *, is_multilabel=False, priors=None, amp_classifier=False, config=None):
        super().__init__()
        self._config = prepare_config(self, config)
        self._num_classes = num_classes
        self._amp_classifier = amp_classifier
        self._distribution = self.DISTRIBUTIONS[self._config["distribution_type"]](config=self._config["distribution_params"])
        self._embedder = self.EMBEDDERS[self._config["embedder_type"]](self._distribution,
                                                                       config=self._config["embedder_params"])
        self._scorer = self.SCORERS[self._config["scorer_type"]](self._distribution, config=self._config["scorer_params"])
        if self.classification:
            self._classifier = self.CLASSIFIERS[self._config["classifier_type"]](
                self._distribution, num_classes,
                is_multilabel=is_multilabel, priors=priors,
                config=self._config["classifier_params"]
            )
            if self._config["freeze_classifier"]:
                freeze(self._classifier)

    @property
    def classification(self):
        """Whether model is classification or just embedder."""
        return self._config["classifier_type"] is not None

    @property
    def distribution(self):
        """Distribution used by the model."""
        return self._distribution

    @property
    def num_classes(self):
        """Number of output classes or None for embedder network."""
        return self._num_classes

    @property
    def classifier(self):
        if not self.classification:
            raise AttributeError("Classifier is not available.")
        return self._classifier

    @property
    def embedder(self):
        """Model for embeddings generation."""
        return self._embedder

    @property
    def scorer(self):
        """Embeddings pairwise scorer."""
        return self._scorer

    @property
    def num_parameters(self):
        total = 0
        for p in self.parameters():
            total += np.prod(list(p.shape))
        return total

    def train(self, mode=True):
        super().train(mode)
        if self.classification and self._config["freeze_classifier"]:
            eval_bn(self._classifier)
        return self

    def forward(self, images, labels=None):
        distributions = self._embedder(images)  # (B, N, D).
        result = {"distributions": distributions}
        if self.classification:
            b, n, d = distributions.shape  # Handle ensemble of N models.
            with disable_amp(not self._amp_classifier):
                logits = self._classifier(distributions.reshape(b * n, d).float(), labels, scorer=self.scorer)  # (B * N, C).
                result["logits"] = logits.reshape(b, n, logits.shape[-1])  # (B, N, C).
        return result

    @property
    def has_final_weights(self):
        return self.classification and self.classifier.has_weight

    def get_final_weights(self):
        if not self.classification:
            raise RuntimeError("Target embeddings are available for classification models only.")
        return self._classifier.weight

    @property
    def has_final_bias(self):
        if not self.classification:
            raise RuntimeError("Target bias is available for classification models only.")
        return self._classifier.has_bias

    def get_final_bias(self):
        if not self.classification:
            raise RuntimeError("Target bias is available for classification models only.")
        return self._classifier.bias

    @property
    def has_final_variance(self):
        if not self.classification:
            raise RuntimeError("Target variance is available for classification models only.")
        return self._classifier.has_variance

    def get_final_variance(self):
        if not self.classification:
            raise RuntimeError("Target variance is available for classification models only.")
        return self._classifier.variance

    @property
    def prior_kld(self):
        prior_klds = [self._embedder.prior_kld]
        if self.classification and hasattr(self._classifier, "prior_kld"):
            prior_klds.append(self._classifier.prior_kld)
        prior_klds = [kld for kld in prior_klds if kld is not None]
        return None if not prior_klds else sum(prior_klds[1:], prior_klds[0])

    def statistics(self, results):
        """Compute useful statistics for logging.

        Args:
            results: Model forward pass results.

        Returns:
            Dictionary with floating-point statistics values.
        """
        parameters = results["distributions"]
        stats = OrderedDict()
        stats.update(self.distribution.statistics(parameters))
        stats.update(self.scorer.statistics())
        if self.classification:
            stats.update(self._classifier.statistics())
            logits = results["logits"].detach()
            stats["logits/mean"] = logits.mean()
            stats["logits/std"] = logits.std()
            if self.has_final_variance:
                stats["output_std"] = self.get_final_variance().sqrt()
        if self._embedder.output_scale is not None:
            stats["output_scale"] = self._embedder.output_scale
        return stats
