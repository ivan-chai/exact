"""Configurable PyTorch layers and modules."""

from .embedder import IdentityEmbedder, CNNEmbedder
from .distribution import DiracDistribution, GMMDistribution, VMFDistribution
from .classifier import LinearClassifier, LinearBayesClassifier, ArcFaceClassifier, CosFaceClassifier
from .classifier import LogLikeClassifier, VMFClassifier, SPEClassifier, ScorerClassifier
from .classifier import BatchSoftmaxClassifier, CentroidFreeFMSClassifier, HyperbolicClassifier
from .scorer import DotProductScorer, CosineScorer, ExpectedCosineScorer, NegativeL2Scorer, HyperbolicScorer
from .scorer import MutualLikelihoodScorer, FeatureMatchingScorer, HIBScorer
