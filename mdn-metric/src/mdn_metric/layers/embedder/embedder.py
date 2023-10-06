from collections import OrderedDict

import torch

from ...config import prepare_config
from ...torch import disable_amp, freeze, freeze_bn, eval_bn
from ..bayes import LinearBayes
from .cnn import ResNetModel, CotrainingModel, EfficientNet, PyramidNet, PMModel, CGDModel, PAModel, VGG, TorchVGGModel
from .pooling import DistributionPooling, EnsemblePooling, FlattenPooling, MultiPool2d


class SequentialFP32(torch.nn.Sequential):
    def forward(self, input):
        with disable_amp():
            return super().forward(input.float())


class IdentityEmbedder(torch.nn.Module):
    """Pass input embeddings to the output.

    Inputs:
        embeddings: Tensor with shape (B, D).

    Outputs:
        Embeddings with shape (B, 1, D), where second dimension is an ensemble size.
    """
    @staticmethod
    def get_default_config(head_normalize=True):
        """Get embedder parameters."""
        return OrderedDict([
            ("head_normalize", head_normalize)
        ])

    def __init__(self, distribution, *, config=None):
        super().__init__()
        self._config = prepare_config(self, config)
        self._distribution = distribution
        self._normalizer = distribution.make_normalizer() if self._config["head_normalize"] else None

    @property
    def input_size(self):
        """Pretrained model input image size."""
        raise NotImplementedError("Input size is unavailable for identity embedder.")

    @property
    def in_channels(self):
        raise NotImplementedError("Input channels are unavailable for identity embedder.")

    @property
    def prior_kld(self):
        return None

    def forward(self, embeddings):
        if embeddings.ndim != 2:
            raise ValueError("Expected embeddings matrix, got tensor with shape {}".format(embeddings.shape))
        if embeddings.shape[-1] != self._distribution.num_parameters:
            raise ValueError("Expected embeddings with dimension {}, got {}".format(self._distribution.num_parameters, embeddings.shape[-1]))
        if self._normalizer is not None:
            embeddings = self._normalizer(embeddings)
        return embeddings.unsqueeze(1)  # (B, 1, D).


class CNNEmbedder(torch.nn.Module):
    """Model to map input images to embeddings.

    Embedder computation pipeline:
    1. Stem (CNN model).
    2. Pooling (CNN output spatial aggregation method).
    3. Head (mapping from CNN output to embedding).
    3*. Extra head for uncertainty prediction.
    4. Normalizer (batchnorm of embeddings for some models).

    Inputs:
        images: Tensor with shape (B, C, H, W).

    Outputs:
        Embeddings with shape (B, N, D), where N is an ensemble size.
    """

    MODELS = {
        "resnet18": lambda pretrained: ResNetModel("resnet18", pretrained=pretrained),
        "resnet34": lambda pretrained: ResNetModel("resnet34", pretrained=pretrained),
        "resnet50": lambda pretrained: ResNetModel("resnet50", pretrained=pretrained),
        "resnet101": lambda pretrained: ResNetModel("resnet101", pretrained=pretrained),
        "wide_resnet16_8": lambda pretrained: CotrainingModel("wide_resnet16_8", pretrained=pretrained),
        "wide_resnet50_2": lambda pretrained: ResNetModel("wide_resnet50_2", pretrained=pretrained),
        "wide_resnet101_2": lambda pretrained: ResNetModel("wide_resnet101_2", pretrained=pretrained),
        "wide_resnet28_10": lambda pretrained: CotrainingModel("wide_resnet28_10", pretrained=pretrained),
        "efficientnet_v2_s": lambda pretrained: EfficientNet("efficientnet_v2_s", pretrained=pretrained),
        "efficientnet_v2_m": lambda pretrained: EfficientNet("efficientnet_v2_m", pretrained=pretrained),
        "efficientnet_v2_l": lambda pretrained: EfficientNet("efficientnet_v2_l", pretrained=pretrained),
        "pyramidnet272": lambda pretrained: PyramidNet("cifar10", depth=272, alpha=200, pretrained=pretrained),
        "bninception": lambda pretrained: PMModel("bninception", pretrained=pretrained),
        "bninception_simple": lambda pretrained: PAModel("bn_inception_simple", pretrained=pretrained),
        "se_resnet50": lambda pretrained: PMModel("se_resnet50", pretrained=pretrained),
        "cgd_se_resnet50": lambda pretrained: CGDModel("cgd_se_resnet50", pretrained=pretrained),
        "vgg_m3": lambda pretrained: VGG("M3", pretrained=pretrained),
        "vgg19": lambda pretrained: TorchVGGModel("vgg19", pretrained=pretrained)
    }

    POOLINGS = {
        "avg": lambda in_channels, distribution, config: torch.nn.AdaptiveAvgPool2d(output_size=(1, 1), **(config or {})),
        "max": lambda in_channels, distribution, config: torch.nn.AdaptiveMaxPool2d(output_size=(1, 1), **(config or {})),
        "multi": lambda in_channels, distribution, config: MultiPool2d(in_channels, **(config or {})),
        "flatten": lambda in_channels, distribution, config: FlattenPooling(in_channels, **(config or {})),
        "distribution": lambda in_channels, distribution, config: DistributionPooling(in_channels, distribution, **(config or {})),
        "ensemble": lambda in_channels, distribution, config: EnsemblePooling(in_channels, **(config or {}))
    }

    @staticmethod
    def get_default_config(model_type="resnet50", pretrained=False, freeze_bn=False,
                           pooling_type="avg", pooling_params=None, pooling_after_head=False, pooling_input_channels=None,
                           dropout=0.0, head_batchnorm=True, head_normalize=True, head_bayes_params=None,
                           extra_head_dim=0, extra_head_layers=3,
                           freeze_stem=False, freeze_head=False, freeze_extra_head=False, freeze_normalizer=False,
                           output_scale=1.0, disable_head=False):
        """Get embedder parameters.

        Args:
            model_type: One of "resnet18", "resnet34", "resnet50", "resnet101", "bninception", "se_resnet50" and "cgd_se_resnet50".
            pretrained: Whether to use ImageNet-pretrained model or start from scratch.
            freeze_bn: Whether to freez batch normalization or not.
            pooling_type: Type of pooling ("avg", "max" or "multi").
            pooling_params: Parameters of the pooling.
            pooling_after_head: Apply pooling after head
            pooling_input_channels: The required number of inputs for pooling applied after head.
            dropout: Head dropout probability during training.
            head_batchnorm: Whether to apply 1-D batchnorm to CNN output or not.
            head_normalize: Whether to apply provided normalizer or not.
            head_bayes_params: Parameters of the Bayesian head. By default Bayesian extension is turned off.
            extra_head_dim: Use additional head (usually for distribution concentration estimation).
                Output embedding is concatenation of the main and extra heads.
            extra_head_layers: The number of FC layers in extra head.
            freeze_stem: Freeze stem during training.
            freeze_head: Freeze main head during training.
            freeze_extra_head: Freeze extra head during training.
            freeze_normalizer: Freeze normalizer during training.
            output_scale: Output embedding multiplier (used in vMF-loss).
            disable_head: Whether to disable head layers.
        """
        return OrderedDict([
            ("model_type", model_type),
            ("pretrained", pretrained),
            ("freeze_bn", freeze_bn),
            ("pooling_type", pooling_type),
            ("pooling_params", pooling_params),
            ("pooling_after_head", pooling_after_head),
            ("pooling_input_channels", pooling_input_channels),
            ("dropout", dropout),
            ("head_batchnorm", head_batchnorm),
            ("head_normalize", head_normalize),
            ("head_bayes_params", head_bayes_params),
            ("extra_head_dim", extra_head_dim),
            ("extra_head_layers", extra_head_layers),
            ("freeze_stem", freeze_stem),
            ("freeze_head", freeze_head),
            ("freeze_extra_head", freeze_extra_head),
            ("freeze_normalizer", freeze_normalizer),
            ("output_scale", output_scale),
            ("disable_head", disable_head)
        ])

    def __init__(self, distribution, *, config=None):
        super().__init__()
        self._config = prepare_config(self, config)
        self._distribution = distribution
        self._stem = self.MODELS[self._config["model_type"]](pretrained=self._config["pretrained"])
        if not self._config["pooling_after_head"]:
            self._pooling = self.POOLINGS[self._config["pooling_type"]](self._stem.channels, distribution, config=self._config["pooling_params"])
            pooling_channels = self._pooling.channels if hasattr(self._pooling, "channels") else self._stem.channels
            out_features = distribution.num_parameters
        else:
            pooling_channels = self._stem.channels
            out_features = self._config["pooling_input_channels"]

        if self._config["disable_head"]:
            actual_out_features = pooling_channels + self._config["extra_head_dim"]
            if out_features != actual_out_features:
                raise ValueError(f"Expected number of output dimensions ({out_features}) doesn't match the actual number"
                                f" ({actual_out_features}) when `disable_head=True`.")

        self._head = self._make_head(pooling_channels, out_features - self._config["extra_head_dim"])
        self._extra_head = self._make_extra_head(pooling_channels, self._config["extra_head_dim"])
        self._normalizer = distribution.make_normalizer() if self._config["head_normalize"] else None
        self.output_scale = self._config["output_scale"]  # Call setter.
        if self._config["pooling_after_head"]:
            self._pooling = self.POOLINGS[self._config["pooling_type"]](out_features, distribution, config=self._config["pooling_params"])
            if self._pooling.channels != distribution.num_parameters:
                raise RuntimeError("Please specify the required number of input channels for pooling, applied after head.")
            out_features = self._pooling.channels
        if self._config["freeze_bn"]:
            freeze_bn(self._stem)
        if self._config["freeze_stem"]:
            freeze(self._stem)
        if self._config["freeze_head"]:
            freeze(self._head)
        if self._config["freeze_extra_head"] and (self._extra_head is not None):
            freeze(self._extra_head)
        if self._config["freeze_normalizer"] and (self._normalizer is not None):
            freeze(self._normalizer)

    @property
    def input_size(self):
        """Pretrained model input image size."""
        return self._stem.input_size

    @property
    def in_channels(self):
        return 3

    @property
    def mean(self):
        """Pretrained model input normalization mean."""
        return self._stem.mean

    @property
    def std(self):
        """Pretrained model input normalization STD."""
        return self._stem.std

    @property
    def output_scale(self):
        """Main head output scale."""
        if (not hasattr(self, "_output_scale")) or (self._output_scale is None):
            return None
        return self._output_scale.item()

    @property
    def prior_kld(self):
        if (not self._config["disable_head"]) and (self._config["head_bayes_params"] is not None):
            return self._head[-1].prior_kld

    @output_scale.setter
    def output_scale(self, scale):
        """Set main head output scale."""
        if hasattr(self, "_output_scale"):
            del self._output_scale
        if scale != 1.0:
            self.register_buffer("_output_scale", torch.full([], scale))

    def forward(self, images):
        cnn_output = self._stem(images)  # (B, C, H, W).
        b, c, h, w = cnn_output.shape
        if not self._config["pooling_after_head"]:
            cnn_output = self._apply_pooling(cnn_output)  # (B, N, C).
        else:
            cnn_output = cnn_output.permute(0, 2, 3, 1).reshape(b, h * w, c)  # (B, N, C).
        _, n, d = cnn_output.shape
        cnn_output = cnn_output.reshape(b * n, d)  # (B * N, D).
        head_output = self._head(cnn_output)
        if head_output.ndim == 3:
            # Increase ensemble size.
            n *= head_output.shape[1]
            head_output = head_output.reshape(b * n, head_output.shape[-1])
        if self.output_scale is not None:
            head_output = head_output * self.output_scale
        if self._extra_head is not None:
            extra_head_output = self._extra_head(cnn_output)
            head_output = torch.cat([head_output, extra_head_output], dim=-1)
        if self._normalizer is not None:
            head_output = self._normalizer(head_output)
        head_output = head_output.reshape(b, n, head_output.shape[-1])  # (B, N, D).
        if self._config["pooling_after_head"]:
            head_output = self._apply_pooling(head_output.permute(0, 2, 1).reshape(b, -1, h, w))
        return head_output

    def train(self, mode):
        super().train(mode)
        if self._config["freeze_bn"] or self._config["freeze_stem"]:
            eval_bn(self._stem)
        if self._config["freeze_head"]:
            eval_bn(self._head)
        if self._config["freeze_extra_head"] and (self._extra_head is not None):
            eval_bn(self._extra_head)
        if self._config["freeze_normalizer"] and (self._normalizer is not None):
            eval_bn(self._normalizer)

    def _make_head(self, in_features, out_features):
        head_layers = []
        if self._config["head_batchnorm"]:
            head_layers.append(torch.nn.BatchNorm1d(in_features))
        if self._config["dropout"] > 0:
            head_layers.append(torch.nn.Dropout(self._config["dropout"]))
        if not self._config["disable_head"]:
            head_cls = ((lambda *args: LinearBayes(*args, config=self._config["head_bayes_params"]))
                        if self._config["head_bayes_params"] is not None
                        else torch.nn.Linear)
            head_layers.append(head_cls(in_features, out_features))
            torch.nn.init.constant_(head_layers[-1].bias, 0)
        return SequentialFP32(*head_layers)

    def _make_extra_head(self, in_features, out_features):
        if out_features == 0:
            return None
        head_layers = []
        for _ in range(self._config["extra_head_layers"] - 1):
            head_layers.append(torch.nn.Linear(in_features, in_features // 2))
            torch.nn.ReLU(inplace=True)
            in_features //= 2
        head_layers.append(torch.nn.Linear(in_features, out_features))
        torch.nn.init.constant_(head_layers[-1].bias, 0)
        return SequentialFP32(*head_layers)

    def _apply_pooling(self, cnn_output):
        cnn_output = self._pooling(cnn_output)  # (B, C) or (B, N, C).
        if cnn_output.shape[-2:] == (1, 1):
            # Standard Avg or Max pooling.
            cnn_output = cnn_output.squeeze(-1).squeeze(-1)
        if cnn_output.ndim == 2:
            cnn_output = cnn_output.unsqueeze(1)  # (B, N, C).
        assert cnn_output.ndim == 3, cnn_output.shape  # (B, N, D).
        return cnn_output
