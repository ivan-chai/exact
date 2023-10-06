from collections import OrderedDict

from catalyst.core.callback import Callback, CallbackNode, CallbackOrder

from ..config import prepare_config


class KLDSchedulerCallback(Callback):
    """Linearly increases weights KLD loss weight during training."""

    @staticmethod
    def get_default_config(max=1.0):
        """Get scheduler parameters."""
        return OrderedDict([
            ("max", max)
        ])

    def __init__(self, num_epochs, *, config=None):
        super().__init__(order=CallbackOrder.scheduler, node=CallbackNode.all)
        self._config = prepare_config(self, config)
        self._num_epochs = num_epochs
        self._criterion = None
        self._initial_weight = None

    def on_stage_start(self, runner):
        self._criterion = runner.criterion
        self._initial_weight = self._criterion._config["weights_prior_kld_weight"]
        self._epoch = 0

    def on_stage_end(self, runner):
        self._criterion = None
        self._initial_weight = None

    def on_epoch_start(self, runner):
        progress = self._epoch / self._num_epochs
        kld_weight = self._initial_weight * (1 - progress) + progress * self._config["max"]
        self._criterion._config = prepare_config(self._criterion._config, {"weights_prior_kld_weight": kld_weight})
        runner.epoch_metrics["_epoch_"]["weights_prior_kld_loss_weight"] = kld_weight
        self._epoch += 1
