#!/usr/bin/env python3
import os
import tempfile
from unittest import TestCase, main

import numpy as np
import torch
import yaml
from scipy import stats

from mdn_metric import commands
from mdn_metric.criterion import Criterion
from mdn_metric.layers import GMMDistribution
from mdn_metric.test import CLIArguments
from mdn_metric.torch import tmp_seed


KLD_CONFIG = {
    "dataset_params": {
        "name": "debug-openset",
        "batch_size": 4,
        "num_workers": 0,
        "num_validation_folds": 2
    },
    "model_params": {
        "embedder_params": {
            "pretrained": False,
            "model_type": "resnet18"
        },
        "distribution_type": "gmm",
        "distribution_params": {
            "dim": 16
        }
    },
    "criterion_params": {
        "prior_kld_weight": 1
    },
    "trainer_params": {
        "num_epochs": 1
    }
}


MLS_CONFIG = {
    "dataset_params": {
        "name": "debug-openset",
        "batch_size": 4,
        "num_workers": 0,
        "num_validation_folds": 2
    },
    "model_params": {
        "embedder_params": {
            "pretrained": False,
            "model_type": "resnet18"
        },
        "classifier_type": None,
        "distribution_type": "gmm",
        "distribution_params": {
            "dim": 16
        }
    },
    "criterion_params": {
        "xent_weight": 0,
        "pfe_weight": 1
    },
    "trainer_params": {
        "num_epochs": 1
    },
    "stages": [
        {"criterion_params": {"pfe_match_self": True}},
        {"criterion_params": {"pfe_match_self": False}}
    ]
}


class TestCriterion(TestCase):
    def test_hinge(self):
        """Test Hinge loss."""
        logits = torch.tensor([
            [0.1, -0.3, 0.2],
            [0.5, 0.0, -0.1],
            [0.0, 0.0, 0.0]
        ])  # (3, 3).
        labels = torch.tensor([1, 0, 2], dtype=torch.long)  # (3).
        criterion = Criterion(config={"xent_weight": 0.0, "hinge_weight": 1.0, "hinge_margin": 0.1})
        loss = criterion(torch.randn(3, 5), labels, logits=logits).item()
        # GT    Deltas.
        # -0.3  0.4, N/A, 0.5
        # 0.5   N/A, -0.5, -0.6
        # 0.0   0.0, 0.0, N/A
        #
        # Losses (margin 0.1).
        # 0.5, N/A, 0.6
        # N/A, 0.0, 0.0
        # 0.1, 0.1, N/A
        #
        loss_gt = np.mean([0.5, 0.6, 0.0, 0.0, 0.1, 0.1])
        self.assertAlmostEqual(loss, loss_gt)

    def test_exact(self):
        criterion = Criterion(config={"xent_weight": 0, "exact_weight": 1, "exact_sample_size": 1024})
        criterion.distribution = GMMDistribution(config={"dim": 2, "max_logivar": None, "parametrization_params": {"type": "exp"}})

        labels = torch.full([1], 1).long()
        mean = torch.ones(2).double()
        concentrations = torch.ones(1).double()
        distributions = criterion.distribution.join_parameters(torch.zeros(1), mean[None], concentrations[None].log())[None]

        # 2 classes.
        targets = torch.tensor([
            [-1., 0.],
            [2., 0.]
        ]).double()
        bias = torch.tensor([0., 1.]).double()
        logits = torch.nn.functional.linear(mean, targets, bias)[None]
        loss = criterion(distributions, labels, logits)
        accuracy = (-loss).exp().item()
        accuracy_gt = stats.norm.cdf(4 / np.sqrt(2))
        self.assertAlmostEqual(accuracy, accuracy_gt, 5)

        self._test_gradients([logits, concentrations],
                             (lambda logits, concentrations:
                              criterion(criterion.distribution.join_parameters(torch.zeros(1), mean[None], concentrations[None].log())[None],
                                        labels, logits)))

    def _test_gradients(self, parameters, loss_fn, eps=1e-3):
        placeholders = [torch.tensor(p.numpy(), requires_grad=True, dtype=torch.double) for p in parameters]
        with tmp_seed(0):
            loss_base = loss_fn(*placeholders)
        loss_base.backward()
        loss_base = loss_base.item()

        grad_norm = self._norm([p.grad for p in placeholders])
        updated_parameters = [p - p.grad * eps / grad_norm for p in placeholders]
        with tmp_seed(0):
            loss_update = loss_fn(*updated_parameters).item()
        self.assertTrue(loss_update < loss_base)

        with torch.no_grad():
            for i, p in enumerate(placeholders):
                shape = p.shape
                p_grad = p.grad.flatten()
                p = p.flatten()
                for j, v in enumerate(p):
                    delta_p = p.clone()
                    delta_p[j] += eps
                    if len(shape) > 1:
                        delta_p = delta_p.reshape(*shape)
                    delta_placeholders = list(placeholders)
                    delta_placeholders[i] = delta_p
                    with tmp_seed(0):
                        loss = loss_fn(*delta_placeholders).item()
                    grad = (loss - loss_base) / eps
                    grad_gt = p_grad[j].item()
                    self.assertAlmostEqual(grad, grad_gt, delta=0.05)

    def _norm(self, parameters):
        return np.sqrt(np.sum([p.square().sum().item() for p in parameters]))


class TestCriterionTraining(TestCase):
    def test_prior_kld(self):
        """Train with KLD loss."""
        with tempfile.TemporaryDirectory() as root:
            config_path = os.path.join(root, "config.yaml")
            with open(config_path, "w") as fp:
                yaml.safe_dump(KLD_CONFIG, fp)
            args = CLIArguments(
                cmd="train",
                data=root,  # Unused.
                config=config_path,
                logger="tensorboard",
                train_root=root
            )
            commands.train(args)

    def test_pfe(self):
        """Train with pair MLS loss."""
        with tempfile.TemporaryDirectory() as root:
            config_path = os.path.join(root, "config.yaml")
            with open(config_path, "w") as fp:
                yaml.safe_dump(MLS_CONFIG, fp)
            args = CLIArguments(
                cmd="train",
                data=root,  # Unused.
                config=config_path,
                logger="tensorboard",
                train_root=root
            )
            commands.train(args)


if __name__ == "__main__":
    main()
