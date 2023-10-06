#!/usr/bin/env python3
import os
import sys
import tempfile
from unittest import TestCase, main

import numpy as np
import torch
from torchvision import transforms

from mdn_metric import commands
from mdn_metric.config import update_config, write_config
from mdn_metric.test import CLIArguments


CONFIG = {
    "dataset_params": {
        "name": "debug-openset",
        "batch_size": 4,
        "num_workers": 0,
        "validation_fold": 0,
        "num_validation_folds": 2
    },
    "model_params": {
        "embedder_params": {
            "pretrained": False,
            "model_type": "vgg_m3"
        }
    },
    "trainer_params": {
        "num_epochs": 2
    },
    "metrics_params": {
        "train_classification_metrics": ["nearest", "scores"]
    }
}


PIPELINES = {
    "asam": {
        "trainer_params": {
            "optimizer_type": "sam"
        }
    },
    "multilabel": {
        "dataset_params": {
            "name": "debug-multi-openset",
            "samples_per_class": None,
            "add_verification_testsets": False
        },
        "criterion_params": {
            "xent_weight": 0.0,
            "bce_weight": 1.0,
            "bce_log_prob": False
        },
        "metrics_params": {
            "test_classification_metrics": ["map"],
            "train_classification_metrics": ["map"]
        }
    },
    "feature-matching-score": {
        "model_params": {
            "distribution_type": "vmf",
            "distribution_params": {
                "dim": 5,
                "components": 3
            },
            "classifier_type": "scorer",
            "scorer_type": "fms"
        },
        "criterion_params": {
            "use_softmax": False
        }
    },
    "distribution-pooling": {
        "model_params": {
            "distribution_type": "gmm",
            "distribution_params": {
                "dim": 176
            },
            "embedder_params": {
                "pooling_type": "distribution"
            }
        }
    },
    "ensemble": {
        "model_params": {
            "distribution_type": "gmm",
            "distribution_params": {
                "spherical": True,
                "dim": 512
            },
            "classifier_type": "arcface",
            "embedder_params": {
                "pooling_type": "ensemble"
            }
        }
    },
    "bayes": {
        "model_params": {
            "embedder_params": {
                "head_bayes_params": {"variational_weight": True}
            }
        },
        "criterion_params": {"weights_prior_kld_weight": 1.0}
    }
}


class TestPipeline(TestCase):
    def test_pipelines(self):
        pipelines = sys.argv[1:] if len(sys.argv) > 1 else list(PIPELINES)
        for pipeline in pipelines:
            if pipeline not in PIPELINES:
                raise ValueError("Unknown pipeline{}. Available pipelines: {}".format(pipeline, list(PIPELINES)))
        for pipeline in pipelines:
            print("Test {}".format(pipeline))
            config = update_config(CONFIG, PIPELINES[pipeline])
            print(config)
            with tempfile.TemporaryDirectory() as root:
                config_path = os.path.join(root, "config.yaml")
                write_config(config, config_path)
                args = CLIArguments(
                    cmd="train",
                    data=root,  # Unused.
                    config=config_path,
                    logger="tensorboard",
                    train_root=root
                )
                commands.train(args)


if __name__ == "__main__":
    main(argv=sys.argv[:1])  # Extract pipeline name manually.
