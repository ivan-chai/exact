# mdn-metric
Probabilistic approch to classifiction and metric learning.

Installation
------------

Library must be installed before execution. It is recommended to use editable installation:
```bash
pip install -e ./mdn-metric
```

You can check installation using tests:
```bash
tox -e py38 --current-env
```

Commands
--------

### Training and testing

To run training use the following command:
```bash
CUDA_VISIBLE_DEVICES=<gpu-index> python3 -m mdn_metric train --config <path-to-config> --train-root <training-root> <path-to-data-root>
```

To use wandb:
```bash
CUDA_VISIBLE_DEVICES=<gpu-index> python3 -m mdn_metric train --config <path-to-config> --logger wandb:<project-name>:<experiment-name> --train-root <training-root> <path-to-data-root>
```

You can compute metrics for the checkpoint using the following command:
```bash
CUDA_VISIBLE_DEVICES=<gpu-index> python3 -m mdn_metric test --config <path-to-config> --checkpoint <path-to-checkpoint> <path-to-data-root>
```

### Cross-validation

You can train and evaluate multiple models using leave-one-out scheme for the training dataset:
```bash
CUDA_VISIBLE_DEVICES=<gpu-index> python3 -m mdn_metric cval --config <path-to-config> --train-root <training-root> <path-to-data-root>
```

### Multi-seed evaluation

You can train and evaluate multiple models using cross-validation and multiple seeds:
```bash
CUDA_VISIBLE_DEVICES=<gpu-index> python3 -m mdn_metric evaluate --config <path-to-config> --train-root <training-root> <path-to-data-root>
```

Use evaluate-cval command to evaluate with cross-validation.

If Wandb logger is used, multiple runs will be grouped together.

### Hyper-parameter tuning

Hyper-parameter tuning is available only for Wandb logger.

Run hyper-parameter tuning with crossvalidation (like `cval` command):
```bash
CUDA_VISIBLE_DEVICES=<gpu-index> python3 -m mdn_metric hopt-cval --config <path-to-config> --logger wandb:<project-name>:<experiment-name> --train-root <training-root> <path-to-data-root>
```

Run hyper-parameter tuning with one-fold or without validation (like `train` command):
```bash
CUDA_VISIBLE_DEVICES=<gpu-index> python3 -m mdn_metric hopt --config <path-to-config> --logger wandb:<project-name>:<experiment-name> --train-root <training-root> <path-to-data-root>
```

All runs for the sweep will be grouped together.

Configs
-------

To generate reality check configs use the following command:
```bash
python3 scripts/configs/generate-from-template.py --best configs/exact/best configs/exact/templates configs/exact
```
