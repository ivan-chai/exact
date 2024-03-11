import argparse
import io
import itertools
import os
from collections import OrderedDict
from copy import deepcopy

import mxnet as mx
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from mdn_metric.config import update_config, as_flat_config, as_nested_config, CONFIG_HOPT
from mdn_metric.io import read_yaml, write_yaml
from mdn_metric.runner import Runner


CONFIG_REMOVE = "_hopt_remove"


def parse_arguments():
    parser = argparse.ArgumentParser("Generate configs from templates.")
    parser.add_argument("templates", help="Templates root.")
    parser.add_argument("dst", help="Target configs root.")
    parser.add_argument("--best", help="Best hopts root.")
    return parser.parse_args()


def get_best_hopts(path):
    """Load best hyperparameters from wandb config.

    If file doesn't exists, returns empty dictionary.
    """
    if not path.exists():
        return {}
    print("Load best parameters from {}.".format(path))
    flat_config = {k: v["value"] for k, v in read_yaml(path).items()
                   if not k.startswith("wandb") and not k.startswith("_")
                   and not k.startswith("dataset_params")
                   and not k.startswith("metrics_params")}
    config = as_nested_config(flat_config)
    config.pop("git_commit", None)
    default_keys = set(Runner.get_default_config())
    for k in config:
        if k not in default_keys:
            raise RuntimeError("Unknown parameter: {}.".format(k))
    return config


def get_templates(src):
    filenames = {path.relative_to(src) for path in src.glob("template-*.yaml")}
    try:
        filenames.remove(Path("template-base.yaml"))
    except KeyError:
        raise FileNotFoundError("Need template-base.yaml.")
    base = read_yaml(src / "template-base.yaml")
    patches = []
    for patch in src.glob("template-patch-*.yaml"):
        filenames.remove(patch.relative_to(src))
        basename = str(os.path.basename(patch)).rsplit(".", 1)[0]
        index = int(basename.split("-")[2])
        patches.append((index, list(read_yaml(patch).items())))
    patches = [patch for _, patch in sorted(patches, key=lambda pair: pair[0])]
    pipelines = [(str(filename.stem).split("-", 1)[1], read_yaml(src / filename)) for filename in filenames]
    return base, patches, pipelines


def main(args):
    src = Path(args.templates)
    dst = Path(args.dst)
    best = Path(args.best) if args.best is not None else None
    base, all_patches, pipelines = get_templates(src)
    for pipeline, pipeline_patch in pipelines:
        print(pipeline)
        for patches in itertools.product(*all_patches):
            config = base
            filename = pipeline
            for name, patch in patches:
                filename = filename + "-" + name
                config = update_config(config, patch)
            filename = filename + ".yaml"
            config = update_config(config, pipeline_patch)
            if best is not None:
                config = update_config(config, get_best_hopts(best / filename), recurse_lists=True)
            if CONFIG_REMOVE in config:
                remove_keys = config.pop(CONFIG_REMOVE)
                flat = as_flat_config(config)
                if CONFIG_HOPT in flat:
                    for key in remove_keys:
                        flat[CONFIG_HOPT].pop(key, None)
                config = as_nested_config(flat)
            write_yaml(config, dst / filename)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
