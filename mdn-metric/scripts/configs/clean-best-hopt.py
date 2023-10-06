import argparse
from pathlib import Path

from mdn_metric.config import read_config, as_flat_config
from mdn_metric.config import CONFIG_HOPT
from mdn_metric.io import read_yaml, write_yaml


def parse_arguments():
    parser = argparse.ArgumentParser("Keep only hyperparameters in best configs.")
    parser.add_argument("--configs", help="Path to full configs.", required=True)
    parser.add_argument("--best", help="Path to best configs.", required=True)
    return parser.parse_args()


def get_value(item):
    try:
        return item["value"]
    except Exception:
        return item


def main(args):
    root = Path(args.configs)
    best_root = Path(args.best)
    for best_path in best_root.iterdir():
        if best_path.suffix.lower() != ".yaml":
            continue
        print(best_path.name)
        config = read_config(root / best_path.relative_to(best_root))
        flat = as_flat_config(config)
        hopt_names = {"seed"} | {name for name in flat.get(CONFIG_HOPT, {})}
        best_config = {k: {"value": get_value(v)} for k, v in read_yaml(best_path).items()
                       if k in hopt_names}
        write_yaml(best_config, best_path)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
