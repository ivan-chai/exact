import argparse
import re
import sys
import warnings
from collections import defaultdict
from datetime import datetime

import numpy as np
import wandb


def parse_date(s):
    utc_dt = datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")
    timestamp = (utc_dt - datetime(1970, 1, 1)).total_seconds()
    return timestamp


def parse_arguments():
    parser = argparse.ArgumentParser("Download metrics from WandB.")
    parser.add_argument("wandb_path", help="Path to the project in format 'entity/project'.")
    parser.add_argument("-f", "--filename", help="Dump output to file.")
    parser.add_argument("--group", help="Group to load metrics from (use '-' to match ungrouped runs).")
    parser.add_argument("--run-regexp", help="Filter runs by name.", nargs="+")
    parser.add_argument("--from-date", help="Select runs starting from the date.")
    parser.add_argument("--first", help="Load only first N runs.", type=int)
    parser.add_argument("--sort-by", help="Metric to sort runs by.")
    parser.add_argument("--metric-regexp", nargs="*", help="Regexp to filter metrics.")
    parser.add_argument("--percent", help="Multiply metrics by 100.", action="store_true")
    parser.add_argument("--precision", help="Number of decimal places.", type=int, default=1)
    parser.add_argument("--separator", help="Fields separator.", default=" ")
    parser.add_argument("--url", help="WandB URL.", default="https://api.wandb.ai")
    return parser.parse_args()


def matches(s, regexps):
    for regexp in regexps:
        if re.search(regexp, s) is not None:
            return True
    return False


def get_runs(api, path, group=None, first=None, regexps=None, from_date=None):
    runs = list(sorted(api.runs(path=path), key=lambda run: parse_date(run.created_at)))
    if group == "-":
        runs = [run for run in runs if not run.group]
    elif group is not None:
        runs = [run for run in runs
                if (run.group is not None) and matches(run.group, [group])]
    if regexps is not None:
        runs = [run for run in runs if matches(run.name, regexps)]
    if from_date is not None:
        start = parse_date(args.from_date)
        runs = [run for run in runs if parse_date(run.created_at) > start]
    if first is not None:
        runs = runs[:first]
    return runs


def get_metrics(run, metric_regexps=None):
    metrics = run.summary
    if metric_regexps is not None:
        metrics = {k: v for k, v in metrics.items()
                   if matches(k, metric_regexps)}
    return metrics


def prepare_metric(metric, percent=False, precision=2):
    if isinstance(metric, str):
        return metric
    if percent:
        metric = metric * 100
    fmt = "{:." + str(precision) + "f}"
    return fmt.format(metric)


def order_metrics(metrics, metric_regexps=None):
    metrics = list(sorted(list(metrics)))
    if metric_regexps is not None:
        ordered = []
        for regexp in metric_regexps:
            for metric in metrics:
                if metric in ordered:
                    continue
                if matches(metric, [regexp]):
                    ordered.append(metric)
        metrics = ordered
    return metrics


def print_metrics(fp, metrics, run_metrics, separator=" ", percent=False, precision=2):
    print(separator.join(["run"] + list(metrics)), file=fp)
    for run in run_metrics:
        tokens = [run["run_id"]]
        for name in metrics:
            mean = run.get(name, "N/A")
            mean = prepare_metric(mean, percent=percent, precision=precision)
            tokens.append(mean)
        print(separator.join(tokens), file=fp)


def get_runs_metrics(runs, metric_regexps):
    """Returns mean/std metrics for best seeds from each group."""
    metrics = []
    for run in runs:
        metrics.append(get_metrics(run, metric_regexps))
        metrics[-1]["run_id"] = run.name
    return metrics


def main(args):
    entity, project = args.wandb_path.split("/")
    api = wandb.apis.public.Api(overrides={"base_url": args.url})

    runs = get_runs(api,"{}/{}".format(entity, project),
                    group=args.group,
                    first=args.first,
                    regexps=args.run_regexp,
                    from_date=args.from_date)
    metrics = get_runs_metrics(runs, args.metric_regexp)
    if args.sort_by:
        metrics = list(sorted(metrics, key=lambda run: run.get(args.sort_by, -1e6)))
    metrics_order = order_metrics(set(sum(map(list, metrics), [])), metric_regexps=args.metric_regexp)
    print_kwargs = {
        "separator": args.separator,
        "percent": args.percent,
        "precision": args.precision
    }
    if args.filename is not None:
        with open(args.filename, "w") as fp:
            print_metrics(fp, metrics_order, metrics, **print_kwargs)
    else:
        print_metrics(sys.stdout, metrics_order, metrics, **print_kwargs)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
