import argparse
import os
import tempfile
import time

import numpy as np
import torch
from tqdm import tqdm

from mdn_metric.criterion import Criterion
from mdn_metric.layers import DiracDistribution, GMMDistribution


def parse_arguments():
    parser = argparse.ArgumentParser("Eval EXACT loss speed.")
    parser.add_argument("-d", "--dim", help="Embedding dim", type=int, default=128)
    parser.add_argument("-c", "--num-classes", help="Number of output classes", type=int, default=32)
    parser.add_argument("-r", "--robust-dims", help="Number of classes to compute with Genz. Use MC for others.", type=int)
    parser.add_argument("-b", "--batch-size", help="Batch size", type=int, default=256)
    parser.add_argument("-s", "--num-samples", help="Number of EXACT samples", type=int, default=16)
    parser.add_argument("-n", "--num-runs", help="Number of evaluation runs", type=int, default=100)
    parser.add_argument("--half", action="store_true", help="Half precision")
    return parser.parse_args()


def main(args):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print("Device", device)
    distribution = DiracDistribution(config={"dim": args.dim})
    criterion = Criterion(config={"xent_weight": 0,
                                  "exact_weight": 1,
                                  "exact_sample_size": args.num_samples,
                                  "exact_robust_dims": args.robust_dims})
    print("Config", criterion._config)
    criterion.distribution = distribution
    b = args.batch_size
    c = args.num_classes
    embeddings = torch.randn(b, distribution.num_parameters, requires_grad=True, device=device)
    logits = torch.randn(b, c, requires_grad=True, device=device)
    labels = torch.randint(0, c, [b], device=device)
    final_weights = torch.randn(c, args.dim, requires_grad=True, device=device)
    final_bias = torch.randn(c, requires_grad=True, device=device)
    criterion(embeddings, labels,
              logits=logits,
              final_weights=final_weights,
              final_bias=final_bias).backward()
    print("Memory usage (MB):", torch.cuda.max_memory_allocated() / 1024 ** 2)
    print("Memory usage per request (MB):", torch.cuda.max_memory_allocated() / 1024 ** 2 / b)

    if args.half:
        loss = criterion(embeddings, labels,
                         logits=logits,
                         final_weights=final_weights,
                         final_bias=final_bias)
        loss.backward()
        print("Full precision result:", loss.item())
        print("Full precision gradients sample:")
        print(embeddings.grad[0][:5])
        print(final_weights.grad[0][:5])
        print(final_bias.grad[:5])
        embeddings = torch.tensor(embeddings.detach().half(), requires_grad=True)
        logits = torch.tensor(logits.detach().half(), requires_grad=True)
        final_weights = torch.tensor(final_weights.detach().half(), requires_grad=True)
        final_bias = torch.tensor(final_bias.detach().half(), requires_grad=True)
        loss = criterion(embeddings, labels,
                         logits=logits,
                         final_weights=final_weights,
                         final_bias=final_bias)
        loss.backward()
        print("Half precision result:", loss.item())
        print("Half precision gradients sample:")
        print(embeddings.grad[0][:5])
        print(final_weights.grad[0][:5])
        print(final_bias.grad[:5])
    start = time.time()
    for _ in range(args.num_runs):
        criterion(embeddings, labels,
                  logits=logits,
                  final_weights=final_weights,
                  final_bias=final_bias).backward()
    total = time.time() - start
    rps = 1 / (total / args.num_runs / b)
    print("BATCH RPS", rps / b)
    print("RPS", rps)
    print("ms per element", 1000 / rps)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
