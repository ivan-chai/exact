"""Tools for testing and debugging."""
import sys
from .__main__ import parse_arguments


def get_default_args():
    """Get default arguments for run script."""
    argv = sys.argv
    sys.argv = [argv[0], "train", "data"]
    try:
        args = parse_arguments()
    finally:
        sys.argv = argv
    args.cmd = None
    args.data = None
    return args


class CLIArguments:
    """Simple CLI arguments wrapper for debug and testing."""
    def __init__(self, **kwargs):
        self.__dict__.update(get_default_args().__dict__)
        self.__dict__.update(kwargs)
