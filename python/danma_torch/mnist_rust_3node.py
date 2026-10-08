"""Backward-compatible alias for the configurable multi-node MNIST benchmark.

Use python -m danma_torch.mnist_rust_multinode --nodes N for new runs.
The default remains three nodes.
"""
from .mnist_rust_multinode import *  # noqa: F401,F403
from .mnist_rust_multinode import main


if __name__ == "__main__":
    raise SystemExit(main())
