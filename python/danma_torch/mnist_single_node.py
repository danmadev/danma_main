"""Configurable-node MNIST 784 -> N -> host ReLU -> 10 DANMA experiment.

The hidden width and process count are configurable without source changes.
Defaults: 1000 hidden neurons and one danma-node process.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from . import mnist_benchmark1000 as base

DEFAULT_HIDDEN_NEURONS = 1000
MAX_HIDDEN_NEURONS = 1024
MAX_NODES = 10


def make_layout(hidden_neurons: int = DEFAULT_HIDDEN_NEURONS, nodes: int = 1) -> base.Layout:
    if type(hidden_neurons) is not int or not 1 <= hidden_neurons <= MAX_HIDDEN_NEURONS:
        raise ValueError(f'hidden_neurons must be an integer in 1..{MAX_HIDDEN_NEURONS}')
    if type(nodes) is not int or not 1 <= nodes <= MAX_NODES:
        raise ValueError(f'nodes must be an integer in 1..{MAX_NODES}')
    return base.Layout(hidden=hidden_neurons, nodes=nodes)


def validate_options(*, hidden_neurons=DEFAULT_HIDDEN_NEURONS, nodes=1,
                     train_samples=20, test_samples=10, epochs=1, batch_size=1,
                     backends=('cpu', 'cuda', 'danma'), require_cuda=False, **_):
    layout = make_layout(hidden_neurons, nodes)
    base.validate_options(
        train_samples=train_samples,
        test_samples=test_samples,
        epochs=epochs,
        batch_size=batch_size,
        nodes=nodes,
        backends=backends,
        require_cuda=require_cuda,
        layout=layout,
    )
    return layout


def run_benchmark(*, hidden_neurons=DEFAULT_HIDDEN_NEURONS, nodes=1,
                  run_dir=None, json_out=None,
                  node_binary=Path('target/release/danma-node'), data_dir=None,
                  train_samples=20, test_samples=10, epochs=1, batch_size=1, seed=7,
                  backends=('cpu', 'cuda', 'danma'), download=True, require_cuda=False,
                  parity_tolerance=2e-5, tolerance_reason=None):
    layout = validate_options(
        hidden_neurons=hidden_neurons,
        nodes=nodes,
        train_samples=train_samples,
        test_samples=test_samples,
        epochs=epochs,
        batch_size=batch_size,
        backends=backends,
        require_cuda=require_cuda,
    )
    if run_dir is None:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        run_dir = Path(f'verify-run/mnist-n{nodes}-h{hidden_neurons}') / stamp
    root = Path(__file__).parent.parent
    provenance_files = [
        Path(__file__),
        root / 'MNIST_SINGLE_NODE.md',
        root / 'benchmark_tests/test_mnist_single_node.py',
    ]
    return base.run_benchmark(
        run_dir=run_dir,
        json_out=json_out,
        node_binary=node_binary,
        data_dir=data_dir,
        train_samples=train_samples,
        test_samples=test_samples,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        nodes=nodes,
        backends=backends,
        download=download,
        require_cuda=require_cuda,
        parity_tolerance=parity_tolerance,
        tolerance_reason=tolerance_reason,
        layout=layout,
        provenance_files=provenance_files,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hidden-neurons', type=int, default=DEFAULT_HIDDEN_NEURONS)
    parser.add_argument('--nodes', type=int, default=1)
    parser.add_argument('--node-binary', type=Path, default=Path('target/release/danma-node'))
    parser.add_argument('--data-dir', type=Path, default=Path.home() / '.cache/danma/mnist')
    parser.add_argument('--train-samples', type=int, default=20)
    parser.add_argument('--test-samples', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--backends', type=lambda value: tuple(value.split(',')), default=('cpu', 'cuda', 'danma'))
    parser.add_argument('--no-download', action='store_true')
    parser.add_argument('--require-cuda', action='store_true')
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--json-out', type=Path)
    parser.add_argument('--smoke', action='store_true', help='one train sample, one test sample, one epoch')
    parser.add_argument('--parity-tolerance', type=float, default=2e-5)
    parser.add_argument('--tolerance-reason')
    args = parser.parse_args(argv)
    if args.smoke:
        args.train_samples = args.test_samples = args.epochs = 1
    try:
        validate_options(**vars(args))
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main(argv=None):
    args = vars(parse_args(argv))
    args['download'] = not args.pop('no_download')
    args.pop('smoke')
    report = run_benchmark(**args)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return report['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
