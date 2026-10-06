"""Experimental single-node MNIST benchmark for the async shard affine executor.

Logical DANMA neurons remain independently addressable and independently
versioned. Compatible neurons owned by one process share only a physical
forward/backward transport + worker envelope.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from .layer import DANMAShardLinear
from . import mnist_benchmark1000 as base

DEFAULT_HIDDEN_NEURONS = 1000
MAX_HIDDEN_NEURONS = 1024


class ShardRemoteModel(torch.nn.Module):
    def __init__(self, client, layout):
        super().__init__()
        if layout.nodes != 1:
            raise ValueError("experimental shard executor currently requires one local node")
        self.hidden = DANMAShardLinear(
            client,
            neuron_ids=layout.hidden_ids,
            input_ids=layout.input_ids,
            feedback_ttl_ms=3000,
            max_batch=1,
        )
        self.output = DANMAShardLinear(
            client,
            neuron_ids=layout.output_ids,
            input_ids=layout.feature_ids,
            feedback_ttl_ms=3000,
            max_batch=1,
        )

    def forward(self, inputs):
        return self.output(F.relu(self.hidden(inputs)))


def make_layout(hidden_neurons: int = DEFAULT_HIDDEN_NEURONS) -> base.Layout:
    if type(hidden_neurons) is not int or not 1 <= hidden_neurons <= MAX_HIDDEN_NEURONS:
        raise ValueError(f"hidden_neurons must be an integer in 1..{MAX_HIDDEN_NEURONS}")
    return base.Layout(hidden=hidden_neurons, nodes=1)


def _write_final_report(report, json_out):
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    destinations = [Path(report["run_dir"]) / "report.json"]
    if json_out is not None and Path(json_out) != destinations[0]:
        destinations.append(Path(json_out))
    for path in destinations:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)


def run_benchmark(
    *,
    hidden_neurons=DEFAULT_HIDDEN_NEURONS,
    run_dir=None,
    json_out=None,
    node_binary=Path("target/release/danma-node"),
    data_dir=None,
    train_samples=20,
    test_samples=10,
    epochs=1,
    batch_size=1,
    seed=7,
    backends=("cpu", "danma"),
    download=True,
    require_cuda=False,
    parity_tolerance=2e-5,
    tolerance_reason=None,
    nodes=1,
):
    if nodes != 1:
        raise ValueError("async local shard v1 requires exactly one node")
    if batch_size != 1:
        raise ValueError("shard MNIST v1 keeps per-example SGD; batch_size must be 1")
    layout = make_layout(hidden_neurons)
    if run_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run_dir = Path(f"verify-run/mnist-shard-h{hidden_neurons}") / stamp

    root = Path(__file__).parent.parent
    provenance_files = (
        Path(__file__),
        root / "MNIST_SHARD.md",
        root / "benchmark_tests/test_mnist_shard.py",
    )
    report = base.run_benchmark(
        run_dir=run_dir,
        json_out=json_out,
        node_binary=node_binary,
        data_dir=data_dir,
        train_samples=train_samples,
        test_samples=test_samples,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        nodes=1,
        backends=backends,
        download=download,
        require_cuda=require_cuda,
        parity_tolerance=parity_tolerance,
        tolerance_reason=tolerance_reason,
        layout=layout,
        provenance_files=provenance_files,
        remote_model_cls=ShardRemoteModel,
    )
    report["executor"] = {
        "kind": "async_local_shard_affine_v1",
        "global_barrier": False,
        "logical_neuron_state": "independent EventID/version/weights/update",
        "physical_forward_envelope": "one local shard request per layer per sample",
        "physical_backward_envelope": "one local shard request per layer per sample",
        "host_boundary": "ReLU, cross entropy, autograd aggregation",
        "expected_train_rpc_per_sample": 4,
        "dense_gemm": False,
        "note": (
            "v1 batches transport and worker dispatch while retaining the existing "
            "per-neuron core math; dense SoA/GEMM is a later optimization."
        ),
    }
    _write_final_report(report, json_out)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--hidden-neurons", type=int, default=DEFAULT_HIDDEN_NEURONS)
    parser.add_argument("--node-binary", type=Path, default=Path("target/release/danma-node"))
    parser.add_argument("--nodes", type=int, choices=[1], default=1,
                        help="v1 local shard executor supports exactly one node")
    parser.add_argument("--data-dir", type=Path, default=Path.home() / ".cache/danma/mnist")
    parser.add_argument("--train-samples", type=int, default=20)
    parser.add_argument("--test-samples", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--backends",
        type=lambda value: tuple(value.split(",")),
        default=("cpu", "danma"),
    )
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--parity-tolerance", type=float, default=2e-5)
    parser.add_argument("--tolerance-reason")
    args = parser.parse_args(argv)
    if args.smoke:
        args.train_samples = args.test_samples = args.epochs = 1
    try:
        make_layout(args.hidden_neurons)
        base.validate_options(
            train_samples=args.train_samples,
            test_samples=args.test_samples,
            epochs=args.epochs,
            batch_size=args.batch_size,
            nodes=1,
            backends=args.backends,
            require_cuda=args.require_cuda,
            layout=make_layout(args.hidden_neurons),
        )
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main(argv=None):
    args = vars(parse_args(argv))
    args["download"] = not args.pop("no_download")
    args.pop("smoke")
    report = run_benchmark(**args)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
