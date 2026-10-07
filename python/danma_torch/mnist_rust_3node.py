"""MNIST CPU vs 3-node DANMA Rust-native shard benchmark.

Model: 784 -> hidden -> ReLU -> 10.  The default hidden width is 1000.
Logical neuron state remains independent; physical shard requests are split by
the owning DANMA process.  Inside each process, the existing Rust bounded
worker mailboxes execute the local shard work.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from collections.abc import Sequence

import torch
from torch.nn import functional as F

from .client import DANMAClient
from .layer import DANMAShardLinear
from . import mnist_benchmark1000 as base

DEFAULT_HIDDEN_NEURONS = 1000
NODES = 3
MAX_HIDDEN_NEURONS = 1024


class MultiNodeShardClient(DANMAClient):
    """Fan one logical shard layer across its actual owning DANMA nodes.

    This object performs no transparent retry.  If one owner commits and a
    later owner fails, the caller receives the failure and the benchmark is
    marked incomplete, matching the existing stateful RPC semantics.
    """

    def __init__(
        self,
        clients: Sequence[DANMAClient],
        owners: dict[int, int],
    ) -> None:
        if len(clients) != NODES or not all(isinstance(client, DANMAClient) for client in clients):
            raise TypeError(f"exactly {NODES} DANMA clients are required")
        super().__init__(
            clients[0].host,
            clients[0].port,
            timeout_seconds=clients[0].timeout_seconds,
        )
        self._clients = tuple(clients)
        self._owners = dict(owners)
        self._phase = "readiness"

    @property
    def phase(self) -> str:
        return self._phase

    @phase.setter
    def phase(self, value: str) -> None:
        self._phase = str(value)
        for client in getattr(self, "_clients", ()):
            if hasattr(client, "phase"):
                client.phase = self._phase

    def request(self, message):
        # Non-shard compatibility operations (inspect/routes/etc.) still enter
        # through node 1 and use the existing routed network protocol.
        primary = self._clients[0]
        if hasattr(primary, "phase"):
            primary.phase = self._phase
        return primary.request(message)

    def _groups(self, neuron_ids: Sequence[int]) -> list[tuple[int, list[int]]]:
        groups: dict[int, list[int]] = {}
        for index, neuron_id in enumerate(neuron_ids):
            owner = self._owners.get(int(neuron_id))
            if owner is None or not 1 <= owner <= len(self._clients):
                raise ValueError(f"missing/invalid owner for neuron {neuron_id}")
            groups.setdefault(owner, []).append(index)
        return sorted(groups.items())

    def forward_shard(
        self,
        *,
        neuron_ids,
        event_ids,
        trace_id,
        inputs,
        training,
    ):
        if len(neuron_ids) != len(event_ids):
            raise ValueError("neuron_ids/event_ids length mismatch")
        outputs: list[float | None] = [None] * len(neuron_ids)
        for owner, indexes in self._groups(neuron_ids):
            client = self._clients[owner - 1]
            if hasattr(client, "phase"):
                client.phase = self._phase
            partial = client.forward_shard(
                neuron_ids=[neuron_ids[i] for i in indexes],
                event_ids=[event_ids[i] for i in indexes],
                trace_id=trace_id,
                inputs=inputs,
                training=training,
            )
            if len(partial) != len(indexes):
                raise RuntimeError("owner returned malformed shard output length")
            for index, value in zip(indexes, partial):
                outputs[index] = float(value)
        if any(value is None for value in outputs):
            raise RuntimeError("multi-node shard forward left an output unset")
        return [float(value) for value in outputs]

    def backward_shard(
        self,
        *,
        neuron_ids,
        event_ids,
        gradients,
        input_ids,
        feedback_ttl_ms,
    ):
        if not (len(neuron_ids) == len(event_ids) == len(gradients)):
            raise ValueError("neuron/event/gradient length mismatch")
        partials: list[list[float]] = []
        for owner, indexes in self._groups(neuron_ids):
            client = self._clients[owner - 1]
            if hasattr(client, "phase"):
                client.phase = self._phase
            partial = client.backward_shard(
                neuron_ids=[neuron_ids[i] for i in indexes],
                event_ids=[event_ids[i] for i in indexes],
                gradients=[gradients[i] for i in indexes],
                input_ids=input_ids,
                feedback_ttl_ms=feedback_ttl_ms,
            )
            if len(partial) != len(input_ids):
                raise RuntimeError("owner returned malformed shard input-gradient length")
            partials.append([float(value) for value in partial])
        return [
            math.fsum(partial[index] for partial in partials)
            for index in range(len(input_ids))
        ]


class ThreeNodeShardRemoteModel(torch.nn.Module):
    def __init__(self, client: DANMAClient, layout: base.Layout):
        super().__init__()
        if layout.nodes != NODES:
            raise ValueError(f"three-node benchmark requires exactly {NODES} nodes")
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

    @classmethod
    def from_cluster(cls, cluster, layout):
        client = MultiNodeShardClient(cluster.clients, layout.owners)
        # The common benchmark updates cluster.client.phase before each stage
        # and snapshots through cluster.client, so install the fanout-aware
        # client as the entrypoint after cluster readiness is proven.
        cluster.client = client
        return cls(client, layout)

    def forward(self, inputs):
        return self.output(F.relu(self.hidden(inputs)))


def make_layout(hidden_neurons: int = DEFAULT_HIDDEN_NEURONS) -> base.Layout:
    if type(hidden_neurons) is not int or not 1 <= hidden_neurons <= MAX_HIDDEN_NEURONS:
        raise ValueError(f"hidden_neurons must be an integer in 1..{MAX_HIDDEN_NEURONS}")
    return base.Layout(hidden=hidden_neurons, nodes=NODES)


def layer_owner_counts(layout: base.Layout) -> dict[str, list[int]]:
    return {
        "hidden": [
            sum(layout.owners[neuron_id] == node for neuron_id in layout.hidden_ids)
            for node in range(1, layout.nodes + 1)
        ],
        "output": [
            sum(layout.owners[neuron_id] == node for neuron_id in layout.output_ids)
            for node in range(1, layout.nodes + 1)
        ],
    }


def expected_rpc_per_sample(layout: base.Layout) -> dict[str, int]:
    hidden_owners = len({layout.owners[neuron_id] for neuron_id in layout.hidden_ids})
    output_owners = len({layout.owners[neuron_id] for neuron_id in layout.output_ids})
    forward = hidden_owners + output_owners
    backward = output_owners + hidden_owners
    return {
        "forward": forward,
        "backward": backward,
        "train_total": forward + backward,
    }


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
    nodes=NODES,
):
    if nodes != NODES:
        raise ValueError(f"benchmark requires exactly {NODES} nodes")
    if batch_size != 1:
        raise ValueError("benchmark preserves per-example SGD; batch_size must be 1")
    layout = make_layout(hidden_neurons)
    if run_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run_dir = Path(f"verify-run/mnist-rust-3node-h{hidden_neurons}") / stamp

    root = Path(__file__).parent.parent
    provenance_files = (
        Path(__file__),
        root / "MNIST_RUST_3NODE.md",
        root / "benchmark_tests/test_mnist_rust_3node.py",
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
        nodes=NODES,
        backends=backends,
        download=download,
        require_cuda=require_cuda,
        parity_tolerance=parity_tolerance,
        tolerance_reason=tolerance_reason,
        layout=layout,
        provenance_files=provenance_files,
        remote_model_cls=ThreeNodeShardRemoteModel,
    )
    rpc = expected_rpc_per_sample(layout)
    report["executor"] = {
        "kind": "rust_native_multinode_shard_affine_v1",
        "nodes": NODES,
        "global_barrier": False,
        "logical_neuron_state": "independent EventID/version/weights/update",
        "local_neuron_transport": "bounded Rust worker mailboxes inside each DANMA process",
        "remote_transport": "one shard RPC per participating owner node and layer",
        "gossip_role": "control-plane route discovery only",
        "host_boundary": "ReLU, cross entropy, autograd aggregation",
        "dense_gemm": False,
        "expected_rpc_per_sample": rpc,
        "placement": {
            "partition_sizes": [len(ids) for ids in layout.partitions],
            "layer_owner_counts": layer_owner_counts(layout),
        },
        "note": (
            "The benchmark fans each affine layer only to nodes that own target neurons. "
            "Within a node, targets retain independent EventID/version semantics and execute "
            "through the Rust shard worker path. Cross-node backward dX is reduced on the host "
            "with math.fsum; this can change floating-point reduction order without changing SGD semantics."
        ),
    }
    _write_final_report(report, json_out)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--hidden-neurons", type=int, default=DEFAULT_HIDDEN_NEURONS)
    parser.add_argument("--node-binary", type=Path, default=Path("target/release/danma-node"))
    parser.add_argument("--nodes", type=int, choices=[NODES], default=NODES)
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
        layout = make_layout(args.hidden_neurons)
        base.validate_options(
            train_samples=args.train_samples,
            test_samples=args.test_samples,
            epochs=args.epochs,
            batch_size=args.batch_size,
            nodes=NODES,
            backends=args.backends,
            require_cuda=args.require_cuda,
            layout=layout,
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
