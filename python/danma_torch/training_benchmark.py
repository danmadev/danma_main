"""Deterministic end-to-end training benchmark for the PyTorch -> DANMA bridge.

The benchmark launches three real DANMA TCP processes, trains one remote output
neuron per process on a tiny linear-regression problem, and runs the equivalent
sequential SGD loop in ordinary PyTorch for numerical comparison.
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from .client import DANMAClient, DANMAError
from .layer import DANMALinear


_INITIAL_WEIGHTS = (
    (2.0, 3.0),
    (-1.0, 4.0),
    (0.5, -2.0),
)
_NEURON_IDS = (11, 21, 31)
_INPUT_IDS = (901, 902)
_FEATURES = torch.tensor(
    [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [-1.0, 0.5],
    ],
    dtype=torch.float32,
)
_TARGET_WEIGHTS = torch.tensor(
    [
        [0.8, -0.4],
        [-0.2, 0.7],
        [0.3, 0.5],
    ],
    dtype=torch.float32,
)
_TARGET_BIAS = torch.tensor([0.1, -0.2, 0.05], dtype=torch.float32)
_TARGETS = F.linear(_FEATURES, _TARGET_WEIGHTS, _TARGET_BIAS)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _ThreeNodeBenchmarkCluster:
    def __init__(self, node_binary: Path) -> None:
        self.node_binary = Path(node_binary)
        self.processes: list[subprocess.Popen[bytes]] = []
        self.ports: list[int] = []

    def __enter__(self) -> "_ThreeNodeBenchmarkCluster":
        if not self.node_binary.is_file():
            raise FileNotFoundError(f"DANMA node binary not found: {self.node_binary}")

        while len(self.ports) < 3:
            candidate = _free_port()
            if candidate not in self.ports:
                self.ports.append(candidate)

        for index, (neuron_id, weights) in enumerate(
            zip(_NEURON_IDS, _INITIAL_WEIGHTS)
        ):
            argv = [
                str(self.node_binary),
                "--id",
                str(index + 1),
                "--listen",
                f"127.0.0.1:{self.ports[index]}",
                "--neuron",
                str(neuron_id),
                "--weight",
                f"{_INPUT_IDS[0]}:{weights[0]}",
                "--weight",
                f"{_INPUT_IDS[1]}:{weights[1]}",
            ]
            for peer_index, port in enumerate(self.ports):
                if peer_index != index:
                    argv.extend(
                        (
                            "--peer",
                            f"{peer_index + 1}@127.0.0.1:{port}",
                        )
                    )
            self.processes.append(
                subprocess.Popen(
                    argv,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            )

        self.client = DANMAClient("127.0.0.1", self.ports[0])
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if any(process.poll() is not None for process in self.processes):
                self.stop()
                raise RuntimeError("DANMA benchmark node exited before route convergence")
            try:
                routes = self.client.routes()
                if all(
                    routes.get(str(neuron_id)) == owner
                    for neuron_id, owner in zip(_NEURON_IDS, (1, 2, 3))
                ):
                    return self
            except (OSError, DANMAError):
                pass
            time.sleep(0.05)

        self.stop()
        raise TimeoutError("DANMA benchmark cluster did not converge")

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.stop()

    def stop(self) -> None:
        for process in self.processes:
            if process.poll() is None:
                process.terminate()
        for process in self.processes:
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)


def _dataset_loss(model: DANMALinear) -> float:
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            prediction = model(_FEATURES)
            return float(F.mse_loss(prediction, _TARGETS).item())
    finally:
        model.train(was_training)


def _reference_loss(model: torch.nn.Linear) -> float:
    with torch.no_grad():
        return float(F.mse_loss(model(_FEATURES), _TARGETS).item())


def run_linear_regression_benchmark(
    *,
    node_binary: Path,
    epochs: int = 30,
) -> dict[str, Any]:
    """Train a tiny distributed regression model and compare with PyTorch SGD."""
    if type(epochs) is not int or epochs <= 0:
        raise ValueError("epochs must be a positive integer")

    torch.manual_seed(0)

    reference = torch.nn.Linear(2, 3, bias=True)
    with torch.no_grad():
        reference.weight.copy_(torch.tensor(_INITIAL_WEIGHTS, dtype=torch.float32))
        reference.bias.zero_()
    optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)

    with _ThreeNodeBenchmarkCluster(Path(node_binary)) as cluster:
        model = DANMALinear(
            cluster.client,
            neuron_ids=_NEURON_IDS,
            input_ids=_INPUT_IDS,
            feedback_ttl_ms=3_000,
        )
        initial_loss = _dataset_loss(model)
        reference_initial_loss = _reference_loss(reference)

        for _ in range(epochs):
            for features, target in zip(_FEATURES, _TARGETS):
                danma_loss = F.mse_loss(model(features), target)
                danma_loss.backward()

                optimizer.zero_grad(set_to_none=True)
                reference_loss = F.mse_loss(reference(features), target)
                reference_loss.backward()
                optimizer.step()

        final_loss = _dataset_loss(model)
        reference_final_loss = _reference_loss(reference)

        versions: list[int] = []
        max_parameter_error = 0.0
        for index, neuron_id in enumerate(_NEURON_IDS):
            state = cluster.client.inspect(neuron_id)
            versions.append(int(state["version"]))
            actual = torch.tensor(
                [
                    float(state["weights"][str(_INPUT_IDS[0])]),
                    float(state["weights"][str(_INPUT_IDS[1])]),
                    float(state["bias"]),
                ],
                dtype=torch.float32,
            )
            expected = torch.cat(
                (
                    reference.weight[index].detach(),
                    reference.bias[index].detach().reshape(1),
                )
            )
            max_parameter_error = max(
                max_parameter_error,
                float(torch.max(torch.abs(actual - expected)).item()),
            )

    return {
        "epochs": epochs,
        "samples_seen": epochs * len(_FEATURES),
        "initial_loss": initial_loss,
        "reference_initial_loss": reference_initial_loss,
        "final_loss": final_loss,
        "reference_final_loss": reference_final_loss,
        "loss_reduction_factor": initial_loss / final_loss,
        "max_parameter_error": max_parameter_error,
        "versions": versions,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the real three-process DANMA PyTorch convergence benchmark"
    )
    parser.add_argument(
        "--node-binary",
        type=Path,
        default=Path("target/debug/danma-node"),
        help="path to the built danma-node binary",
    )
    parser.add_argument("--epochs", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_linear_regression_benchmark(
        node_binary=args.node_binary,
        epochs=args.epochs,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
