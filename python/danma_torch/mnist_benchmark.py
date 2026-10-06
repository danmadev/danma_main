"""MNIST quality/speed benchmark: PyTorch CPU vs CUDA vs distributed DANMA."""

from __future__ import annotations

import argparse
import json
import math
import platform
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from .client import DANMAClient, DANMAError
from .layer import DANMALinear

INPUT_IDS = tuple(range(20_001, 20_001 + 784))
NEURON_IDS = tuple(range(10_001, 10_011))
LR = 0.1


def _balanced(images: torch.Tensor, labels: torch.Tensor, count: int, seed: int):
    """Return exactly count stratified examples, as balanced as capacity allows.

    MNIST's 10k test split is not class-balanced. The previous implementation
    silently returned fewer rows when a requested per-class quota exceeded the
    available class capacity. This allocator redistributes that deficit to
    classes with spare examples while preserving deterministic sampling.
    """
    if not 1 <= count <= len(labels):
        raise ValueError("subset size outside dataset")
    if labels.ndim != 1 or len(images) != len(labels):
        raise ValueError("images/labels length mismatch")

    capacities = torch.bincount(labels, minlength=10).tolist()
    if len(capacities) != 10:
        raise ValueError("labels must contain MNIST classes 0..9")

    per_class, extra = divmod(count, 10)
    quotas = [min(capacities[cls], per_class + int(cls < extra)) for cls in range(10)]
    deficit = count - sum(quotas)

    while deficit:
        progressed = False
        for cls in range(10):
            if quotas[cls] < capacities[cls]:
                quotas[cls] += 1
                deficit -= 1
                progressed = True
                if deficit == 0:
                    break
        if not progressed:
            raise ValueError("cannot allocate requested stratified subset")

    g = torch.Generator().manual_seed(seed)
    chunks = []
    for cls, take in enumerate(quotas):
        candidates = torch.nonzero(labels == cls, as_tuple=False).flatten()
        order = torch.randperm(len(candidates), generator=g)[:take]
        chunks.append(candidates[order])
    index = torch.cat(chunks)
    if len(index) != count:
        raise AssertionError("stratified sampler did not return the requested count")
    index = index[torch.randperm(len(index), generator=g)]
    return images[index].contiguous(), labels[index].contiguous()


def load_mnist(data_dir: Path, train_samples: int, test_samples: int, seed: int, download: bool):
    try:
        from torchvision.datasets import MNIST
    except ImportError as exc:
        raise RuntimeError(
            "MNIST benchmark requires torchvision; install a version matching PyTorch"
        ) from exc

    train = MNIST(root=data_dir, train=True, download=download)
    test = MNIST(root=data_dir, train=False, download=download)
    train_x = train.data.reshape(-1, 784).float().div_(255.0)
    test_x = test.data.reshape(-1, 784).float().div_(255.0)
    train_y = train.targets.long()
    test_y = test.targets.long()
    train_x, train_y = _balanced(train_x, train_y, train_samples, seed)
    test_x, test_y = _balanced(test_x, test_y, test_samples, seed + 1)
    return train_x, train_y, test_x, test_y


def _orders(count: int, epochs: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    return [torch.randperm(count, generator=g) for _ in range(epochs)]


def _batches(order: torch.Tensor, batch_size: int):
    for start in range(0, len(order), batch_size):
        yield order[start : start + batch_size]


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    xs = sorted(values)

    def q(p: float) -> float:
        pos = (len(xs) - 1) * p
        lo, hi = math.floor(pos), math.ceil(pos)
        if lo == hi:
            return xs[lo]
        return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)

    return {
        "mean": sum(xs) / len(xs),
        "p50": q(0.50),
        "p95": q(0.95),
        "max": xs[-1],
    }


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _Cluster:
    def __init__(self, binary: Path, nodes: int):
        if not 1 <= nodes <= 10:
            raise ValueError("nodes must be 1..10")
        self.binary = Path(binary)
        self.nodes = nodes
        self.processes: list[subprocess.Popen[bytes]] = []
        self.ports: list[int] = []
        self.startup_seconds = 0.0

    def __enter__(self):
        if not self.binary.is_file():
            raise FileNotFoundError(self.binary)
        started = time.perf_counter()
        while len(self.ports) < self.nodes:
            port = _free_port()
            if port not in self.ports:
                self.ports.append(port)

        owners: dict[int, int] = {}
        for node_index in range(self.nodes):
            argv = [
                str(self.binary), "--id", str(node_index + 1),
                "--listen", f"127.0.0.1:{self.ports[node_index]}",
            ]
            for output_index, neuron in enumerate(NEURON_IDS):
                if output_index % self.nodes != node_index:
                    continue
                owners[neuron] = node_index + 1
                argv += ["--neuron", str(neuron)]
                for source in INPUT_IDS:
                    argv += ["--weight", f"{source}:0"]
            for peer_index, port in enumerate(self.ports):
                if peer_index != node_index:
                    argv += ["--peer", f"{peer_index + 1}@127.0.0.1:{port}"]
            self.processes.append(
                subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            )

        self.client = DANMAClient("127.0.0.1", self.ports[0], timeout_seconds=10.0)
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if any(p.poll() is not None for p in self.processes):
                self.stop()
                raise RuntimeError("DANMA node exited before route convergence")
            try:
                routes = self.client.routes()
                if all(routes.get(str(n)) == owner for n, owner in owners.items()):
                    self.startup_seconds = time.perf_counter() - started
                    return self
            except (OSError, DANMAError):
                pass
            time.sleep(0.05)
        self.stop()
        raise TimeoutError("DANMA MNIST cluster did not converge")

    def __exit__(self, *_):
        self.stop()

    def stop(self):
        for p in self.processes:
            if p.poll() is None:
                p.terminate()
        for p in self.processes:
            try:
                p.wait(timeout=2)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(timeout=2)


def _torch_eval(model, images, labels, device, batch_size):
    model.eval()
    total_loss = 0.0
    correct = 0
    latencies: list[float] = []
    cuda_events = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        for index in _batches(torch.arange(len(labels)), batch_size):
            t0 = time.perf_counter()
            if device.type == "cuda":
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                begin.record()
            x, y = images[index].to(device), labels[index].to(device)
            logits = model(x)
            total_loss += float(F.cross_entropy(logits, y, reduction="sum").item())
            correct += int((logits.argmax(1) == y).sum().item())
            if device.type == "cuda":
                end.record()
                cuda_events.append((begin, end))
            else:
                latencies.append(time.perf_counter() - t0)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        latencies = [a.elapsed_time(b) / 1000 for a, b in cuda_events]
    elapsed = time.perf_counter() - started
    return total_loss / len(labels), correct / len(labels), elapsed, latencies


def _run_torch(backend, train_x, train_y, test_x, test_y, orders, batch_size):
    device = torch.device(backend)
    model = torch.nn.Linear(784, 10).to(device)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)

    epoch_seconds: list[float] = []
    latencies: list[float] = []
    final_train_loss = 0.0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_started = time.perf_counter()
    for order in orders:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_started = time.perf_counter()
        total_loss = 0.0
        seen = 0
        cuda_events = []
        model.train()
        for index in _batches(order, batch_size):
            t0 = time.perf_counter()
            if device.type == "cuda":
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                begin.record()
            x, y = train_x[index].to(device), train_y[index].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            if device.type == "cuda":
                end.record()
                cuda_events.append((begin, end))
            else:
                latencies.append(time.perf_counter() - t0)
            total_loss += float(loss.detach().item()) * len(index)
            seen += len(index)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            latencies.extend(a.elapsed_time(b) / 1000 for a, b in cuda_events)
        final_train_loss = total_loss / seen
        epoch_seconds.append(time.perf_counter() - epoch_started)
    train_seconds = time.perf_counter() - train_started

    test_loss, accuracy, eval_seconds, eval_latencies = _torch_eval(
        model, test_x, test_y, device, batch_size
    )
    state = torch.cat(
        [model.weight.detach().cpu().flatten(), model.bias.detach().cpu()]
    )
    metrics: dict[str, Any] = {
        "status": "ok",
        "device": str(device),
        "final_train_loss": final_train_loss,
        "test_loss": test_loss,
        "test_accuracy": accuracy,
        "train_seconds": train_seconds,
        "eval_seconds": eval_seconds,
        "epoch_seconds": epoch_seconds,
        "train_samples_per_second": len(train_y) * len(orders) / train_seconds,
        "eval_samples_per_second": len(test_y) / eval_seconds,
        "train_batch_latency_seconds": _summary(latencies),
        "eval_batch_latency_seconds": _summary(eval_latencies),
    }
    if device.type == "cuda":
        metrics.update({
            "cuda_device_name": torch.cuda.get_device_name(device),
            "cuda_runtime": torch.version.cuda,
            "cuda_compute_capability": list(torch.cuda.get_device_capability(device)),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        })
    return metrics, state


def _danma_eval(model, images, labels, batch_size):
    model.eval()
    total_loss = 0.0
    correct = 0
    latencies: list[float] = []
    started = time.perf_counter()
    with torch.no_grad():
        for index in _batches(torch.arange(len(labels)), batch_size):
            t0 = time.perf_counter()
            logits = model(images[index])
            y = labels[index]
            total_loss += float(F.cross_entropy(logits, y, reduction="sum").item())
            correct += int((logits.argmax(1) == y).sum().item())
            latencies.append(time.perf_counter() - t0)
    elapsed = time.perf_counter() - started
    return total_loss / len(labels), correct / len(labels), elapsed, latencies


def _run_danma(binary, nodes, train_x, train_y, test_x, test_y, orders, batch_size):
    with _Cluster(binary, nodes) as cluster:
        model = DANMALinear(
            cluster.client,
            neuron_ids=NEURON_IDS,
            input_ids=INPUT_IDS,
            feedback_ttl_ms=3_000,
            max_batch=9,
        )
        epoch_seconds: list[float] = []
        latencies: list[float] = []
        final_train_loss = 0.0
        train_started = time.perf_counter()
        for order in orders:
            epoch_started = time.perf_counter()
            total_loss = 0.0
            seen = 0
            model.train()
            for index in _batches(order, batch_size):
                t0 = time.perf_counter()
                y = train_y[index]
                loss = F.cross_entropy(model(train_x[index]), y)
                loss.backward()
                latencies.append(time.perf_counter() - t0)
                total_loss += float(loss.detach().item()) * len(index)
                seen += len(index)
            final_train_loss = total_loss / seen
            epoch_seconds.append(time.perf_counter() - epoch_started)
        train_seconds = time.perf_counter() - train_started

        test_loss, accuracy, eval_seconds, eval_latencies = _danma_eval(
            model, test_x, test_y, batch_size
        )
        states = [cluster.client.inspect(n) for n in NEURON_IDS]
        weights, biases, versions = [], [], []
        for state in states:
            weights.extend(float(state["weights"][str(source)]) for source in INPUT_IDS)
            biases.append(float(state["bias"]))
            versions.append(int(state["version"]))
        parameter_state = torch.tensor(weights + biases, dtype=torch.float32)

        updates = len(train_y) * len(orders)
        train_requests = 2 * updates * len(NEURON_IDS)
        metrics = {
            "status": "ok",
            "device": "danma-remote-cpu",
            "nodes": nodes,
            "cluster_startup_seconds": cluster.startup_seconds,
            "end_to_end_seconds": cluster.startup_seconds + train_seconds + eval_seconds,
            "final_train_loss": final_train_loss,
            "test_loss": test_loss,
            "test_accuracy": accuracy,
            "train_seconds": train_seconds,
            "eval_seconds": eval_seconds,
            "epoch_seconds": epoch_seconds,
            "train_samples_per_second": updates / train_seconds,
            "eval_samples_per_second": len(test_y) / eval_seconds,
            "train_batch_latency_seconds": _summary(latencies),
            "eval_batch_latency_seconds": _summary(eval_latencies),
            "versions": versions,
            "expected_version_per_neuron": updates,
            "train_transport_requests_per_second": train_requests / train_seconds,
            "transport_requests": {
                "train_forward": updates * len(NEURON_IDS),
                "train_backward": updates * len(NEURON_IDS),
                "eval_forward": len(test_y) * len(NEURON_IDS),
                "inspect": len(NEURON_IDS),
            },
        }
        return metrics, parameter_state


def _comparisons(results, states):
    out: dict[str, Any] = {}
    cpu = results.get("cpu")
    if cpu and cpu.get("status") == "ok":
        for name, result in results.items():
            if name == "cpu" or result.get("status") != "ok":
                continue
            item = {
                "train_throughput_ratio_vs_cpu":
                    result["train_samples_per_second"] / cpu["train_samples_per_second"],
                "eval_throughput_ratio_vs_cpu":
                    result["eval_samples_per_second"] / cpu["eval_samples_per_second"],
                "test_accuracy_gap_vs_cpu_percentage_points":
                    100 * (result["test_accuracy"] - cpu["test_accuracy"]),
                "test_loss_delta_vs_cpu": result["test_loss"] - cpu["test_loss"],
            }
            if name in states:
                item["max_parameter_error_vs_cpu"] = float(
                    torch.max(torch.abs(states[name] - states["cpu"])).item()
                )
            out[name] = item
    cuda, danma = results.get("cuda"), results.get("danma")
    if cuda and danma and cuda.get("status") == danma.get("status") == "ok":
        out["danma_vs_cuda"] = {
            "train_throughput_ratio":
                danma["train_samples_per_second"] / cuda["train_samples_per_second"],
            "eval_throughput_ratio":
                danma["eval_samples_per_second"] / cuda["eval_samples_per_second"],
            "test_accuracy_gap_percentage_points":
                100 * (danma["test_accuracy"] - cuda["test_accuracy"]),
            "test_loss_delta": danma["test_loss"] - cuda["test_loss"],
        }
    return out


def run_mnist_benchmark(
    *,
    node_binary: Path,
    data_dir: Path,
    train_samples: int = 1024,
    test_samples: int = 512,
    epochs: int = 3,
    batch_size: int = 8,
    seed: int = 7,
    nodes: int = 3,
    backends: tuple[str, ...] = ("cpu", "cuda", "danma"),
    download: bool = True,
    require_cuda: bool = False,
):
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if not 1 <= batch_size <= 9:
        raise ValueError("batch_size must be 1..9 for a DANMA-comparable run")
    unknown = set(backends) - {"cpu", "cuda", "danma"}
    if unknown:
        raise ValueError(f"unknown backends: {sorted(unknown)}")

    loaded = time.perf_counter()
    train_x, train_y, test_x, test_y = load_mnist(
        data_dir, train_samples, test_samples, seed, download
    )
    data_seconds = time.perf_counter() - loaded
    orders = _orders(len(train_y), epochs, seed + 2)

    results, states = {}, {}
    for backend in backends:
        if backend == "cuda" and not torch.cuda.is_available():
            if require_cuda:
                raise RuntimeError("CUDA requested but unavailable")
            results["cuda"] = {"status": "unavailable", "reason": "CUDA unavailable"}
            continue
        if backend == "cuda":
            torch.cuda.reset_peak_memory_stats()
        if backend in {"cpu", "cuda"}:
            metrics, state = _run_torch(
                backend, train_x, train_y, test_x, test_y, orders, batch_size
            )
        else:
            metrics, state = _run_danma(
                node_binary, nodes, train_x, train_y, test_x, test_y, orders, batch_size
            )
        results[backend], states[backend] = metrics, state

    return {
        "benchmark": "mnist-linear-784x10",
        "model": {
            "input_features": 784,
            "classes": 10,
            "initialization": "zeros",
            "optimizer": "SGD",
            "learning_rate": LR,
            "loss": "cross_entropy",
        },
        "dataset": {
            "train_samples": len(train_y),
            "test_samples": len(test_y),
            "balanced_subset": True,
            "load_and_subset_seconds": data_seconds,
        },
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "seed": seed,
            "same_epoch_order_for_all_backends": True,
        },
        "timing_semantics": {
            "train_seconds":
                "wall time including host-to-backend staging/serialization",
            "eval_seconds":
                "wall time including host-to-backend staging/serialization",
            "danma_cluster_startup_excluded_from_train_seconds": True,
            "cuda_batch_latency":
                "CUDA-event device latency; epoch wall time synchronizes at boundaries",
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "cpu_threads": torch.get_num_threads(),
        },
        "results": results,
        "comparisons": _comparisons(results, states),
    }


def _parse_backends(value: str):
    result = tuple(x.strip().lower() for x in value.split(",") if x.strip())
    if not result or len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("backends must be a non-empty unique list")
    return result


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--node-binary", type=Path, default=Path("target/debug/danma-node"))
    parser.add_argument("--data-dir", type=Path, default=Path.home() / ".cache/danma/mnist")
    parser.add_argument("--train-samples", type=int, default=1024)
    parser.add_argument("--test-samples", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--nodes", type=int, default=3)
    parser.add_argument("--backends", type=_parse_backends, default=("cpu", "cuda", "danma"))
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    report = run_mnist_benchmark(
        node_binary=args.node_binary,
        data_dir=args.data_dir,
        train_samples=args.train_samples,
        test_samples=args.test_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        nodes=args.nodes,
        backends=args.backends,
        download=not args.no_download,
        require_cuda=args.require_cuda,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
