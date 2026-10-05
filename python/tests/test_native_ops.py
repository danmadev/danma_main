"""Native PyTorch operators backed by remote Rust DANMA tensor storage."""

from __future__ import annotations

import os
import socket
import subprocess
import time
import unittest
from pathlib import Path

import torch

from danma_torch import enable_privateuse1, privateuse1_stats


ROOT = Path(__file__).resolve().parents[2]
NODE_BINARY = Path(os.environ.get("DANMA_NODE_BIN", ROOT / "target/debug/danma-node"))


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class SingleNode:
    def __init__(self) -> None:
        self.port = free_port()
        self.process = subprocess.Popen(
            [
                str(NODE_BINARY),
                "--id", "1",
                "--listen", f"127.0.0.1:{self.port}",
                "--neuron", "1",
                "--weight", "99:1",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError("DANMA node exited during startup")
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.1):
                    return
            except OSError:
                time.sleep(0.025)
        self.stop()
        raise TimeoutError("DANMA node did not start")

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=2)


class NativeAtenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Register PrivateUse1 before the first autograd Engine use.
        enable_privateuse1()

    def setUp(self) -> None:
        self.node = SingleNode()
        enable_privateuse1("127.0.0.1", self.node.port)

    def tearDown(self) -> None:
        self.node.stop()

    def test_mm_and_add_execute_in_remote_tensor_runtime(self) -> None:
        device = torch.device("danma:0")
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32).to(device)
        b = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32).to(device)
        bias = torch.tensor([1.0, -1.0], dtype=torch.float32).to(device)

        result = torch.mm(a, b) + bias
        self.assertEqual(result.device.type, "danma")
        torch.testing.assert_close(
            result.to("cpu"),
            torch.tensor([[20.0, 21.0], [44.0, 49.0]], dtype=torch.float32),
        )

        stats = privateuse1_stats()
        self.assertEqual(stats["storage"], "remote_rust")
        self.assertEqual(stats["host_payload_bytes"], 0)
        self.assertGreaterEqual(int(stats["remote_mm_ops"]), 1)
        self.assertGreaterEqual(int(stats["remote_add_ops"]), 1)

        # Mul is deliberately unsupported: native ops must remain fail-closed.
        with self.assertRaises((RuntimeError, NotImplementedError)):
            _ = a * b

    def test_ordinary_nn_linear_trains_without_danma_linear(self) -> None:
        device = torch.device("danma:0")
        layer = torch.nn.Linear(2, 2)
        with torch.no_grad():
            layer.weight.copy_(
                torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
            )
            layer.bias.copy_(torch.tensor([0.5, -0.5], dtype=torch.float32))
        layer = layer.to(device)

        x = torch.tensor([[1.0, 2.0]], dtype=torch.float32).to(device)
        x.requires_grad_(True)

        y = layer(x)
        self.assertEqual(y.device.type, "danma")
        torch.testing.assert_close(
            y.to("cpu"),
            torch.tensor([[5.5, 10.5]], dtype=torch.float32),
        )

        y.to("cpu").sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.device.type, "danma")
        torch.testing.assert_close(
            x.grad.to("cpu"),
            torch.tensor([[4.0, 6.0]], dtype=torch.float32),
        )
        torch.testing.assert_close(
            layer.weight.grad.to("cpu"),
            torch.tensor([[1.0, 2.0], [1.0, 2.0]], dtype=torch.float32),
        )
        torch.testing.assert_close(
            layer.bias.grad.to("cpu"),
            torch.tensor([1.0, 1.0], dtype=torch.float32),
        )

        optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)
        optimizer.step()

        torch.testing.assert_close(
            layer.weight.to("cpu"),
            torch.tensor([[0.9, 1.8], [2.9, 3.8]], dtype=torch.float32),
            atol=1e-6,
            rtol=0,
        )
        torch.testing.assert_close(
            layer.bias.to("cpu"),
            torch.tensor([0.4, -0.6], dtype=torch.float32),
            atol=1e-6,
            rtol=0,
        )

        with torch.no_grad():
            trained = layer(x.detach()).to("cpu")
        torch.testing.assert_close(
            trained,
            torch.tensor([[4.9, 9.9]], dtype=torch.float32),
            atol=1e-6,
            rtol=0,
        )

    def test_native_operator_fails_when_remote_tensor_runtime_is_down(self) -> None:
        device = torch.device("danma:0")
        a = torch.tensor([[1.0]], dtype=torch.float32).to(device)
        b = torch.tensor([[2.0]], dtype=torch.float32).to(device)
        self.node.stop()

        with self.assertRaises(RuntimeError):
            _ = torch.mm(a, b)


if __name__ == "__main__":
    unittest.main()
