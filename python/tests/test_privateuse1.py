"""Fail-closed PrivateUse1 proof for the DANMA PyTorch integration.

These tests prove three separate properties:
1. a tensor genuinely carries the PrivateUse1/DANMA dispatch key;
2. unsupported operators cannot silently execute through a CPU fallback; and
3. DANMALinear forward/backward requires and mutates the real Rust neuron.
"""

from __future__ import annotations

import os
import socket
import subprocess
import time
import unittest
from pathlib import Path

import torch

from danma_torch import (
    DANMAClient,
    DANMALinear,
    DANMATransportError,
    enable_privateuse1,
    privateuse1_stats,
)


ROOT = Path(__file__).resolve().parents[2]
NODE_BINARY = Path(os.environ.get("DANMA_NODE_BIN", ROOT / "target/debug/danma-node"))


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class SingleNode:
    def __init__(self) -> None:
        port = free_port()
        self.process = subprocess.Popen(
            [
                str(NODE_BINARY),
                "--id", "1",
                "--listen", f"127.0.0.1:{port}",
                "--neuron", "11",
                "--weight", "901:2",
                "--weight", "902:3",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.client = DANMAClient("127.0.0.1", port)

        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError("DANMA node exited during startup")
            try:
                if self.client.inspect(11)["version"] == 0:
                    return
            except Exception:
                pass
            time.sleep(0.05)
        self.stop()
        raise TimeoutError("DANMA node did not become ready")

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=2)


class PrivateUse1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # PyTorch 2.8 sizes accelerator autograd queues when its Engine first
        # starts. Register DANMA before the first backward in this process.
        enable_privateuse1()

    def test_danma_device_is_privateuse1_and_has_no_cpu_operator_fallback(self) -> None:
        device = torch.device("danma:0")
        host = torch.tensor([1.25, -2.5], dtype=torch.float32)
        tensor = host.to(device)

        self.assertEqual(tensor.device.type, "danma")
        self.assertFalse(tensor.is_cpu)
        torch.testing.assert_close(tensor.to("cpu"), host)

        stats = privateuse1_stats()
        self.assertFalse(stats["cpu_fallback"])
        self.assertGreaterEqual(int(stats["allocations"]), 1)
        self.assertGreaterEqual(int(stats["copies"]), 2)

        # Deliberately unsupported: a generic CPU fallback would make this
        # succeed. DANMA must fail closed instead.
        with self.assertRaises((RuntimeError, NotImplementedError)):
            _ = tensor + tensor

    def test_danma_forward_has_no_local_fallback_when_rust_node_is_down(self) -> None:
        cluster = SingleNode()
        model = DANMALinear(
            cluster.client,
            neuron_ids=(11,),
            input_ids=(901, 902),
        )
        inputs = torch.tensor([1.0, 2.0], dtype=torch.float32).to("danma:0")
        cluster.stop()

        with self.assertRaises(DANMATransportError):
            model(inputs)

    def test_danma_tensor_forward_backward_mutates_remote_rust_neuron(self) -> None:
        cluster = SingleNode()
        try:
            model = DANMALinear(
                cluster.client,
                neuron_ids=(11,),
                input_ids=(901, 902),
            )
            inputs = torch.tensor(
                [1.0, 2.0],
                dtype=torch.float32,
            ).to("danma:0")
            inputs.requires_grad_(True)

            before = cluster.client.inspect(11)
            self.assertEqual(before["version"], 0)

            output = model(inputs)
            self.assertEqual(output.device.type, "danma")
            torch.testing.assert_close(
                output.to("cpu"),
                torch.tensor([8.0], dtype=torch.float32),
            )

            # Loss reduction is ordinary PyTorch CPU bookkeeping. The affine
            # forward/backward itself crosses the DANMA TCP boundary.
            output.to("cpu").sum().backward()

            after = cluster.client.inspect(11)
            self.assertEqual(after["version"], 1)
            self.assertAlmostEqual(float(after["weights"]["901"]), 1.9, places=6)
            self.assertAlmostEqual(float(after["weights"]["902"]), 2.8, places=6)
            self.assertAlmostEqual(float(after["bias"]), -0.1, places=6)

            self.assertIsNotNone(inputs.grad)
            self.assertEqual(inputs.grad.device.type, "danma")
            torch.testing.assert_close(
                inputs.grad.to("cpu"),
                torch.tensor([2.0, 3.0], dtype=torch.float32),
            )

            # A second inference sees the new Rust-owned parameters, not a
            # local cached torch.nn.Linear parameter set.
            model.eval()
            with torch.no_grad():
                updated = model(inputs.detach()).to("cpu")
            torch.testing.assert_close(
                updated,
                torch.tensor([7.4], dtype=torch.float32),
                atol=1e-6,
                rtol=0,
            )
        finally:
            cluster.stop()


if __name__ == "__main__":
    unittest.main()
