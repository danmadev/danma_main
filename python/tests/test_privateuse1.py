"""Fail-closed PrivateUse1 proof for the DANMA PyTorch integration.

These tests prove two separate properties:
1. a tensor can genuinely carry the PrivateUse1/DANMA dispatch key; and
2. DANMALinear changes the remote Rust neuron's state, so the affine math
   cannot be satisfied by an ordinary PyTorch CPU linear fallback.
"""

from __future__ import annotations

import os
import socket
import subprocess
import time
import unittest
from pathlib import Path

import torch

from danma_torch import DANMAClient, DANMALinear, enable_privateuse1


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
        enable_privateuse1()

    def test_danma_device_is_privateuse1_and_has_no_cpu_operator_fallback(self) -> None:
        device = torch.device("danma:0")
        host = torch.tensor([1.25, -2.5], dtype=torch.float32)
        tensor = host.to(device)

        self.assertEqual(tensor.device.type, "danma")
        self.assertFalse(tensor.is_cpu)
        torch.testing.assert_close(tensor.to("cpu"), host)

        # Deliberately unsupported: a fallback to CPU would make this succeed.
        # DANMA must fail closed instead of silently executing ATen math on CPU.
        with self.assertRaises((RuntimeError, NotImplementedError)):
            _ = tensor + tensor

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

            # Loss math is intentionally ordinary CPU PyTorch. The affine
            # forward/backward itself must still cross the DANMA TCP boundary.
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
        finally:
            cluster.stop()


if __name__ == "__main__":
    unittest.main()
