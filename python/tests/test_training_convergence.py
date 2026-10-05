"""End-to-end convergence test for the real PyTorch -> DANMA training path."""

from __future__ import annotations

import os
from pathlib import Path
import unittest

from danma_torch.training_benchmark import run_linear_regression_benchmark


ROOT = Path(__file__).resolve().parents[2]
NODE_BINARY = Path(os.environ.get("DANMA_NODE_BIN", ROOT / "target/debug/danma-node"))


class TrainingConvergenceTests(unittest.TestCase):
    def test_distributed_regression_converges_and_matches_torch_sgd(self) -> None:
        result = run_linear_regression_benchmark(
            node_binary=NODE_BINARY,
            epochs=30,
        )

        self.assertEqual(result["epochs"], 30)
        self.assertEqual(result["samples_seen"], 120)
        self.assertEqual(result["versions"], [120, 120, 120])

        self.assertLess(
            result["final_loss"],
            result["initial_loss"] * 0.02,
            "DANMA training should reduce the held-out full-dataset loss by at least 50x",
        )
        self.assertAlmostEqual(
            result["final_loss"],
            result["reference_final_loss"],
            delta=2e-5,
            msg="DANMA convergence should track the equivalent sequential PyTorch SGD run",
        )
        self.assertLess(
            result["max_parameter_error"],
            2e-5,
            "remote DANMA weights/bias should match the PyTorch CPU reference",
        )


if __name__ == "__main__":
    unittest.main()
