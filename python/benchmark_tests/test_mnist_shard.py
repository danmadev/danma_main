import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from danma_torch import DANMAShardLinear
from danma_torch import mnist_benchmark1000 as base
from danma_torch import mnist_shard


class ShardMNISTUnitTests(unittest.TestCase):
    @staticmethod
    def data():
        train_x = torch.zeros((1, 784), dtype=torch.float32)
        train_y = torch.tensor([0], dtype=torch.int64)
        test_x = torch.ones((1, 784), dtype=torch.float32) * 0.01
        test_y = torch.tensor([0], dtype=torch.int64)
        return train_x, train_y, test_x, test_y

    def test_layout_is_one_local_node(self):
        layout = mnist_shard.make_layout(1000)
        self.assertEqual(layout.nodes, 1)
        self.assertEqual(len(layout.hidden_ids), 1000)
        self.assertEqual(len(layout.output_ids), 10)

    def test_remote_model_uses_shard_layers(self):
        client = mock.Mock(spec=base.DANMAClient)
        # Constructor checks exact DANMAClient type through the inherited layer.
        client.__class__ = base.DANMAClient
        layout = mnist_shard.make_layout(3)
        model = mnist_shard.ShardRemoteModel(client, layout)
        self.assertIsInstance(model.hidden, DANMAShardLinear)
        self.assertIsInstance(model.output, DANMAShardLinear)

    def test_cpu_only_wrapper_reports_executor_contract(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            base, "load_mnist", return_value=self.data()
        ):
            report = mnist_shard.run_benchmark(
                hidden_neurons=3,
                run_dir=Path(directory),
                backends=("cpu",),
                train_samples=1,
                test_samples=1,
                epochs=1,
                download=False,
            )
        self.assertEqual(report["status"], "ok")
        self.assertFalse(report["executor"]["global_barrier"])
        self.assertEqual(report["executor"]["expected_train_rpc_per_sample"], 4)
        self.assertFalse(report["executor"]["dense_gemm"])


@unittest.skipUnless(
    Path(os.environ.get("DANMA_NODE_BIN", "target/debug/danma-node")).is_file(),
    "real node binary missing",
)
class RealShardMNISTTests(unittest.TestCase):
    def test_small_cpu_and_shard_danma_run_have_numerical_parity(self):
        train_x = torch.zeros((1, 784), dtype=torch.float32)
        train_y = torch.tensor([0], dtype=torch.int64)
        test_x = torch.ones((1, 784), dtype=torch.float32) * 0.01
        test_y = torch.tensor([0], dtype=torch.int64)
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            base,
            "load_mnist",
            return_value=(train_x, train_y, test_x, test_y),
        ):
            report = mnist_shard.run_benchmark(
                hidden_neurons=3,
                run_dir=Path(directory),
                node_binary=Path(
                    os.environ.get("DANMA_NODE_BIN", "target/debug/danma-node")
                ),
                backends=("cpu", "danma"),
                train_samples=1,
                test_samples=1,
                epochs=1,
                download=False,
            )
        self.assertEqual(report["status"], "ok")
        comparison = report["comparisons"]["danma_vs_cpu"]
        self.assertTrue(comparison["numerical_parity_passed"])
        requests = report["results"]["danma"]["logical_requests"]
        self.assertEqual(requests.get("train:forward_shard"), 2)
        self.assertEqual(requests.get("train:backward_shard"), 2)
        versions = report["results"]["danma"]["version_checks"][-1]
        self.assertTrue(versions["all_expected"])
        self.assertEqual(versions["expected"], 1)


if __name__ == "__main__":
    unittest.main()
