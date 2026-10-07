import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from danma_torch.client import DANMAClient
from danma_torch import mnist_benchmark1000 as base
from danma_torch import mnist_rust_multinode as bench


class RecordingClient(DANMAClient):
    def __init__(self, node):
        super().__init__("127.0.0.1", 12000 + node)
        self.node = node
        self.phase = "test"
        self.forward_calls = []
        self.backward_calls = []

    def request(self, message):
        raise AssertionError("unit fake must not use generic request")

    def forward_shard(self, **kwargs):
        self.forward_calls.append(kwargs)
        return [float(neuron_id) for neuron_id in kwargs["neuron_ids"]]

    def backward_shard(self, **kwargs):
        self.backward_calls.append(kwargs)
        return [float(self.node)] * len(kwargs["input_ids"])


class RustMultiNodeMNISTUnitTests(unittest.TestCase):
    @staticmethod
    def data():
        train_x = torch.zeros((1, 784), dtype=torch.float32)
        train_y = torch.tensor([0], dtype=torch.int64)
        test_x = torch.ones((1, 784), dtype=torch.float32) * 0.01
        test_y = torch.tensor([0], dtype=torch.int64)
        return train_x, train_y, test_x, test_y

    def test_1000_hidden_layout_accepts_variable_node_counts(self):
        for nodes in (1, 2, 3, 4, 10, 100, 1010):
            with self.subTest(nodes=nodes):
                layout = bench.make_layout(1000, nodes)
                self.assertEqual(layout.nodes, nodes)
                self.assertEqual(len(layout.partitions), nodes)
                self.assertEqual(sum(map(len, layout.partitions)), 1010)
                self.assertTrue(all(layout.partitions))
        for nodes in (0, -1, 1011, True, 1.5, "3"):
            with self.subTest(nodes=nodes):
                with self.assertRaises(ValueError):
                    bench.make_layout(1000, nodes)

    def test_three_node_placement_and_rpc_count_remain_expected(self):
        layout = bench.make_layout(1000, 3)
        self.assertEqual([len(ids) for ids in layout.partitions], [337, 337, 336])
        self.assertEqual(
            bench.layer_owner_counts(layout),
            {"hidden": [337, 337, 326], "output": [0, 0, 10]},
        )
        self.assertEqual(
            bench.expected_rpc_per_sample(layout),
            {
                "hidden_owners": 3,
                "output_owners": 1,
                "forward": 4,
                "backward": 4,
                "train_total": 8,
            },
        )

    def test_rpc_count_is_derived_from_actual_placement(self):
        expected = {
            1: 4,
            3: 8,
            4: 10,
            10: 22,
            1010: 2020,
        }
        for nodes, train_total in expected.items():
            with self.subTest(nodes=nodes):
                layout = bench.make_layout(1000, nodes)
                self.assertEqual(
                    bench.expected_rpc_per_sample(layout)["train_total"],
                    train_total,
                )

    def test_multinode_client_partitions_targets_and_reduces_dx(self):
        clients = [RecordingClient(node) for node in (1, 2, 3, 4)]
        owners = {11: 1, 12: 2, 13: 3, 14: 4, 15: 1}
        client = bench.MultiNodeShardClient(clients, owners)
        client.phase = "train"

        output = client.forward_shard(
            neuron_ids=(11, 12, 13, 14, 15),
            event_ids=(101, 102, 103, 104, 105),
            trace_id=500,
            inputs=((201, 1.0), (202, 2.0)),
            training=True,
        )
        self.assertEqual(output, [11.0, 12.0, 13.0, 14.0, 15.0])
        self.assertEqual([len(item.forward_calls) for item in clients], [1, 1, 1, 1])
        self.assertEqual(clients[0].forward_calls[0]["neuron_ids"], [11, 15])
        self.assertTrue(all(item.phase == "train" for item in clients))

        dx = client.backward_shard(
            neuron_ids=(11, 12, 13, 14, 15),
            event_ids=(101, 102, 103, 104, 105),
            gradients=(1.0, 2.0, 3.0, 4.0, 5.0),
            input_ids=(201, 202),
            feedback_ttl_ms=3000,
        )
        self.assertEqual(dx, [10.0, 10.0])
        self.assertEqual([len(item.backward_calls) for item in clients], [1, 1, 1, 1])

    def test_cli_accepts_variable_nodes_and_rejects_abbreviation(self):
        for nodes in (1, 3, 4, 10, 1010):
            args = bench.parse_args(
                ["--nodes", str(nodes), "--backends", "cpu", "--no-download"]
            )
            self.assertEqual(args.nodes, nodes)
        with self.assertRaises(SystemExit):
            bench.parse_args(["--nodes", "0"])
        with self.assertRaises(SystemExit):
            bench.parse_args(["--nodes", "1011"])
        with self.assertRaises(SystemExit):
            bench.parse_args(["--node", "3"])

    def test_cpu_only_report_describes_selected_node_count(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            base, "load_mnist", return_value=self.data()
        ):
            report = bench.run_benchmark(
                hidden_neurons=1000,
                nodes=4,
                run_dir=Path(directory),
                backends=("cpu",),
                train_samples=1,
                test_samples=1,
                epochs=1,
                download=False,
            )
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["model"]["nodes"], 4)
        self.assertEqual(report["executor"]["nodes"], 4)
        self.assertFalse(report["executor"]["global_barrier"])
        self.assertEqual(report["executor"]["expected_rpc_per_sample"]["train_total"], 10)
        self.assertEqual(
            report["executor"]["placement"]["partition_sizes"],
            [253, 253, 252, 252],
        )


@unittest.skipUnless(
    Path(os.environ.get("DANMA_NODE_BIN", "target/debug/danma-node")).is_file(),
    "real node binary missing",
)
class RealRustMultiNodeMNISTTests(unittest.TestCase):
    def _run(self, nodes):
        train_x = torch.zeros((1, 784), dtype=torch.float32)
        train_y = torch.tensor([0], dtype=torch.int64)
        test_x = torch.ones((1, 784), dtype=torch.float32) * 0.01
        test_y = torch.tensor([0], dtype=torch.int64)
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            base,
            "load_mnist",
            return_value=(train_x, train_y, test_x, test_y),
        ):
            return bench.run_benchmark(
                hidden_neurons=1000,
                nodes=nodes,
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

    def test_1000_hidden_cpu_vs_configurable_danma_nodes_have_parity(self):
        for nodes in (1, 3, 4):
            with self.subTest(nodes=nodes):
                report = self._run(nodes)
                self.assertEqual(report["status"], "ok")
                comparison = report["comparisons"]["danma_vs_cpu"]
                self.assertTrue(comparison["numerical_parity_passed"])
                self.assertLessEqual(comparison["max_parameter_error"], 2e-5)
                self.assertEqual(report["model"]["nodes"], nodes)

                expected = bench.expected_rpc_per_sample(
                    bench.make_layout(1000, nodes)
                )
                requests = report["results"]["danma"]["logical_requests"]
                self.assertEqual(
                    requests.get("train:forward_shard"),
                    expected["forward"],
                )
                self.assertEqual(
                    requests.get("train:backward_shard"),
                    expected["backward"],
                )
                versions = report["results"]["danma"]["version_checks"][-1]
                self.assertTrue(versions["all_expected"])
                self.assertEqual(versions["count"], 1010)
                self.assertEqual(versions["expected"], 1)


if __name__ == "__main__":
    unittest.main()
