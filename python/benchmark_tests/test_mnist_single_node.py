import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from danma_torch import mnist_benchmark1000 as base
from danma_torch import mnist_single_node as single


class SingleNodeBenchmarkTests(unittest.TestCase):
    @staticmethod
    def data():
        train_x = torch.zeros((1, 784), dtype=torch.float32)
        train_y = torch.tensor([0], dtype=torch.int64)
        test_x = torch.ones((1, 784), dtype=torch.float32) * 0.01
        test_y = torch.tensor([0], dtype=torch.int64)
        return train_x, train_y, test_x, test_y

    def test_default_layout_is_one_process_with_1000_hidden(self):
        layout = single.make_layout()
        self.assertEqual(layout.hidden, 1000)
        self.assertEqual(layout.nodes, 1)
        self.assertEqual(len(layout.neuron_ids), 1010)
        self.assertEqual(len(layout.partitions), 1)
        self.assertEqual(len(layout.partitions[0]), 1010)

    def test_hidden_width_is_configurable_with_explicit_bounds(self):
        for hidden in (1, 32, 512, 1000, 1024):
            with self.subTest(hidden=hidden):
                self.assertEqual(single.make_layout(hidden).hidden, hidden)
        for hidden in (0, 1025, -1, True, 1.5, "1000"):
            with self.subTest(hidden=hidden):
                with self.assertRaises(ValueError):
                    single.make_layout(hidden)

    def test_node_count_is_configurable(self):
        for nodes in (1, 2, 4, 10):
            with self.subTest(nodes=nodes):
                layout = single.make_layout(1000, nodes)
                self.assertEqual(layout.nodes, nodes)
                self.assertEqual(len(layout.partitions), nodes)
                self.assertEqual(sum(map(len, layout.partitions)), 1010)
        for nodes in (0, 11, -1, True, 1.5, "4"):
            with self.subTest(nodes=nodes):
                with self.assertRaises(ValueError):
                    single.make_layout(1000, nodes)

    def test_default_width_fits_runtime_startup_bounds(self):
        layout = single.make_layout(1000)
        total_weights = layout.inputs * layout.hidden + layout.hidden * layout.outputs
        self.assertLessEqual(len(layout.neuron_ids), base.MAX_CONFIG_NEURONS)
        self.assertLessEqual(total_weights, base.MAX_CONFIG_TOTAL_WEIGHTS)
        self.assertEqual(total_weights, 794000)

    def test_small_one_node_config_is_written_as_one_file(self):
        layout = base.Layout(inputs=4, hidden=3, outputs=2, nodes=1)
        initial = base.shared_initialization(7, layout)
        with tempfile.TemporaryDirectory() as directory:
            paths, metadata = base.write_configs(Path(directory), layout, initial)
            self.assertEqual(len(paths), 1)
            self.assertEqual(metadata[0]['neurons'], 5)
            data = json.loads(paths[0].read_text())
            self.assertEqual(len(data['neurons']), 5)
            base.validate_config(data)

    def test_cpu_only_run_reports_selected_hidden_width_and_nodes(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            base, 'load_mnist', return_value=self.data()
        ):
            report = single.run_benchmark(
                hidden_neurons=3,
                nodes=4,
                run_dir=Path(directory),
                backends=('cpu',),
                train_samples=1,
                test_samples=1,
                epochs=1,
                download=False,
            )
        self.assertEqual(report['status'], 'ok')
        self.assertEqual(report['model']['hidden_features'], 3)
        self.assertEqual(report['model']['nodes'], 4)
        self.assertEqual(report['model']['logical_neurons'], 13)
        self.assertEqual(report['model']['parameters'], 2395)

    def test_parser_defaults_and_accepts_nodes_flag(self):
        args = single.parse_args(['--backends', 'cpu', '--no-download'])
        self.assertEqual((args.hidden_neurons, args.nodes), (1000, 1))
        layout = single.validate_options(**vars(args))
        self.assertEqual(layout.nodes, 1)

        args = single.parse_args(['--nodes', '4', '--backends', 'cpu', '--no-download'])
        self.assertEqual(args.nodes, 4)
        self.assertEqual(single.validate_options(**vars(args)).nodes, 4)


@unittest.skipUnless(
    Path(os.environ.get('DANMA_NODE_BIN', 'target/debug/danma-node')).is_file(),
    'real node binary missing',
)
class RealVariableNodeBenchmarkTests(unittest.TestCase):
    def test_four_node_cpu_and_danma_run_have_numerical_parity(self):
        train_x = torch.zeros((1, 784), dtype=torch.float32)
        train_y = torch.tensor([0], dtype=torch.int64)
        test_x = torch.ones((1, 784), dtype=torch.float32) * 0.01
        test_y = torch.tensor([0], dtype=torch.int64)
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            base, 'load_mnist', return_value=(train_x, train_y, test_x, test_y)
        ):
            report = single.run_benchmark(
                hidden_neurons=3,
                nodes=4,
                run_dir=Path(directory),
                node_binary=Path(os.environ.get('DANMA_NODE_BIN', 'target/debug/danma-node')),
                backends=('cpu', 'danma'),
                train_samples=1,
                test_samples=1,
                epochs=1,
                download=False,
            )
        self.assertEqual(report['status'], 'ok')
        self.assertEqual(report['model']['nodes'], 4)
        self.assertTrue(report['comparisons']['danma_vs_cpu']['numerical_parity_passed'])


if __name__ == '__main__':
    unittest.main()
