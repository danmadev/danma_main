"""Bounded tests; remote parity below uses real child processes, not mocks."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import torch
from torch.nn import functional as F

from danma_torch import mnist_benchmark1000 as b


class ContractTests(unittest.TestCase):
    def test_shared_initialization_and_rng_isolation(self):
        rng = torch.random.get_rng_state().clone()
        a, c = b.shared_initialization(7), b.shared_initialization(7)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertEqual(b.tensor_digest(a), b.tensor_digest(c))
        self.assertNotEqual(b.tensor_digest(a), b.tensor_digest(b.shared_initialization(8)))
        model = b.LocalModel(a, 'cpu')
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        for name, tensor in model.state_dict().items():
            self.assertTrue(torch.equal(tensor, a[name]))
        self.assertEqual(tuple(a['w1'].shape), (1000, 784))
        self.assertEqual(tuple(a['w2'].shape), (10, 1000))
        self.assertGreater(float(a['w1'].std()), 0)

    def test_full_layout_configs_and_roundtrip(self):
        layout = b.Layout()
        self.assertEqual(len(layout.neuron_ids), 1010)
        self.assertEqual(layout.hidden_ids, tuple(range(10001, 11001)))
        self.assertEqual(layout.output_ids, tuple(range(11001, 11011)))
        self.assertFalse(set(layout.aliases) & set(layout.neuron_ids))
        init = b.shared_initialization(7)
        with tempfile.TemporaryDirectory() as d:
            paths, metadata = b.write_configs(Path(d), layout, init)
            self.assertEqual(len(paths), 10)
            recovered = {}
            for p in paths:
                data = json.loads(p.read_text())
                b.validate_config(data)
                neurons = data['neurons']
                self.assertEqual(len(neurons), 101)
                self.assertEqual(sorted(sum(n['id'] % 2 == w for n in neurons) for w in (0, 1)), [50, 51])
                self.assertLess(p.stat().st_size, 16 * 1024 * 1024)
                self.assertLessEqual(sum(len(n['weights']) for n in neurons), b.MAX_CONFIG_TOTAL_WEIGHTS)
                for n in neurons:
                    self.assertEqual(n['activation'], 'linear')
                    recovered[n['id']] = n
            for layer, ids, sources in [('1', layout.hidden_ids, layout.input_ids), ('2', layout.output_ids, layout.feature_ids)]:
                w = torch.tensor([[n['weight'] for n in recovered[i]['weights']] for i in ids], dtype=torch.float32)
                self.assertTrue(torch.equal(w, init['w' + layer]))
                self.assertEqual([n['source'] for n in recovered[ids[0]]['weights']], list(sources))
            self.assertEqual(len(metadata), 10)
            self.assertEqual(sum(n['id'] < 11001 for n in json.loads(paths[-1].read_text())['neurons']), 91)

    def test_colliding_aliases_fail_preflight(self):
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            b.Layout(inputs=4, hidden=3, outputs=2, nodes=1, feature_start=10001)

    def test_options(self):
        for option in [{'batch_size': 2}, {'nodes': 2}, {'train_samples': 0}, {'test_samples': 0}, {'epochs': 0}, {'backends': ()}, {'backends': ('cpu', 'cpu')}, {'backends': ('bad',)}, {'require_cuda': True, 'backends': ('cpu',)}]:
            with self.subTest(option=option), self.assertRaises(ValueError):
                b.validate_options(**option)
        args = b.parse_args(['--backends', 'cpu', '--smoke'])
        self.assertEqual((args.train_samples, args.test_samples, args.epochs, args.batch_size, args.nodes), (1, 1, 1, 1, 10))

    def test_schema_rejects_invalid(self):
        layout = b.Layout(inputs=4, hidden=3, outputs=2, nodes=1)
        with tempfile.TemporaryDirectory() as d:
            paths, _ = b.write_configs(Path(d), layout, b.shared_initialization(7, layout))
            original = json.loads(paths[0].read_text())
            for change in ['missing', 'extra', 'bool', 'nan', 'activation', 'duplicate']:
                data = json.loads(json.dumps(original))
                if change == 'missing': del data['settings']['learning_rate']
                if change == 'extra': data['axons'] = []
                if change == 'bool': data['neurons'][0]['id'] = True
                if change == 'nan': data['neurons'][0]['bias'] = float('nan')
                if change == 'activation': data['neurons'][0]['activation'] = 'relu'
                if change == 'duplicate': data['neurons'][0]['weights'].append(data['neurons'][0]['weights'][0])
                with self.subTest(change=change), self.assertRaises(ValueError):
                    b.validate_config(data)

    def test_logical_counter_records_failed_attempt(self):
        counts = {}
        client = b.CountingClient(12345, counts)
        client.phase = 'train'
        with mock.patch.object(b.DANMAClient, 'request', side_effect=b.DANMAError('failed')):
            with self.assertRaises(b.DANMAError):
                client.request({'kind': 'backward'})
        self.assertEqual(counts, {'train:backward': 1})

    def test_nondefault_tolerance_requires_reason(self):
        with self.assertRaisesRegex(ValueError, 'reason'):
            b.run_benchmark(parity_tolerance=1e-3)

    def test_parity_is_measured_not_assumed(self):
        layout = b.Layout(inputs=4, hidden=3, outputs=2, nodes=1)
        initial = b.shared_initialization(7, layout)
        changed = {key: value.clone() for key, value in initial.items()}
        changed['w1'][0, 0] += .01
        metrics = dict(train_samples_per_second=1., eval_samples_per_second=1., test_loss=1., test_accuracy=.5)
        comparison = b.comparisons({'cpu': metrics, 'danma': metrics}, {'cpu': initial, 'danma': changed}, 2e-5)
        self.assertFalse(comparison['danma_vs_cpu']['numerical_parity_passed'])
        self.assertGreater(comparison['danma_vs_cpu']['max_parameter_error'], .009)


class MetricsTests(unittest.TestCase):
    def data(self):
        return (torch.full((1, 784), .2), torch.tensor([0]), torch.full((1, 784), .1), torch.tensor([1]))

    def test_cpu_metrics_and_incremental_report(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(b, 'load_mnist', return_value=self.data()):
            report = b.run_benchmark(run_dir=Path(d), backends=('cpu',), train_samples=1, test_samples=1)
            self.assertEqual(report['status'], 'ok')
            cpu = report['results']['cpu']
            self.assertEqual(cpu['completed_train_samples'], 1)
            self.assertEqual(len(cpu['epoch_train_losses']), 1)
            self.assertEqual(cpu['eval_updates'], 0)
            self.assertTrue(cpu['parameters_finite'])
            self.assertGreater(cpu['train_seconds'], 0)
            self.assertIn('initial_train', cpu['evaluations'])
            self.assertIn('final_train', cpu['evaluations'])
            self.assertEqual(json.loads((Path(d) / 'report.json').read_text())['status'], 'ok')

    def test_cuda_unavailable_and_required(self):
        for required in (False, True):
            with tempfile.TemporaryDirectory() as d, mock.patch.object(b, 'load_mnist', return_value=self.data()), mock.patch.object(torch.cuda, 'is_available', return_value=False):
                r = b.run_benchmark(run_dir=Path(d), backends=('cpu', 'cuda'), train_samples=1, test_samples=1, require_cuda=required)
                self.assertEqual(r['results']['cpu']['status'], 'ok')
                self.assertEqual(r['results']['cuda']['status'], 'failed' if required else 'unavailable')
                self.assertEqual(r['exit_code'], 1 if required else 0)

    def test_failed_report_preserves_prior_backend_and_counters(self):
        def fail(*args, **kwargs):
            metrics = args[1]
            metrics['logical_requests'] = {'train:forward': 3}
            kwargs['checkpoint']()
            raise RuntimeError('partial remote update')
        with tempfile.TemporaryDirectory() as d, mock.patch.object(b, 'load_mnist', return_value=self.data()), mock.patch.object(b, 'run_remote', side_effect=fail):
            r = b.run_benchmark(run_dir=Path(d), backends=('cpu', 'danma'), train_samples=1, test_samples=1)
            self.assertEqual(r['results']['cpu']['status'], 'ok')
            self.assertEqual(r['results']['danma']['status'], 'failed')
            self.assertEqual(r['results']['danma']['logical_requests']['train:forward'], 3)
            self.assertNotIn('test_loss', r['results']['danma'])
            self.assertEqual(r['exit_code'], 1)


class LifecycleTests(unittest.TestCase):
    def cluster(self, root, binary):
        layout = b.Layout(inputs=4, hidden=3, outputs=2, nodes=2)
        return b.Cluster(binary, Path(root), layout, b.shared_initialization(7, layout), startup_timeout=0.3)

    def test_partial_spawn_cleanup(self):
        with tempfile.TemporaryDirectory() as d:
            cluster = self.cluster(d, Path('/bin/true'))
            child = mock.Mock()
            child.poll.return_value = None
            with mock.patch.object(subprocess, 'Popen', side_effect=[child, OSError('spawn')]):
                with self.assertRaises(OSError): cluster.start()
            child.terminate.assert_called_once()
            child.wait.assert_called_once()
            self.assertTrue((Path(d) / 'configs/node-1.json').exists())

    def test_dead_child_cleanup(self):
        with tempfile.TemporaryDirectory() as d:
            cluster = self.cluster(d, Path('/bin/false'))
            with self.assertRaisesRegex(RuntimeError, 'exited'): cluster.start()
            self.assertTrue(all(p.poll() is not None for p in cluster.processes))

    def test_malformed_binary(self):
        with tempfile.TemporaryDirectory() as d:
            binary = Path(d) / 'bad'; binary.write_text('not executable format'); binary.chmod(0o755)
            cluster = self.cluster(d, binary)
            with self.assertRaises(OSError): cluster.start()
            self.assertTrue(all(p.poll() is not None for p in cluster.processes))

    def test_readiness_errors_obey_deadline(self):
        with tempfile.TemporaryDirectory() as d:
            cluster = self.cluster(d, Path('/bin/true'))
            child = mock.Mock()
            child.poll.return_value = None
            with mock.patch.object(subprocess, 'Popen', return_value=child), mock.patch.object(b.CountingClient, 'routes', side_effect=b.DANMAError('not ready')):
                with self.assertRaises(TimeoutError): cluster.start()
            child.terminate.assert_called()
            self.assertGreater(cluster.timings['cluster_start_seconds'], .25)


@unittest.skipUnless(Path(os.environ.get('DANMA_NODE_BIN', 'target/debug/danma-node')).is_file(), 'real node binary missing')
class RealTCPTests(unittest.TestCase):
    def test_one_and_two_node_multilayer_parity(self):
        torch.set_num_threads(1)
        for nodes in (1, 2):
            with self.subTest(nodes=nodes), tempfile.TemporaryDirectory() as d:
                layout = b.Layout(inputs=4, hidden=3, outputs=2, nodes=nodes)
                init = b.shared_initialization(7, layout)
                init['b1'][0] = -2.0  # always inactive ReLU branch
                init['b1'][1:] = 0.5
                local = b.LocalModel(init, 'cpu')
                optimizer = torch.optim.SGD(local.parameters(), lr=.1)
                cluster = b.Cluster(Path(os.environ.get('DANMA_NODE_BIN', 'target/debug/danma-node')), Path(d), layout, init)
                try:
                    cluster.start()
                    remote = b.RemoteModel(cluster.client, layout)
                    initial, versions = cluster.snapshot('initial_inspect')
                    self.assertEqual(b.tensor_digest(initial), b.tensor_digest(init))
                    self.assertEqual(set(versions), {0})
                    for step in (1, 2):
                        x = torch.tensor([[.2, -.4, .6, .1]], requires_grad=True)
                        rx = x.detach().clone().requires_grad_()
                        optimizer.zero_grad(set_to_none=True)
                        lh = local.hidden(x); rh = remote.hidden(rx)
                        torch.testing.assert_close(lh, rh, atol=2e-5, rtol=2e-5)
                        self.assertLess(float(rh[0, 0].detach()), 0)
                        ll = local.output(F.relu(lh)); rl = remote.output(F.relu(rh))
                        torch.testing.assert_close(ll, rl, atol=2e-5, rtol=2e-5)
                        loss = F.cross_entropy(ll, torch.tensor([1])); rloss = F.cross_entropy(rl, torch.tensor([1]))
                        torch.testing.assert_close(loss, rloss, atol=2e-5, rtol=2e-5)
                        loss.backward(); rloss.backward(); optimizer.step()
                        torch.testing.assert_close(x.grad, rx.grad, atol=2e-5, rtol=2e-5)
                        state, versions = cluster.snapshot('step_inspect')
                        self.assertEqual(set(versions), {step})
                        for key, tensor in local.state_dict().items():
                            torch.testing.assert_close(tensor, state[key], atol=2e-5, rtol=2e-5)
                        self.assertTrue(torch.equal(state['w1'][0], init['w1'][0]))
                    remote.eval()
                    with torch.no_grad(): remote(rx.detach())
                    _, versions = cluster.snapshot('eval_inspect')
                    self.assertEqual(set(versions), {2})
                    cluster.processes[-1].terminate(); cluster.processes[-1].wait(timeout=2)
                    with self.assertRaisesRegex(RuntimeError, 'exited'): cluster.check_alive()
                finally:
                    cluster.stop()
                self.assertTrue(all(p.poll() is not None for p in cluster.processes))

    def test_remote_metrics_counts_and_all_eval_versions(self):
        layout = b.Layout(inputs=4, hidden=3, outputs=2, nodes=2)
        init = b.shared_initialization(7, layout)
        dataset = (torch.tensor([[.2, -.4, .6, .1]]), torch.tensor([1]),
                   torch.tensor([[.1, .2, .3, .4]]), torch.tensor([0]))
        with tempfile.TemporaryDirectory() as d:
            cluster = b.Cluster(Path(os.environ.get('DANMA_NODE_BIN', 'target/debug/danma-node')), Path(d), layout, init)
            metrics = dict(backend='danma')
            try:
                cluster.start()
                state, versions = cluster.snapshot('initial_inspect')
                self.assertEqual(set(versions), {0})
                remote = b.RemoteModel(cluster.client, layout)
                result = b.train_and_measure(remote, init, dataset, [torch.tensor([0])], torch.device('cpu'), metrics, checkpoint=lambda: None, cluster=cluster)
                self.assertEqual(cluster.counts['train:forward'], 5)
                self.assertEqual(cluster.counts['train:backward'], 5)
                for phase in ('initial_train', 'initial_test', 'final_train', 'final_test'):
                    self.assertEqual(cluster.counts[phase + ':forward'], 5)
                    self.assertNotIn(phase + ':backward', cluster.counts)
                self.assertTrue(all(v['all_expected'] for v in metrics['version_checks']))
                self.assertEqual([v['expected'] for v in metrics['version_checks']], [0, 0, 1, 1, 1])
                local = b.LocalModel(init, 'cpu')
                local_metrics = dict(backend='cpu')
                expected = b.train_and_measure(local, init, dataset, [torch.tensor([0])], torch.device('cpu'), local_metrics, checkpoint=lambda: None)
                for key in b.PARAMETER_KEYS:
                    torch.testing.assert_close(result[key], expected[key], atol=2e-5, rtol=2e-5)
            finally:
                cluster.stop()


if __name__ == '__main__':
    unittest.main()
