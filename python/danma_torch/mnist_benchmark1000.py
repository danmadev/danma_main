"""Independent fixed MNIST 784 -> 1000 -> host ReLU -> 10 benchmark.

Both DANMA affine operators and SGD updates execute in Rust. Host feature
aliases deliberately have no routes; CPU autograd applies ReLU exactly once.
No training command is retried after an uncertain or partial remote outcome.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time
from typing import Any

import torch
from torch.nn import functional as F

from .client import DANMAClient, DANMAError, checked_id
from .layer import DANMALinear
from .mnist_benchmark import load_mnist, _orders, _summary

LR = 0.1
SETTINGS = dict(learning_rate=LR, activation_ttl_ms=120000,
                replay_retention_ms=10000, max_live_events=4096,
                max_staleness_versions=8)
PARAMETER_KEYS = ('w1', 'b1', 'w2', 'b2')
MAX_CONFIG_BYTES = 64 * 1024 * 1024
MAX_CONFIG_NEURONS = 2_048
MAX_CONFIG_TOTAL_WEIGHTS = 1_048_576


@dataclass(frozen=True)
class Layout:
    """Dimensions are internal test seams, never public model CLI options."""
    inputs: int = 784
    hidden: int = 1000
    outputs: int = 10
    nodes: int = 10
    feature_start: int = 30001

    def __post_init__(self):
        for value in (self.inputs, self.hidden, self.outputs):
            if type(value) is not int or not 1 <= value <= 1024:
                raise ValueError('dimensions must be 1..1024')
        if type(self.nodes) is not int or not 1 <= self.nodes <= 10:
            raise ValueError('internal nodes must be 1..10')
        groups = (self.neuron_ids, self.input_ids, self.feature_ids)
        for group in groups:
            for value in group:
                checked_id(value, 'layout ID')
        if any(set(a) & set(c) for i, a in enumerate(groups) for c in groups[i + 1:]):
            raise ValueError('neuron IDs and both host alias spaces must be disjoint')
        if self.hidden + self.outputs < self.nodes or math.ceil((self.hidden + self.outputs) / self.nodes) > MAX_CONFIG_NEURONS:
            raise ValueError('layout must fit nonempty bounded neuron config files')

    @property
    def hidden_ids(self):
        return tuple(range(10001, 10001 + self.hidden))

    @property
    def output_ids(self):
        return tuple(range(10001 + self.hidden, 10001 + self.hidden + self.outputs))

    @property
    def neuron_ids(self):
        return self.hidden_ids + self.output_ids

    @property
    def input_ids(self):
        return tuple(range(20001, 20001 + self.inputs))

    @property
    def feature_ids(self):
        return tuple(range(self.feature_start, self.feature_start + self.hidden))

    @property
    def aliases(self):
        return self.input_ids + self.feature_ids

    @property
    def partitions(self):
        count, extra = divmod(len(self.neuron_ids), self.nodes)
        chunks, start = [], 0
        for i in range(self.nodes):
            end = start + count + int(i < extra)
            chunks.append(self.neuron_ids[start:end])
            start = end
        return chunks

    @property
    def owners(self):
        return {n: i + 1 for i, ids in enumerate(self.partitions) for n in ids}


def finite(tensor: torch.Tensor, name: str):
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f'{name} must be finite')


def shared_initialization(seed: int, layout: Layout | None = None):
    layout = layout or Layout()
    generator = torch.Generator(device='cpu').manual_seed(seed)
    state = {}
    for layer, fan_in, fan_out in [('1', layout.inputs, layout.hidden), ('2', layout.hidden, layout.outputs)]:
        # nn.Linear-equivalent uniform fan-in scale, explicit generator, zero bias.
        state['w' + layer] = torch.empty((fan_out, fan_in), dtype=torch.float32, device='cpu').uniform_(
            -1 / math.sqrt(fan_in), 1 / math.sqrt(fan_in), generator=generator)
        state['b' + layer] = torch.zeros(fan_out, dtype=torch.float32, device='cpu')
    return state


def tensor_digest(tensors):
    digest = hashlib.sha256()
    items = tensors.items() if isinstance(tensors, dict) else enumerate(tensors)
    for name, tensor in items:
        value = tensor.detach().cpu().contiguous()
        digest.update(str(name).encode())
        digest.update(str((str(value.dtype), tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class LocalModel(torch.nn.Module):
    def __init__(self, initial, device):
        super().__init__()
        for key in PARAMETER_KEYS:
            self.register_parameter(key, torch.nn.Parameter(initial[key].clone().to(device)))

    def hidden(self, inputs):
        return F.linear(inputs, self.w1, self.b1)

    def output(self, inputs):
        return F.linear(inputs, self.w2, self.b2)

    def forward(self, inputs):
        return self.output(F.relu(self.hidden(inputs)))


class RemoteModel(torch.nn.Module):
    def __init__(self, client, layout):
        super().__init__()
        self.hidden = DANMALinear(client, neuron_ids=layout.hidden_ids, input_ids=layout.input_ids,
                                  feedback_ttl_ms=3000, max_batch=1)
        self.output = DANMALinear(client, neuron_ids=layout.output_ids, input_ids=layout.feature_ids,
                                  feedback_ttl_ms=3000, max_batch=1)

    def forward(self, inputs):
        return self.output(F.relu(self.hidden(inputs)))


def validate_config(data):
    """Validate generated subset of runtime v1 (linear only for this hybrid)."""
    def keys(value, expected):
        if type(value) is not dict or set(value) != set(expected):
            raise ValueError('configuration has missing/unknown fields')

    def integer(value, lower, upper):
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError('configuration integer outside range')

    def number(value):
        if type(value) not in (int, float) or not math.isfinite(value) or abs(value) > torch.finfo(torch.float32).max:
            raise ValueError('configuration number must be finite float32')

    keys(data, ('schema_version', 'settings', 'neurons'))
    integer(data['schema_version'], 1, 1)
    settings = data['settings']
    keys(settings, SETTINGS)
    number(settings['learning_rate'])
    if not 0 < float(torch.tensor(settings['learning_rate'], dtype=torch.float32)) <= 1:
        raise ValueError('invalid learning rate')
    for key in ('activation_ttl_ms', 'replay_retention_ms'):
        integer(settings[key], 1, 600000)
    integer(settings['max_live_events'], 1, 4096)
    integer(settings['max_staleness_versions'], 0, 8)
    neurons = data['neurons']
    if type(neurons) is not list or not 1 <= len(neurons) <= MAX_CONFIG_NEURONS:
        raise ValueError('invalid neuron count')
    ids, total = set(), 0
    for neuron in neurons:
        keys(neuron, ('id', 'bias', 'activation', 'weights'))
        checked_id(neuron['id'], 'neuron ID')
        if neuron['id'] in ids:
            raise ValueError('duplicate neuron ID')
        ids.add(neuron['id'])
        number(neuron['bias'])
        if neuron['activation'] != 'linear':
            raise ValueError('host ReLU requires all remote activations linear')
        weights = neuron['weights']
        if type(weights) is not list or len(weights) > 1024:
            raise ValueError('invalid weights')
        sources = set()
        for weight in weights:
            keys(weight, ('source', 'weight'))
            checked_id(weight['source'], 'source ID')
            if weight['source'] in sources:
                raise ValueError('duplicate source')
            sources.add(weight['source'])
            number(weight['weight'])
        total += len(weights)
    if total > MAX_CONFIG_TOTAL_WEIGHTS:
        raise ValueError('too many total weights')


def write_configs(directory, layout, initial):
    directory.mkdir(parents=True, exist_ok=True)
    specs = {}
    for layer, ids, sources in [('1', layout.hidden_ids, layout.input_ids), ('2', layout.output_ids, layout.feature_ids)]:
        weights, biases = initial['w' + layer], initial['b' + layer]
        if tuple(weights.shape) != (len(ids), len(sources)) or tuple(biases.shape) != (len(ids),):
            raise ValueError('initial parameter shape does not match layout')
        for tensor in (weights, biases):
            if tensor.dtype != torch.float32 or tensor.device.type != 'cpu':
                raise ValueError('shared initialization must be CPU float32')
            finite(tensor, 'initial parameters')
        for i, neuron in enumerate(ids):
            specs[neuron] = dict(id=neuron, bias=float(biases[i]), activation='linear',
                                weights=[dict(source=source, weight=value) for source, value in zip(sources, weights[i].tolist())])
    paths, metadata = [], []
    for index, ids in enumerate(layout.partitions):
        data = dict(schema_version=1, settings=SETTINGS.copy(), neurons=[specs[n] for n in ids])
        validate_config(data)
        encoded = json.dumps(data, allow_nan=False, separators=(',', ':')).encode()
        if len(encoded) > MAX_CONFIG_BYTES:
            raise ValueError(f'configuration exceeds {MAX_CONFIG_BYTES} bytes')
        path = directory / f'node-{index + 1}.json'
        with path.open('xb') as stream:
            stream.write(encoded)
        paths.append(path)
        metadata.append(dict(node=index + 1, path=str(path), bytes=len(encoded),
                             sha256=hashlib.sha256(encoded).hexdigest(), neurons=len(ids),
                             worker_neurons=[sum(n % 2 == w for n in ids) for w in (0, 1)]))
    return paths, metadata


class CountingClient(DANMAClient):
    """Logical attempts at this client, not wire relays or a runtime profiler."""
    def __init__(self, port, counts):
        super().__init__('127.0.0.1', port, timeout_seconds=4)
        self.counts = counts
        self.phase = 'readiness'

    def request(self, message):
        key = f"{self.phase}:{message['kind']}"
        self.counts[key] = self.counts.get(key, 0) + 1
        return super().request(message)


class Cluster:
    def __init__(self, binary, run_dir, layout, initial, *, startup_timeout=30.0):
        self.binary = Path(binary).resolve()
        self.run_dir, self.layout, self.initial = Path(run_dir), layout, initial
        self.startup_timeout = startup_timeout
        self.processes, self.logs, self.reservations = [], [], []
        self.counts = {}
        self.timings = {}
        self.metadata = []
        self.clients = []

    def start(self):
        started = time.perf_counter()
        try:
            if not self.binary.is_file():
                raise FileNotFoundError(self.binary)
            before = time.perf_counter()
            configs, self.metadata = write_configs(self.run_dir / 'configs', self.layout, self.initial)
            self.timings['config_write_seconds'] = time.perf_counter() - before
            ports = []
            # Reserve all distinct ports until just before each child is spawned.
            # Exec cannot inherit the listener in current Rust CLI: a small race remains.
            for _ in range(self.layout.nodes):
                sock = socket.socket()
                self.reservations.append(sock)
                sock.bind(('127.0.0.1', 0))
                ports.append(sock.getsockname()[1])
            self.clients = [CountingClient(port, self.counts) for port in ports]
            self.client = self.clients[0]
            log_dir = self.run_dir / 'logs'
            log_dir.mkdir(parents=True, exist_ok=True)
            for index, port in enumerate(ports):
                argv = [str(self.binary), '--id', str(index + 1), '--listen', f'127.0.0.1:{port}',
                        '--workers', '2', '--mailbox', '256', '--neuron-config', str(configs[index])]
                for peer, peer_port in enumerate(ports):
                    if peer != index:
                        argv.extend(['--peer', f'{peer + 1}@127.0.0.1:{peer_port}'])
                log = (log_dir / f'node-{index + 1}.log').open('xb')
                self.logs.append(log)
                self.reservations[index].close()
                self.processes.append(subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT))
            deadline = time.monotonic() + self.startup_timeout
            while time.monotonic() < deadline:
                self.check_alive()
                ready = True
                for client in self.clients:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        ready = False
                        break
                    client.timeout_seconds = min(.25, remaining)
                    try:
                        routes = client.routes()
                        if set(map(str, self.layout.aliases)) & set(routes):
                            raise ValueError('host aliases unexpectedly routed')
                        ready = ready and all(routes.get(str(n)) == owner for n, owner in self.layout.owners.items())
                    except (OSError, DANMAError):
                        ready = False
                if ready:
                    self.check_alive()
                    for client in self.clients:
                        client.timeout_seconds = 4
                    self.timings['cluster_start_seconds'] = time.perf_counter() - started
                    return self
                time.sleep(.025)
            raise TimeoutError('DANMA routes did not converge on every node before deadline')
        except BaseException:
            self.timings['cluster_start_seconds'] = time.perf_counter() - started
            self.stop()
            raise

    def check_alive(self):
        for index, process in enumerate(self.processes):
            if process.poll() is not None:
                raise RuntimeError(f'DANMA node {index + 1} exited with {process.returncode}; see retained logs')

    def snapshot(self, phase, deadline=None):
        started = time.perf_counter()
        self.client.phase = phase
        state, versions = {}, []
        for layer, ids, sources in [('1', self.layout.hidden_ids, self.layout.input_ids), ('2', self.layout.output_ids, self.layout.feature_ids)]:
            rows, biases = [], []
            for neuron in ids:
                self.check_alive()
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError('error inspection deadline')
                    self.client.timeout_seconds = min(.2, remaining)
                reply = self.client.inspect(neuron)
                if set(reply['weights']) != set(map(str, sources)) or reply.get('axons') != []:
                    raise ValueError('remote state has unexpected synapses/axons')
                rows.append([reply['weights'][str(source)] for source in sources])
                biases.append(reply['bias'])
                version = reply['version']
                if type(version) is not int or version < 0:
                    raise ValueError('invalid remote version')
                versions.append(version)
            state['w' + layer] = torch.tensor(rows, dtype=torch.float32)
            state['b' + layer] = torch.tensor(biases, dtype=torch.float32)
        for value in state.values():
            finite(value, 'remote parameters')
        self.timings[phase + '_seconds'] = self.timings.get(phase + '_seconds', 0) + time.perf_counter() - started
        return state, versions

    def stop(self):
        before = time.perf_counter()
        for sock in self.reservations:
            sock.close()
        for process in self.processes:
            if process.poll() is None:
                process.terminate()
        for process in self.processes:
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        for log in self.logs:
            log.close()
        self.timings['cleanup_seconds'] = self.timings.get('cleanup_seconds', 0) + time.perf_counter() - before


def validate_options(*, train_samples=20, test_samples=10, epochs=1, batch_size=1,
                     nodes=10, backends=('cpu', 'cuda', 'danma'), require_cuda=False,
                     layout=None, **_):
    if any(type(n) is not int or n <= 0 for n in (train_samples, test_samples, epochs)):
        raise ValueError('train_samples, test_samples and epochs must be positive integers')
    if type(batch_size) is not int or batch_size != 1:
        raise ValueError('batch_size must be 1: multilayer batch update fairness is not proven')
    if layout is None:
        if type(nodes) is not int or nodes != 10:
            raise ValueError('public benchmark requires exactly 10 nodes')
    else:
        if not isinstance(layout, Layout):
            raise ValueError('layout must be a Layout instance')
        if type(nodes) is not int or nodes != layout.nodes:
            raise ValueError('nodes must match supplied layout')
        if (layout.inputs, layout.outputs) != (784, 10):
            raise ValueError('MNIST benchmark requires 784 inputs and 10 outputs')
    if not backends or len(set(backends)) != len(backends) or set(backends) - {'cpu', 'cuda', 'danma'}:
        raise ValueError('backends must be a nonempty unique list of cpu,cuda,danma')
    if require_cuda and 'cuda' not in backends:
        raise ValueError('require_cuda requires cuda in backends')


def progress(message):
    print(message, file=sys.stderr, flush=True)


def local_state(model):
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    for value in state.values():
        finite(value, 'local parameters')
    return state


def assert_versions(versions, expected, phase, metrics):
    observation = dict(stage=phase, expected=expected, count=len(versions),
                       min=min(versions), max=max(versions), all_expected=all(v == expected for v in versions))
    metrics.setdefault('version_checks', []).append(observation)
    if not observation['all_expected']:
        raise ValueError(f'{phase}: not all remote versions equal {expected}')


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def evaluate(model, images, labels, device, name, checkpoint, metrics, cluster=None):
    model.eval()
    if cluster:
        cluster.client.phase = name
    latencies, events, total_loss, correct = [], [], 0., 0
    synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        for i in range(len(labels)):
            if cluster:
                cluster.check_alive()
            before = time.perf_counter()
            if device.type == 'cuda':
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
            logits = model(images[i:i + 1].to(device))
            finite(logits, 'evaluation logits')
            loss = F.cross_entropy(logits, labels[i:i + 1].to(device))
            finite(loss, 'evaluation loss')
            total_loss += float(loss.item())
            correct += int((logits.argmax(1) == labels[i:i + 1].to(device)).sum().item())
            if device.type == 'cuda':
                end.record(); synchronize(device)
                events.append(begin.elapsed_time(end) / 1000)
            latencies.append(time.perf_counter() - before)
            metrics['progress'] = dict(stage=name, completed_samples=i + 1, total_samples=len(labels))
            progress(f'{metrics["backend"]} {name} {i + 1}/{len(labels)}')
            checkpoint()
    seconds = time.perf_counter() - started
    result = dict(loss=total_loss / len(labels), accuracy=correct / len(labels), seconds=seconds,
                  samples=len(labels), samples_per_second=len(labels) / seconds,
                  batch_wall_latency_seconds=_summary(latencies))
    if events:
        result['batch_cuda_event_seconds'] = _summary(events)
    return result


def train_and_measure(model, initial, dataset, orders, device, metrics, *, checkpoint, cluster=None):
    train_x, train_y, test_x, test_y = dataset
    optimizer = None if cluster else torch.optim.SGD(model.parameters(), lr=LR, momentum=0, weight_decay=0)
    metrics.update(completed_train_samples=0, epoch_train_losses=[], epoch_seconds=[], evaluations={}, eval_updates=0)
    expected = 0

    def check_state(phase):
        if cluster:
            state, versions = cluster.snapshot(phase + '_inspect')
            assert_versions(versions, expected, phase, metrics)
        else:
            state = local_state(model)
        metrics['parameters_finite'] = True
        return state

    for name, x, y in [('initial_train', train_x, train_y), ('initial_test', test_x, test_y)]:
        metrics['evaluations'][name] = evaluate(model, x, y, device, name, checkpoint, metrics, cluster)
        state = check_state(name)
        if tensor_digest(state) != tensor_digest(initial):
            raise ValueError('initial evaluation mutated shared parameters')
        checkpoint()
    wall_latencies, cuda_latencies, forward_seconds, backward_seconds = [], [], [], []
    synchronize(device)
    train_started = time.perf_counter()
    for epoch, order in enumerate(orders):
        model.train()
        if cluster:
            cluster.client.phase = 'train'
        epoch_started, total_loss = time.perf_counter(), 0.
        for sample, index in enumerate(order.tolist()):
            if cluster:
                cluster.check_alive()
            before = time.perf_counter()
            if device.type == 'cuda':
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
            if optimizer:
                optimizer.zero_grad(set_to_none=True)
            x, y = train_x[index:index + 1].to(device), train_y[index:index + 1].to(device)
            forward_started = time.perf_counter()
            logits = model(x)
            finite(logits, 'training logits')
            loss = F.cross_entropy(logits, y)
            finite(loss, 'training loss')
            synchronize(device)
            forward_seconds.append(time.perf_counter() - forward_started)
            backward_started = time.perf_counter()
            loss.backward()
            if optimizer:
                optimizer.step()
            if device.type == 'cuda':
                end.record(); synchronize(device)
                cuda_latencies.append(begin.elapsed_time(end) / 1000)
            backward_seconds.append(time.perf_counter() - backward_started)
            wall_latencies.append(time.perf_counter() - before)
            total_loss += float(loss.detach().item())
            expected += 1
            metrics['completed_train_samples'] = expected
            metrics['train_batch_wall_latency_seconds'] = _summary(wall_latencies)
            metrics['forward_wall_seconds'] = _summary(forward_seconds)
            metrics['backward_wall_seconds'] = _summary(backward_seconds)
            # Conservative oldest activation lifetime bound includes both layers.
            metrics['oldest_activation_wall_bound_seconds'] = max(wall_latencies)
            metrics['progress'] = dict(stage='train', epoch=epoch + 1, completed_samples=sample + 1, total_samples=len(order))
            progress(f'{metrics["backend"]} train epoch {epoch + 1} {sample + 1}/{len(order)}')
            checkpoint()
        metrics['epoch_seconds'].append(time.perf_counter() - epoch_started)
        metrics['epoch_train_losses'].append(total_loss / len(order))
        state = check_state(f'epoch_{epoch + 1}')
        checkpoint()
        if cluster:
            cluster.client.phase = 'train'
    metrics['train_seconds'] = time.perf_counter() - train_started
    metrics['train_samples_per_second'] = expected / metrics['train_seconds']
    if cuda_latencies:
        metrics['train_batch_cuda_event_seconds'] = _summary(cuda_latencies)
    for name, x, y in [('final_train', train_x, train_y), ('final_test', test_x, test_y)]:
        before_digest = tensor_digest(state)
        metrics['evaluations'][name] = evaluate(model, x, y, device, name, checkpoint, metrics, cluster)
        state = check_state(name)
        if tensor_digest(state) != before_digest:
            raise ValueError(f'{name}: evaluation mutated parameters')
        checkpoint()
    final = metrics['evaluations']['final_test']
    metrics.update(final_train_loss=metrics['evaluations']['final_train']['loss'],
                   test_loss=final['loss'], test_accuracy=final['accuracy'], eval_seconds=final['seconds'],
                   eval_samples_per_second=final['samples_per_second'],
                   final_state_sha256=tensor_digest(state))
    return state


def run_remote(context, metrics, *, checkpoint):
    layout = context.get('layout') or Layout()
    cluster = Cluster(context['node_binary'], context['run_dir'], layout, context['initial'])
    metrics['logical_requests'] = cluster.counts
    metrics['runtime_timings'] = cluster.timings
    started = time.perf_counter()
    try:
        progress('danma cluster startup/configuration')
        cluster.start()
        metrics['configs'] = cluster.metadata
        progress('danma inspect every initial neuron')
        state, versions = cluster.snapshot('initial_inspect')
        assert_versions(versions, 0, 'initial_inspect', metrics)
        if tensor_digest(state) != tensor_digest(context['initial']):
            raise ValueError('remote initialization does not exactly roundtrip shared float32 values')
        metrics['initial_state_sha256'] = tensor_digest(state)
        checkpoint()
        model_cls = context.get('remote_model_cls', RemoteModel)
        model = model_cls(cluster.client, layout)
        return train_and_measure(model, context['initial'], context['dataset'], context['orders'],
                                 torch.device('cpu'), metrics, checkpoint=checkpoint, cluster=cluster)
    except Exception:
        if cluster.clients and cluster.processes and all(p.poll() is None for p in cluster.processes):
            # Read-only, best effort, strict overall bound; never retry updates.
            before = time.perf_counter()
            try:
                state, versions = cluster.snapshot('error_inspect', deadline=time.monotonic() + 2)
                metrics['error_inspection'] = dict(state_sha256=tensor_digest(state), version_counts=dict(Counter(versions)))
            except Exception as inspection_error:
                metrics['error_inspection'] = dict(error=str(inspection_error))
            finally:
                cluster.timings['error_inspect_seconds'] = time.perf_counter() - before
        raise
    finally:
        metrics['configs'] = cluster.metadata
        cluster.stop()
        metrics['end_to_end_seconds'] = time.perf_counter() - started
        metrics['children_stopped'] = all(p.poll() is not None for p in cluster.processes)
        metrics['logical_request_total'] = sum(cluster.counts.values())
        kind_totals = Counter()
        for key, count in cluster.counts.items():
            kind_totals[key.rsplit(':', 1)[-1]] += count
        metrics['logical_request_kind_totals'] = dict(kind_totals)
        if metrics.get('train_seconds'):
            train_requests = sum(
                cluster.counts.get(f'train:{kind}', 0)
                for kind in ('forward', 'backward', 'forward_shard', 'backward_shard')
            )
            metrics['logical_train_requests_per_second'] = train_requests / metrics['train_seconds']
        checkpoint()


def source_provenance(run_dir, binary, extra_files=()):
    def git(*args):
        return subprocess.check_output(['git', *args], stderr=subprocess.STDOUT)
    result = {}
    try:
        result['git_head'] = git('rev-parse', 'HEAD').decode().strip()
        result['git_status'] = git('status', '--porcelain=v1').decode()
        for name, args in [('working', ('diff', '--binary')), ('staged', ('diff', '--cached', '--binary'))]:
            data = git(*args)
            path = run_dir / f'source-{name}.patch'
            path.write_bytes(data)
            result[name + '_snapshot'] = dict(path=str(path), sha256=hashlib.sha256(data).hexdigest())
    except (OSError, subprocess.CalledProcessError) as exc:
        result['git_error'] = str(exc)
    files = [Path(__file__), Path(__file__).with_name('layer.py'), Path(__file__).with_name('client.py'), Path(__file__).with_name('mnist_benchmark.py'),
             Path(__file__).parent.parent / 'MNIST_BENCHMARK1000.md',
             Path(__file__).parent.parent / 'benchmark_tests/test_mnist_benchmark1000.py']
    for extra in extra_files:
        path = Path(extra)
        if path not in files:
            files.append(path)
    result['source_sha256'] = {}
    snapshot_dir = run_dir / 'source-snapshot'
    snapshot_dir.mkdir()
    for path in files:
        if path.is_file():
            data = path.read_bytes()
            result['source_sha256'][str(path)] = hashlib.sha256(data).hexdigest()
            (snapshot_dir / path.name).write_bytes(data)
    result['node_binary'] = str(binary)
    if binary.is_file():
        result['node_binary_sha256'] = hashlib.sha256(binary.read_bytes()).hexdigest()
    return result


def comparisons(results, states, tolerance):
    output = {}
    for reference, name in [('cpu', 'cuda'), ('cpu', 'danma'), ('cuda', 'danma')]:
        if reference not in states or name not in states:
            continue
        a, c = results[reference], results[name]
        errors = {key: float((states[name][key] - states[reference][key]).abs().max()) for key in PARAMETER_KEYS}
        train_ratio = c['train_samples_per_second'] / a['train_samples_per_second']
        eval_ratio = c['eval_samples_per_second'] / a['eval_samples_per_second']
        loss_delta = c['test_loss'] - a['test_loss']
        passed = (all(torch.allclose(states[name][key], states[reference][key], atol=tolerance, rtol=tolerance) for key in PARAMETER_KEYS)
                  and abs(loss_delta) <= tolerance * (1 + abs(a['test_loss']))
                  and c['test_accuracy'] == a['test_accuracy'])
        output[f'{name}_vs_{reference}'] = dict(max_parameter_error=max(errors.values()),
            per_parameter_max_error=errors, per_layer_max_error={layer: max(errors['w' + layer], errors['b' + layer]) for layer in ('1', '2')},
            test_loss_delta=loss_delta, test_accuracy_gap_percentage_points=100 * (c['test_accuracy'] - a['test_accuracy']),
            train_throughput_ratio=train_ratio, eval_throughput_ratio=eval_ratio,
            train_slowdown=1 / train_ratio, eval_slowdown=1 / eval_ratio,
            numerical_parity_passed=passed)
    return output


def run_benchmark(*, run_dir=None, json_out=None, node_binary=Path('target/release/danma-node'),
                  data_dir=None, train_samples=20, test_samples=10,
                  epochs=1, batch_size=1, seed=7, nodes=10, backends=('cpu', 'cuda', 'danma'),
                  download=True, require_cuda=False, parity_tolerance=2e-5, tolerance_reason=None,
                  layout=None, provenance_files=(), remote_model_cls=RemoteModel):
    validate_options(train_samples=train_samples, test_samples=test_samples, epochs=epochs,
                     batch_size=batch_size, nodes=nodes, backends=backends, require_cuda=require_cuda,
                     layout=layout)
    layout = layout or Layout()
    if not math.isfinite(parity_tolerance) or parity_tolerance <= 0:
        raise ValueError('parity tolerance must be finite and positive')
    if parity_tolerance != 2e-5 and not tolerance_reason:
        raise ValueError('nondefault tolerance requires a documented tolerance_reason')
    started = time.perf_counter()
    run_dir = Path(run_dir) if run_dir else Path('verify-run/mnist1000-implementation/python') / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / 'report.json').exists():
        raise FileExistsError('run_dir already contains a report; choose a fresh run directory')
    node_binary = Path(node_binary)
    partition_sizes = [len(ids) for ids in layout.partitions]
    neurons_per_process = partition_sizes[0] if len(set(partition_sizes)) == 1 else partition_sizes
    parameter_count = (layout.inputs + 1) * layout.hidden + (layout.hidden + 1) * layout.outputs
    report: dict[str, Any] = dict(benchmark=f'mnist-{layout.inputs}-{layout.hidden}-relu-{layout.outputs}', schema_version=1, status='running', exit_code=1,
        run_dir=str(run_dir), requested_backends=list(backends),
        model=dict(input_features=layout.inputs, hidden_features=layout.hidden, output_features=layout.outputs,
                   logical_neurons=len(layout.neuron_ids), parameters=parameter_count, nodes=layout.nodes,
                   neurons_per_process=neurons_per_process, workers_per_process=2,
                   initialization='seeded CPU float32 uniform +/-1/sqrt(fan_in), zero biases',
                   optimizer='per-example SGD', learning_rate=LR, momentum=0, weight_decay=0,
                   loss='cross_entropy', precision='float32, no AMP, IEEE CUDA matmul'),
        hybrid_boundary=dict(remote='both affine forward/backward/SGD updates', host='ReLU, cross entropy, autograd aggregation',
                             input_aliases=[layout.input_ids[0], layout.input_ids[-1]],
                             feature_aliases=[layout.feature_ids[0], layout.feature_ids[-1]],
                             all_remote_activations='linear', local_affine_fallback=False),
        training=dict(epochs=epochs, batch_size=1, seed=seed, order_seed=seed + 2),
        quality_criteria=dict(parameter_atol=parity_tolerance, parameter_rtol=parity_tolerance,
                              loss_atol=parity_tolerance, loss_rtol=parity_tolerance,
                              accuracy_gap_percentage_points=0, tolerance_reason=tolerance_reason,
                              note='only completed pairwise comparisons can pass; small N is not statistical accuracy evidence'),
        runtime=dict(settings=SETTINGS, feedback_ttl_ms=3000, request_timeout_seconds=4,
                     logical_counts='observed request attempts at Python client; excludes runtime relays/gossip',
                     retries='none for training', mailbox=256),
        timing_semantics=dict(batch_wall='staging, forward, loss, backward/update and CUDA synchronization; excludes progress/report writes',
            train_wall='epoch loop including progress/report overhead and epoch state inspections; excludes initial/final evaluation',
            cuda_event='device timeline including staging and stream waits; separate from batch wall',
            end_to_end='direct backend timer including setup, all quality evaluations, inspections and cleanup',
            overall='whole experiment including dataset loading, provenance and all backends'),
        environment=dict(python=platform.python_version(), torch=torch.__version__, platform=platform.platform(), cpu_threads=torch.get_num_threads()),
        results={name: dict(backend=name, status='pending') for name in backends}, comparisons={})

    def checkpoint():
        report['overall_seconds'] = time.perf_counter() - started
        text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n'
        destinations = [run_dir / 'report.json']
        if json_out is not None and Path(json_out) != destinations[0]:
            destinations.append(Path(json_out))
        for path in destinations:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + '.tmp')
            temporary.write_text(text, encoding='utf-8')
            temporary.replace(path)

    states = {}
    checkpoint()
    try:
        report['provenance'] = source_provenance(run_dir, node_binary, provenance_files)
        progress('load shared balanced MNIST subsets')
        before = time.perf_counter()
        dataset = load_mnist(Path(data_dir) if data_dir else Path.home() / '.cache/danma/mnist',
                             train_samples, test_samples, seed, download)
        for tensor in dataset:
            finite(tensor, 'dataset')
        for x, y, count in [(dataset[0], dataset[1], train_samples), (dataset[2], dataset[3], test_samples)]:
            if tuple(x.shape) != (count, 784) or tuple(y.shape) != (count,) or x.dtype != torch.float32 or y.dtype != torch.int64:
                raise ValueError('MNIST loader returned invalid shape/dtype')
            if bool(((y < 0) | (y >= 10)).any()):
                raise ValueError('MNIST labels must be 0..9')
        train_counts = torch.bincount(dataset[1], minlength=10).tolist()
        test_counts = torch.bincount(dataset[3], minlength=10).tolist()
        report['dataset'] = dict(
            train_samples=train_samples,
            test_samples=test_samples,
            stratified_subset=True,
            balanced_subset=(max(train_counts) - min(train_counts) <= 1
                             and max(test_counts) - min(test_counts) <= 1),
            train_balanced=max(train_counts) - min(train_counts) <= 1,
            test_balanced=max(test_counts) - min(test_counts) <= 1,
            data_seconds=time.perf_counter() - before,
            sha256=tensor_digest(dataset),
            train_class_counts=train_counts,
            test_class_counts=test_counts,
        )
        if min(train_samples, test_samples) < 10:
            report['dataset']['warning'] = 'less than ten samples omits classes; smoke proves execution only'
            progress(report['dataset']['warning'])
        initial = shared_initialization(seed, layout)
        report['initialization_sha256'] = tensor_digest(initial)
        torch.save(initial, run_dir / 'initial-state.pt')
        orders = _orders(train_samples, epochs, seed + 2)
        report['training']['order_sha256'] = tensor_digest(orders)
        context = dict(
            run_dir=run_dir,
            node_binary=node_binary,
            initial=initial,
            dataset=dataset,
            orders=orders,
            layout=layout,
            remote_model_cls=remote_model_cls,
        )
        checkpoint()
        for name in backends:
            metrics = report['results'][name]
            metrics['status'] = 'running'
            checkpoint()
            before = time.perf_counter()
            try:
                if name == 'cuda' and not torch.cuda.is_available():
                    metrics.update(status='failed' if require_cuda else 'unavailable', reason='CUDA unavailable')
                    continue
                if name == 'danma':
                    state = run_remote(context, metrics, checkpoint=checkpoint)
                else:
                    device = torch.device(name)
                    # Use only the API available in this PyTorch version, never mix precision APIs.
                    precision = None
                    if name == 'cuda':
                        if hasattr(torch.backends.cuda.matmul, 'fp32_precision'):
                            precision = torch.backends.cuda.matmul.fp32_precision
                            torch.backends.cuda.matmul.fp32_precision = 'ieee'
                        else:
                            precision = torch.backends.cuda.matmul.allow_tf32
                            torch.backends.cuda.matmul.allow_tf32 = False
                        torch.cuda.reset_peak_memory_stats(device)
                        metrics.update(cuda_device_name=torch.cuda.get_device_name(device), cuda_runtime=torch.version.cuda,
                                       cuda_compute_capability=list(torch.cuda.get_device_capability(device)))
                    try:
                        model = LocalModel(initial, device)
                        metrics['initial_state_sha256'] = tensor_digest(local_state(model))
                        if metrics['initial_state_sha256'] != report['initialization_sha256']:
                            raise ValueError('local shared initialization mismatch')
                        state = train_and_measure(model, initial, dataset, orders, device, metrics, checkpoint=checkpoint)
                        if name == 'cuda':
                            metrics['peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
                    finally:
                        if name == 'cuda' and precision is not None:
                            if isinstance(precision, str):
                                torch.backends.cuda.matmul.fp32_precision = precision
                            else:
                                torch.backends.cuda.matmul.allow_tf32 = precision
                states[name] = state
                torch.save(state, run_dir / f'{name}-final-state.pt')
                metrics['status'] = 'ok'
            except Exception as exc:
                metrics.update(status='failed', error=dict(type=type(exc).__name__, message=str(exc)),
                               warning='incomplete backend; remote updates may be partial; no completed quality claim')
                progress(f'{name} failed: {exc}')
                for key in ('test_loss', 'test_accuracy', 'final_train_loss'):
                    metrics.pop(key, None)
            finally:
                metrics.setdefault('end_to_end_seconds', time.perf_counter() - before)
                checkpoint()
        report['comparisons'] = comparisons(report['results'], states, parity_tolerance)
        failed = any(r['status'] == 'failed' for r in report['results'].values())
        mismatch = any(not c['numerical_parity_passed'] for c in report['comparisons'].values())
        report.update(status='failed' if failed or mismatch else 'ok', exit_code=int(failed or mismatch))
    except Exception as exc:
        report.update(status='failed', exit_code=1, error=dict(type=type(exc).__name__, message=str(exc)))
        for metrics in report['results'].values():
            if metrics['status'] == 'pending':
                metrics.update(status='not_run', reason='experiment setup failed')
        progress(f'experiment failed: {exc}')
    finally:
        checkpoint()
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--node-binary', type=Path, default=Path('target/release/danma-node'))
    parser.add_argument('--data-dir', type=Path, default=Path.home() / '.cache/danma/mnist')
    parser.add_argument('--train-samples', type=int, default=20)
    parser.add_argument('--test-samples', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--nodes', type=int, choices=[10], default=10)
    parser.add_argument('--backends', type=lambda value: tuple(value.split(',')), default=('cpu', 'cuda', 'danma'))
    parser.add_argument('--no-download', action='store_true')
    parser.add_argument('--require-cuda', action='store_true')
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--json-out', type=Path)
    parser.add_argument('--smoke', action='store_true', help='actual fixed model, one train/test sample, one epoch')
    parser.add_argument('--parity-tolerance', type=float, default=2e-5)
    parser.add_argument('--tolerance-reason')
    args = parser.parse_args(argv)
    if args.smoke:
        args.train_samples = args.test_samples = args.epochs = 1
    try:
        validate_options(**vars(args))
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main(argv=None):
    args = vars(parse_args(argv))
    args['download'] = not args.pop('no_download')
    args.pop('smoke')
    report = run_benchmark(**args)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return report['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
