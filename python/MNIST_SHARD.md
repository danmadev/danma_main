# DANMA async local shard affine v1

This branch experiments with a different **physical execution granularity**
without changing DANMA's logical neuron semantics.

## Invariant

A neuron remains the logical unit of state and learning:

- its own NeuronID;
- its own weights and bias;
- its own parameter version;
- its own activation EventID;
- its own TTL/staleness decision;
- its own duplicate-feedback decision.

There is no global forward barrier, global backward barrier, or shared
`optimizer.step()`.

The optimization is only this:

```text
before:
  one TCP request -> one neuron

v1 shard executor:
  one TCP request -> many compatible neurons owned by one local node
```

The local CPU shard groups those independent operations into one mailbox
command per worker. Workers can progress independently. A delayed feedback for
one EventID is still applied/rejected independently of the other events that
happened to share its transport envelope.

## Current MNIST path

For the one-node model:

```text
784 -> 1000 -> host ReLU -> 10
```

training one sample previously needs approximately:

```text
1000 hidden forward RPC
10 output forward RPC
10 output backward RPC
1000 hidden backward RPC
--------------------------------
2020 RPC / sample
```

With `DANMAShardLinear`:

```text
hidden forward_shard
output forward_shard
output backward_shard
hidden backward_shard
--------------------------------
4 RPC / sample
```

Every target neuron inside those four envelopes still has a distinct EventID
and version update.

## Important scope boundary

This is **not yet a dense GEMM kernel**.

v1 removes transport and mailbox granularity first. The Rust workers still call
the existing per-neuron `Neuron::forward` / `Neuron::backward` logic inside
the local batch command. This deliberately isolates how much of the current
slowdown comes from thousands of TCP/JSON round trips.

A later v2 can replace the worker's internal representation with dense SoA /
matrix kernels while preserving the same per-neuron EventID/version contract.

## Host-bound backward

`backward_shard` is currently intended for affine layers whose predecessor
features are host aliases, as in the MNIST hybrid benchmark. The node verifies
that the supplied input IDs are neither local neurons nor routed remote
neurons, applies every feedback independently, and only then aggregates the
returned host `dX` vector.

If one item expires, is stale, is duplicated, or fails, the response is
`partial`. The client fails closed and does not retry, because earlier neuron
updates may already have committed.

## Build

```bash
cargo build --release --locked -p danma-net --bin danma-node
```

## Fast parity smoke

```bash
PYTHONPATH=python python -m danma_torch.mnist_shard \
  --hidden-neurons 32 \
  --backends cpu,danma \
  --node-binary target/release/danma-node \
  --smoke
```

## 1000-neuron comparison

```bash
PYTHONPATH=python python -m danma_torch.mnist_shard \
  --hidden-neurons 1000 \
  --backends cpu,danma \
  --node-binary target/release/danma-node \
  --train-samples 20 \
  --test-samples 10 \
  --epochs 1
```

Compare this directly with the previous per-neuron RPC runner:

```bash
PYTHONPATH=python python -m danma_torch.mnist_single_node \
  --hidden-neurons 1000 \
  --nodes 1 \
  --backends cpu,danma \
  --node-binary target/release/danma-node
```

The first performance question is whether train latency drops from roughly
seconds per sample to the local-compute/serialization regime while numerical
parity remains within the existing benchmark tolerance.

## Tests

```bash
cargo test --locked -p danma-shard -p danma-net --all-targets

DANMA_NODE_BIN=target/debug/danma-node PYTHONPATH=python \
python -m unittest discover -s python/benchmark_tests \
  -p "test_mnist_shard.py" -v
```

## Next optimization after v1

If v1 confirms that RPC granularity dominates, implement a dense worker kernel:

```text
Y  = X W^T + b
dX = dY W
```

while retaining a metadata vector indexed by logical neuron for EventID,
version, TTL, deduplication and local learning state.

That is the point where DANMA gets matrix/SIMD efficiency without turning the
logical network into a globally synchronous ordinary neural network.
