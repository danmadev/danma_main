# MNIST: CPU vs DANMA Rust-native 3-node shard runtime

This benchmark compares the same MNIST model and per-example SGD trajectory on:

1. ordinary PyTorch CPU;
2. three DANMA Rust node processes using shard RPC fan-out to the node that
   actually owns each target neuron.

## Model

```text
784 inputs -> 1000 hidden linear neurons -> host ReLU -> 10 output linear neurons
```

The benchmark therefore has 1000 hidden neurons and 10 output neurons, or 1010
logical DANMA neurons total.

All backends share:

- the same float32 initialization;
- the same MNIST subset;
- the same training order;
- learning rate 0.1;
- per-example SGD, batch size 1;
- cross entropy;
- no momentum or weight decay.

## Three-node placement

The existing deterministic layout partitions all 1010 logical neurons into:

```text
node 1: 337 neurons
node 2: 337 neurons
node 3: 336 neurons
```

For the two layers this is:

```text
             hidden   output
node 1         337       0
node 2         337       0
node 3         326      10
```

So one training sample uses:

```text
hidden forward : 3 shard RPC
output forward : 1 shard RPC
output backward: 1 shard RPC
hidden backward: 3 shard RPC
-----------------------------
total           8 shard RPC / sample
```

Inside each DANMA process, the shard request is dispatched through bounded Rust
worker mailboxes. There is no per-neuron TCP request within the process.

This benchmark still has the current hybrid boundary:

```text
remote DANMA: affine forward/backward + SGD state updates
host PyTorch: ReLU + cross entropy + autograd aggregation
```

It is therefore a runtime/parity benchmark, not yet the final fully autonomous
DANMA graph.

## Build

```bash
cargo build --release --locked -p danma-net --bin danma-node
```

## Fast correctness smoke

```bash
PYTHONPATH=python python -m danma_torch.mnist_rust_3node \
  --hidden-neurons 1000 \
  --nodes 3 \
  --backends cpu,danma \
  --node-binary target/release/danma-node \
  --smoke
```

This runs one train and one test sample. It proves execution, routing, version
updates and CPU numerical parity, not MNIST accuracy.

## Practical comparison run

```bash
PYTHONPATH=python python -m danma_torch.mnist_rust_3node \
  --hidden-neurons 1000 \
  --nodes 3 \
  --backends cpu,danma \
  --node-binary target/release/danma-node \
  --train-samples 1000 \
  --test-samples 1000 \
  --epochs 1
```

## Larger accuracy/performance run

```bash
PYTHONPATH=python python -m danma_torch.mnist_rust_3node \
  --hidden-neurons 1000 \
  --nodes 3 \
  --backends cpu,danma \
  --node-binary target/release/danma-node \
  --train-samples 10000 \
  --test-samples 10000 \
  --epochs 3
```

This is deliberately expensive because training remains per-example SGD.

## Output

The JSON report contains the existing CPU/DANMA comparison:

- train throughput and slowdown;
- evaluation throughput and slowdown;
- test loss and accuracy;
- parameter max error;
- per-layer and per-parameter error;
- numerical parity result;
- per-batch forward/backward latency;
- node startup/config timings;
- logical request counts;
- per-neuron version checks.

It also adds an `executor` section containing:

- three-node placement;
- hidden/output neuron counts by owner;
- expected shard RPC count per sample;
- local Rust transport description;
- remote transport description;
- explicit statement that there is no global barrier.

## Correctness acceptance

The run is accepted only when:

```text
CPU and DANMA test accuracy are identical
parameter tensors match within 2e-5
test loss matches within benchmark tolerance
all 1010 neuron versions equal the expected training update count
no DANMA backend failure is reported
```

The multi-node backward sums one `dX` vector returned by each owner node with
`math.fsum`. This changes floating-point reduction grouping compared with the
single-node path, so bit-identical state is not required; numerical parity is.

## Automated test

```bash
DANMA_NODE_BIN=target/debug/danma-node PYTHONPATH=python \
python -m unittest discover -s python/benchmark_tests \
  -p "test_mnist_rust_3node.py" -v
```

The real-node CI test uses 1000 hidden neurons and one sample so that every
commit checks the exact requested topology without turning CI into a long
training run.
