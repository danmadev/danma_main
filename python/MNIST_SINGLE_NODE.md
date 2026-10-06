# Single-node configurable-hidden MNIST experiment

This experiment runs the existing hybrid DANMA MNIST model as **one Rust
`danma-node` process**. The default topology is:

```text
784 host pixel aliases
        |
        v
1000 DANMA hidden neurons
        |
    host ReLU
        |
        v
10 DANMA output neurons
```

All 1010 logical neurons are owned by the same process. The hidden width is a
CLI parameter, so changing the neuron count does not require editing source.

## Supported width

`--hidden-neurons N` accepts **1..1024**, default **1000**. The upper bound is
intentional: every output neuron receives all hidden features and the current
DANMA core bounds one neuron to 1024 dendrites.

Examples:

```bash
--hidden-neurons 128
--hidden-neurons 512
--hidden-neurons 1000
--hidden-neurons 1024
```

The startup config remains bounded. Runtime v1 allows up to 64 MiB per neuron
config, 2048 neurons per file, 1024 weights per neuron, and 1,048,576 total
weights. The default 1000-hidden graph needs 1010 logical neurons and 794,000
weights, so it fits those bounds without splitting across processes.

## Build

```bash
cargo build --release --locked -p danma-net --bin danma-node
```

## Full 1000-hidden single-node smoke

```bash
PYTHONPATH=python python -m danma_torch.mnist_single_node \
  --hidden-neurons 1000 \
  --backends cpu,danma \
  --node-binary target/release/danma-node \
  --smoke \
  --run-dir verify-run/mnist-single-node-1000
```

A smoke run still exercises all 1000 hidden neurons for one train and one test
sample, so it is substantially heavier than a small-width harness check.

## Change only the neuron count

For a faster iteration:

```bash
PYTHONPATH=python python -m danma_torch.mnist_single_node \
  --hidden-neurons 128 \
  --backends cpu,danma \
  --node-binary target/release/danma-node \
  --smoke \
  --run-dir verify-run/mnist-single-node-128
```

For a longer experiment omit `--smoke` and set `--train-samples`,
`--test-samples` and `--epochs`.

## Semantics

The experiment preserves the current hybrid boundary: both affine layers and
their SGD updates execute in Rust-owned DANMA neurons; ReLU, cross-entropy and
autograd aggregation remain on the host. There is no local affine fallback.
The old fixed 10-process `mnist_benchmark1000` CLI is unchanged.
