# Configurable-node, configurable-hidden MNIST experiment

This experiment runs the existing hybrid DANMA MNIST model across a configurable
number of Rust `danma-node` processes. Defaults are **1000 hidden neurons** and
**1 node**.

The topology stays:

```text
784 host pixel aliases
        |
        v
N DANMA hidden neurons
        |
    host ReLU
        |
        v
10 DANMA output neurons
```

`--hidden-neurons N` accepts **1..1024**, default **1000**.
`--nodes K` accepts **1..10**, default **1**. Logical neurons are partitioned
as evenly as possible across the selected nodes.

Examples:

```bash
--hidden-neurons 1000 --nodes 1
--hidden-neurons 1000 --nodes 4
--hidden-neurons 512  --nodes 8
--hidden-neurons 1024 --nodes 10
```

The hidden-width upper bound is intentional: every output neuron receives all
hidden features and the current DANMA core bounds one neuron to 1024 dendrites.

The startup config remains bounded. Runtime v1 allows up to 64 MiB per neuron
config, 2048 neurons per file, 1024 weights per neuron, and 1,048,576 total
weights per file. The 1000-hidden model contains 1010 logical neurons and
794,000 weights in total.

## Build

```bash
cargo build --release --locked -p danma-net --bin danma-node
```

## 1000 hidden neurons on one node

```bash
PYTHONPATH=python python -m danma_torch.mnist_single_node \
  --hidden-neurons 1000 \
  --nodes 1 \
  --backends cpu,danma \
  --node-binary target/release/danma-node \
  --smoke \
  --run-dir verify-run/mnist-n1-h1000
```

## The same 1000 hidden neurons on four nodes

```bash
PYTHONPATH=python python -m danma_torch.mnist_single_node \
  --hidden-neurons 1000 \
  --nodes 4 \
  --backends cpu,danma \
  --node-binary target/release/danma-node \
  --smoke \
  --run-dir verify-run/mnist-n4-h1000
```

For a longer experiment omit `--smoke` and set `--train-samples`,
`--test-samples` and `--epochs`.

The module name `mnist_single_node` is kept for compatibility even though the
runner now supports multiple nodes.

## Semantics

Both affine layers and their SGD updates execute in Rust-owned DANMA neurons;
ReLU, cross-entropy and autograd aggregation remain on the host. There is no
local affine fallback. The old fixed 10-process `mnist_benchmark1000` CLI
remains unchanged.
