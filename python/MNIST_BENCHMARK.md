# MNIST quality + speed benchmark

`danma_torch.mnist_benchmark` compares the same 10-class MNIST classifier on PyTorch CPU, PyTorch CUDA, and the real distributed DANMA Rust runtime.

The first benchmark deliberately uses a single linear classifier, `784 -> 10`, with zero initialization, cross-entropy and SGD with learning rate `0.1`. That maps exactly to ten DANMA output neurons and keeps the architecture identical across backends. It is a correctness and transport-performance benchmark, not yet a deep-network MNIST benchmark.

## Install benchmark dependency

The core package still depends only on PyTorch. MNIST loading is optional and uses torchvision:

```bash
python -m pip install "torchvision"
```

On a GPU host, install CUDA-enabled PyTorch/torchvision builds that match each other.

## Run

Build an optimized DANMA node:

```bash
cargo build --release --locked -p danma-net --bin danma-node
```

Run the comparison:

```bash
PYTHONPATH=python python -m danma_torch.mnist_benchmark \
  --node-binary target/release/danma-node \
  --train-samples 1024 \
  --test-samples 512 \
  --epochs 3 \
  --batch-size 8 \
  --backends cpu,cuda,danma \
  --json-out mnist-benchmark.json
```

Use `--require-cuda` on a GPU benchmark machine to fail rather than recording CUDA as unavailable.

For a quick CPU + DANMA smoke run:

```bash
PYTHONPATH=python python -m danma_torch.mnist_benchmark \
  --node-binary target/debug/danma-node \
  --train-samples 100 --test-samples 100 --epochs 1 \
  --backends cpu,danma
```

## Fairness rules

All backends receive the same balanced train/test subsets, the same per-epoch sample order, the same zero initialization, the same `784 -> 10` model, the same cross-entropy loss, the same batch size and SGD learning rate `0.1`.

DANMA's current staleness contract limits a directly comparable batch to at most 9; the default is 8. CPU/CUDA use `torch.nn.Linear`. DANMA uses ten remote `DANMALinear` neurons distributed round-robin over three real `danma-node` processes.

## Quality metrics

The JSON report records:

- final training loss;
- test loss;
- test accuracy;
- maximum final parameter error versus CPU;
- accuracy gap and loss delta versus CPU;
- DANMA-vs-CUDA accuracy/loss deltas when CUDA is available.

## Speed metrics

The report records:

- training wall time and evaluation wall time;
- per-epoch wall time;
- train and evaluation samples/second;
- mean/p50/p95/max batch latency;
- CUDA device/runtime/compute capability and peak allocated bytes;
- DANMA cluster startup time and end-to-end time;
- DANMA train transport requests/second;
- exact DANMA forward/backward/evaluation request counts;
- throughput ratios relative to CPU and, when available, CUDA.

`train_seconds` includes host-to-backend staging and DANMA serialization. DANMA cluster startup is reported separately and excluded from `train_seconds`, but included in `end_to_end_seconds`. CUDA throughput synchronizes at epoch boundaries; CUDA batch latency is measured with device events.

## Interpretation

The current DANMA v1 path uses synchronous JSON/TCP request/response traffic for each sample and output neuron. The benchmark is therefore expected to expose transport overhead strongly. That is useful: it separates **learning quality** from **current systems throughput** and provides a baseline for persistent connections, batching and shard/tensor kernels.

Do not interpret this linear benchmark as proof of parity for deep models. The next benchmark tier should add hidden layers and nonlinear activation after the transport path is optimized enough that communication overhead does not dominate the experiment.
