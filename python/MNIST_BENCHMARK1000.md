# Independent MNIST 784 → 1000 → ReLU → 10 benchmark

This benchmark is separate from the unchanged [original linear benchmark](danma_torch/mnist_benchmark.py).
The implementation is [mnist_benchmark1000.py](danma_torch/mnist_benchmark1000.py).
The public architecture is fixed: 784 pixels, 1000 hidden features, host ReLU,
and **10 class outputs**, not 1000 outputs. DANMA owns **1010 logical neurons
in 10 localhost processes**, 101 neurons per process, with two workers and a
256-entry mailbox per process. There are 794,000 weights and 1010 biases.

## Execution and fairness contract

- Shared seeded CPU float32 initialization: each weight is uniform within
  ±1/√fan-in (the linear-layer weight initializer scale); biases are zero.
  Nonzero random weights break hidden-neuron symmetry. No model constructor
  consumes random initialization. CPU, CUDA, and serialized remote state clone
  the same tensors; the report and retained initial state contain their digest.
- The existing balanced MNIST loader is imported without modifications.
  All backends use identical subsets, normalization to [0,1], and epoch order
  generated from seed + 2. Defaults: 20 train, 10 test, one epoch, seed 7.
- Batch size **1 only**. CPU/CUDA and DANMA use per-example SGD, learning rate
  0.1, no momentum, weight decay, or AMP. CUDA matmul is IEEE float32 (TF32
  disabled), with the original precision setting restored afterward.
- DANMA composes two existing remote affine adapters. **Both affine forwards,
  backwards, and parameter updates execute in Rust**. CPU computes ReLU,
  cross entropy, and autograd aggregation. There is no local affine fallback.
  This is a **hybrid CPU/Rust benchmark**, not an autonomous remote graph or
  a generic DANMA-device operator benchmark.
- Hidden IDs are 10001–11000; output IDs 11001–11010. Pixel aliases
  20001–20784 and interlayer host-feature aliases 30001–31000 are disjoint from
  all neuron IDs and must have no routes. Reusing hidden IDs as output-input
  sources would cause automatic Rust upstream routing instead of host gradient
  aggregation. All remote activations are linear: host ReLU applies its
  derivative exactly once. Tiny real-TCP tests verify gradients use forward
  weights even though output parameters update before hidden backward.
- Ownership is consecutive blocks of 101 IDs. The last node owns 91 hidden
  neurons plus 10 outputs. ID modulo two assigns 50/51 neurons to its workers.
  Internal smaller layouts exist only for tests, not public architecture flags.

## Prerequisites and commands

Use the existing environment with compatible PyTorch and torchvision. No
dependency installation or environment modification is performed by this tool.
Build the runtime with file-loader support (short startup arguments):

```bash
cargo build --release --locked -p danma-net --bin danma-node
```

From the repository root, use a **fresh run directory every time**.
CPU only:

```bash
PYTHONPATH=python .venv/bin/python -m danma_torch.mnist_benchmark1000 \
  --backends cpu --seed 7 --run-dir verify-run/mnist1000-cpu
```

DANMA only:

```bash
PYTHONPATH=python .venv/bin/python -m danma_torch.mnist_benchmark1000 \
  --backends danma --node-binary target/release/danma-node \
  --seed 7 --run-dir verify-run/mnist1000-danma
```

CPU/CUDA/DANMA with CUDA required:

```bash
PYTHONPATH=python .venv/bin/python -m danma_torch.mnist_benchmark1000 \
  --backends cpu,cuda,danma --require-cuda --nodes 10 --batch-size 1 \
  --train-samples 20 --test-samples 10 --epochs 1 --seed 7 \
  --node-binary target/release/danma-node \
  --run-dir verify-run/mnist1000-all --json-out verify-run/mnist1000-all-copy.json
```

Actual full-model/topology smoke, bounded to one training and one test sample:

```bash
PYTHONPATH=python .venv/bin/python -m danma_torch.mnist_benchmark1000 \
  --smoke --backends cpu,danma --seed 7 \
  --node-binary target/release/danma-node --run-dir verify-run/mnist1000-smoke
```

Repeat the identical seed explicitly with a different directory:

```bash
PYTHONPATH=python .venv/bin/python -m danma_torch.mnist_benchmark1000 \
  --smoke --backends cpu,danma --seed 7 \
  --node-binary target/release/danma-node --run-dir verify-run/mnist1000-smoke-repeat
```

The smoke still performs initial train/test evaluation and final train/test
evaluation in addition to its single SGD sample. Subsets smaller than ten
omit classes (a one-sample balanced subset selects class zero): a warning is
recorded. Small default and smoke datasets are execution/parity experiments,
**not statistical evidence of MNIST classification quality**.

Other options: dataset location and no-download, epochs, train/test counts,
backend list, seed, report-copy destination, and run directory. Omitted run
directories receive a unique UTC timestamp under
`verify-run/mnist1000-implementation/python/`. Only ten nodes are accepted;
batch sizes above one and zero sample counts fail validation. An unavailable
optional CUDA backend has explicit unavailable status; requiring CUDA makes
the report and process fail. The default binary is the release node.

## Runtime lifecycle and retained artifacts

Every process reads its retained configuration from the run directory's
`configs/`, using schema version 1, linear activation, learning rate 0.1,
activation TTL 120000 ms, replay retention 10000 ms, live-event bound 4096,
and staleness bound 8. Runtime v1 now permits up to 64 MiB, 2048 neurons,
1024 fan-in per neuron and 1,048,576 total weights per file. This fixed ten-node
benchmark still writes only 101 neurons per file and contains no axons. All nine other processes
are bootstrap peers. Per-process combined logs remain in `logs/`.

Ports are reserved together, then released immediately before each spawn.
The current CLI cannot inherit a bound listener, so a small reservation-to-exec
race remains: startup failure is bounded and explicit, never hidden by retries.
Readiness requires every process alive, every node's route table owning all
1010 neurons correctly, and all host aliases absent. Request exceptions are
polled only during read-only readiness within a 30-second deadline.

Every initial neuron is inspected before training: all versions must be zero,
weights/biases must exactly roundtrip shared float32 initialization, and there
must be no extra weights or axons. All neurons are re-inspected after every
completed epoch and every complete quality evaluation; versions must equal
completed training samples (ultimately train samples × epochs). Evaluation
must not mutate initial parameters or advance any remote version. Parameters,
inputs, logits, and losses are checked for finiteness.

The feedback TTL remains 3000 ms; the normal client timeout remains four
seconds per request. The report records per-example forward/backward wall
times and a conservative full-example bound on the oldest saved activation's
age, for comparison with the 120-second activation TTL. These are **different
limits**: extending activation retention does not extend routed feedback TTL.
An expiry, transport error, or partial backward fails the backend. **No stateful
request is retried**, and no completed quality metric is claimed for it.

Reports are atomically updated at stages, samples, epochs, and backend exits.
Requested backend names always remain present; prior completed backends and
observed request attempts survive a later failure. Error inspection is best
effort with a two-second total deadline, before unconditional child cleanup.
Partial startup is also cleaned up. Completed final states and initial state
are retained as tensor artifacts. Provenance includes Git HEAD, dirty status,
working/staged diffs (read only), copies/hashes of Python sources, binary hash,
config hashes, normalized subset hash/class counts, initialization digest,
and epoch-order digest. The index is never staged, committed, or reset.

## Metrics and interpretation

The native report name is **mnist-784-1000-relu-10**. It records:

- Initial train/test loss and accuracy; online per-epoch mean training losses;
  final training-set loss from a separate inference pass; final test loss and
  accuracy. Online epoch loss is not the final-state training-set loss.
- Train/evaluation wall time and samples/second, per-epoch wall time,
  mean/p50/p95/max per-example wall latency. CPU/CUDA batch wall includes
  staging and updates; CUDA additionally records separate event latency,
  synchronization, device/runtime/capability, and peak allocated memory.
- Configuration writing, cluster startup (includes configuration writing),
  each inspection stage, cleanup, directly timed backend end-to-end, dataset
  loading, and whole-experiment wall time. Training wall includes progress,
  report instrumentation, and epoch inspections; batch wall does not include
  report writes. Startup/initial/final quality evaluations are outside training
  wall. End-to-end includes all these stages and cleanup. Timings overlap as
  documented and must not be blindly summed.
- Observed **logical Python-client attempts**, grouped by phase and request
  kind. They exclude gossip and server relays and are not a wire profiler.
  Training forwards and backwards, four quality-evaluation forwards, readiness,
  initial/epoch/evaluation inspections, and error inspections are all separate.
  A successful one-epoch smoke normally has 1010 train forwards + 1010 train
  backwards + 4040 quality forwards + 6060 inspections, plus variable
  readiness calls. Do not estimate request count as simply twice sample count.
- Every complete version check with count/min/max/expected and assertion.
  Pairwise CPU/CUDA/DANMA parameter errors (whole model, layer, and tensor),
  test-loss and accuracy deltas, throughput ratios and reciprocal slowdowns.

Default numerical parity criteria: parameters close with absolute and relative
tolerance 2e-5, test loss within 2e-5 × (1 + reference loss), and exactly matching
small-subset test accuracy. This accommodates float32 accumulation order in
tiny correctness tests, **not** a promise of large-run equivalence. Numerical
comparison failure yields nonzero exit even if each backend executed fully.
A backend-only run cannot claim cross-backend parity. A changed tolerance
requires both the tolerance option and a reason, retained in the report.
No exception alone is a quality PASS. Run completion and measured numerical
parity are separate fields. Accuracy ties/near-zero ReLU boundaries may amplify
float32 discrepancies on larger experiments; investigate rather than silently
raising tolerance.

## Tests and limitations

Trackable [tests](benchmark_tests/test_mnist_benchmark1000.py) use the standard
unittest runner, not the ignored legacy test directory:

```bash
PYTHONPATH=python DANMA_NODE_BIN=target/debug/danma-node \
  .venv/bin/python -m unittest discover -s python/benchmark_tests \
  -p test_mnist_benchmark1000.py -v
```

The one-/two-process 4→3→2 tests use real Rust TCP for both affine layers,
pre/post-ReLU logits, loss, input gradients, one/two SGD steps, biases, weights,
inactive branches, versions, evaluation, and metric request accounting.
Lifecycle/error-report tests use mocks only where noted, never as remote
numerical-parity evidence. Missing binaries skip real-TCP tests explicitly.
Debug builds are suitable for correctness only. Full release topology smoke
and longer quality/throughput experiments are separate validation steps.

No Rust schema changes, remote batching, fallback, recovery, tensor protocol
optimization, autonomous axons, or general DANMA operator support are added.
Rollback consists of removing only the three new source/document/test files;
the original benchmark and runtime are independent and unchanged by this work.
