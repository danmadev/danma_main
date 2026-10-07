# MVP roadmap

## 1. CUDA Integration Plan
Phase 1: Infrastructure & Setup
 Add Dependency: Update root Cargo.toml to include neuromorph-driver in the [dependencies] section.
Definition of Done: cargo build passes and neuromorph-driver is accessible in src/main.rs.
 Initialize Driver: Add neuromorph_driver::init() call at the start of main() in src/main.rs.
Definition of Done: Application starts successfully without panicking on initialization.
Phase 2: Memory Management Integration
 Refactor Neuron Struct: Modify Neuron struct to hold DeviceMemory handles for weights and bias.
Definition of Done: Neuron struct compiles with neuromorph_driver::DeviceMemory fields replacing or augmenting Vec<i32>.
 Implement Data Transfer: Create helper methods in Neuron to copy weights/bias to device upon creation (new).
Definition of Done: Unit test confirms data is correctly copied to device memory (verified by copying back and comparing).
Phase 3: Kernel Execution Integration
 Create Dummy Kernel: Create a dummy byte array representing a compiled kernel to be loaded by Kernel::from_bytes.
Definition of Done: Kernel::from_bytes returns Ok with the dummy data.
 Implement Forward Pass on Device: Rewrite Neuron::forward to:
Allocate device memory for input.
Copy input data from host to device.
Launch the kernel using Kernel::launch.
Copy the result back to host.
Definition of Done: forward method compiles and runs using neuromorph-driver APIs without errors.
Phase 4: Integration & Verification
 Update WebSocket Handler: Ensure handle_client calls the new CUDA-backed forward method.
Definition of Done: WebSocket server responds to input (e.g., "1,2,3") with a result calculated via the driver.
 Integration Test: Create a test in tests/ that starts the server, connects via WebSocket, sends data, and asserts a valid response is received.
Definition of Done: cargo test passes the new integration test.

## 2. NN frameworks integration 
Phase 1: Computational Engine (Data Plane)
Goal: Enable the simulator to actually execute mathematical operations, moving beyond "mock" execution.

1.1 Define Tensor Metadata Structure

Task: Modify neuromorphMalloc in simulator.rs to allocate a structured Tensor object (containing shape, strides, data type, and data pointer) instead of just raw bytes.
Definition of Done:
neuromorphMalloc allocates a structure that tracks dimensions (e.g., [3, 3]) and type (e.g., f32).
neuromorphMemcpy validates that the source/destination sizes match the allocated tensor's capacity.
Unit tests confirm metadata is preserved across allocations.
1.2 Implement Kernel Registry System

Task: Replace the opaque Vec<u8> kernel blob with a registry of built-in Rust functions. Create an enum or ID system to dispatch kernels by name/ID.
Definition of Done:
A KernelRegistry struct exists in simulator.rs.
neuromorphLaunchKernel accepts a Kernel ID, looks up the corresponding Rust function, and executes it.
Unknown Kernel IDs return a specific error code.
1.3 Implement Dense Matrix Multiplication (GEMM) Kernel

Task: Implement a Rust function that performs C = alpha _ (A @ B) + beta _ C and register it in the kernel registry.
Definition of Done:
A unit test allocates three tensors (A, B, C) on the "device".
neuromorphLaunchKernel is called with the GEMM kernel ID.
The result in C matches the expected matrix multiplication result (verified against a CPU reference).
1.4 Implement Element-wise Operations (Add, Mul, ReLU)

Task: Implement kernels for basic arithmetic and activation functions.
Definition of Done:
Kernels for Add, Mul, and ReLU are registered.
Unit tests verify correctness for each operation on random input data.
Phase 2: Training Primitives (Autograd Support)
Goal: Enable the calculation of gradients required for backpropagation.

2.1 Implement Backward Pass for GEMM (MatMulGradient)

Task: Implement the gradient computation for Matrix Multiplication: given dC (gradient of output), compute dA = dC @ B^T and dB = A^T @ dC.
Definition of Done:
A MatMulBackward kernel is registered.
Gradient checks (finite difference comparison) confirm the kernel computes correct gradients for inputs A and B.
2.2 Implement Backward Pass for Element-wise Ops

Task: Implement derivative kernels for ReLU (step function), Add (pass-through), and Mul.
Definition of Done:
Backward kernels are registered for all Phase 1 operators.
Unit tests verify that gradients propagate correctly through a chain of operations (e.g., ReLU(Add(A, B))).
2.3 Implement SGD Optimizer Kernel

Task: Implement a kernel that updates weights in-place: param -= learning_rate \* grad.
Definition of Done:
An SGD kernel is registered.
A test case shows a tensor's values changing in the direction of the negative gradient after execution.
2.4 Implement Random Number Generation (RNG) Kernel

Task: Implement a kernel to fill a tensor with random values (uniform/normal distribution) for weight initialization.
Definition of Done:
An RNG kernel is registered.
Calling it multiple times produces different sequences (unless seeded deterministically).
Statistical tests (mean/variance) confirm the distribution matches the request.
Phase 3: Python & PyTorch Integration
Goal: Connect the Rust backend to the Python ecosystem.

3.1 Create neuromorph-python PyO3 Bindings

Task: Create a new crate that exposes neuromorph-driver functionality to Python using PyO3.
Definition of Done:
pip install . works.
Python script can import neuromorph, allocate memory, and launch a dummy kernel.
3.2 Implement PyTorch PrivateUse1 Dispatch Key — MVP DONE (2026-10-05)

Implemented for the active DANMA path as a C++ extension registered under
PrivateUse1 and renamed to `danma`. `torch.device("danma:0")` works and
DANMA tensors report a non-CPU device. The backend is fail-closed: unsupported
ATen operators raise instead of silently using CPU kernels. CI verifies
forward/backward against a real Rust node and proves that stopping the node
prevents local fallback.

3.3 Register Allocator with PyTorch (c10::Allocator) — MVP PARTIAL

A PrivateUse1 `c10::Allocator` is registered and currently uses host memory as
explicit staging storage. CPU <-> DANMA copies, device guard and synchronous
events are implemented. Device-resident DANMA tensor storage and allocator
telemetry comparable to CUDA remain future work.

3.4 Register Core Operators (aten::mm, aten::add) — PARTIAL / NEXT

The current supported compute operator is `DANMALinear`: its affine
forward/backward and local SGD execute on Rust-owned DANMA neuron state through
the TCP data plane, while PyTorch supplies autograd orchestration and can
compute the loss on CPU. Generic `aten::mm`, `aten::add` and arbitrary ATen
coverage are intentionally not registered yet; using such operations directly
on a DANMA tensor fails closed.


## 3. Adaptive Neuron Lifecycle / Neurogenesis

Goal: Allow DANMA to continuously remove persistently ineffective neurons and create new neurons without stopping the network or introducing a global training epoch.

Important distinction: the earlier idea of periodically replacing a fraction of the worst connections applies to dendrites/edges. Neuron replacement is a separate, more conservative lifecycle and must not blindly reuse a fixed "replace 25%" rule.

### Phase 1: Neuron Utility Telemetry

#### 3.1 Track local neuron effectiveness

Task: Add bounded, windowed/EWMA statistics to each neuron or its shard-owned metadata. At minimum track activation count, completed feedback count, useful downstream contribution, gradient/update magnitude, age/sample count, and enough information to detect prolonged inactivity or redundancy.

Selection must not classify a neuron as ineffective merely because it fires rarely; rare but strongly useful neurons must survive.

Definition of Done:
- Per-neuron utility telemetry is bounded in memory and updated without global synchronization.
- Utility statistics are inspectable through the existing shard/runtime observability path.
- Tests cover frequent-use, rare-but-useful, inactive, and persistently low-contribution neurons.

#### 3.2 Define a configurable `utility_score` and retirement policy

Task: Introduce a local policy that ranks retirement candidates using multiple signals rather than one threshold. Include minimum neuron age / minimum sample count, hysteresis, a configurable evaluation window, and a bounded replacement budget per window.

Definition of Done:
- Newly created or insufficiently sampled neurons cannot be retired prematurely.
- Rare but high-impact neurons are not selected solely due to low activation count.
- Replacement budget is configurable and bounded; neuron replacement has no hard-coded 25% default.
- Deterministic tests verify candidate ranking for synthetic histories.

### Phase 2: Safe Retirement State Machine

#### 3.3 Add neuron lifecycle states

Task: Add explicit lifecycle states such as `ACTIVE -> DRAINING -> RETIRED`.

When a neuron becomes `DRAINING`, routing must stop sending new forward activations to it, while already accepted activations remain valid until their feedback completes or their existing TTL expires.

Definition of Done:
- No new forward event is accepted after a neuron enters `DRAINING`.
- Existing pending `EventID` records can still complete normally.
- Retirement does not violate current feedback deduplication, replay-retention, or TTL invariants.

#### 3.4 Make late messages safe

Task: Preserve enough retirement/tombstone information so that delayed or duplicated forward/feedback packets for an old neuron generation cannot mutate a new neuron.

A retired `NeuronID` must never be immediately reused for a different neuron. Prefer globally unique/monotonic IDs; if generations are ever introduced, messages and routes must include the generation/epoch and reject stale generations.

Definition of Done:
- Delayed feedback after retirement is rejected or classified as expired/stale, never applied to another neuron.
- Duplicate delivery remains idempotent.
- Property/integration tests cover delayed, reordered, duplicated and post-retirement packets.

### Phase 3: Dynamic Shard Membership and Routing

#### 3.5 Support runtime add/drain/remove operations

Task: Replace the current startup-only immutable neuron ownership assumption with a controlled mutable lifecycle API. Add shard/runtime operations to register a neuron, begin drain, finalize removal, and inspect lifecycle state.

Definition of Done:
- A running shard can add a neuron without restart.
- A running shard can drain and remove a neuron after outstanding activations are resolved or expired.
- Ownership changes are serialized/fenced so stale workers cannot resurrect or update a retired neuron.
- Existing static startup path remains supported.

#### 3.6 Version route updates

Task: Propagate neuron creation/retirement and ownership changes through the DANMA control plane / routing mechanism using versioned route records. Gossip may distribute discovery/update information, but the data plane should still use direct delivery once the owner is known.

Definition of Done:
- Peers eventually learn that a retired neuron is unavailable and that a new neuron has an owner.
- Stale route information cannot cause state mutation on the wrong neuron generation.
- Tests cover concurrent route update, node delay, duplicate gossip, and stale-route delivery.

### Phase 4: Neurogenesis

#### 3.7 Create replacement neurons

Task: After a neuron reaches `RETIRED` and its slot/capacity becomes available, create a replacement with a new `NeuronID`.

Initial connectivity should combine exploitation and exploration:
- exploitation: sample useful nearby/upstream/downstream structure from successful neurons;
- exploration: add randomized valid connections so the network can discover new representations.

The exploitation/exploration ratio must be configurable and experimentally measured rather than hard-coded as an architectural invariant.

Definition of Done:
- New neuron receives a fresh identity, initialized bias/weights, valid bounded dendrites/axons, and a registered owner/route.
- New connectivity respects `MAX_DENDRITES_PER_NEURON` and `MAX_AXONS_PER_NEURON`.
- Seeded tests reproduce topology generation; unseeded mode produces diversity.
- No dangling references remain to a physically removed neuron.

#### 3.8 Add probation / maturation period

Task: Give a newly created neuron a configurable probation period during which it learns normally but is protected from immediate retirement. After sufficient age/samples it enters the normal utility competition.

Definition of Done:
- New neurons survive the minimum maturation period.
- Their utility history starts cleanly and cannot inherit stale statistics from a retired neuron.
- After maturation, they are evaluated by the same policy as all other active neurons.

### Phase 5: Verification and Experiments

#### 3.9 Prove lifecycle safety under asynchronous execution

Task: Add deterministic simulation/property tests for lifecycle transitions while forward/backward traffic is in flight.

Required scenarios:
- retire while feedback is delayed;
- duplicate feedback during drain;
- TTL expiration during drain;
- stale route after retirement;
- new neuron created while old packets are still in the network;
- shard/node restart during `DRAINING`;
- repeated create/retire cycles under bounded capacity.

Definition of Done:
- No feedback is ever applied to the wrong neuron identity/generation.
- One legitimate feedback contribution is applied at most once according to the existing contribution identity rules.
- No neuron disappears while it still owns an unexpired activation unless recovery semantics explicitly preserve that activation.
- Memory used by lifecycle metadata, tombstones and telemetry remains bounded.

#### 3.10 Evaluate whether neurogenesis improves learning

Task: Compare fixed-topology DANMA with adaptive-neurogenesis DANMA on the same workloads and seeds.

Measure at least:
- validation loss / accuracy;
- convergence speed;
- neuron utilization distribution;
- number of retirements and births;
- topology churn;
- network traffic;
- CPU/RAM overhead;
- recovery from deliberately degraded or redundant neuron populations.

Definition of Done:
- Experiment artifacts make it possible to determine whether adaptive neuron replacement improves model quality or resource efficiency.
- Neurogenesis can be disabled with a feature/config flag so fixed-topology behavior remains a reference baseline.
