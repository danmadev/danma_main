# danma-torch — PyTorch bridge to distributed DANMA

This package connects PyTorch to the real distributed DANMA CPU runtime. It now
supports both the original CPU-tensor autograd bridge and an explicit,
**fail-closed PyTorch PrivateUse1 device** registered as `danma`.

The important boundary is:

- PyTorch may hold/stage tensor bytes in host memory and may compute the loss
  or unrelated CPU layers.
- The `DANMALinear` affine forward, local learning update, and upstream
  gradient are executed by the addressable Rust DANMA neurons through the TCP
  data plane.
- There is **no generic ATen CPU fallback for the `danma` dispatch key**.
  An unsupported operator on a DANMA tensor fails instead of silently running
  the operator in PyTorch on CPU.
- If the Rust DANMA node is unavailable, `DANMALinear` fails with a transport
  error; it cannot substitute a local `torch.nn.Linear`.

This is therefore a real `torch.device("danma:0")` integration for the
currently supported DANMA layer path, not yet a general-purpose PyTorch device
implementing arbitrary ATen operators.

## Install and verify

Build the real Rust TCP node:

    cargo build --locked -p danma-net --bin danma-node

Install CPU PyTorch 2.8 and compile the PrivateUse1 extension:

    python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.8.0"
    python -m pip install -e ./python --no-build-isolation --no-deps

Run the fail-closed device proof **in a fresh Python process**:

    DANMA_NODE_BIN=target/debug/danma-node \
      python -m unittest discover -s python/tests -p "test_privateuse1.py" -v

Then run the ordinary CPU-tensor bridge and convergence suites:

    DANMA_NODE_BIN=target/debug/danma-node \
      python -m unittest discover -s python/tests -p "test_autograd.py" -v

    DANMA_NODE_BIN=target/debug/danma-node \
      python -m unittest discover -s python/tests -p "test_training_convergence.py" -v

PyTorch 2.8 initializes accelerator autograd worker queues when its autograd
Engine first starts. Call `enable_privateuse1()` before the first backward in
a process that will use DANMA tensors.

## Explicit DANMA device

    import torch
    from danma_torch import DANMAClient, DANMALinear, enable_privateuse1

    enable_privateuse1()
    device = torch.device("danma:0")

    client = DANMAClient("127.0.0.1", 9101)
    layer = DANMALinear(
        client,
        neuron_ids=(11, 21, 31),
        input_ids=(901, 902),
    )

    x = torch.tensor([1.0, 2.0], dtype=torch.float32).to(device)
    x.requires_grad_(True)

    y = layer(x)
    assert y.device.type == "danma"
    assert not y.is_cpu

    # Loss math is outside the DANMA affine operator and can run on CPU.
    loss = y.to("cpu").square().sum()
    loss.backward()

    # The gradient crossed the DANMA backward path.
    assert x.grad.device.type == "danma"

    # This is intentionally unsupported and MUST fail rather than use CPU:
    # z = y + y

The PrivateUse1 allocator currently uses host memory as **staging storage**.
That does not mean PyTorch CPU computes the DANMA affine layer: input values
are serialized to the Rust node, its neuron state computes forward and applies
the SGD update during backward, and returned upstream gradients are copied
back into the DANMA tensor. The storage implementation can later be replaced
without changing this execution invariant.

## What CI proves

The PrivateUse1 integration test starts a real Rust `danma-node` with neuron
11 and weights `[2, 3]`, then checks all of the following:

1. `torch.device("danma:0")` is a genuine PrivateUse1 device and
   `tensor.is_cpu == False`.
2. CPU <-> DANMA staging copies preserve data.
3. An unsupported `tensor + tensor` raises; there is no generic CPU
   operator fallback.
4. Stopping the Rust node makes `DANMALinear` fail instead of computing
   locally.
5. Forward on `[1, 2]` returns the Rust neuron's value `8`.
6. Backward increments the Rust neuron's version and changes remote weights
   from `[2, 3]` to `[1.9, 2.8]` and bias to `-0.1`.
7. The input gradient returns as a DANMA tensor with value `[2, 3]`.
8. A following inference observes the newly updated remote parameters and
   returns `7.4`.

The separate multi-epoch convergence benchmark still trains three output
neurons owned by three separate DANMA processes for 30 epochs / 120 updates
per neuron and compares their final state with equivalent sequential PyTorch
SGD.

## Start a simple three-node layer

The first node owns neuron 11 with host inputs 901 and 902:

    cargo run -p danma-net --bin danma-node -- \
      --id 1 --listen 127.0.0.1:9101 \
      --neuron 11 --weight 901:2 --weight 902:3 \
      --peer 2@127.0.0.1:9102 --peer 3@127.0.0.1:9103

The second node owns neuron 21:

    cargo run -p danma-net --bin danma-node -- \
      --id 2 --listen 127.0.0.1:9102 \
      --neuron 21 --weight 901:-1 --weight 902:4 \
      --peer 1@127.0.0.1:9101 --peer 3@127.0.0.1:9103

The third node owns neuron 31:

    cargo run -p danma-net --bin danma-node -- \
      --id 3 --listen 127.0.0.1:9103 \
      --neuron 31 --weight 901:0.5 --weight 902:-2 \
      --peer 1@127.0.0.1:9101 --peer 2@127.0.0.1:9102

## Training contract

* A neuron is one output feature; all input synapses already exist in
  `danma-node`. The current CLI initializes bias to zero, linear activation,
  and local SGD with learning rate 0.1.
* The v1 adapter accepts at most 1024 input and 1024 output features; each
  neuron is capped at 1024 dendrites and 1024 axons.
* Forward creates unique EventIDs. Backward sends the teacher gradient to the
  remote output neuron. The owning DANMA neuron applies the state update and
  produces upstream gradient contributions.
* Remote weights are not `nn.Parameter` objects. `torch.optim.step()` does
  not own or update them; the current DANMA local SGD is applied during remote
  backward.
* CPU tensors remain supported for composition with existing PyTorch modules.
  DANMA tensors make the device boundary explicit.
* Each differentiable forward supports one backward. TTL, duplicate feedback,
  stale activation and routing checks remain enforced by the Rust runtime.

## Current limits

PrivateUse1 currently implements only the minimum tensor/device infrastructure
needed for the DANMA bridge: allocator/staging, device guard, synchronous
events, CPU <-> DANMA copies and tensor creation. Arbitrary ATen operators,
`torch.compile`, `vmap`, higher-order gradients, CUDA/PTX compatibility,
checkpoint/recovery and untrusted cross-host networking remain out of scope.

The transport still sends synchronous TCP requests using JSON framing, and
PrivateUse1 staging memory is host-resident. Performance work requires a batch
protocol and shard/tensor kernels; the purpose of this slice is correctness
and an unambiguous execution boundary.
