# DANMA TCP node — distributed CPU MVP slice

This is a **working three-process localhost prototype**, not a public P2P
network or a production device driver. It depends on the standalone
[danma-core](../danma-core/README.md) single-writer neuron state machine.
CUDA and GPU support are out of scope.

## Architecture

- Each OS process owns a **multi-neuron CPU shard**, served by one TCP
  listener and a fixed, bounded number of CPU workers. NeuronId selects a
  local worker mailbox; a neuron has no individual process, thread or socket.
  Different neurons on the same process propagate feedback locally.
- A small, static allowlist supplies the identities and addresses of bootstrap
  peers. Round-robin gossip (one peer per 100 ms) exchanges bounded neuron
  owner/epoch advertisements. Forward activations and backward feedback are
  sent by direct TCP to the discovered owner, rather than flooded by gossip.
- Neurons can declare logical outgoing axons with `--axon EDGE:TARGET`.
  After a neuron fires, the runtime resolves each target through the route table
  and sends a forward signal locally or by direct TCP. Downstream neurons collect
  fan-in by TraceID/EventID and fire only when all configured dendrite sources
  have arrived. Terminal outputs are returned in the top-level `terminals` list.
  Wire validation permits up to **1024 incoming dendrites and 1024 expected/outgoing branches per neuron**; the development route table is capped at 2048 entries.
- The protocol frames each JSON request with a 4-byte, big-endian frame length
  and caps the payload at 256 KiB. There is one request and one response per
  connection. The listener limits concurrent connections to 32; input fan-in,
  feedback fan-out and route table size are bounded separately.
- The wire protocol v1 uses u64 EventIDs; danma-core retains u128 identifiers.
  Migration to an unambiguous globally unique wire ID is required before
  large-scale dynamic network deployment.
- An activation specifies expected downstream contributions before dispatch.
  A neuron aggregates distinct contributions and applies one weight update;
  repeated delivery of an already-seen contribution does not update twice.
  Relative feedback TTL decreases across relays and is checked again by the
  CPU worker after mailbox waiting; gradient hops are separate from routing
  hops. Routes are neither globally consistent nor transactional.

## Run a local three-node cluster

From the repo root, in three terminals:

    cargo run -p danma-net --bin danma-node -- \
      --id 1 --listen 127.0.0.1:9101 \
      --neuron 1 --weight 99:2 --axon 12:2 \
      --peer 2@127.0.0.1:9102 --peer 3@127.0.0.1:9103

    cargo run -p danma-net --bin danma-node -- \
      --id 2 --listen 127.0.0.1:9102 \
      --neuron 2 --weight 1:3 --axon 23:3 \
      --peer 1@127.0.0.1:9101 --peer 3@127.0.0.1:9103

    cargo run -p danma-net --bin danma-node -- \
      --id 3 --listen 127.0.0.1:9103 --neuron 3 --weight 2:4 \
      --peer 1@127.0.0.1:9101 --peer 2@127.0.0.1:9102

The --neuron/--weight pair can be repeated for multiple neurons within one
process. For example, Node 1 can also host neuron 4 with a local input from
neuron 1:

    cargo run -p danma-net --bin danma-node -- \
      --id 1 --listen 127.0.0.1:9101 --workers 2 --mailbox 16 \
      --neuron 1 --weight 99:2 --neuron 4 --weight 1:1 \
      --peer 2@127.0.0.1:9102 --peer 3@127.0.0.1:9103

To inspect gossip convergence from another shell:

    python3 - <<'PY'
    import json, socket, struct
    request = json.dumps({"kind": "routes"}).encode()
    with socket.create_connection(("127.0.0.1", 9101), timeout=2) as s:
        s.sendall(struct.pack("!I", len(request)) + request)
        size = struct.unpack("!I", s.recv(4))[0]
        data = bytearray()
        while len(data) < size:
            part = s.recv(size - len(data))
            if not part:
                raise EOFError("incomplete DANMA frame")
            data.extend(part)
        print(json.loads(data))
    PY

A routed inspect request returns a neuron's current version, bias and
weights. During an active training activation, a routed trace request such as
{"kind":"trace","target":2,"event_id":50,"route_hops":4} returns the
TraceID, output, weight version, expiry and number of received/expected
feedback contributions. Completed traces are removed from active memory;
there is no durable per-activation history yet.

The integration tests launch three **separate child processes**, including
a six-neuron layout with two neurons and two CPU workers per process. They
exercise A→B→C forward, C→B→A backward, local and remote gradient hops,
retry/dedup, TTL expiry, invalid contributor, gossip convergence, remote
traces and malformed frames:

    cargo test --locked -p danma-core -p danma-shard -p danma-net --all-targets

## Bounded neuron configuration file (v1)

For dense initializations, use `--neuron-config PATH` instead of expanding
every weight into command-line arguments (which can exceed OS argument limits):

    cargo run --locked -p danma-net --bin danma-node -- \
      --id 1 --listen 127.0.0.1:9101 --workers 2 --mailbox 256 \
      --neuron-config node-1.json --peer 2@127.0.0.1:9102

A valid complete document is:

```json
{
  "schema_version": 1,
  "settings": {
    "learning_rate": 0.1,
    "activation_ttl_ms": 120000,
    "replay_retention_ms": 10000,
    "max_live_events": 4096,
    "max_staleness_versions": 8
  },
  "neurons": [
    {
      "id": 10001,
      "bias": 0.0,
      "activation": "linear",
      "weights": [{"source": 20001, "weight": 0.001}]
    }
  ]
}
```

All displayed keys are **required**; no defaults or additional keys are allowed
at any object level. Duplicate object keys, duplicate neuron IDs, and duplicate
sources within one neuron are rejected. The top level is an object, settings is
an object, neurons and weights are arrays of objects. The version must be the
JSON integer `1`; activation must be exactly the string `"linear"` or `"relu"`.
Version 1 has **no axons**; file neurons have empty outgoing axon lists. Existing
legacy `--axon` support is unchanged.

Limits (inclusive unless stated otherwise):

| Field/resource | Contract |
| --- | --- |
| File | At most 64 MiB (67,108,864 bytes), including whitespace |
| Neurons | 1–2048 per file |
| Weights | 0–1024 per neuron; at most 1,048,576 total per file |
| id, source | JSON integers, 1–18,446,744,073,709,551,615; exact unsigned 64-bit parsing, never through a float |
| bias, weight | JSON numbers finite and within ±float32 maximum before narrowing; decimal/scientific notation and integers accepted |
| learning_rate | Same numeric decoder, then float32 value strictly positive and at most 1 |
| activation_ttl_ms, replay_retention_ms | JSON integers, 1–600,000 milliseconds |
| max_live_events | JSON integer, 1–4096; shared capacity for traces and tombstones per neuron |
| max_staleness_versions | JSON integer, 0–8 |

Numeric strings, booleans, nulls, objects and arrays are not numbers. Fractional
or scientific tokens are not accepted for integer fields, even if mathematically
integral. Float fields use the existing wire decoder's finite float64 range
check before conversion to float32, with accurate JSON float parsing. Normal
float32 rounding applies; very small magnitudes may underflow to zero (which
is invalid for learning_rate). Empty weight arrays permit bias-only neurons,
matching the core constructor contract; an empty neuron array is invalid.

The reader consumes at most limit+1 bytes and does not trust file metadata.
JSON parsing is directly into strict typed structs with bounded array visitors;
it rejects excess array elements without storing them and retains the parser's
normal nesting limit. The byte buffer, parsed vectors and core weight maps
still incur bounded memory/CPU costs; 16 MiB is an input cap, **not** a promise
that total startup memory is 16 MiB. The whole document is validated before
core construction, shard workers or the TCP listener. Failure exits nonzero;
there is no fallback to legacy flags. This is startup-only loading, not reload
or checkpoint support.

`--neuron-config` can appear only once and is mutually exclusive with every
`--neuron`, `--weight` and `--axon` flag. Node identity, listen address, peers,
workers and mailbox remain CLI-only. Legacy defaults are unchanged: zero bias,
linear activation, learning rate 0.1, activation TTL 10,000 ms, replay retention
10,000 ms, 4096 live events and staleness 8.

The authorized 784→1000→ReLU→10 layout has 1010 logical neurons across ten
processes (101 per file): hidden neurons have 784 weights and output neurons
1000 weights, totaling 794,000 weights plus 1010 biases = 795,010 parameters.
These fit the file bounds. The parent experiment must generate the shared
float32 initialization and assign neuron ownership; the loader does neither.

The experimental [single-node MNIST runner](../python/MNIST_SINGLE_NODE.md) uses the same bounded file format with one process. Its default 1000-hidden layout owns 1010 logical neurons and 794,000 weights in one node; `--hidden-neurons` can vary the hidden width from 1 to 1024 without source changes.
Use `"relu"` for hidden neurons and `"linear"` for outputs. A 120,000 ms trace
TTL can accommodate a long sequential forward pass, but must be measured by
the experiment; it does **not** extend a feedback packet's relative deadline.
No MNIST performance or completion claim is made by this loader.

## Known gaps and non-goals

**Security:** the entire v1 network is restricted to loopback addresses.
A static peer allowlist is not authentication: the incoming claimed node ID
is not cryptographically bound to the TCP source. No TLS, signed gossip,
Sybil resistance, or authorization of teachers is implemented. Do not expose
this prototype to untrusted networks or accept arbitrary user traffic.

**Delivery and crash recovery:** outbound feedback is sent only after the local
weight update. If the process crashes or the network fails in between, there
is no durable outbox; a timeout means the result may be unknown. The bounded
in-memory activation and dedup ledger does not survive restart. The response
returns unrouted/uncertain upstream gradients rather than silently dropping
them. A production version needs atomic journaling, retry/replay, epochs,
dedup persistence and versioned owner fencing.

**Topology:** gossip operates over a known three-peer mesh; there is no DHT,
dynamic admission, signed advertisement, liveness suspicion or route lease
expiry. Routing state is bounded but not guaranteed to be fresh. All nodes
must agree on their bootstrap peer identities and network addresses.

**Time and training:** each host currently derives the local core deadline
from wall-clock Unix milliseconds, while the transport carries a relative
remaining TTL. For independent remote hosts, clock-skew handling and a
monotonic per-node activation clock are required. Asynchronous training
convergence has not been established.

**Framework integration:** the Python adapter now registers a fail-closed
PyTorch PrivateUse1 device as `danma:0` for the DANMALinear path. Its storage
is host staging memory, while affine forward/backward and local SGD go through
this TCP data plane to Rust-owned neuron state. Unsupported DANMA tensor
operators fail instead of falling back to ATen CPU. TensorFlow PluggableDevice,
broad ATen coverage and device-resident tensor storage remain future work.
