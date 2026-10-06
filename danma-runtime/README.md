# DANMA Rust-native runtime

This crate is the process-local data plane for DANMA.

It has **no TCP, JSON, database or storage dependency**. Logical neurons are
owned by `danma-shard` workers and communicate inside one process through
bounded Rust mailboxes. Only work that crosses a process boundary is returned
as typed egress for `danma-net`.

## Execution layers

```text
logical neuron
    |
    v
danma-core
EventID / TTL / version / learning semantics
    |
    v
danma-shard
single-writer workers + bounded in-memory mailboxes
    |
    v
danma-runtime
local forward/backward graph cascade
    |
    +------ local target ------> Rust mailbox only
    |
    +------ remote target -----> typed egress
                                   |
                                   v
                               danma-net
                               TCP gateway
```

Gossip remains a **control-plane** mechanism. It is not used for local
neuron-to-neuron data delivery.

## 100-billion-neuron address contract

`NeuronId` is already a `u64`, so no identifier-width migration is required
for 100 billion logical neurons.

The default logical partitioning is:

```text
100,000,000,000 neurons
/ 1,000,000 neurons per logical shard
= 100,000 shard placement entries
```

`AddressSpace::locate()` computes:

```text
NeuronID -> ShardID + local_offset
```

with O(1) arithmetic and no per-neuron global route table.

The one-million-neuron shard size is a **logical addressing default**, not a
claim that the current BTreeMap-backed core should run one million hot neurons
per process. Physical shard size must be measured against RAM, fan-in/out and
activation working set.

## Placement gossip

`PlacementDirectory` stores:

```text
ShardID -> NodeID + PlacementEpoch
```

rather than:

```text
NeuronID -> NodeID
```

It rejects:

- zero/invalid IDs;
- oversized gossip batches;
- capacity overflow;
- two different owners claiming the same shard at the same epoch.

Older epochs are ignored. A newer epoch replaces an older placement.

This directory is **discovery state, not consensus**. A future migration /
ownership controller must establish fencing authority before publishing the
new epoch.

## Local forward path

Once one local neuron fires, `LocalDataPlane::cascade_forward()` walks local
descendants directly through the Rust shard API.

```text
Neuron A fires
   |
   +-- B is local --> shard.signal(B) --> maybe fires C
   |                                    |
   |                                    +-- C local --> Rust only
   |
   +-- X is remote --> RemoteForward typed egress
```

No local hop is converted into a network `Message`, JSON frame or TCP request.

The cascade is bounded by:

- `max_deliveries`;
- forward hop budget;
- `max_remote_egress`;
- the shard's bounded per-worker mailbox;
- the neuron's existing EventID/activation capacities.

## Local backward path

After one neuron applies feedback, its upstream feedback dispatches are walked
locally in the same way.

```text
feedback at C
   |
   +-- upstream B local --> shard.backward_live(B)
   |                       |
   |                       +-- upstream A local --> Rust only
   |
   +-- upstream remote --> RemoteFeedback typed egress
```

Duplicate EventIDs, TTL expiry, staleness and local versions remain governed by
`danma-core`; the runtime does not invent a second learning model.

## What this version deliberately does not solve

1. **Dense/SoA/CSR storage.** Current neurons still use the existing
   BTreeMap-based core representation.
2. **Persistence.** There is no ScyllaDB, WAL or checkpoint layer in this
   branch.
3. **Shard migration authority.** Placement epochs are modeled, but ownership
   consensus/fencing is not yet implemented.
4. **Persistent remote connections / binary wire protocol.** Remote boundaries
   still use the existing `danma-net` development transport.
5. **Wave/microbatch local scheduling.** The first Rust-native data plane
   removes protocol recursion. A subsequent optimization can group ready local
   signals by worker/kernel without changing logical event semantics.

## Required invariants

- A local edge never requires a TCP hop.
- A remote edge is emitted exactly as typed egress from the local runtime.
- Same EventID feedback cannot train the same closed activation twice.
- Independent valid feedback contributions remain distinct.
- Every queue/cascade has a hard bound.
- Global route-state cardinality grows with shard count, not neuron count.
- Gossip does not become a global synchronization barrier.

## Next implementation milestones

```text
Rust-local data plane            <-- this branch
        |
        v
local signal wave batching
        |
        v
SoA / CSR neuron storage
        |
        v
SIMD / GEMM compatible kernels
        |
        v
shard-level gossip in danma-net
        |
        v
binary persistent inter-shard transport
        |
        v
checkpoint / WAL / recovery
        |
        v
1M -> 10M -> 100M -> 1B -> 100B scale tests
```
