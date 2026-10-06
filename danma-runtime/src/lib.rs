//! Rust-native execution primitives for DANMA.
//!
//! This crate deliberately contains no socket, JSON, database, or storage
//! dependency. A process-local shard communicates through bounded Rust
//! mailboxes owned by `danma-shard`. Only events that leave the process are
//! returned as typed egress records for a network gateway.
//!
//! The scale model is shard-first: 100 billion logical neurons fit in u64 IDs,
//! while the control plane tracks O(shards), not O(neurons), placement entries.

use danma_core::{
    derived_event_id, Axon, EdgeId, EventId, Feedback, FeedbackDispatch, FeedbackStatus,
    ForwardSignal, NeuronId, SignalStatus, SynapticInput, TraceId,
};
use danma_shard::{Shard, ShardError};
use std::{
    collections::{BTreeMap, VecDeque},
    sync::Arc,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

/// Design ceiling for the current logical address contract.
///
/// This is not a promise that one deployment can host this many live neurons;
/// it ensures identifiers and shard arithmetic do not need redesign before
/// experiments at that order of magnitude.
pub const MAX_LOGICAL_NEURONS: u64 = 100_000_000_000;

/// One million logical neurons per shard gives 100,000 placement entries for
/// 100 billion neurons. The actual physical shard size remains workload- and
/// memory-dependent.
pub const DEFAULT_NEURONS_PER_SHARD: u64 = 1_000_000;

pub const DEFAULT_MAX_LOCAL_DELIVERIES: usize = 1_000_000;
pub const DEFAULT_MAX_REMOTE_EGRESS: usize = 65_536;
pub const DEFAULT_MAX_PLACEMENTS: usize = 1_000_000;
pub const DEFAULT_MAX_GOSSIP_BATCH: usize = 4_096;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct ShardId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct NodeId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct PlacementEpoch(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LogicalAddress {
    pub shard_id: ShardId,
    /// Zero-based offset inside the deterministic logical shard range.
    pub local_offset: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddressError {
    InvalidConfiguration,
    UnknownNeuron(NeuronId),
}

/// Deterministic logical partitioning. It performs O(1) arithmetic and never
/// allocates a route entry per neuron.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AddressSpace {
    max_neurons: u64,
    neurons_per_shard: u64,
}

impl AddressSpace {
    pub fn new(max_neurons: u64, neurons_per_shard: u64) -> Result<Self, AddressError> {
        if max_neurons == 0
            || max_neurons > MAX_LOGICAL_NEURONS
            || neurons_per_shard == 0
        {
            return Err(AddressError::InvalidConfiguration);
        }
        Ok(Self {
            max_neurons,
            neurons_per_shard,
        })
    }

    pub fn hundred_billion_default() -> Self {
        Self {
            max_neurons: MAX_LOGICAL_NEURONS,
            neurons_per_shard: DEFAULT_NEURONS_PER_SHARD,
        }
    }

    pub fn max_neurons(&self) -> u64 {
        self.max_neurons
    }

    pub fn neurons_per_shard(&self) -> u64 {
        self.neurons_per_shard
    }

    pub fn shard_count(&self) -> u64 {
        (self.max_neurons - 1) / self.neurons_per_shard + 1
    }

    pub fn locate(&self, neuron_id: NeuronId) -> Result<LogicalAddress, AddressError> {
        if neuron_id == 0 || neuron_id > self.max_neurons {
            return Err(AddressError::UnknownNeuron(neuron_id));
        }
        let zero_based = neuron_id - 1;
        Ok(LogicalAddress {
            shard_id: ShardId(zero_based / self.neurons_per_shard + 1),
            local_offset: zero_based % self.neurons_per_shard,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Placement {
    pub node_id: NodeId,
    pub epoch: PlacementEpoch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlacementAdvert {
    pub shard_id: ShardId,
    pub node_id: NodeId,
    pub epoch: PlacementEpoch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlacementError {
    InvalidConfiguration,
    CapacityExceeded,
    GossipBatchTooLarge,
    EpochConflict,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PlacementMerge {
    pub inserted: usize,
    pub updated: usize,
    pub unchanged: usize,
    pub stale_ignored: usize,
}

/// Gossip-derived shard directory.
///
/// This is discovery state, not an ownership consensus protocol. A future
/// migration controller must establish the new epoch/fencing authority before
/// advertising it. The directory only refuses same-epoch conflicting owners
/// and ignores stale advertisements.
#[derive(Debug, Clone)]
pub struct PlacementDirectory {
    entries: BTreeMap<ShardId, Placement>,
    max_entries: usize,
    max_gossip_batch: usize,
}

impl PlacementDirectory {
    pub fn new(max_entries: usize, max_gossip_batch: usize) -> Result<Self, PlacementError> {
        if max_entries == 0 || max_gossip_batch == 0 || max_gossip_batch > max_entries {
            return Err(PlacementError::InvalidConfiguration);
        }
        Ok(Self {
            entries: BTreeMap::new(),
            max_entries,
            max_gossip_batch,
        })
    }

    pub fn scalable_default() -> Self {
        Self {
            entries: BTreeMap::new(),
            max_entries: DEFAULT_MAX_PLACEMENTS,
            max_gossip_batch: DEFAULT_MAX_GOSSIP_BATCH,
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn get(&self, shard_id: ShardId) -> Option<Placement> {
        self.entries.get(&shard_id).copied()
    }

    /// Validate the complete gossip batch before mutating the table so an
    /// invalid advert cannot publish a partial control-plane update.
    pub fn merge_gossip(
        &mut self,
        adverts: &[PlacementAdvert],
    ) -> Result<PlacementMerge, PlacementError> {
        if adverts.len() > self.max_gossip_batch {
            return Err(PlacementError::GossipBatchTooLarge);
        }

        let mut staged: BTreeMap<ShardId, Placement> = BTreeMap::new();
        for advert in adverts {
            if advert.shard_id.0 == 0 || advert.node_id.0 == 0 || advert.epoch.0 == 0 {
                return Err(PlacementError::InvalidConfiguration);
            }
            let placement = Placement {
                node_id: advert.node_id,
                epoch: advert.epoch,
            };
            if let Some(previous) = staged.insert(advert.shard_id, placement) {
                if previous != placement {
                    return Err(PlacementError::EpochConflict);
                }
            }
        }

        let new_entries = staged
            .keys()
            .filter(|shard_id| !self.entries.contains_key(shard_id))
            .count();
        if self.entries.len().saturating_add(new_entries) > self.max_entries {
            return Err(PlacementError::CapacityExceeded);
        }

        let mut result = PlacementMerge::default();
        let mut updates = Vec::new();
        for (shard_id, incoming) in staged {
            match self.entries.get(&shard_id).copied() {
                None => {
                    result.inserted += 1;
                    updates.push((shard_id, incoming));
                }
                Some(current) if incoming.epoch < current.epoch => {
                    result.stale_ignored += 1;
                }
                Some(current) if incoming.epoch == current.epoch => {
                    if incoming.node_id != current.node_id {
                        return Err(PlacementError::EpochConflict);
                    }
                    result.unchanged += 1;
                }
                Some(_) => {
                    result.updated += 1;
                    updates.push((shard_id, incoming));
                }
            }
        }

        for (shard_id, placement) in updates {
            self.entries.insert(shard_id, placement);
        }
        Ok(result)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LocalLimits {
    pub max_deliveries: usize,
    pub max_remote_egress: usize,
}

impl Default for LocalLimits {
    fn default() -> Self {
        Self {
            max_deliveries: DEFAULT_MAX_LOCAL_DELIVERIES,
            max_remote_egress: DEFAULT_MAX_REMOTE_EGRESS,
        }
    }
}

impl LocalLimits {
    fn valid(self) -> bool {
        self.max_deliveries > 0 && self.max_remote_egress > 0
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RemoteForward {
    pub target: NeuronId,
    pub event_id: EventId,
    pub trace_id: TraceId,
    pub edge_id: EdgeId,
    pub from: NeuronId,
    pub source_event_id: EventId,
    pub value: f32,
    pub training: bool,
    pub forward_hops: u8,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TerminalActivation {
    pub neuron_id: NeuronId,
    pub event_id: EventId,
    pub output: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForwardFailureKind {
    HopLimitExhausted,
    Expired,
    Shard(ShardError),
    DeliveryBudgetExceeded,
    RemoteEgressBudgetExceeded,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ForwardFailure {
    pub target: NeuronId,
    pub edge_id: EdgeId,
    pub kind: ForwardFailureKind,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ForwardCascade {
    pub terminals: Vec<TerminalActivation>,
    pub remote: Vec<RemoteForward>,
    pub failures: Vec<ForwardFailure>,
    pub local_deliveries: usize,
    pub ignored_duplicates: usize,
    pub pending_inputs: usize,
    pub truncated: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RemoteFeedback {
    pub target: NeuronId,
    pub feedback: Feedback,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackwardFailureKind {
    Expired,
    Stale,
    Shard(ShardError),
    DeliveryBudgetExceeded,
    RemoteEgressBudgetExceeded,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackwardFailure {
    pub target: NeuronId,
    pub event_id: EventId,
    pub kind: BackwardFailureKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AppliedFeedback {
    pub target: NeuronId,
    pub event_id: EventId,
    pub version: u64,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct BackwardCascade {
    pub remote: Vec<RemoteFeedback>,
    pub applied: Vec<AppliedFeedback>,
    pub failures: Vec<BackwardFailure>,
    pub local_deliveries: usize,
    pub pending: usize,
    pub ignored_duplicates: usize,
    pub truncated: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeError {
    InvalidConfiguration,
}

#[derive(Debug, Clone, Copy)]
struct ForwardWork {
    source: NeuronId,
    source_event_id: EventId,
    trace_id: TraceId,
    value: f32,
    training: bool,
    remaining_hops: u8,
    axon: Axon,
}

/// Process-local DANMA data plane.
///
/// It owns no sockets and serializes no protocol frames. Same-process neuron
/// delivery goes directly through the shard's bounded Rust mailboxes. Only
/// cross-process work is emitted to the caller as typed egress.
#[derive(Clone)]
pub struct LocalDataPlane {
    shard: Arc<Shard>,
    limits: LocalLimits,
}

impl LocalDataPlane {
    pub fn new(shard: Arc<Shard>, limits: LocalLimits) -> Result<Self, RuntimeError> {
        if !limits.valid() {
            return Err(RuntimeError::InvalidConfiguration);
        }
        Ok(Self { shard, limits })
    }

    pub fn shard(&self) -> &Arc<Shard> {
        &self.shard
    }

    pub fn contains(&self, neuron_id: NeuronId) -> bool {
        self.shard.contains(neuron_id)
    }

    /// Continue a forward activation from a neuron that has already fired.
    ///
    /// Local descendants are processed iteratively in this process. Remote
    /// descendants are returned to the network gateway and are never encoded
    /// here. The delivery budget bounds accidental cycles even if hop limits
    /// were configured too generously.
    pub async fn cascade_forward(
        &self,
        source: NeuronId,
        source_event_id: EventId,
        trace_id: TraceId,
        output: f32,
        training: bool,
        forward_hops: u8,
        axons: &[Axon],
    ) -> ForwardCascade {
        let mut result = ForwardCascade::default();
        if axons.is_empty() {
            result.terminals.push(TerminalActivation {
                neuron_id: source,
                event_id: source_event_id,
                output,
            });
            return result;
        }

        let mut queue = VecDeque::new();
        for axon in axons {
            queue.push_back(ForwardWork {
                source,
                source_event_id,
                trace_id,
                value: output,
                training,
                remaining_hops: forward_hops,
                axon: *axon,
            });
        }

        while let Some(work) = queue.pop_front() {
            if result.local_deliveries + result.remote.len() >= self.limits.max_deliveries {
                result.failures.push(ForwardFailure {
                    target: work.axon.to,
                    edge_id: work.axon.edge_id,
                    kind: ForwardFailureKind::DeliveryBudgetExceeded,
                });
                result.truncated = true;
                break;
            }
            if work.remaining_hops == 0 {
                result.failures.push(ForwardFailure {
                    target: work.axon.to,
                    edge_id: work.axon.edge_id,
                    kind: ForwardFailureKind::HopLimitExhausted,
                });
                continue;
            }

            let child_event = derived_event_id(work.trace_id, work.axon.to);
            let child_hops = work.remaining_hops - 1;
            if !self.shard.contains(work.axon.to) {
                if result.remote.len() >= self.limits.max_remote_egress {
                    result.failures.push(ForwardFailure {
                        target: work.axon.to,
                        edge_id: work.axon.edge_id,
                        kind: ForwardFailureKind::RemoteEgressBudgetExceeded,
                    });
                    result.truncated = true;
                    break;
                }
                result.remote.push(RemoteForward {
                    target: work.axon.to,
                    event_id: child_event,
                    trace_id: work.trace_id,
                    edge_id: work.axon.edge_id,
                    from: work.source,
                    source_event_id: work.source_event_id,
                    value: work.value,
                    training: work.training,
                    forward_hops: child_hops,
                });
                continue;
            }

            result.local_deliveries += 1;
            let signal = ForwardSignal {
                event_id: child_event,
                trace_id: work.trace_id,
                now_ms: epoch_millis(),
                input: SynapticInput {
                    from: work.source,
                    source_event_id: work.source_event_id,
                    value: work.value,
                },
                training: work.training,
            };
            match self.shard.signal(work.axon.to, signal).await {
                Err(error) => result.failures.push(ForwardFailure {
                    target: work.axon.to,
                    edge_id: work.axon.edge_id,
                    kind: ForwardFailureKind::Shard(error),
                }),
                Ok(outcome) => match outcome.status {
                    SignalStatus::Pending { .. } => result.pending_inputs += 1,
                    SignalStatus::IgnoredDuplicate => result.ignored_duplicates += 1,
                    SignalStatus::Expired => result.failures.push(ForwardFailure {
                        target: work.axon.to,
                        edge_id: work.axon.edge_id,
                        kind: ForwardFailureKind::Expired,
                    }),
                    SignalStatus::Fired { output } => {
                        if outcome.axons.is_empty() {
                            result.terminals.push(TerminalActivation {
                                neuron_id: work.axon.to,
                                event_id: child_event,
                                output,
                            });
                        } else {
                            for axon in outcome.axons {
                                queue.push_back(ForwardWork {
                                    source: work.axon.to,
                                    source_event_id: child_event,
                                    trace_id: work.trace_id,
                                    value: output,
                                    training: work.training,
                                    remaining_hops: child_hops,
                                    axon,
                                });
                            }
                        }
                    }
                },
            }
        }

        result
    }

    /// Continue locally generated feedback until it reaches either local
    /// completion or a process boundary.
    pub async fn cascade_backward(
        &self,
        initial: impl IntoIterator<Item = FeedbackDispatch>,
        deadline: Instant,
    ) -> BackwardCascade {
        let mut result = BackwardCascade::default();
        let mut queue: VecDeque<_> = initial.into_iter().collect();

        while let Some(dispatch) = queue.pop_front() {
            if result.local_deliveries + result.remote.len() >= self.limits.max_deliveries {
                result.failures.push(BackwardFailure {
                    target: dispatch.target_neuron_id,
                    event_id: dispatch.feedback.event_id,
                    kind: BackwardFailureKind::DeliveryBudgetExceeded,
                });
                result.truncated = true;
                break;
            }

            if !self.shard.contains(dispatch.target_neuron_id) {
                if result.remote.len() >= self.limits.max_remote_egress {
                    result.failures.push(BackwardFailure {
                        target: dispatch.target_neuron_id,
                        event_id: dispatch.feedback.event_id,
                        kind: BackwardFailureKind::RemoteEgressBudgetExceeded,
                    });
                    result.truncated = true;
                    break;
                }
                result.remote.push(RemoteFeedback {
                    target: dispatch.target_neuron_id,
                    feedback: dispatch.feedback,
                });
                continue;
            }

            result.local_deliveries += 1;
            let event_id = dispatch.feedback.event_id;
            match self
                .shard
                .backward_live(dispatch.target_neuron_id, dispatch.feedback, deadline)
                .await
            {
                Err(error) => result.failures.push(BackwardFailure {
                    target: dispatch.target_neuron_id,
                    event_id,
                    kind: BackwardFailureKind::Shard(error),
                }),
                Ok(FeedbackStatus::Pending { .. }) => result.pending += 1,
                Ok(FeedbackStatus::IgnoredDuplicate) => result.ignored_duplicates += 1,
                Ok(FeedbackStatus::Expired) => result.failures.push(BackwardFailure {
                    target: dispatch.target_neuron_id,
                    event_id,
                    kind: BackwardFailureKind::Expired,
                }),
                Ok(FeedbackStatus::Stale) => result.failures.push(BackwardFailure {
                    target: dispatch.target_neuron_id,
                    event_id,
                    kind: BackwardFailureKind::Stale,
                }),
                Ok(FeedbackStatus::Applied { upstream, version }) => {
                    result.applied.push(AppliedFeedback {
                        target: dispatch.target_neuron_id,
                        event_id,
                        version,
                    });
                    queue.extend(upstream);
                }
            }
        }

        result
    }
}

fn epoch_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}
