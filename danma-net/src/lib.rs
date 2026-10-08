//! Process-boundary transport for the DANMA CPU-first prototype.
//!
//! Same-process neuron cascades are executed by danma-runtime through bounded
//! Rust mailboxes; they never recurse through TCP/JSON. This gateway is used
//! only when work crosses a process boundary. Gossip remains control-plane
//! discovery. Neither gossip nor an acknowledgement provides durable
//! exactly-once effects.
mod tensor;

use danma_core::{
    derived_event_id, Feedback, FeedbackSource, FeedbackStatus, Forward, ForwardSignal, Neuron, SignalStatus,
    SynapticInput, MAX_AXONS_PER_NEURON, MAX_DENDRITES_PER_NEURON,
};
use danma_runtime::{
    BackwardFailureKind, FiredActivation, ForwardFailureKind, LocalDataPlane, LocalLimits,
};
use danma_shard::Shard;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    io,
    net::{IpAddr, SocketAddr},
    sync::Arc,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    sync::{RwLock, Semaphore},
    time::timeout,
};

const MAX_FRAME_BYTES: usize = 256 * 1024;
const MAX_ADVERTISED_ROUTES: usize = 2_048;
const MAX_IO_WAIT: Duration = Duration::from_secs(4);
const MAX_CONCURRENT_CONNECTIONS: usize = 32;
const MAX_INCOMING_INPUTS: usize = MAX_DENDRITES_PER_NEURON;
const MAX_EXPECTED_BRANCHES: usize = MAX_AXONS_PER_NEURON;
const MAX_SHARD_TARGETS: usize = 1_024;
const DEFAULT_ROUTE_HOPS: u8 = 4;
const DEFAULT_FORWARD_HOPS: u8 = 32;

fn default_forward_hops() -> u8 {
    DEFAULT_FORWARD_HOPS
}

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn elapsed_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}

async fn read_frame_bytes(stream: &mut TcpStream) -> io::Result<Vec<u8>> {
    let length = stream.read_u32().await? as usize;
    if length == 0 || length > MAX_FRAME_BYTES {
        return Err(invalid("frame exceeds protocol size limit"));
    }
    let mut bytes = vec![0_u8; length];
    stream.read_exact(&mut bytes).await?;
    Ok(bytes)
}

async fn write_frame_bytes(stream: &mut TcpStream, bytes: &[u8]) -> io::Result<()> {
    if bytes.is_empty() || bytes.len() > MAX_FRAME_BYTES {
        return Err(invalid("outgoing frame exceeds size limit"));
    }
    stream.write_u32(bytes.len() as u32).await?;
    stream.write_all(bytes).await?;
    stream.flush().await
}

async fn read_frame(stream: &mut TcpStream) -> io::Result<Value> {
    let bytes = read_frame_bytes(stream).await?;
    serde_json::from_slice(&bytes).map_err(|_| invalid("invalid JSON frame"))
}

async fn write_frame(stream: &mut TcpStream, value: &Value) -> io::Result<()> {
    let encoded = serde_json::to_vec(value).map_err(|_| invalid("cannot serialize response"))?;
    write_frame_bytes(stream, &encoded).await
}

/// One request per TCP connection. A timeout is an UNKNOWN delivery outcome,
/// not evidence that a remote learning update was rolled back.
pub async fn request(address: SocketAddr, value: &Value) -> io::Result<Value> {
    timeout(MAX_IO_WAIT, async {
        let mut stream = TcpStream::connect(address).await?;
        write_frame(&mut stream, value).await?;
        read_frame(&mut stream).await
    })
    .await
    .map_err(|_| io::Error::new(io::ErrorKind::TimedOut, "DANMA request timed out"))?
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Advert {
    pub owner: u64,
    pub neuron: u64,
    pub epoch: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum Source {
    // Empty struct variants enforce deny_unknown_fields; unit variants do not.
    Teacher {},
    Neuron { neuron_id: u64, event_id: u64 },
}

impl From<Source> for FeedbackSource {
    fn from(source: Source) -> Self {
        match source {
            Source::Teacher {} => FeedbackSource::Teacher,
            Source::Neuron {
                neuron_id,
                event_id,
            } => FeedbackSource::Neuron {
                neuron_id,
                event_id: u128::from(event_id),
            },
        }
    }
}

impl TryFrom<FeedbackSource> for Source {
    type Error = io::Error;

    fn try_from(source: FeedbackSource) -> io::Result<Self> {
        Ok(match source {
            FeedbackSource::Teacher => Self::Teacher {},
            FeedbackSource::Neuron {
                neuron_id,
                event_id,
            } => Self::Neuron {
                neuron_id,
                event_id: u64::try_from(event_id)
                    .map_err(|_| invalid("EventID exceeds v1 wire range"))?,
            },
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    from: u64,
    source_event_id: u64,
    #[serde(deserialize_with = "deserialize_finite_f32")]
    value: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ShardInput {
    from: u64,
    #[serde(deserialize_with = "deserialize_finite_f32")]
    value: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ShardForwardTarget {
    target: u64,
    event_id: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ShardGradient {
    target: u64,
    event_id: u64,
    #[serde(deserialize_with = "deserialize_finite_f32")]
    gradient: f32,
}

fn deserialize_finite_f32<'de, D>(deserializer: D) -> Result<f32, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = f64::deserialize(deserializer)?;
    // Check before narrowing: values just beyond f32::MAX can round down to it.
    if !value.is_finite() || value.abs() > f64::from(f32::MAX) {
        return Err(serde::de::Error::custom("expected a finite f32"));
    }
    Ok(value as f32)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum Message {
    Gossip {
        from_node: u64,
        routes: Vec<Advert>,
    },
    // Unlike a unit variant, an empty struct enforces deny_unknown_fields.
    Routes {},
    Inspect {
        target: u64,
        route_hops: u8,
    },
    Trace {
        target: u64,
        event_id: u64,
        route_hops: u8,
    },
    Forward {
        target: u64,
        event_id: u64,
        trace_id: u64,
        route_hops: u8,
        #[serde(default = "default_forward_hops")]
        forward_hops: u8,
        inputs: Vec<Input>,
        expected: Vec<Source>,
    },
    ForwardShard {
        targets: Vec<ShardForwardTarget>,
        trace_id: u64,
        #[serde(default = "default_forward_hops")]
        forward_hops: u8,
        inputs: Vec<ShardInput>,
        training: bool,
    },
    Signal {
        target: u64,
        event_id: u64,
        trace_id: u64,
        edge_id: u64,
        from: u64,
        source_event_id: u64,
        #[serde(deserialize_with = "deserialize_finite_f32")]
        value: f32,
        training: bool,
        forward_hops: u8,
        route_hops: u8,
    },
    Backward {
        target: u64,
        event_id: u64,
        from: Source,
        #[serde(deserialize_with = "deserialize_finite_f32")]
        gradient: f32,
        ttl_ms: u64,
        gradient_hops: u8,
        route_hops: u8,
    },
    BackwardShard {
        targets: Vec<ShardGradient>,
        input_ids: Vec<u64>,
        ttl_ms: u64,
        gradient_hops: u8,
    },
}

#[derive(Clone)]
pub struct Peer {
    pub id: u64,
    pub address: SocketAddr,
}

pub struct NodeConfig {
    pub id: u64,
    pub address: SocketAddr,
    pub neurons: Vec<Neuron>,
    pub worker_threads: usize,
    pub mailbox_capacity: usize,
    /// Explicitly permitted bootstrap peers. Only loopback is accepted by v1.
    pub peers: Vec<Peer>,
}

#[derive(Clone, Copy)]
struct Route {
    owner: u64,
    epoch: u64,
}

#[derive(Clone, Copy)]
struct ForwardEmission {
    source: u64,
    source_event_id: u64,
    trace_id: u64,
    output: f32,
    training: bool,
    forward_hops: u8,
}

struct NodeState {
    id: u64,
    shard: Arc<Shard>,
    local: LocalDataPlane,
    peers: BTreeMap<u64, SocketAddr>,
    routes: RwLock<BTreeMap<u64, Route>>,
    tensors: RwLock<tensor::TensorRuntime>,
}

impl NodeState {
    async fn target_address(&self, target: u64) -> Option<SocketAddr> {
        let route = *self.routes.read().await.get(&target)?;
        if route.owner == self.id {
            return None;
        }
        self.peers.get(&route.owner).copied()
    }

    async fn route_or_error(&self, target: u64, hops: u8) -> Result<SocketAddr, Value> {
        if hops < 2 {
            return Err(error_response("route_hops_exhausted"));
        }
        self.target_address(target)
            .await
            .ok_or_else(|| error_response("route_unknown"))
    }

    async fn cascade_forward(
        &self,
        emission: ForwardEmission,
        axons: &[danma_core::Axon],
    ) -> (Vec<Value>, Vec<Value>) {
        let local = self
            .local
            .cascade_forward(
                FiredActivation {
                    source: emission.source,
                    source_event_id: u128::from(emission.source_event_id),
                    trace_id: u128::from(emission.trace_id),
                    output: emission.output,
                    training: emission.training,
                    forward_hops: emission.forward_hops,
                },
                axons,
            )
            .await;

        let mut terminals = Vec::new();
        let mut unrouted = Vec::new();
        for terminal in local.terminals {
            match u64::try_from(terminal.event_id) {
                Ok(event_id) => terminals.push(json!({
                    "neuron":terminal.neuron_id,
                    "event_id":event_id,
                    "output":terminal.output
                })),
                Err(_) => unrouted.push(json!({
                    "target":terminal.neuron_id,
                    "reason":"terminal_event_id_out_of_range"
                })),
            }
        }
        unrouted.extend(local.failures.into_iter().map(|failure| {
            let reason = match failure.kind {
                ForwardFailureKind::HopLimitExhausted => "forward_hops_exhausted",
                ForwardFailureKind::Expired => "expired",
                ForwardFailureKind::DeliveryBudgetExceeded => "local_delivery_budget_exceeded",
                ForwardFailureKind::RemoteEgressBudgetExceeded => "remote_egress_budget_exceeded",
                ForwardFailureKind::Shard(_) => "local_shard_error",
            };
            json!({
                "target":failure.target,
                "edge_id":failure.edge_id,
                "reason":reason,
                "detail":format!("{:?}", failure.kind)
            })
        }));

        for remote in local.remote {
            let event_id = match u64::try_from(remote.event_id) {
                Ok(value) => value,
                Err(_) => {
                    unrouted.push(json!({
                        "target":remote.target,
                        "edge_id":remote.edge_id,
                        "reason":"event_id_out_of_range"
                    }));
                    continue;
                }
            };
            let trace_id = match u64::try_from(remote.trace_id) {
                Ok(value) => value,
                Err(_) => {
                    unrouted.push(json!({
                        "target":remote.target,
                        "edge_id":remote.edge_id,
                        "reason":"trace_id_out_of_range"
                    }));
                    continue;
                }
            };
            let source_event_id = match u64::try_from(remote.source_event_id) {
                Ok(value) => value,
                Err(_) => {
                    unrouted.push(json!({
                        "target":remote.target,
                        "edge_id":remote.edge_id,
                        "reason":"source_event_id_out_of_range"
                    }));
                    continue;
                }
            };
            let address = match self.target_address(remote.target).await {
                Some(address) => address,
                None => {
                    unrouted.push(json!({
                        "target":remote.target,
                        "edge_id":remote.edge_id,
                        "reason":"no_route"
                    }));
                    continue;
                }
            };
            let next = Message::Signal {
                target: remote.target,
                event_id,
                trace_id,
                edge_id: remote.edge_id,
                from: remote.from,
                source_event_id,
                value: remote.value,
                training: remote.training,
                forward_hops: remote.forward_hops,
                route_hops: DEFAULT_ROUTE_HOPS,
            };
            let reply = self.relay(address, &next).await;
            if reply["kind"] == "signal_result" {
                if let Some(extra) = reply["terminals"].as_array() {
                    terminals.extend(extra.iter().cloned());
                }
                if let Some(extra) = reply["unrouted"].as_array() {
                    unrouted.extend(extra.iter().cloned());
                }
            } else {
                unrouted.push(json!({
                    "target":remote.target,
                    "edge_id":remote.edge_id,
                    "reason":"delivery_not_confirmed",
                    "response":reply
                }));
            }
        }

        (terminals, unrouted)
    }

    async fn process(&self, msg: Message) -> Value {
        match msg {
            Message::Routes {} => {
                let routes: BTreeMap<_, _> = self
                    .routes
                    .read()
                    .await
                    .iter()
                    .map(|(neuron, route)| (neuron.to_string(), route.owner))
                    .collect();
                json!({"kind":"routes_result","routes":routes})
            }
            Message::Gossip { from_node, routes } => {
                if !self.peers.contains_key(&from_node) || routes.len() > MAX_ADVERTISED_ROUTES {
                    return error_response("untrusted_or_oversized_gossip");
                }
                let mut table = self.routes.write().await;
                // Stage the full advertisement batch: bad routes must not
                // partially mutate the control-plane routing table.
                let mut staged = table.clone();
                for adv in routes {
                    if adv.owner == 0
                        || adv.neuron == 0
                        || adv.epoch == 0
                        || !(self.peers.contains_key(&adv.owner)
                            || (adv.owner == self.id && self.shard.contains(adv.neuron)))
                    {
                        return error_response("invalid_route_owner");
                    }
                    if let Some(current) = staged.get(&adv.neuron) {
                        if current.owner != adv.owner {
                            return error_response("route_owner_conflict");
                        }
                        if current.epoch >= adv.epoch {
                            continue;
                        }
                    }
                    if staged.len() >= MAX_ADVERTISED_ROUTES && !staged.contains_key(&adv.neuron) {
                        return error_response("route_table_full");
                    }
                    staged.insert(
                        adv.neuron,
                        Route {
                            owner: adv.owner,
                            epoch: adv.epoch,
                        },
                    );
                }
                *table = staged;
                json!({"kind":"gossip_accepted"})
            }
            Message::Inspect { target, route_hops } => {
                if !self.shard.contains(target) {
                    let address = match self.route_or_error(target, route_hops).await {
                        Ok(address) => address,
                        Err(response) => return response,
                    };
                    return self
                        .relay(
                            address,
                            &Message::Inspect {
                                target,
                                route_hops: route_hops - 1,
                            },
                        )
                        .await;
                }
                match self.shard.inspect(target).await {
                    Ok(neuron) => json!({
                        "kind":"inspect_result",
                        "target":target,
                        "version":neuron.version,
                        "bias":neuron.bias,
                        "weights":neuron.weights,
                        "axons":neuron.axons.iter().map(|axon| json!({
                            "edge_id":axon.edge_id,
                            "to":axon.to
                        })).collect::<Vec<_>>()
                    }),
                    Err(err) => error_response(&format!("inspect_{err:?}")),
                }
            }
            Message::Trace {
                target,
                event_id,
                route_hops,
            } => {
                if !self.shard.contains(target) {
                    let address = match self.route_or_error(target, route_hops).await {
                        Ok(address) => address,
                        Err(response) => return response,
                    };
                    return self
                        .relay(
                            address,
                            &Message::Trace {
                                target,
                                event_id,
                                route_hops: route_hops - 1,
                            },
                        )
                        .await;
                }
                let trace = match self.shard.trace(target, u128::from(event_id)).await {
                    Ok(trace) => trace,
                    Err(err) => return error_response(&format!("trace_{err:?}")),
                };
                let trace = trace.map(|trace| {
                    json!({
                        "trace_id":trace.trace_id,
                        "output":trace.output,
                        "parameter_version":trace.parameter_version,
                        "expires_at_ms":trace.expires_at_ms,
                        "expected_contributions":trace.expected_contributions,
                        "received_contributions":trace.received_contributions
                    })
                });
                json!({"kind":"trace_result","target":target,"event_id":event_id,"trace":trace})
            }
            Message::Forward {
                target,
                event_id,
                trace_id,
                route_hops,
                forward_hops,
                inputs,
                expected,
            } => {
                if inputs.len() > MAX_INCOMING_INPUTS || expected.len() > MAX_EXPECTED_BRANCHES {
                    return error_response("activation_too_large");
                }
                if !self.shard.contains(target) {
                    let address = match self.route_or_error(target, route_hops).await {
                        Ok(address) => address,
                        Err(response) => return response,
                    };
                    return self
                        .relay(
                            address,
                            &Message::Forward {
                                target,
                                event_id,
                                trace_id,
                                route_hops: route_hops - 1,
                                forward_hops,
                                inputs,
                                expected,
                            },
                        )
                        .await;
                }
                let training = !expected.is_empty();
                let forward = Forward {
                    event_id: u128::from(event_id),
                    trace_id: u128::from(trace_id),
                    now_ms: elapsed_millis(),
                    inputs: inputs
                        .into_iter()
                        .map(|input| SynapticInput {
                            from: input.from,
                            source_event_id: u128::from(input.source_event_id),
                            value: input.value,
                        })
                        .collect(),
                    expected: expected.into_iter().map(FeedbackSource::from).collect(),
                };
                match self.shard.forward_outcome(target, forward).await {
                    Ok(outcome) => {
                        let (terminals, unrouted) = self
                            .cascade_forward(
                                ForwardEmission {
                                    source: target,
                                    source_event_id: event_id,
                                    trace_id,
                                    output: outcome.output,
                                    training,
                                    forward_hops,
                                },
                                &outcome.axons,
                            )
                            .await;
                        json!({
                            "kind":"forward_result",
                            "output":outcome.output,
                            "terminals":terminals,
                            "unrouted":unrouted
                        })
                    }
                    Err(err) => error_response(&format!("forward_{err:?}")),
                }
            }
            Message::ForwardShard {
                targets,
                trace_id,
                forward_hops,
                inputs,
                training,
            } => {
                let process_started = Instant::now();
                if targets.is_empty()
                    || targets.len() > MAX_SHARD_TARGETS
                    || inputs.is_empty()
                    || inputs.len() > MAX_INCOMING_INPUTS
                    || trace_id == 0
                {
                    return error_response("invalid_shard_forward");
                }

                let mut seen_targets = BTreeMap::new();
                let mut seen_events = BTreeMap::new();
                for item in &targets {
                    if item.target == 0
                        || item.event_id == 0
                        || !self.shard.contains(item.target)
                        || seen_targets.insert(item.target, ()).is_some()
                        || seen_events.insert(item.event_id, ()).is_some()
                    {
                        return error_response("invalid_or_nonlocal_shard_target");
                    }
                }
                let mut seen_inputs = BTreeMap::new();
                for input in &inputs {
                    if input.from == 0 || seen_inputs.insert(input.from, ()).is_some() {
                        return error_response("invalid_shard_input");
                    }
                }

                let validation_us = process_started.elapsed().as_micros() as u64;
                let build_started = Instant::now();
                let items = targets
                    .iter()
                    .map(|item| {
                        (
                            item.target,
                            Forward {
                                event_id: u128::from(item.event_id),
                                trace_id: u128::from(trace_id),
                                now_ms: elapsed_millis(),
                                inputs: inputs
                                    .iter()
                                    .map(|input| SynapticInput {
                                        from: input.from,
                                        source_event_id: u128::from(item.event_id),
                                        value: input.value,
                                    })
                                    .collect(),
                                expected: if training {
                                    vec![FeedbackSource::Teacher]
                                } else {
                                    Vec::new()
                                },
                            },
                        )
                    })
                    .collect();
                let build_items_us = build_started.elapsed().as_micros() as u64;

                let shard_started = Instant::now();
                let outcomes = match self.shard.forward_batch(items).await {
                    Ok(outcomes) => outcomes,
                    Err(err) => return error_response(&format!("forward_shard_{err:?}")),
                };
                let shard_batch_us = shard_started.elapsed().as_micros() as u64;
                let result_started = Instant::now();
                let mut by_target = BTreeMap::new();
                for (target, result) in outcomes {
                    by_target.insert(target, result);
                }

                let mut all_ok = true;
                let mut results = Vec::with_capacity(targets.len());
                for item in targets {
                    match by_target.remove(&item.target) {
                        Some(Ok(outcome)) => {
                            let (terminals, unrouted) = self
                                .cascade_forward(
                                    ForwardEmission {
                                        source: item.target,
                                        source_event_id: item.event_id,
                                        trace_id,
                                        output: outcome.output,
                                        training,
                                        forward_hops,
                                    },
                                    &outcome.axons,
                                )
                                .await;
                            results.push(json!({
                                "target":item.target,
                                "event_id":item.event_id,
                                "status":"ok",
                                "output":outcome.output,
                                "terminals":terminals,
                                "unrouted":unrouted
                            }));
                        }
                        Some(Err(err)) => {
                            all_ok = false;
                            results.push(json!({
                                "target":item.target,
                                "event_id":item.event_id,
                                "status":"error",
                                "code":format!("{err:?}")
                            }));
                        }
                        None => {
                            all_ok = false;
                            results.push(json!({
                                "target":item.target,
                                "event_id":item.event_id,
                                "status":"error",
                                "code":"missing_worker_result"
                            }));
                        }
                    }
                }
                let result_build_us = result_started.elapsed().as_micros() as u64;
                json!({
                    "kind":"forward_shard_result",
                    "status":if all_ok {"ok"} else {"partial"},
                    "results":results,
                    "timing_us":{
                        "validation":validation_us,
                        "build_items":build_items_us,
                        "shard_batch":shard_batch_us,
                        "result_build":result_build_us,
                        "process_total":process_started.elapsed().as_micros() as u64
                    }
                })
            }
            Message::Signal {
                target,
                event_id,
                trace_id,
                edge_id,
                from,
                source_event_id,
                value,
                training,
                forward_hops,
                route_hops,
            } => {
                if edge_id == 0
                    || u128::from(event_id)
                        != derived_event_id(u128::from(trace_id), target)
                {
                    return error_response("invalid_forward_signal");
                }
                if !self.shard.contains(target) {
                    let address = match self.route_or_error(target, route_hops).await {
                        Ok(address) => address,
                        Err(response) => return response,
                    };
                    return self
                        .relay(
                            address,
                            &Message::Signal {
                                target,
                                event_id,
                                trace_id,
                                edge_id,
                                from,
                                source_event_id,
                                value,
                                training,
                                forward_hops,
                                route_hops: route_hops - 1,
                            },
                        )
                        .await;
                }
                let input = ForwardSignal {
                    event_id: u128::from(event_id),
                    trace_id: u128::from(trace_id),
                    now_ms: elapsed_millis(),
                    input: SynapticInput {
                        from,
                        source_event_id: u128::from(source_event_id),
                        value,
                    },
                    training,
                };
                match self.shard.signal(target, input).await {
                    Err(err) => error_response(&format!("signal_{err:?}")),
                    Ok(outcome) => match outcome.status {
                        SignalStatus::Pending { remaining } => json!({
                            "kind":"signal_result",
                            "status":"pending",
                            "remaining":remaining,
                            "terminals":[],
                            "unrouted":[]
                        }),
                        SignalStatus::IgnoredDuplicate => json!({
                            "kind":"signal_result",
                            "status":"ignored_duplicate",
                            "terminals":[],
                            "unrouted":[]
                        }),
                        SignalStatus::Expired => json!({
                            "kind":"signal_result",
                            "status":"expired",
                            "terminals":[],
                            "unrouted":[]
                        }),
                        SignalStatus::Fired { output } => {
                            let (terminals, unrouted) = self
                                .cascade_forward(
                                    ForwardEmission {
                                        source: target,
                                        source_event_id: event_id,
                                        trace_id,
                                        output,
                                        training,
                                        forward_hops,
                                    },
                                    &outcome.axons,
                                )
                                .await;
                            json!({
                                "kind":"signal_result",
                                "status":"fired",
                                "output":output,
                                "terminals":terminals,
                                "unrouted":unrouted
                            })
                        }
                    },
                }
            }
            Message::Backward {
                target,
                event_id,
                from,
                gradient,
                ttl_ms,
                gradient_hops,
                route_hops,
            } => {
                if ttl_ms == 0 {
                    return json!({"kind":"backward_result","status":"expired"});
                }
                let deadline = match Instant::now().checked_add(Duration::from_millis(ttl_ms)) {
                    Some(deadline) => deadline,
                    None => return error_response("invalid_ttl"),
                };
                if !self.shard.contains(target) {
                    let address = match self.route_or_error(target, route_hops).await {
                        Ok(address) => address,
                        Err(response) => return response,
                    };
                    let remaining = remaining_ms(deadline);
                    if remaining == 0 {
                        return json!({"kind":"backward_result","status":"expired"});
                    }
                    return self
                        .relay(
                            address,
                            &Message::Backward {
                                target,
                                event_id,
                                from,
                                gradient,
                                ttl_ms: remaining,
                                gradient_hops,
                                route_hops: route_hops - 1,
                            },
                        )
                        .await;
                }
                let now = elapsed_millis();
                let packet = Feedback {
                    event_id: u128::from(event_id),
                    from: from.into(),
                    gradient,
                    expires_at_ms: now.saturating_add(ttl_ms),
                    hops_left: gradient_hops,
                };
                let result = self.shard.backward_live(target, packet, deadline).await;
                match result {
                    Err(err) => error_response(&format!("backward_{err:?}")),
                    Ok(FeedbackStatus::Pending { remaining }) => {
                        json!({"kind":"backward_result","status":"pending","remaining":remaining})
                    }
                    Ok(FeedbackStatus::IgnoredDuplicate) => {
                        json!({"kind":"backward_result","status":"ignored_duplicate"})
                    }
                    Ok(FeedbackStatus::Expired) => {
                        json!({"kind":"backward_result","status":"expired"})
                    }
                    Ok(FeedbackStatus::Stale) => {
                        json!({"kind":"backward_result","status":"stale"})
                    }
                    Ok(FeedbackStatus::Applied { upstream, version }) => {
                        let local = self.local.cascade_backward(upstream, deadline).await;
                        let mut unrouted = local
                            .failures
                            .into_iter()
                            .map(|failure| {
                                let reason = match failure.kind {
                                    BackwardFailureKind::Expired => "expired",
                                    BackwardFailureKind::Stale => "stale",
                                    BackwardFailureKind::DeliveryBudgetExceeded => {
                                        "local_delivery_budget_exceeded"
                                    }
                                    BackwardFailureKind::RemoteEgressBudgetExceeded => {
                                        "remote_egress_budget_exceeded"
                                    }
                                    BackwardFailureKind::Shard(_) => "local_shard_error",
                                };
                                json!({
                                    "target":failure.target,
                                    "event_id":failure.event_id.to_string(),
                                    "reason":reason,
                                    "detail":format!("{:?}", failure.kind)
                                })
                            })
                            .collect::<Vec<_>>();

                        for remote in local.remote {
                            let dest = remote.target;
                            let core_packet = remote.feedback;
                            let remaining = remaining_ms(deadline).min(
                                core_packet.expires_at_ms.saturating_sub(elapsed_millis()),
                            );
                            if remaining == 0 {
                                unrouted.push(json!({"target":dest,"reason":"expired"}));
                                continue;
                            }
                            let address = match self.target_address(dest).await {
                                Some(address) => address,
                                None => {
                                    unrouted.push(json!({
                                        "target":dest,
                                        "reason":"no_route",
                                        "event_id":core_packet.event_id.to_string(),
                                        "gradient":core_packet.gradient
                                    }));
                                    continue;
                                }
                            };
                            let next_source = match Source::try_from(core_packet.from) {
                                Ok(source) => source,
                                Err(_) => {
                                    unrouted.push(json!({
                                        "target":dest,
                                        "reason":"event_id_out_of_range"
                                    }));
                                    continue;
                                }
                            };
                            let parent_event = match u64::try_from(core_packet.event_id) {
                                Ok(id) => id,
                                Err(_) => {
                                    unrouted.push(json!({
                                        "target":dest,
                                        "reason":"event_id_out_of_range"
                                    }));
                                    continue;
                                }
                            };
                            let next = Message::Backward {
                                target: dest,
                                event_id: parent_event,
                                from: next_source,
                                gradient: core_packet.gradient,
                                ttl_ms: remaining,
                                gradient_hops: core_packet.hops_left,
                                route_hops: DEFAULT_ROUTE_HOPS,
                            };
                            let reply = self.relay(address, &next).await;
                            if reply["kind"] == "backward_result"
                                && (reply["status"] == "applied"
                                    || reply["status"] == "ignored_duplicate"
                                    || reply["status"] == "pending")
                            {
                                if let Some(extra) = reply["unrouted"].as_array() {
                                    unrouted.extend(extra.iter().cloned());
                                }
                            } else {
                                unrouted.push(json!({
                                    "target":dest,
                                    "reason":"delivery_not_confirmed",
                                    "response":reply
                                }));
                            }
                        }
                        json!({
                            "kind":"backward_result",
                            "status":"applied",
                            "version":version,
                            "unrouted":unrouted
                        })
                    }
                }
            }
            Message::BackwardShard {
                targets,
                input_ids,
                ttl_ms,
                gradient_hops,
            } => {
                let process_started = Instant::now();
                if targets.is_empty()
                    || targets.len() > MAX_SHARD_TARGETS
                    || input_ids.is_empty()
                    || input_ids.len() > MAX_INCOMING_INPUTS
                    || ttl_ms == 0
                {
                    return error_response("invalid_shard_backward");
                }

                let mut seen_targets = BTreeMap::new();
                let mut seen_events = BTreeMap::new();
                for item in &targets {
                    if item.target == 0
                        || item.event_id == 0
                        || !self.shard.contains(item.target)
                        || seen_targets.insert(item.target, ()).is_some()
                        || seen_events.insert(item.event_id, ()).is_some()
                    {
                        return error_response("invalid_or_nonlocal_shard_target");
                    }
                }

                let mut input_positions = BTreeMap::new();
                for (index, input_id) in input_ids.iter().copied().enumerate() {
                    if input_id == 0
                        || input_positions.insert(input_id, index).is_some()
                        || self.shard.contains(input_id)
                        || self.target_address(input_id).await.is_some()
                    {
                        return error_response("shard_backward_requires_unrouted_host_inputs");
                    }
                }

                let validation_us = process_started.elapsed().as_micros() as u64;
                let build_started = Instant::now();
                let deadline = match Instant::now().checked_add(Duration::from_millis(ttl_ms)) {
                    Some(deadline) => deadline,
                    None => return error_response("invalid_ttl"),
                };
                let packets = targets
                    .iter()
                    .map(|item| {
                        let now = elapsed_millis();
                        (
                            item.target,
                            Feedback {
                                event_id: u128::from(item.event_id),
                                from: FeedbackSource::Teacher,
                                gradient: item.gradient,
                                expires_at_ms: now.saturating_add(ttl_ms),
                                hops_left: gradient_hops,
                            },
                        )
                    })
                    .collect();
                let build_packets_us = build_started.elapsed().as_micros() as u64;

                let shard_started = Instant::now();
                let outcomes = match self.shard.backward_batch_live(packets, deadline).await {
                    Ok(outcomes) => outcomes,
                    Err(err) => return error_response(&format!("backward_shard_{err:?}")),
                };
                let shard_batch_us = shard_started.elapsed().as_micros() as u64;
                let aggregate_started = Instant::now();
                let mut by_target = BTreeMap::new();
                for (target, result) in outcomes {
                    by_target.insert(target, result);
                }

                let mut all_applied = true;
                let mut accumulated = vec![0.0_f32; input_ids.len()];
                let mut results = Vec::with_capacity(targets.len());
                for item in targets {
                    match by_target.remove(&item.target) {
                        Some(Ok(FeedbackStatus::Applied { upstream, version })) => {
                            let mut valid_upstream = upstream.len() == input_ids.len();
                            let mut seen_inputs = BTreeMap::new();
                            for dispatch in upstream {
                                if let Some(index) = input_positions.get(&dispatch.target_neuron_id) {
                                    if seen_inputs
                                        .insert(dispatch.target_neuron_id, ())
                                        .is_some()
                                    {
                                        valid_upstream = false;
                                    } else {
                                        accumulated[*index] += dispatch.feedback.gradient;
                                    }
                                } else {
                                    valid_upstream = false;
                                }
                            }
                            valid_upstream =
                                valid_upstream && seen_inputs.len() == input_ids.len();
                            if valid_upstream {
                                results.push(json!({
                                    "target":item.target,
                                    "event_id":item.event_id,
                                    "status":"applied",
                                    "version":version
                                }));
                            } else {
                                all_applied = false;
                                results.push(json!({
                                    "target":item.target,
                                    "event_id":item.event_id,
                                    "status":"error",
                                    "code":"unexpected_upstream_gradient"
                                }));
                            }
                        }
                        Some(Ok(FeedbackStatus::Pending { remaining })) => {
                            all_applied = false;
                            results.push(json!({"target":item.target,"event_id":item.event_id,"status":"pending","remaining":remaining}));
                        }
                        Some(Ok(FeedbackStatus::IgnoredDuplicate)) => {
                            all_applied = false;
                            results.push(json!({"target":item.target,"event_id":item.event_id,"status":"ignored_duplicate"}));
                        }
                        Some(Ok(FeedbackStatus::Expired)) => {
                            all_applied = false;
                            results.push(json!({"target":item.target,"event_id":item.event_id,"status":"expired"}));
                        }
                        Some(Ok(FeedbackStatus::Stale)) => {
                            all_applied = false;
                            results.push(json!({"target":item.target,"event_id":item.event_id,"status":"stale"}));
                        }
                        Some(Err(err)) => {
                            all_applied = false;
                            results.push(json!({
                                "target":item.target,
                                "event_id":item.event_id,
                                "status":"error",
                                "code":format!("{err:?}")
                            }));
                        }
                        None => {
                            all_applied = false;
                            results.push(json!({
                                "target":item.target,
                                "event_id":item.event_id,
                                "status":"error",
                                "code":"missing_worker_result"
                            }));
                        }
                    }
                }
                let aggregate_us = aggregate_started.elapsed().as_micros() as u64;
                json!({
                    "kind":"backward_shard_result",
                    "status":if all_applied {"applied"} else {"partial"},
                    "results":results,
                    "input_gradients":accumulated,
                    "timing_us":{
                        "validation":validation_us,
                        "build_packets":build_packets_us,
                        "shard_batch":shard_batch_us,
                        "aggregate":aggregate_us,
                        "process_total":process_started.elapsed().as_micros() as u64
                    }
                })
            }
        }
    }

    async fn relay(&self, address: SocketAddr, msg: &Message) -> Value {
        let request_value = match serde_json::to_value(msg) {
            Ok(value) => value,
            Err(_) => return error_response("message_encoding_error"),
        };
        match request(address, &request_value).await {
            Ok(reply) => reply,
            Err(_) => error_response("peer_delivery_uncertain"),
        }
    }
}

fn remaining_ms(deadline: Instant) -> u64 {
    let remaining = deadline.saturating_duration_since(Instant::now()).as_millis();
    u64::try_from(remaining).unwrap_or(u64::MAX)
}

fn error_response(code: &str) -> Value {
    json!({"kind":"error","code":code})
}

fn decode_message(incoming: &[u8]) -> Result<Message, serde_json::Error> {
    let value = serde_json::from_slice::<Value>(incoming)?;
    serde_json::from_value(value)
}

async fn handle_connection(mut stream: TcpStream, state: Arc<NodeState>) -> io::Result<()> {
    let incoming = timeout(MAX_IO_WAIT, read_frame_bytes(&mut stream))
        .await
        .map_err(|_| io::Error::new(io::ErrorKind::TimedOut, "frame timed out"))??;

    if tensor::TensorRuntime::is_frame(&incoming) {
        let reply = state.tensors.write().await.process(&incoming);
        return timeout(MAX_IO_WAIT, write_frame_bytes(&mut stream, &reply))
            .await
            .map_err(|_| io::Error::new(io::ErrorKind::TimedOut, "response timed out"))?;
    }

    let reply = match decode_message(&incoming) {
        Ok(message) => state.process(message).await,
        Err(_) => error_response("invalid_protocol_message"),
    };
    timeout(MAX_IO_WAIT, write_frame(&mut stream, &reply))
        .await
        .map_err(|_| io::Error::new(io::ErrorKind::TimedOut, "response timed out"))?
}

async fn gossip_loop(state: Arc<NodeState>) {
    if state.peers.is_empty() {
        return;
    }
    let peers: Vec<_> = state.peers.values().copied().collect();
    let mut index = 0_usize;
    let mut tick = tokio::time::interval(Duration::from_millis(100));
    loop {
        tick.tick().await;
        let routes: Vec<_> = state
            .routes
            .read()
            .await
            .iter()
            .take(MAX_ADVERTISED_ROUTES)
            .map(|(neuron, route)| Advert {
                owner: route.owner,
                neuron: *neuron,
                epoch: route.epoch,
            })
            .collect();
        let frame = json!({
            "kind":"gossip",
            "from_node":state.id,
            "routes":routes
        });
        let _ = request(peers[index % peers.len()], &frame).await;
        index = index.wrapping_add(1);
    }
}

/// Run one multi-neuron CPU shard behind one TCP listener. Multiple
/// processes form a trusted development cluster through bounded gossip.
pub async fn serve(config: NodeConfig) -> io::Result<()> {
    if config.id == 0
        || config.neurons.is_empty()
        || config.neurons.len() > MAX_ADVERTISED_ROUTES
        || !is_loopback(config.address.ip())
        || config.peers.len() > MAX_ADVERTISED_ROUTES
    {
        return Err(invalid("invalid node identity, address or peer count"));
    }
    let mut peers = BTreeMap::new();
    for peer in config.peers {
        if peer.id == 0
            || peer.id == config.id
            || !is_loopback(peer.address.ip())
            || peers.insert(peer.id, peer.address).is_some()
        {
            return Err(invalid("invalid or duplicate bootstrap peer"));
        }
    }
    let shard = Arc::new(
        Shard::new(
            config.neurons,
            config.worker_threads,
            config.mailbox_capacity,
        )
        .map_err(|_| invalid("invalid CPU shard configuration"))?,
    );
    let local = LocalDataPlane::new(Arc::clone(&shard), LocalLimits::default())
        .map_err(|_| invalid("invalid local data-plane configuration"))?;
    let mut routes = BTreeMap::new();
    for neuron_id in shard.neuron_ids() {
        routes.insert(
            neuron_id,
            Route {
                owner: config.id,
                epoch: 1,
            },
        );
    }
    let state = Arc::new(NodeState {
        id: config.id,
        shard,
        local,
        peers,
        routes: RwLock::new(routes),
        tensors: RwLock::new(tensor::TensorRuntime::default()),
    });
    let listener = TcpListener::bind(config.address).await?;
    let slots = Arc::new(Semaphore::new(MAX_CONCURRENT_CONNECTIONS));
    tokio::spawn(gossip_loop(Arc::clone(&state)));
    loop {
        // Acquire before accept so overload reaches the TCP backlog rather
        // than spawning unbounded tasks.
        let permit = Arc::clone(&slots)
            .acquire_owned()
            .await
            .map_err(|_| invalid("connection pool closed"))?;
        let (socket, _) = listener.accept().await?;
        let node = Arc::clone(&state);
        tokio::spawn(async move {
            let _permit = permit;
            let _ = handle_connection(socket, node).await;
        });
    }
}

fn is_loopback(ip: IpAddr) -> bool {
    ip.is_loopback()
}

#[cfg(test)]
mod protocol_tests;
