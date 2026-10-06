use danma_core::{
    derived_event_id, Activation, Axon, Config, Feedback, FeedbackSource, FeedbackStatus, Forward,
    Neuron, SynapticInput,
};
use danma_runtime::{
    AddressSpace, FiredActivation, LocalDataPlane, LocalLimits, NodeId, PlacementAdvert, PlacementDirectory,
    PlacementEpoch, PlacementError, ShardId, MAX_LOGICAL_NEURONS,
};
use danma_shard::Shard;
use std::{sync::Arc, time::{Duration, Instant, SystemTime, UNIX_EPOCH}};

fn config() -> Config {
    Config {
        activation: Activation::Linear,
        learning_rate: 0.1,
        activation_ttl_ms: 10_000,
        replay_retention_ms: 10_000,
        max_live_events: 64,
        max_staleness_versions: 8,
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

fn neuron(id: u64, incoming: u64, weight: f32, axons: Vec<Axon>) -> Neuron {
    Neuron::new_with_axons(id, 0.0, [(incoming, weight)], axons, config()).unwrap()
}

fn plane(neurons: Vec<Neuron>) -> LocalDataPlane {
    let shard = Arc::new(Shard::new(neurons, 2, 32).unwrap());
    LocalDataPlane::new(
        shard,
        LocalLimits {
            max_deliveries: 64,
            max_remote_egress: 16,
        },
    )
    .unwrap()
}

#[test]
fn hundred_billion_address_space_needs_only_one_hundred_thousand_logical_shards() {
    let space = AddressSpace::hundred_billion_default();
    assert_eq!(space.max_neurons(), MAX_LOGICAL_NEURONS);
    assert_eq!(space.shard_count(), 100_000);
    assert_eq!(space.locate(1).unwrap().shard_id, ShardId(1));
    assert_eq!(space.locate(1).unwrap().local_offset, 0);
    assert_eq!(space.locate(1_000_000).unwrap().shard_id, ShardId(1));
    assert_eq!(
        space.locate(1_000_001).unwrap(),
        danma_runtime::LogicalAddress {
            shard_id: ShardId(2),
            local_offset: 0,
        }
    );
    assert_eq!(
        space.locate(MAX_LOGICAL_NEURONS).unwrap(),
        danma_runtime::LogicalAddress {
            shard_id: ShardId(100_000),
            local_offset: 999_999,
        }
    );
}

#[test]
fn placement_gossip_is_shard_level_epoch_fenced_and_atomic() {
    let mut directory = PlacementDirectory::new(8, 4).unwrap();
    let first = directory
        .merge_gossip(&[
            PlacementAdvert {
                shard_id: ShardId(1),
                node_id: NodeId(10),
                epoch: PlacementEpoch(1),
            },
            PlacementAdvert {
                shard_id: ShardId(2),
                node_id: NodeId(20),
                epoch: PlacementEpoch(3),
            },
        ])
        .unwrap();
    assert_eq!(first.inserted, 2);

    let mixed = directory
        .merge_gossip(&[
            PlacementAdvert {
                shard_id: ShardId(1),
                node_id: NodeId(99),
                epoch: PlacementEpoch(0),
            },
        ]);
    assert_eq!(mixed, Err(PlacementError::InvalidConfiguration));
    assert_eq!(directory.get(ShardId(1)).unwrap().node_id, NodeId(10));

    let newer = directory
        .merge_gossip(&[
            PlacementAdvert {
                shard_id: ShardId(1),
                node_id: NodeId(11),
                epoch: PlacementEpoch(2),
            },
            PlacementAdvert {
                shard_id: ShardId(2),
                node_id: NodeId(20),
                epoch: PlacementEpoch(2),
            },
        ])
        .unwrap();
    assert_eq!(newer.updated, 1);
    assert_eq!(newer.stale_ignored, 1);
    assert_eq!(directory.get(ShardId(1)).unwrap().node_id, NodeId(11));

    let conflict = directory.merge_gossip(&[
        PlacementAdvert {
            shard_id: ShardId(1),
            node_id: NodeId(12),
            epoch: PlacementEpoch(2),
        },
    ]);
    assert_eq!(conflict, Err(PlacementError::EpochConflict));
    assert_eq!(directory.get(ShardId(1)).unwrap().node_id, NodeId(11));
}

#[tokio::test]
async fn forward_chain_stays_in_process_until_a_real_boundary() {
    let runtime = plane(vec![
        neuron(1, 99, 2.0, vec![Axon { edge_id: 12, to: 2 }]),
        neuron(2, 1, 3.0, vec![Axon { edge_id: 23, to: 3 }]),
        neuron(3, 2, 4.0, vec![]),
    ]);
    let start = runtime
        .shard()
        .forward_outcome(
            1,
            Forward {
                event_id: 10,
                trace_id: 77,
                now_ms: now_ms(),
                inputs: vec![SynapticInput {
                    from: 99,
                    source_event_id: 1,
                    value: 1.0,
                }],
                expected: vec![],
            },
        )
        .await
        .unwrap();

    let cascade = runtime
        .cascade_forward(
            FiredActivation {
                source: 1,
                source_event_id: 10,
                trace_id: 77,
                output: start.output,
                training: false,
                forward_hops: 8,
            },
            &start.axons,
        )
        .await;
    assert_eq!(start.output, 2.0);
    assert_eq!(cascade.local_deliveries, 2);
    assert!(cascade.remote.is_empty());
    assert!(cascade.failures.is_empty());
    assert_eq!(cascade.terminals.len(), 1);
    assert_eq!(cascade.terminals[0].neuron_id, 3);
    assert_eq!(cascade.terminals[0].output, 24.0);
}

#[tokio::test]
async fn only_cross_process_forward_work_becomes_typed_egress() {
    let runtime = plane(vec![
        Neuron::new_with_axons(
            1,
            0.0,
            [(99, 2.0)],
            [
                Axon { edge_id: 12, to: 2 },
                Axon {
                    edge_id: 19,
                    to: 9_000_000_001,
                },
            ],
            config(),
        )
        .unwrap(),
        neuron(2, 1, 3.0, vec![]),
    ]);
    let start = runtime
        .shard()
        .forward_outcome(
            1,
            Forward {
                event_id: 10,
                trace_id: 77,
                now_ms: now_ms(),
                inputs: vec![SynapticInput {
                    from: 99,
                    source_event_id: 1,
                    value: 1.0,
                }],
                expected: vec![],
            },
        )
        .await
        .unwrap();

    let cascade = runtime
        .cascade_forward(
            FiredActivation {
                source: 1,
                source_event_id: 10,
                trace_id: 77,
                output: start.output,
                training: false,
                forward_hops: 8,
            },
            &start.axons,
        )
        .await;
    assert_eq!(cascade.local_deliveries, 1);
    assert_eq!(cascade.terminals.len(), 1);
    assert_eq!(cascade.terminals[0].neuron_id, 2);
    assert_eq!(cascade.remote.len(), 1);
    assert_eq!(cascade.remote[0].target, 9_000_000_001);
    assert!(cascade.failures.is_empty());
}

#[tokio::test]
async fn backward_feedback_walks_local_chain_without_network_and_updates_once() {
    let runtime = plane(vec![
        neuron(1, 99, 2.0, vec![Axon { edge_id: 12, to: 2 }]),
        neuron(2, 1, 3.0, vec![Axon { edge_id: 23, to: 3 }]),
        neuron(3, 2, 4.0, vec![]),
    ]);
    let trace_id = 77_u128;
    let event_2 = derived_event_id(trace_id, 2);
    let event_3 = derived_event_id(trace_id, 3);
    let start = runtime
        .shard()
        .forward_outcome(
            1,
            Forward {
                event_id: 10,
                trace_id,
                now_ms: now_ms(),
                inputs: vec![SynapticInput {
                    from: 99,
                    source_event_id: 1,
                    value: 1.0,
                }],
                expected: vec![FeedbackSource::Neuron {
                    neuron_id: 2,
                    event_id: event_2,
                }],
            },
        )
        .await
        .unwrap();
    let forward = runtime
        .cascade_forward(
            FiredActivation {
                source: 1,
                source_event_id: 10,
                trace_id,
                output: start.output,
                training: true,
                forward_hops: 8,
            },
            &start.axons,
        )
        .await;
    assert!(forward.failures.is_empty());
    assert_eq!(forward.terminals[0].event_id, event_3);

    let deadline = Instant::now() + Duration::from_secs(2);
    let teacher = Feedback {
        event_id: event_3,
        from: FeedbackSource::Teacher,
        gradient: 1.0,
        expires_at_ms: now_ms() + 5_000,
        hops_left: 8,
    };
    let end = runtime
        .shard()
        .backward_live(3, teacher.clone(), deadline)
        .await
        .unwrap();
    let FeedbackStatus::Applied { upstream, version } = end else {
        panic!("terminal neuron must learn");
    };
    assert_eq!(version, 1);

    let backward = runtime.cascade_backward(upstream, deadline).await;
    assert_eq!(backward.local_deliveries, 2);
    assert_eq!(backward.applied.len(), 2);
    assert_eq!(backward.remote.len(), 1);
    assert_eq!(backward.remote[0].target, 99);
    assert!(backward.failures.is_empty());
    assert_eq!(runtime.shard().inspect(1).await.unwrap().version, 1);
    assert_eq!(runtime.shard().inspect(2).await.unwrap().version, 1);
    assert_eq!(runtime.shard().inspect(3).await.unwrap().version, 1);

    assert_eq!(
        runtime
            .shard()
            .backward_live(3, teacher, Instant::now() + Duration::from_secs(2))
            .await
            .unwrap(),
        FeedbackStatus::IgnoredDuplicate
    );
    assert_eq!(runtime.shard().inspect(3).await.unwrap().version, 1);
}

#[tokio::test]
async fn local_delivery_budget_bounds_cycles_and_bad_topologies() {
    let runtime = LocalDataPlane::new(
        Arc::new(
            Shard::new(
                vec![
                    neuron(1, 2, 1.0, vec![Axon { edge_id: 12, to: 2 }]),
                    neuron(2, 1, 1.0, vec![Axon { edge_id: 21, to: 1 }]),
                ],
                2,
                8,
            )
            .unwrap(),
        ),
        LocalLimits {
            max_deliveries: 1,
            max_remote_egress: 4,
        },
    )
    .unwrap();

    let cascade = runtime
        .cascade_forward(
            FiredActivation {
                source: 1,
                source_event_id: 10,
                trace_id: 77,
                output: 1.0,
                training: false,
                forward_hops: 8,
            },
            &[Axon { edge_id: 12, to: 2 }],
        )
        .await;
    assert!(cascade.truncated);
    assert_eq!(cascade.local_deliveries, 1);
    assert!(!cascade.failures.is_empty());
}
