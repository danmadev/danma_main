use danma_core::{Activation, Config, Error as CoreError, Feedback, FeedbackSource, FeedbackStatus, Forward, Neuron, SynapticInput};
use danma_shard::{Shard, ShardError};

fn config() -> Config {
    Config {
        activation: Activation::Linear,
        learning_rate: 0.1,
        activation_ttl_ms: 2_000,
        replay_retention_ms: 2_000,
        max_live_events: 16,
        max_staleness_versions: 8,
    }
}

fn neuron(id: u64, incoming: u64, weight: f32) -> Neuron {
    Neuron::new(id, 0.0, [(incoming, weight)], config()).unwrap()
}

fn forward(id: u128, source: u64, source_event: u128, value: f32, expected: Vec<FeedbackSource>) -> Forward {
    Forward {
        event_id: id,
        trace_id: 77,
        now_ms: 1_000,
        inputs: vec![SynapticInput { from: source, source_event_id: source_event, value }],
        expected,
    }
}

#[tokio::test]
async fn two_workers_host_four_independently_addressable_neurons() {
    let shard = Shard::new(vec![
        neuron(1, 99, 2.0),
        neuron(2, 1, 3.0),
        neuron(3, 2, 4.0),
        neuron(4, 3, 5.0),
    ], 2, 8).unwrap();
    assert_eq!(shard.worker_count(), 2);
    assert_eq!(shard.neuron_ids(), vec![1, 2, 3, 4]);
    assert!(shard.contains(4));
    assert!(!shard.contains(99));

    let out = shard.forward(1, forward(10, 99, 1, 1.0, vec![
        FeedbackSource::Neuron { neuron_id: 2, event_id: 20 },
    ])).await.unwrap();
    assert_eq!(out, 2.0);
    assert_eq!(shard.forward(2, forward(20, 1, 10, out, vec![
        FeedbackSource::Teacher,
    ])).await.unwrap(), 6.0);
    let trace = shard.trace(1, 10).await.unwrap().unwrap();
    assert_eq!(trace.trace_id, 77);
    assert_eq!(trace.expected_contributions, 1);
    assert_eq!(shard.inspect(1).await.unwrap().weights.get(&99), Some(&2.0));
    assert_eq!(shard.inspect(4).await.unwrap().version, 0);
}

#[tokio::test]
async fn feedback_routes_between_workers_without_retraining_an_event() {
    let shard = Shard::new(vec![neuron(1, 99, 2.0), neuron(2, 1, 3.0)], 2, 4).unwrap();
    let source = FeedbackSource::Neuron { neuron_id: 2, event_id: 20 };
    shard.forward(1, forward(10, 99, 1, 1.0, vec![source])).await.unwrap();
    shard.forward(2, forward(20, 1, 10, 2.0, vec![FeedbackSource::Teacher])).await.unwrap();

    let end = shard.backward(2, Feedback {
        event_id: 20, from: FeedbackSource::Teacher, gradient: 6.0,
        expires_at_ms: 1_500, hops_left: 4,
    }, 1_001).await.unwrap();
    let FeedbackStatus::Applied { upstream, .. } = end else { panic!("downstream must learn"); };
    assert_eq!(upstream.len(), 1);
    assert_eq!(upstream[0].target_neuron_id, 1);
    assert!(matches!(
        shard.backward(1, upstream[0].feedback.clone(), 1_002).await.unwrap(),
        FeedbackStatus::Applied { .. }
    ));
    assert_eq!(shard.backward(1, upstream[0].feedback.clone(), 1_003).await.unwrap(), FeedbackStatus::IgnoredDuplicate);
    assert_eq!(shard.inspect(1).await.unwrap().version, 1);
    assert_eq!(shard.inspect(2).await.unwrap().version, 1);
    assert!(shard.trace(1, 10).await.unwrap().is_none());
}

#[tokio::test]
async fn malformed_layout_and_unknown_neuron_fail_before_side_effect() {
    assert!(matches!(Shard::new(vec![], 2, 8), Err(ShardError::InvalidConfiguration)));
    assert!(matches!(
        Shard::new(vec![neuron(1, 99, 2.0)], 0, 8),
        Err(ShardError::InvalidConfiguration)
    ));
    assert!(matches!(
        Shard::new(vec![neuron(1, 99, 2.0)], 1, 0),
        Err(ShardError::InvalidConfiguration)
    ));
    assert!(matches!(
        Shard::new(vec![neuron(1, 99, 2.0), neuron(1, 99, 3.0)], 2, 8),
        Err(ShardError::DuplicateNeuron(1))
    ));
    let shard = Shard::new(vec![neuron(1, 99, 2.0)], 2, 8).unwrap();
    assert_eq!(shard.worker_count(), 1);
    assert!(matches!(shard.inspect(999).await, Err(ShardError::UnknownNeuron(999))));
    assert!(matches!(
        shard.backward(1, Feedback {
            event_id: 999, from: FeedbackSource::Teacher, gradient: 1.0,
            expires_at_ms: 2_000, hops_left: 2,
        }, 1_001).await,
        Err(ShardError::Core(CoreError::UnknownEvent))
    ));
    assert_eq!(shard.inspect(1).await.unwrap().version, 0);
}

#[tokio::test]
async fn elapsed_transport_deadline_is_checked_by_cpu_worker_before_learning() {
    use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

    let shard = Shard::new(vec![neuron(1, 99, 2.0)], 1, 4).unwrap();
    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH).unwrap().as_millis() as u64;
    shard.forward(1, Forward {
        event_id: 10,
        trace_id: 77,
        now_ms,
        inputs: vec![SynapticInput { from: 99, source_event_id: 1, value: 1.0 }],
        expected: vec![FeedbackSource::Teacher],
    }).await.unwrap();
    let packet = Feedback {
        event_id: 10, from: FeedbackSource::Teacher, gradient: 1.0,
        expires_at_ms: now_ms + 5_000, hops_left: 2,
    };
    let expired = Instant::now().checked_sub(Duration::from_millis(1)).unwrap();
    assert_eq!(
        shard.backward_live(1, packet.clone(), expired).await.unwrap(),
        FeedbackStatus::Expired
    );
    assert_eq!(shard.inspect(1).await.unwrap().version, 0);

    let valid = Instant::now() + Duration::from_secs(2);
    assert!(matches!(
        shard.backward_live(1, packet, valid).await.unwrap(),
        FeedbackStatus::Applied { .. }
    ));
    assert_eq!(shard.inspect(1).await.unwrap().version, 1);
}


#[tokio::test]
async fn local_batch_preserves_independent_event_and_version_semantics() {
    use std::collections::BTreeMap;
    use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

    let shard = Shard::new(
        vec![neuron(1, 99, 2.0), neuron(2, 99, 3.0)],
        2,
        8,
    )
    .unwrap();
    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;

    let make_forward = |event_id| Forward {
        event_id,
        trace_id: 77,
        now_ms,
        inputs: vec![SynapticInput {
            from: 99,
            source_event_id: event_id,
            value: 4.0,
        }],
        expected: vec![FeedbackSource::Teacher],
    };
    let forwarded = shard
        .forward_batch(vec![(1, make_forward(10)), (2, make_forward(20))])
        .await
        .unwrap();
    let outputs: BTreeMap<_, _> = forwarded
        .into_iter()
        .map(|(target, result)| (target, result.unwrap().output))
        .collect();
    assert_eq!(outputs.get(&1), Some(&8.0));
    assert_eq!(outputs.get(&2), Some(&12.0));

    let make_feedback = |event_id| Feedback {
        event_id,
        from: FeedbackSource::Teacher,
        gradient: 1.0,
        expires_at_ms: now_ms + 5_000,
        hops_left: 2,
    };
    let learned = shard
        .backward_batch_live(
            vec![(1, make_feedback(10)), (2, make_feedback(20))],
            Instant::now() + Duration::from_secs(2),
        )
        .await
        .unwrap();
    assert_eq!(learned.len(), 2);
    for (_, result) in learned {
        assert!(matches!(result.unwrap(), FeedbackStatus::Applied { .. }));
    }
    assert_eq!(shard.inspect(1).await.unwrap().version, 1);
    assert_eq!(shard.inspect(2).await.unwrap().version, 1);

    // A redelivery of only one contribution is still deduplicated per neuron.
    let duplicate = shard
        .backward_batch_live(
            vec![(1, make_feedback(10))],
            Instant::now() + Duration::from_secs(2),
        )
        .await
        .unwrap();
    assert_eq!(duplicate[0].1.as_ref().unwrap(), &FeedbackStatus::IgnoredDuplicate);
    assert_eq!(shard.inspect(1).await.unwrap().version, 1);
    assert_eq!(shard.inspect(2).await.unwrap().version, 1);
}
