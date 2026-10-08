use super::*;
use danma_core::{Error, Feedback, FeedbackSource, FeedbackStatus, Forward, SynapticInput};
use serde_json::{json, Value};
use std::io::Cursor;

fn document() -> Value {
    json!({
        "schema_version":1,
        "settings":{
            "learning_rate":0.25,"activation_ttl_ms":120000,
            "replay_retention_ms":10000,"max_live_events":4096,"max_staleness_versions":8
        },
        "neurons":[{"id":10001,"bias":0.5,"activation":"linear",
            "weights":[{"source":20001,"weight":0.125}]}]
    })
}

fn parse(value: &Value) -> Result<Vec<Neuron>, String> {
    read(Cursor::new(serde_json::to_vec(value).unwrap()))
}

fn forward(event_id: u128, now_ms: u64, value: f32) -> Forward {
    Forward {
        event_id,
        trace_id: 7,
        now_ms,
        inputs: vec![SynapticInput {
            from: 20001,
            source_event_id: 8,
            value,
        }],
        expected: vec![FeedbackSource::Teacher],
    }
}

fn feedback(event_id: u128) -> Feedback {
    Feedback {
        event_id,
        from: FeedbackSource::Teacher,
        gradient: 1.0,
        expires_at_ms: 1_000_000,
        hops_left: 1,
    }
}

#[test]
fn happy_file_creates_core_values_and_shared_settings() {
    let mut value = document();
    let mut second = value["neurons"][0].clone();
    second["id"] = json!(10002);
    second["activation"] = json!("relu");
    value["neurons"].as_array_mut().unwrap().push(second);
    let mut neurons = parse(&value).unwrap();
    assert_eq!(neurons[0].id(), 10001);
    assert_eq!(neurons[0].bias(), 0.5);
    assert_eq!(neurons[0].weight(20001), Some(0.125));
    assert!(neurons.iter().all(|neuron| neuron.axons().is_empty()));
    assert_eq!(neurons[0].forward(forward(1, 1000, -8.0)).unwrap(), -0.5);
    assert_eq!(neurons[1].forward(forward(1, 1000, -8.0)).unwrap(), 0.0);
    for neuron in &neurons {
        assert_eq!(neuron.trace(1).unwrap().expires_at_ms, 121000);
    }
    neurons[0].backward(feedback(1), 1001).unwrap();
    assert_eq!(neurons[0].bias(), 0.25);
    assert_eq!(neurons[0].weight(20001), Some(2.125));
    neurons[1].backward(feedback(1), 1001).unwrap();
    assert_eq!(neurons[1].bias(), 0.5); // ReLU's negative preactivation blocks learning.
}

#[test]
fn decimal_scientific_values_are_equivalent() {
    let decimal = serde_json::to_string(&document()).unwrap();
    let scientific = decimal
        .replace("0.125", "1.25e-1")
        .replace("0.5", "5e-1")
        .replace("0.25", "2.5e-1");
    let mut a = read(decimal.as_bytes()).unwrap();
    let mut b = read(scientific.as_bytes()).unwrap();
    assert_eq!(a[0].weights(), b[0].weights());
    assert_eq!(a[0].bias(), b[0].bias());
    assert_eq!(
        a[0].forward(forward(1, 1000, 2.0)),
        b[0].forward(forward(1, 1000, 2.0))
    );
    a[0].backward(feedback(1), 1001).unwrap();
    b[0].backward(feedback(1), 1001).unwrap();
    assert_eq!(a[0].bias(), b[0].bias());
    assert_eq!(a[0].weights(), b[0].weights());
}

#[test]
fn captured_scientific_regression_matches_decimal_in_every_float_field() {
    let base = serde_json::to_string(&document()).unwrap();
    for (field, initial) in [
        ("weight", "0.125"),
        ("bias", "0.5"),
        ("learning_rate", "0.25"),
    ] {
        let literal = format!("\"{field}\":{initial}");
        let decimal = base.replace(&literal, &format!("\"{field}\":0.00005251302718534134"));
        let scientific = base.replace(&literal, &format!("\"{field}\":5.251302718534134e-05"));
        let mut a = read(decimal.as_bytes()).unwrap();
        let mut b = read(scientific.as_bytes()).unwrap();
        assert_eq!(a[0].weights(), b[0].weights(), "{field}");
        assert_eq!(a[0].bias(), b[0].bias(), "{field}");
        assert_eq!(
            a[0].forward(forward(1, 1000, 2.0)),
            b[0].forward(forward(1, 1000, 2.0)),
            "{field}"
        );
        a[0].backward(feedback(1), 1001).unwrap();
        b[0].backward(feedback(1), 1001).unwrap();
        assert_eq!(a[0].weights(), b[0].weights(), "{field}");
        assert_eq!(a[0].bias(), b[0].bias(), "{field}");
    }
}

#[test]
fn required_unknown_and_wrongly_typed_fields_are_rejected_at_every_level() {
    let objects = ["", "/settings", "/neurons/0", "/neurons/0/weights/0"];
    for pointer in objects {
        let base = document();
        let keys: Vec<_> = base
            .pointer(pointer)
            .unwrap()
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect();
        for key in keys {
            let mut value = base.clone();
            value
                .pointer_mut(pointer)
                .unwrap()
                .as_object_mut()
                .unwrap()
                .remove(&key);
            assert!(parse(&value).is_err(), "missing {pointer}/{key}");
            for bad in [Value::Null, json!(true), json!({}), json!([])] {
                if key == "weights" && bad == json!([]) {
                    continue; // Empty weights are explicitly valid, not a type error.
                }
                let mut value = base.clone();
                value.pointer_mut(pointer).unwrap()[&key] = bad;
                assert!(
                    parse(&value).is_err(),
                    "wrong type {pointer}/{key}: {value}"
                );
            }
        }
        let mut value = base;
        value.pointer_mut(pointer).unwrap()["unknown"] = json!(1);
        assert!(parse(&value).is_err(), "unknown {pointer}");
    }
    let mut value = document();
    value["neurons"][0]["axons"] = json!([]);
    assert!(parse(&value).is_err());
}

#[test]
fn positional_arrays_cannot_impersonate_objects() {
    let base = document();
    let settings = json!([0.25, 120000, 10000, 4096, 8]);
    let neuron = json!([10001, 0.5, "linear", base["neurons"][0]["weights"]]);
    let weight = json!([20001, 0.125]);
    assert!(parse(&json!([1, base["settings"], base["neurons"]])).is_err());
    for (pointer, sequence) in [
        ("/settings", settings),
        ("/neurons/0", neuron),
        ("/neurons/0/weights/0", weight),
    ] {
        let mut value = base.clone();
        *value.pointer_mut(pointer).unwrap() = sequence;
        assert!(parse(&value).is_err(), "{pointer}");
    }
}

#[test]
fn duplicate_object_fields_are_rejected_including_equal_values() {
    let base = serde_json::to_string(&document()).unwrap();
    for (field, token) in [
        ("schema_version", "1"),
        ("learning_rate", "0.25"),
        ("activation_ttl_ms", "120000"),
        ("replay_retention_ms", "10000"),
        ("max_live_events", "4096"),
        ("max_staleness_versions", "8"),
        ("id", "10001"),
        ("bias", "0.5"),
        ("activation", "\"linear\""),
        ("source", "20001"),
        ("weight", "0.125"),
    ] {
        let field_text = format!("\"{field}\":{token}");
        let duplicate = base.replace(&field_text, &format!("{field_text},{field_text}"));
        assert!(read(duplicate.as_bytes()).is_err(), "duplicate {field}");
    }
    for field in ["settings", "neurons", "weights"] {
        let duplicate = base.replace(
            &format!("\"{field}\":"),
            &format!("\"{field}\":null,\"{field}\":"),
        );
        assert!(read(duplicate.as_bytes()).is_err(), "duplicate {field}");
    }
}

#[test]
fn ids_are_exact_nonzero_u64_and_duplicates_fail() {
    let mut value = document();
    value["neurons"][0]["id"] = json!(u64::MAX);
    value["neurons"][0]["weights"] = json!([
        {"source":9007199254740993_u64,"weight":1},
        {"source":u64::MAX,"weight":2}
    ]);
    let neurons = parse(&value).unwrap();
    assert_eq!(neurons[0].id(), u64::MAX);
    assert_eq!(neurons[0].weight(9_007_199_254_740_993), Some(1.0));
    assert_eq!(neurons[0].weight(u64::MAX), Some(2.0));
    let text = serde_json::to_string(&document()).unwrap();
    for (field, valid) in [("id", "10001"), ("source", "20001")] {
        for bad in [
            "0",
            "-1",
            "1.0",
            "1e0",
            "9007199254740993.0",
            "18446744073709551616",
            "true",
            "\"1\"",
            "null",
            "{}",
        ] {
            let invalid = text.replace(
                &format!("\"{field}\":{valid}"),
                &format!("\"{field}\":{bad}"),
            );
            assert!(read(invalid.as_bytes()).is_err(), "{field}={bad}");
        }
    }
    let mut duplicate = document();
    let neuron = duplicate["neurons"][0].clone();
    duplicate["neurons"].as_array_mut().unwrap().push(neuron);
    assert!(parse(&duplicate).is_err());
    let mut duplicate = document();
    let weight = duplicate["neurons"][0]["weights"][0].clone();
    duplicate["neurons"][0]["weights"]
        .as_array_mut()
        .unwrap()
        .push(weight);
    assert!(parse(&duplicate).is_err());
}

#[test]
fn finite_f32_boundaries_and_overflow_are_checked_before_narrowing() {
    let base = serde_json::to_string(&document()).unwrap();
    for token in [
        "3.4028234663852886e38",
        "-3.4028234663852886e38",
        "340282346638528859811704183484516925440.0",
        "1.401298464324817e-45",
        "0",
        "-0.0",
        "1e-300",
    ] {
        let value = base.replace("0.125", token).replace("0.5", token);
        assert!(read(value.as_bytes()).is_ok(), "{token}");
    }
    for (field, valid) in [
        ("weight", "0.125"),
        ("bias", "0.5"),
        ("learning_rate", "0.25"),
    ] {
        for token in [
            "3.402823466385289e38",
            "-3.402823466385289e38",
            "3.5e38",
            "1e400",
            "true",
            "false",
            "\"0.125\"",
            "null",
            "{}",
            "[]",
            "{\"$serde_json::private::Number\":\"0.125\"}",
        ] {
            let value = base.replace(
                &format!("\"{field}\":{valid}"),
                &format!("\"{field}\":{token}"),
            );
            assert!(read(value.as_bytes()).is_err(), "{field}={token}");
        }
    }
}

#[test]
fn versions_and_activations_are_strict() {
    for bad in [json!(0), json!(2), json!(1.0), json!("1"), json!(u64::MAX)] {
        let mut value = document();
        value["schema_version"] = bad;
        assert!(parse(&value).is_err());
    }
    for bad in [
        json!("ReLU"),
        json!("sigmoid"),
        json!({"linear":null}),
        json!(1),
        json!(false),
    ] {
        let mut value = document();
        value["neurons"][0]["activation"] = bad;
        assert!(parse(&value).is_err());
    }
}

#[test]
fn settings_boundaries_are_inclusive_and_invalid_settings_fail() {
    for (field, valid, invalid) in [
        (
            "learning_rate",
            vec![json!(f32::MIN_POSITIVE), json!(1)],
            vec![
                json!(0),
                json!(-0.1),
                json!(1.00001),
                json!(1e-300),
                json!("0.1"),
            ],
        ),
        (
            "activation_ttl_ms",
            vec![json!(1), json!(600000)],
            vec![json!(0), json!(600001), json!(-1), json!(1.0), json!("1")],
        ),
        (
            "replay_retention_ms",
            vec![json!(1), json!(600000)],
            vec![json!(0), json!(600001), json!(-1), json!(1.0), json!("1")],
        ),
        (
            "max_live_events",
            vec![json!(1), json!(4096)],
            vec![json!(0), json!(4097), json!(-1), json!(1.0), json!("1")],
        ),
        (
            "max_staleness_versions",
            vec![json!(0), json!(8)],
            vec![json!(9), json!(-1), json!(1.0), json!("1")],
        ),
    ] {
        for token in valid {
            let mut value = document();
            value["settings"][field] = token;
            assert!(parse(&value).is_ok(), "valid {field}: {value}");
        }
        for token in invalid {
            let mut value = document();
            value["settings"][field] = token;
            assert!(parse(&value).is_err(), "invalid {field}: {value}");
        }
    }
}

#[test]
fn max_neurons_and_weights_per_neuron_are_bounded_independently() {
    let mut value = document();
    value["neurons"] = Value::Array(
        (1..=MAX_NEURONS)
            .map(|id| json!({"id":id,"bias":0,"activation":"linear","weights":[]}))
            .collect(),
    );
    let neurons = parse(&value).unwrap();
    assert_eq!(neurons.len(), MAX_NEURONS);

    value["neurons"].as_array_mut().unwrap().push(json!({
        "id": (MAX_NEURONS as u64) + 1,
        "bias": 0,
        "activation": "linear",
        "weights": []
    }));
    assert!(parse(&value).is_err());

    let weights: Vec<_> = (1..=MAX_DENDRITES_PER_NEURON)
        .map(|source| json!({"source":source,"weight":0}))
        .collect();
    let mut value = document();
    value["neurons"][0]["weights"] = Value::Array(weights);
    assert!(parse(&value).is_ok());
    value["neurons"][0]["weights"]
        .as_array_mut()
        .unwrap()
        .push(json!({"source":(MAX_DENDRITES_PER_NEURON as u64) + 1,"weight":0}));
    assert!(parse(&value).is_err());
}

#[test]
fn single_node_mnist_1000_fits_startup_bounds() {
    const INPUTS: usize = 784;
    const HIDDEN: usize = 1_000;
    const OUTPUTS: usize = 10;
    let logical_neurons = HIDDEN + OUTPUTS;
    let total_weights = INPUTS * HIDDEN + HIDDEN * OUTPUTS;

    assert!(logical_neurons <= MAX_NEURONS);
    // The 1024-dendrite boundary is exercised by the independent max-weight test above.
    assert!(total_weights <= MAX_TOTAL_WEIGHTS);
    assert_eq!(logical_neurons, 1_010);
    assert_eq!(total_weights, 794_000);
}

#[test]
fn empty_neuron_array_fails_but_bias_only_neuron_matches_core_contract() {
    let mut value = document();
    value["neurons"] = json!([]);
    assert!(parse(&value).is_err());
    let mut value = document();
    value["neurons"][0]["weights"] = json!([]);
    assert!(parse(&value).is_ok());
}

#[test]
fn bounded_reader_accepts_exact_limit_and_reads_only_limit_plus_one() {
    let mut bytes = serde_json::to_vec(&document()).unwrap();
    bytes.resize(MAX_FILE_BYTES, b' ');
    assert!(read(Cursor::new(&bytes)).is_ok());
    bytes.extend_from_slice(b"  ignored trailing data");
    let mut cursor = Cursor::new(&bytes);
    assert!(read(&mut cursor).unwrap_err().contains("exceeds"));
    assert_eq!(cursor.position(), (MAX_FILE_BYTES + 1) as u64);
    for invalid in [
        b"".as_slice(),
        b"{}",
        b"[]",
        b"null",
        b"{",
        b"{} {}",
        b"\xff",
    ] {
        assert!(read(invalid).is_err());
    }
}

#[test]
fn loaded_retention_capacity_and_staleness_govern_core_behavior() {
    let mut value = document();
    value["settings"]["max_live_events"] = json!(1);
    value["settings"]["activation_ttl_ms"] = json!(10);
    value["settings"]["replay_retention_ms"] = json!(20);
    let mut neurons = parse(&value).unwrap();
    let n = &mut neurons[0];
    n.forward(forward(1, 1000, 1.0)).unwrap();
    assert_eq!(
        n.forward(forward(2, 1001, 1.0)),
        Err(Error::CapacityExceeded)
    );
    assert_eq!(
        n.backward(feedback(1), 1010).unwrap(),
        FeedbackStatus::Expired
    );
    assert_eq!(
        n.forward(forward(2, 1029, 1.0)),
        Err(Error::CapacityExceeded)
    );
    assert!(n.forward(forward(2, 1030, 1.0)).is_ok());
    let mut value = document();
    value["settings"]["max_staleness_versions"] = json!(0);
    let mut neurons = parse(&value).unwrap();
    let n = &mut neurons[0];
    n.forward(forward(1, 1000, 1.0)).unwrap();
    n.forward(forward(2, 1000, 1.0)).unwrap();
    n.backward(feedback(1), 1001).unwrap();
    assert_eq!(
        n.backward(feedback(2), 1001).unwrap(),
        FeedbackStatus::Stale
    );
}

#[test]
fn file_cli_conflicts_and_legacy_defaults_are_preserved() {
    let prefix = ["--id", "1", "--listen", "127.0.0.1:9101"];
    for suffix in [
        vec!["--neuron-config", "missing", "--neuron", "1"],
        vec!["--neuron-config", "missing", "--weight", "2:1"],
        vec!["--neuron-config", "missing", "--axon", "3:4"],
        vec!["--neuron", "1", "--neuron-config", "missing"],
        vec![
            "--neuron",
            "1",
            "--weight",
            "2:1",
            "--neuron-config",
            "missing",
        ],
        vec![
            "--neuron",
            "1",
            "--axon",
            "3:4",
            "--neuron-config",
            "missing",
        ],
        vec!["--neuron-config", "missing", "--neuron-config", "missing"],
    ] {
        let result = crate::parse_args(prefix.into_iter().chain(suffix).map(str::to_owned));
        assert!(result.is_err());
    }
    assert!(crate::parse_args(["--neuron-config"].map(str::to_owned)).is_err());
    let mut config = crate::parse_args(
        prefix
            .into_iter()
            .chain([
                "--neuron",
                "10001",
                "--weight",
                "20001:0.125",
                "--axon",
                "3:4",
                "--peer",
                "2@127.0.0.1:9102",
            ])
            .map(str::to_owned),
    )
    .unwrap();
    assert_eq!(config.worker_threads, 2);
    assert_eq!(config.mailbox_capacity, 256);
    assert_eq!(config.peers[0].id, 2);
    let n = &mut config.neurons[0];
    assert_eq!(n.bias(), 0.0);
    assert_eq!(n.axons(), &[danma_core::Axon { edge_id: 3, to: 4 }]);
    assert_eq!(n.forward(forward(1, 1000, -2.0)).unwrap(), -0.25);
    assert_eq!(n.trace(1).unwrap().expires_at_ms, 11000);
    n.backward(feedback(1), 1001).unwrap();
    assert_eq!(n.bias(), -0.1);
    assert_eq!(n.weight(20001), Some(0.325));
}
