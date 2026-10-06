use super::*;

const CAPTURED: &str = r#"{"kind":"backward","target":10002,"event_id":9720822987784714125,"from":{"kind":"teacher"},"gradient":5.251302718534134e-05,"ttl_ms":3000,"gradient_hops":2,"route_hops":4}"#;

fn backward(token: &str) -> String {
    CAPTURED.replace("5.251302718534134e-05", token)
}

fn forward(token: &str) -> String {
    format!(
        r#"{{"kind":"forward","target":1,"event_id":2,"trace_id":3,"route_hops":4,"forward_hops":32,"inputs":[{{"from":99,"source_event_id":5,"value":{token}}}],"expected":[{{"kind":"neuron","neuron_id":6,"event_id":7}}]}}"#
    )
}

fn signal(token: &str) -> String {
    format!(
        r#"{{"kind":"signal","target":1,"event_id":2,"trace_id":3,"edge_id":4,"from":99,"source_event_id":5,"value":{token},"training":true,"forward_hops":32,"route_hops":4}}"#
    )
}

fn numeric_value(text: &str) -> f32 {
    match decode_message(text.as_bytes()).unwrap_or_else(|error| panic!("{text}: {error}")) {
        Message::Backward { gradient, .. } => gradient,
        Message::Forward { inputs, .. } => inputs[0].value,
        Message::Signal { value, .. } => value,
        other => panic!("unexpected message: {other:?}"),
    }
}

#[test]
fn numeric_protocol_captured_decimal_control() {
    let decimal = backward("0.00005251302718534134");
    assert_eq!(numeric_value(&decimal), 5.251_302_7e-5_f32);
}

#[test]
fn numeric_protocol_captured_scientific_regression() {
    assert_eq!(CAPTURED.len(), 171);
    let decimal = numeric_value(&backward("0.00005251302718534134"));
    assert_eq!(numeric_value(CAPTURED).to_bits(), decimal.to_bits());
    match decode_message(CAPTURED.as_bytes()).unwrap() {
        Message::Backward {
            event_id, target, ..
        } => {
            assert_eq!(event_id, 9_720_822_987_784_714_125);
            assert_eq!(target, 10002);
        }
        _ => unreachable!(),
    }
}

#[test]
fn numeric_protocol_equivalent_notation_all_float_fields() {
    for make in [backward, forward, signal] {
        for (scientific, decimal) in [
            ("1e-5", "0.00001"),
            ("1.25e-1", "0.125"),
            ("-1.25E-1", "-0.125"),
            ("1E+2", "100"),
            ("0e0", "0"),
            ("-0e0", "-0.0"),
        ] {
            let control = numeric_value(&make(decimal));
            assert_eq!(
                numeric_value(&make(scientific)).to_bits(),
                control.to_bits()
            );
        }
    }
}

#[test]
fn numeric_protocol_finite_float_limits() {
    for make in [backward, forward, signal] {
        for (token, expected) in [
            ("3.4028234663852886e38", f32::MAX),
            ("-3.4028234663852886e38", -f32::MAX),
            ("1.1754943508222875e-38", f32::MIN_POSITIVE),
            ("1.401298464324817e-45", f32::from_bits(1)),
            ("1e-50", 0.0),
        ] {
            assert_eq!(numeric_value(&make(token)).to_bits(), expected.to_bits());
        }
    }
}

#[test]
fn numeric_protocol_rejects_overflow_and_nonfinite() {
    for make in [backward, forward, signal] {
        for token in [
            "3.4028235e38",
            "-3.4028235e38",
            "3.5e38",
            "-3.5e38",
            "1e39",
            "1e400",
            "NaN",
            "Infinity",
            "-Infinity",
        ] {
            let text = make(token);
            assert!(decode_message(text.as_bytes()).is_err(), "accepted {text}");
        }
    }
}

#[test]
fn numeric_protocol_rejects_non_numeric_types() {
    for make in [backward, forward, signal] {
        for token in [
            r#""0.125""#,
            "{}",
            "[]",
            "null",
            "true",
            "false",
            r#"{"$serde_json::private::Number":"0.125"}"#,
        ] {
            let text = make(token);
            assert!(decode_message(text.as_bytes()).is_err(), "accepted {text}");
        }
    }
}

#[test]
fn numeric_protocol_strict_schema() {
    for mut value in [
        serde_json::from_str::<Value>(&backward("0.125")).unwrap(),
        serde_json::from_str::<Value>(&forward("0.125")).unwrap(),
        serde_json::from_str::<Value>(&signal("0.125")).unwrap(),
        json!({"kind":"routes"}),
    ] {
        value["extra"] = json!(1);
        assert!(decode_message(&serde_json::to_vec(&value).unwrap()).is_err());
    }
    let mut nested = serde_json::from_str::<Value>(&forward("0.125")).unwrap();
    for pointer in ["/inputs/0", "/expected/0"] {
        nested.pointer_mut(pointer).unwrap()["extra"] = json!(1);
        assert!(decode_message(&serde_json::to_vec(&nested).unwrap()).is_err());
        nested
            .pointer_mut(pointer)
            .unwrap()
            .as_object_mut()
            .unwrap()
            .remove("extra");
    }
    let mut source = serde_json::from_str::<Value>(&backward("0.125")).unwrap();
    source["from"]["extra"] = json!(1);
    assert!(decode_message(&serde_json::to_vec(&source).unwrap()).is_err());
    assert!(decode_message(
        br#"{"kind":"gossip","from_node":1,"routes":[{"owner":2,"neuron":3,"epoch":4,"extra":5}]}"#
    )
    .is_err());
    assert!(decode_message(br#"{"kind":"inspect","target":1}"#).is_err());
    assert!(decode_message(br#"{"kind":"unknown"}"#).is_err());
    assert!(decode_message(b"null").is_err());
}

#[test]
fn numeric_protocol_integer_fidelity_and_rejection() {
    let cases = [
        (
            backward("0.125"),
            vec!["/target", "/event_id", "/ttl_ms"],
            vec!["/gradient_hops", "/route_hops"],
        ),
        (
            forward("0.125"),
            vec![
                "/target",
                "/event_id",
                "/trace_id",
                "/inputs/0/from",
                "/inputs/0/source_event_id",
                "/expected/0/neuron_id",
                "/expected/0/event_id",
            ],
            vec!["/route_hops", "/forward_hops"],
        ),
        (
            signal("0.125"),
            vec![
                "/target",
                "/event_id",
                "/trace_id",
                "/edge_id",
                "/from",
                "/source_event_id",
            ],
            vec!["/forward_hops", "/route_hops"],
        ),
        (
            r#"{"kind":"gossip","from_node":1,"routes":[{"owner":2,"neuron":3,"epoch":4}]}"#.into(),
            vec![
                "/from_node",
                "/routes/0/owner",
                "/routes/0/neuron",
                "/routes/0/epoch",
            ],
            vec![],
        ),
        (
            r#"{"kind":"inspect","target":1,"route_hops":4}"#.into(),
            vec!["/target"],
            vec!["/route_hops"],
        ),
        (
            r#"{"kind":"trace","target":1,"event_id":2,"route_hops":4}"#.into(),
            vec!["/target", "/event_id"],
            vec!["/route_hops"],
        ),
        (
            backward("0.125").replace(
                r#"{"kind":"teacher"}"#,
                r#"{"kind":"neuron","neuron_id":6,"event_id":7}"#,
            ),
            vec!["/from/neuron_id", "/from/event_id"],
            vec![],
        ),
    ];
    for (text, ids, counters) in cases {
        let original: Value = serde_json::from_str(&text).unwrap();
        for (paths, max, overflow) in [
            (ids, u64::MAX, "18446744073709551616"),
            (counters, u8::MAX.into(), "256"),
        ] {
            for pointer in paths {
                for integer in [0, 1, max] {
                    let mut value = original.clone();
                    *value.pointer_mut(pointer).unwrap() = json!(integer);
                    let decoded = decode_message(&serde_json::to_vec(&value).unwrap()).unwrap();
                    let roundtrip = serde_json::to_value(decoded).unwrap();
                    assert_eq!(roundtrip.pointer(pointer).unwrap().as_u64(), Some(integer));
                }
                if max == u64::MAX {
                    let mut value = original.clone();
                    *value.pointer_mut(pointer).unwrap() = json!(9_007_199_254_740_993_u64);
                    let decoded = decode_message(&serde_json::to_vec(&value).unwrap()).unwrap();
                    assert_eq!(
                        serde_json::to_value(decoded)
                            .unwrap()
                            .pointer(pointer)
                            .unwrap()
                            .as_u64(),
                        Some(9_007_199_254_740_993)
                    );
                }
                for token in [
                    overflow,
                    "-1",
                    "0.5",
                    "1.0",
                    "1e0",
                    "1.000000000000000000001",
                    "18446744073709551615.0",
                    r#""1""#,
                    "null",
                    "true",
                    "{}",
                ] {
                    // Insert literal text without rounding it through a test-side float.
                    let mut value = original.clone();
                    *value.pointer_mut(pointer).unwrap() = json!("NUMERIC_TOKEN");
                    let candidate = serde_json::to_string(&value)
                        .unwrap()
                        .replace(r#""NUMERIC_TOKEN""#, token);
                    assert!(
                        decode_message(candidate.as_bytes()).is_err(),
                        "accepted {pointer}={token}: {candidate}"
                    );
                }
            }
        }
    }
}
