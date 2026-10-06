//! Literal JSON over the production TCP decoder; no test-side float reformatting.
use danma_core::derived_event_id;
use danma_net::request;
use serde_json::{json, Value};
use std::net::{SocketAddr, TcpListener};
use std::process::{Child, Command, Stdio};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::time::{sleep, timeout};

static NODE_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

struct Node(Child);

impl Drop for Node {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

async fn node() -> (Node, SocketAddr) {
    let reserved = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = reserved.local_addr().unwrap();
    drop(reserved);
    let process = Command::new(env!("CARGO_BIN_EXE_danma-node"))
        .args([
            "--id",
            "1",
            "--listen",
            &address.to_string(),
            "--neuron",
            "1",
            "--weight",
            "99:2",
            "--neuron",
            "2",
            "--weight",
            "99:2",
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .unwrap();
    let node = Node(process);
    timeout(Duration::from_secs(8), async {
        loop {
            if let Ok(response) = request(address, &json!({"kind":"routes"})).await {
                if response["kind"] == "routes_result" {
                    break;
                }
            }
            sleep(Duration::from_millis(25)).await;
        }
    })
    .await
    .expect("node readiness");
    (node, address)
}

async fn literal(address: SocketAddr, text: &str) -> Value {
    timeout(Duration::from_secs(4), async {
        let mut stream = TcpStream::connect(address).await.unwrap();
        stream.write_u32(text.len() as u32).await.unwrap();
        stream.write_all(text.as_bytes()).await.unwrap();
        let length = stream.read_u32().await.unwrap() as usize;
        assert!(length <= 256 * 1024);
        let mut response = vec![0; length];
        stream.read_exact(&mut response).await.unwrap();
        serde_json::from_slice(&response).unwrap()
    })
    .await
    .expect("literal request timeout")
}

fn forward(target: u64, value: &str) -> String {
    format!(
        r#"{{"kind":"forward","target":{target},"event_id":18446744073709551615,"trace_id":9007199254740993,"route_hops":4,"inputs":[{{"from":99,"source_event_id":18446744073709551615,"value":{value}}}],"expected":[{{"kind":"teacher"}}]}}"#
    )
}

fn backward(target: u64, event: u64, gradient: &str) -> String {
    format!(
        r#"{{"kind":"backward","target":{target},"event_id":{event},"from":{{"kind":"teacher"}},"gradient":{gradient},"ttl_ms":3000,"gradient_hops":1,"route_hops":4}}"#
    )
}

async fn assert_same_parameters(address: SocketAddr) {
    let mut states = Vec::new();
    for target in [1, 2] {
        let state = request(
            address,
            &json!({"kind":"inspect","target":target,"route_hops":4}),
        )
        .await
        .unwrap();
        assert_eq!(state["kind"], "inspect_result");
        assert_eq!(state["version"], 1);
        assert_ne!(state["bias"].as_f64(), Some(0.0));
        states.push(state);
    }
    assert_eq!(states[0]["weights"], states[1]["weights"]);
    assert_eq!(states[0]["bias"], states[1]["bias"]);
}

async fn train_pair(input_tokens: [&str; 2], gradient_tokens: [&str; 2]) {
    let _guard = NODE_LOCK.lock().await;
    let (_node, address) = node().await;
    let mut outputs = Vec::new();
    for (index, target) in [1, 2].into_iter().enumerate() {
        let output = literal(address, &forward(target, input_tokens[index])).await;
        assert_eq!(output["kind"], "forward_result", "{output}");
        outputs.push(output["output"].clone());
        let trace = request(
            address,
            &json!({"kind":"trace","target":target,"event_id":u64::MAX,"route_hops":4}),
        )
        .await
        .unwrap();
        assert_eq!(trace["event_id"].as_u64(), Some(u64::MAX));
        assert_eq!(
            trace["trace"]["trace_id"].as_u64(),
            Some(9_007_199_254_740_993)
        );
        let trained = literal(address, &backward(target, u64::MAX, gradient_tokens[index])).await;
        assert_eq!(trained["kind"], "backward_result", "{trained}");
        assert_eq!(trained["status"], "applied");
        let duplicate = literal(address, &backward(target, u64::MAX, gradient_tokens[index])).await;
        assert_eq!(duplicate["status"], "ignored_duplicate");
    }
    assert_eq!(outputs[0], outputs[1]);
    assert_same_parameters(address).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn numeric_protocol_tcp_decimal_training_control() {
    train_pair(
        ["0.125", "0.125"],
        ["0.00005251302718534134", "0.00005251302718534134"],
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn numeric_protocol_tcp_scientific_gradient_matches_decimal() {
    train_pair(
        ["0.125", "0.125"],
        ["0.00005251302718534134", "5.251302718534134e-05"],
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn numeric_protocol_tcp_scientific_input_matches_decimal() {
    train_pair(["0.125", "1.25e-1"], ["0.125", "0.125"]).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn numeric_protocol_tcp_scientific_signal_matches_decimal() {
    let _guard = NODE_LOCK.lock().await;
    let (_node, address) = node().await;
    let mut outputs = Vec::new();
    for (target, token) in [(1, "-0.125"), (2, "-1.25e-1")] {
        let event = u64::try_from(derived_event_id(9007199254740993, target)).unwrap();
        let text = format!(
            r#"{{"kind":"signal","target":{target},"event_id":{event},"trace_id":9007199254740993,"edge_id":1,"from":99,"source_event_id":18446744073709551615,"value":{token},"training":true,"forward_hops":32,"route_hops":4}}"#
        );
        let output = literal(address, &text).await;
        assert_eq!(output["kind"], "signal_result", "{output}");
        assert_eq!(output["status"], "fired");
        outputs.push(output["output"].clone());
        let trained = literal(address, &backward(target, event, "0.125")).await;
        assert_eq!(trained["status"], "applied", "{trained}");
    }
    assert_eq!(outputs[0], outputs[1]);
    assert_same_parameters(address).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn numeric_protocol_tcp_invalid_numbers_cannot_mutate_state() {
    let _guard = NODE_LOCK.lock().await;
    let (_node, address) = node().await;
    for token in [
        "3.5e38",
        "-3.5e38",
        "1e400",
        "null",
        "true",
        r#""0.125""#,
        r#"{"$serde_json::private::Number":"0.125"}"#,
    ] {
        let rejected = literal(address, &forward(1, token)).await;
        assert_eq!(
            rejected,
            json!({"kind":"error","code":"invalid_protocol_message"})
        );
    }
    let created = literal(address, &forward(1, "0.125")).await;
    assert_eq!(created["kind"], "forward_result");
    for token in ["3.5e38", "-3.5e38", "1e400", "null", "false", "{}"] {
        let rejected = literal(address, &backward(1, u64::MAX, token)).await;
        assert_eq!(
            rejected,
            json!({"kind":"error","code":"invalid_protocol_message"})
        );
    }
    let state = request(
        address,
        &json!({"kind":"inspect","target":1,"route_hops":4}),
    )
    .await
    .unwrap();
    assert_eq!(state["version"], 0);
    assert_eq!(state["bias"].as_f64(), Some(0.0));
    assert_eq!(state["weights"]["99"].as_f64(), Some(2.0));
    let trained = literal(address, &backward(1, u64::MAX, "5.251302718534134e-05")).await;
    assert_eq!(trained["status"], "applied", "{trained}");
}
