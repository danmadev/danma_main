//! File startup through the real binary, shard, and TCP protocol.
use danma_net::request;
use serde_json::{json, Value};
use std::{
    fs,
    net::{SocketAddr, TcpListener, TcpStream},
    path::PathBuf,
    process::{Child, Command, Stdio},
    sync::atomic::{AtomicUsize, Ordering},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::time::{sleep, timeout};

struct ConfigFile(PathBuf);
impl ConfigFile {
    fn new(bytes: &[u8]) -> Self {
        static SEQUENCE: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "danma-neuron-config-{}-{}",
            std::process::id(),
            SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&dir).unwrap();
        let path = dir.join("neurons.json");
        fs::write(&path, bytes).unwrap();
        Self(path)
    }
}
impl Drop for ConfigFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
        let _ = fs::remove_dir(self.0.parent().unwrap());
    }
}

struct Node(Child);
impl Drop for Node {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn address() -> SocketAddr {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
}

fn document() -> Value {
    json!({"schema_version":1,"settings":{
        "learning_rate":0.25,"activation_ttl_ms":120000,"replay_retention_ms":10000,
        "max_live_events":4096,"max_staleness_versions":8
    },"neurons":[
        {"id":10001,"bias":0.5,"activation":"linear","weights":[{"source":20001,"weight":0.125}]},
        {"id":10002,"bias":0.5,"activation":"relu","weights":[{"source":20001,"weight":0.125}]}
    ]})
}

fn spawn(file: &ConfigFile, addr: SocketAddr, id: u64, peer: Option<(u64, SocketAddr)>) -> Node {
    let mut command = Command::new(env!("CARGO_BIN_EXE_danma-node"));
    command.args([
        "--id",
        &id.to_string(),
        "--listen",
        &addr.to_string(),
        "--workers",
        "2",
        "--mailbox",
        "16",
        "--neuron-config",
    ]);
    command.arg(&file.0);
    if let Some((id, peer)) = peer {
        command.args(["--peer", &format!("{id}@{peer}")]);
    }
    Node(
        command
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap(),
    )
}

async fn call(addr: SocketAddr, message: Value) -> Value {
    request(addr, &message).await.unwrap()
}

fn epoch_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn file_startup_inspect_forward_trace_and_teacher_learning_over_real_tcp() {
    let first = address();
    let mut second = address();
    while second == first {
        second = address();
    }
    // Use literal scientific notation through the file parser, not a Value roundtrip.
    let text = serde_json::to_string(&document())
        .unwrap()
        .replace("0.125", "1.25e-1");
    let file = ConfigFile::new(text.as_bytes());
    let mut output = document();
    output["neurons"] = json!([{"id":10003,"bias":0.25,"activation":"linear",
        "weights":[{"source":10001,"weight":2.0}]}]);
    let output_file = ConfigFile::new(&serde_json::to_vec(&output).unwrap());
    let _first = spawn(&file, first, 1, Some((2, second)));
    let _second = spawn(&output_file, second, 2, Some((1, first)));
    timeout(Duration::from_secs(10), async {
        loop {
            let mut ready = true;
            for addr in [first, second] {
                match request(addr, &json!({"kind":"routes"})).await {
                    Ok(value)
                        if value["routes"]["10001"] == 1
                            && value["routes"]["10002"] == 1
                            && value["routes"]["10003"] == 2 => {}
                    _ => ready = false,
                }
            }
            if ready {
                break;
            }
            sleep(Duration::from_millis(25)).await;
        }
    })
    .await
    .expect("file nodes and routes ready");
    for target in [10001, 10002] {
        let inspected = call(
            second,
            json!({"kind":"inspect","target":target,"route_hops":4}),
        )
        .await;
        assert_eq!(inspected["bias"], 0.5);
        assert_eq!(inspected["weights"]["20001"], 0.125);
        assert_eq!(inspected["axons"], json!([]));
        println!("file initial inspect: {inspected}");
        let before = epoch_ms();
        let result = call(
            second,
            json!({"kind":"forward","target":target,"event_id":target,
            "trace_id":9,"route_hops":4,"inputs":[{"from":20001,"source_event_id":8,"value":-8}],
            "expected":[{"kind":"teacher"}]}),
        )
        .await;
        let after = epoch_ms();
        assert_eq!(result["kind"], "forward_result");
        assert_eq!(result["output"], if target == 10001 { -0.5 } else { 0.0 });
        let trace = call(
            second,
            json!({"kind":"trace","target":target,"event_id":target,"route_hops":4}),
        )
        .await;
        let expiry = trace["trace"]["expires_at_ms"].as_u64().unwrap();
        assert!((before + 120000..=after + 120000).contains(&expiry));
        println!("file forward: {result}; trace: {trace}");
    }
    // The file TTL does not extend or disable the per-packet feedback deadline.
    let expired = call(
        first,
        json!({"kind":"backward","target":10001,"event_id":10001,
        "from":{"kind":"teacher"},"gradient":1,"ttl_ms":0,"gradient_hops":1,"route_hops":4}),
    )
    .await;
    assert_eq!(expired["status"], "expired");
    for target in [10001, 10002] {
        let result = call(
            second,
            json!({"kind":"backward","target":target,"event_id":target,
            "from":{"kind":"teacher"},"gradient":1,"ttl_ms":3000,"gradient_hops":1,"route_hops":4}),
        )
        .await;
        assert_eq!(result["status"], "applied", "{result}");
        let inspected = call(
            second,
            json!({"kind":"inspect","target":target,"route_hops":4}),
        )
        .await;
        assert_eq!(inspected["bias"], if target == 10001 { 0.25 } else { 0.5 });
        assert_eq!(
            inspected["weights"]["20001"],
            if target == 10001 { 2.125 } else { 0.125 }
        );
        println!("file learned inspect: {inspected}");
    }
    let result = call(
        first,
        json!({"kind":"forward","target":10003,"event_id":10003,
        "trace_id":9,"route_hops":4,"inputs":[{"from":10001,"source_event_id":10001,"value":-0.5}],
        "expected":[]}),
    )
    .await;
    assert_eq!(result["output"], -0.75);
    println!("routed output neuron forward: {result}");
}

fn assert_invalid_startup(file: &ConfigFile, extra: &[&str]) {
    let addr = address();
    let mut command = Command::new(env!("CARGO_BIN_EXE_danma-node"));
    command
        .args([
            "--id",
            "1",
            "--listen",
            &addr.to_string(),
            "--neuron-config",
        ])
        .arg(&file.0)
        .args(extra)
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    let mut node = Node(command.spawn().unwrap());
    let deadline = Instant::now() + Duration::from_secs(5);
    let status = loop {
        assert!(
            TcpStream::connect_timeout(&addr, Duration::from_millis(10)).is_err(),
            "invalid startup opened listener"
        );
        if let Some(status) = node.0.try_wait().unwrap() {
            break status;
        }
        assert!(Instant::now() < deadline, "invalid startup did not exit");
        std::thread::sleep(Duration::from_millis(5));
    };
    assert_eq!(status.code(), Some(2));
    assert!(TcpListener::bind(addr).is_ok(), "listener left behind");
    let mut stderr = String::new();
    use std::io::Read;
    node.0
        .stderr
        .take()
        .unwrap()
        .read_to_string(&mut stderr)
        .unwrap();
    assert!(stderr.contains("DANMA node configuration:"), "{stderr}");
    println!(
        "invalid startup exit={status}, no listener: {}",
        stderr.lines().next().unwrap()
    );
}

#[test]
fn invalid_file_and_cli_startup_exit_nonzero_without_listener() {
    let mut invalids = vec![b"{broken".to_vec(), b"{}".to_vec()];
    for (pointer, bad) in [
        ("/schema_version", json!(2)),
        ("/neurons/0/id", json!(0)),
        ("/neurons/0/weights/0/source", json!(1.0)),
        ("/neurons/0/activation", json!("sigmoid")),
        ("/settings/activation_ttl_ms", json!(600001)),
        ("/settings/learning_rate", json!(0)),
    ] {
        let mut value = document();
        *value.pointer_mut(pointer).unwrap() = bad;
        invalids.push(serde_json::to_vec(&value).unwrap());
    }
    let text = serde_json::to_string(&document()).unwrap();
    invalids.push(
        text.replace("\"bias\":0.5", "\"bias\":0.5,\"bias\":0.5")
            .into_bytes(),
    );
    invalids.push(
        text.replace("\"bias\":0.5", "\"unknown\":1,\"bias\":0.5")
            .into_bytes(),
    );
    invalids.push(text.replace("0.125", "3.402823466385289e38").into_bytes());
    let mut duplicate = document();
    duplicate["neurons"][1]["id"] = json!(10001);
    invalids.push(serde_json::to_vec(&duplicate).unwrap());
    invalids.push(vec![b' '; 16 * 1024 * 1024 + 1]);
    for bytes in invalids {
        assert_invalid_startup(&ConfigFile::new(&bytes), &[]);
    }
    let file = ConfigFile::new(text.as_bytes());
    for extra in [
        ["--neuron", "1"],
        ["--weight", "2:1"],
        ["--axon", "3:4"],
        ["--neuron-config", "another"],
    ] {
        assert_invalid_startup(&file, &extra);
    }
    let missing = ConfigFile(file.0.with_file_name("missing.json"));
    assert_invalid_startup(&missing, &[]);
}
