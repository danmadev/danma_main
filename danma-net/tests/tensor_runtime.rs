//! Remote tensor protocol proof: payload lives in Rust, and mm/add execute there.
use std::net::{SocketAddr, TcpListener};
use std::process::{Child, Command, Stdio};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::time::{sleep, timeout};

const MAGIC: &[u8; 4] = b"DNT1";
const ALLOC: u8 = 1;
const FREE: u8 = 2;
const UPLOAD: u8 = 3;
const DOWNLOAD: u8 = 4;
const ADD: u8 = 5;
const MM: u8 = 6;
const FILL: u8 = 7;
const COPY: u8 = 8;
const SUM_ROWS: u8 = 9;
const STATS: u8 = 10;

fn free_port() -> u16 {
    let socket = TcpListener::bind("127.0.0.1:0").expect("bind test port");
    socket.local_addr().unwrap().port()
}

struct Node(Child);

impl Drop for Node {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn push_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_be_bytes());
}

fn push_f32(out: &mut Vec<u8>, value: f32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn read_u64(input: &[u8], offset: &mut usize) -> u64 {
    let bytes: [u8; 8] = input[*offset..*offset + 8].try_into().unwrap();
    *offset += 8;
    u64::from_be_bytes(bytes)
}

fn read_f32(input: &[u8], offset: &mut usize) -> f32 {
    let bytes: [u8; 4] = input[*offset..*offset + 4].try_into().unwrap();
    *offset += 4;
    f32::from_le_bytes(bytes)
}

async fn raw_call(addr: SocketAddr, opcode: u8, body: Vec<u8>) -> Vec<u8> {
    let mut frame = Vec::with_capacity(5 + body.len());
    frame.extend_from_slice(MAGIC);
    frame.push(opcode);
    frame.extend_from_slice(&body);

    let mut stream = TcpStream::connect(addr).await.expect("connect tensor runtime");
    stream.write_u32(frame.len() as u32).await.unwrap();
    stream.write_all(&frame).await.unwrap();
    stream.flush().await.unwrap();

    let len = stream.read_u32().await.unwrap() as usize;
    let mut reply = vec![0; len];
    stream.read_exact(&mut reply).await.unwrap();
    assert_eq!(reply.first().copied(), Some(0), "tensor runtime error: {reply:?}");
    reply
}

async fn alloc(addr: SocketAddr, id: u64, numel: u64) {
    let mut body = Vec::new();
    push_u64(&mut body, id);
    push_u64(&mut body, numel);
    raw_call(addr, ALLOC, body).await;
}

async fn upload(addr: SocketAddr, id: u64, values: &[f32]) {
    let mut body = Vec::new();
    push_u64(&mut body, id);
    push_u64(&mut body, values.len() as u64);
    for value in values {
        push_f32(&mut body, *value);
    }
    raw_call(addr, UPLOAD, body).await;
}

async fn download(addr: SocketAddr, id: u64) -> Vec<f32> {
    let mut body = Vec::new();
    push_u64(&mut body, id);
    let reply = raw_call(addr, DOWNLOAD, body).await;
    let mut offset = 1;
    let count = read_u64(&reply, &mut offset) as usize;
    (0..count).map(|_| read_f32(&reply, &mut offset)).collect()
}

async fn wait_ready(addr: SocketAddr) {
    timeout(Duration::from_secs(8), async {
        loop {
            if TcpStream::connect(addr).await.is_ok() {
                break;
            }
            sleep(Duration::from_millis(25)).await;
        }
    })
    .await
    .expect("node did not start");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn remote_tensor_mm_add_fill_copy_sum_and_stats() {
    let port = free_port();
    let addr: SocketAddr = format!("127.0.0.1:{port}").parse().unwrap();
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_danma-node"));
    cmd.args([
        "--id", "1",
        "--listen", &addr.to_string(),
        "--neuron", "1",
        "--weight", "99:1",
    ]);
    cmd.stdout(Stdio::null()).stderr(Stdio::inherit());
    let _node = Node(cmd.spawn().expect("spawn node"));
    wait_ready(addr).await;

    // A = [[1,2],[3,4]], B = [[5,6],[7,8]]
    alloc(addr, 1001, 4).await;
    alloc(addr, 1002, 4).await;
    alloc(addr, 1003, 4).await;
    upload(addr, 1001, &[1., 2., 3., 4.]).await;
    upload(addr, 1002, &[5., 6., 7., 8.]).await;

    let mut mm = Vec::new();
    for id in [1001_u64, 1002, 1003] {
        push_u64(&mut mm, id);
    }
    for dim in [2_u64, 2, 2] {
        push_u64(&mut mm, dim);
    }
    mm.push(0); // lhs not transposed
    mm.push(0); // rhs not transposed
    raw_call(addr, MM, mm).await;
    assert_eq!(download(addr, 1003).await, vec![19., 22., 43., 50.]);

    // Bias broadcast over the last dimension.
    alloc(addr, 1004, 2).await;
    alloc(addr, 1005, 4).await;
    upload(addr, 1004, &[1., -1.]).await;
    let mut add = Vec::new();
    for id in [1003_u64, 1004, 1005] {
        push_u64(&mut add, id);
    }
    push_u64(&mut add, 4); // lhs numel
    push_u64(&mut add, 2); // rhs numel
    push_f32(&mut add, 1.0);
    raw_call(addr, ADD, add).await;
    assert_eq!(download(addr, 1005).await, vec![20., 21., 44., 49.]);

    // sum rows is required by bias autograd.
    alloc(addr, 1006, 2).await;
    let mut sum = Vec::new();
    for id in [1005_u64, 1006] {
        push_u64(&mut sum, id);
    }
    push_u64(&mut sum, 2);
    push_u64(&mut sum, 2);
    raw_call(addr, SUM_ROWS, sum).await;
    assert_eq!(download(addr, 1006).await, vec![64., 70.]);

    // fill and copy remain remote; no tensor payload is required in the client allocator.
    alloc(addr, 1007, 2).await;
    let mut fill = Vec::new();
    push_u64(&mut fill, 1007);
    push_f32(&mut fill, 3.5);
    raw_call(addr, FILL, fill).await;
    assert_eq!(download(addr, 1007).await, vec![3.5, 3.5]);

    alloc(addr, 1008, 2).await;
    let mut copy = Vec::new();
    push_u64(&mut copy, 1007);
    push_u64(&mut copy, 1008);
    raw_call(addr, COPY, copy).await;
    assert_eq!(download(addr, 1008).await, vec![3.5, 3.5]);

    let stats = raw_call(addr, STATS, Vec::new()).await;
    let mut offset = 1;
    let tensor_count = read_u64(&stats, &mut offset);
    let remote_bytes = read_u64(&stats, &mut offset);
    let mm_ops = read_u64(&stats, &mut offset);
    let add_ops = read_u64(&stats, &mut offset);
    assert!(tensor_count >= 8);
    assert!(remote_bytes >= 24 * 4);
    assert_eq!(mm_ops, 1);
    assert_eq!(add_ops, 1);

    let mut free = Vec::new();
    push_u64(&mut free, 1008);
    raw_call(addr, FREE, free).await;
}
