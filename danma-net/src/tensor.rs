//! Bounded remote tensor store used by the PrivateUse1 DANMA device.
//!
//! Tensor payload bytes live only in this Rust process. The PyTorch allocator
//! keeps an opaque handle and never stores the f32 payload in host-side tensor
//! storage.

use std::collections::BTreeMap;

pub const MAGIC: &[u8; 4] = b"DNT1";
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

const MAX_TENSOR_ELEMENTS: usize = 60_000;
const MAX_REMOTE_BYTES: usize = 64 * 1024 * 1024;

#[derive(Default)]
pub(crate) struct TensorRuntime {
    tensors: BTreeMap<u64, Vec<f32>>,
    remote_bytes: usize,
    allocations: u64,
    uploads: u64,
    downloads: u64,
    add_ops: u64,
    mm_ops: u64,
}

struct Cursor<'a> {
    input: &'a [u8],
    offset: usize,
}

impl<'a> Cursor<'a> {
    fn new(input: &'a [u8]) -> Self {
        Self { input, offset: 0 }
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8], &'static str> {
        let end = self.offset.checked_add(len).ok_or("tensor_frame_overflow")?;
        let slice = self.input.get(self.offset..end).ok_or("tensor_frame_truncated")?;
        self.offset = end;
        Ok(slice)
    }

    fn u8(&mut self) -> Result<u8, &'static str> {
        Ok(self.take(1)?[0])
    }

    fn u64(&mut self) -> Result<u64, &'static str> {
        Ok(u64::from_be_bytes(
            self.take(8)?.try_into().map_err(|_| "tensor_bad_u64")?,
        ))
    }

    fn f32(&mut self) -> Result<f32, &'static str> {
        Ok(f32::from_le_bytes(
            self.take(4)?.try_into().map_err(|_| "tensor_bad_f32")?,
        ))
    }

    fn done(&self) -> bool {
        self.offset == self.input.len()
    }
}

fn ok() -> Vec<u8> {
    vec![0]
}

fn error(code: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(code.len() + 1);
    out.push(1);
    out.extend_from_slice(code.as_bytes());
    out
}

fn push_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_be_bytes());
}

fn push_f32(out: &mut Vec<u8>, value: f32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn checked_numel(value: u64) -> Result<usize, &'static str> {
    let value = usize::try_from(value).map_err(|_| "tensor_size_overflow")?;
    if value > MAX_TENSOR_ELEMENTS {
        return Err("tensor_too_large");
    }
    Ok(value)
}

impl TensorRuntime {
    pub(crate) fn is_frame(frame: &[u8]) -> bool {
        frame.len() >= 5 && &frame[..4] == MAGIC
    }

    pub(crate) fn process(&mut self, frame: &[u8]) -> Vec<u8> {
        match self.process_inner(frame) {
            Ok(reply) => reply,
            Err(code) => error(code),
        }
    }

    fn process_inner(&mut self, frame: &[u8]) -> Result<Vec<u8>, &'static str> {
        if !Self::is_frame(frame) {
            return Err("not_tensor_frame");
        }
        let mut cursor = Cursor::new(&frame[4..]);
        let opcode = cursor.u8()?;
        let reply = match opcode {
            ALLOC => {
                let id = cursor.u64()?;
                let numel = checked_numel(cursor.u64()?)?;
                if id == 0 || !cursor.done() || self.tensors.contains_key(&id) {
                    return Err("tensor_alloc_invalid");
                }
                let bytes = numel.checked_mul(4).ok_or("tensor_size_overflow")?;
                if self.remote_bytes.saturating_add(bytes) > MAX_REMOTE_BYTES {
                    return Err("tensor_memory_limit");
                }
                self.tensors.insert(id, vec![0.0; numel]);
                self.remote_bytes += bytes;
                self.allocations = self.allocations.saturating_add(1);
                ok()
            }
            FREE => {
                let id = cursor.u64()?;
                if !cursor.done() {
                    return Err("tensor_free_invalid");
                }
                if let Some(old) = self.tensors.remove(&id) {
                    self.remote_bytes = self.remote_bytes.saturating_sub(old.len() * 4);
                }
                ok()
            }
            UPLOAD => {
                let id = cursor.u64()?;
                let numel = checked_numel(cursor.u64()?)?;
                let tensor = self.tensors.get_mut(&id).ok_or("tensor_unknown")?;
                if tensor.len() != numel {
                    return Err("tensor_upload_size_mismatch");
                }
                for value in tensor.iter_mut() {
                    *value = cursor.f32()?;
                    if !value.is_finite() {
                        return Err("tensor_non_finite");
                    }
                }
                if !cursor.done() {
                    return Err("tensor_upload_trailing_bytes");
                }
                self.uploads = self.uploads.saturating_add(1);
                ok()
            }
            DOWNLOAD => {
                let id = cursor.u64()?;
                if !cursor.done() {
                    return Err("tensor_download_invalid");
                }
                let tensor = self.tensors.get(&id).ok_or("tensor_unknown")?;
                let mut out = Vec::with_capacity(1 + 8 + tensor.len() * 4);
                out.push(0);
                push_u64(&mut out, tensor.len() as u64);
                for value in tensor {
                    push_f32(&mut out, *value);
                }
                self.downloads = self.downloads.saturating_add(1);
                out
            }
            ADD => {
                let lhs_id = cursor.u64()?;
                let rhs_id = cursor.u64()?;
                let out_id = cursor.u64()?;
                let lhs_numel = checked_numel(cursor.u64()?)?;
                let rhs_numel = checked_numel(cursor.u64()?)?;
                let alpha = cursor.f32()?;
                if !alpha.is_finite() || !cursor.done() || rhs_numel == 0 {
                    return Err("tensor_add_invalid");
                }
                let (lhs, rhs) = {
                    let lhs = self.tensors.get(&lhs_id).ok_or("tensor_unknown")?;
                    let rhs = self.tensors.get(&rhs_id).ok_or("tensor_unknown")?;
                    if lhs.len() != lhs_numel
                        || rhs.len() != rhs_numel
                        || lhs_numel % rhs_numel != 0
                    {
                        return Err("tensor_add_shape_mismatch");
                    }
                    (lhs.clone(), rhs.clone())
                };
                let out = self.tensors.get_mut(&out_id).ok_or("tensor_unknown")?;
                if out.len() != lhs_numel {
                    return Err("tensor_add_output_mismatch");
                }
                for index in 0..lhs_numel {
                    out[index] = lhs[index] + alpha * rhs[index % rhs_numel];
                }
                self.add_ops = self.add_ops.saturating_add(1);
                ok()
            }
            MM => {
                let lhs_id = cursor.u64()?;
                let rhs_id = cursor.u64()?;
                let out_id = cursor.u64()?;
                let m = checked_numel(cursor.u64()?)?;
                let k = checked_numel(cursor.u64()?)?;
                let n = checked_numel(cursor.u64()?)?;
                let lhs_t = cursor.u8()? != 0;
                let rhs_t = cursor.u8()? != 0;
                if !cursor.done() {
                    return Err("tensor_mm_invalid");
                }
                let lhs_len = m.checked_mul(k).ok_or("tensor_size_overflow")?;
                let rhs_len = k.checked_mul(n).ok_or("tensor_size_overflow")?;
                let out_len = m.checked_mul(n).ok_or("tensor_size_overflow")?;
                let (lhs, rhs) = {
                    let lhs = self.tensors.get(&lhs_id).ok_or("tensor_unknown")?;
                    let rhs = self.tensors.get(&rhs_id).ok_or("tensor_unknown")?;
                    if lhs.len() != lhs_len || rhs.len() != rhs_len {
                        return Err("tensor_mm_shape_mismatch");
                    }
                    (lhs.clone(), rhs.clone())
                };
                let out = self.tensors.get_mut(&out_id).ok_or("tensor_unknown")?;
                if out.len() != out_len {
                    return Err("tensor_mm_output_mismatch");
                }
                for row in 0..m {
                    for col in 0..n {
                        let mut value = 0.0_f32;
                        for inner in 0..k {
                            let left = if lhs_t {
                                lhs[inner * m + row]
                            } else {
                                lhs[row * k + inner]
                            };
                            let right = if rhs_t {
                                rhs[col * k + inner]
                            } else {
                                rhs[inner * n + col]
                            };
                            value += left * right;
                        }
                        out[row * n + col] = value;
                    }
                }
                self.mm_ops = self.mm_ops.saturating_add(1);
                ok()
            }
            FILL => {
                let id = cursor.u64()?;
                let value = cursor.f32()?;
                if !value.is_finite() || !cursor.done() {
                    return Err("tensor_fill_invalid");
                }
                let tensor = self.tensors.get_mut(&id).ok_or("tensor_unknown")?;
                tensor.fill(value);
                ok()
            }
            COPY => {
                let src = cursor.u64()?;
                let dst = cursor.u64()?;
                if !cursor.done() {
                    return Err("tensor_copy_invalid");
                }
                let values = self.tensors.get(&src).ok_or("tensor_unknown")?.clone();
                let output = self.tensors.get_mut(&dst).ok_or("tensor_unknown")?;
                if values.len() != output.len() {
                    return Err("tensor_copy_shape_mismatch");
                }
                output.copy_from_slice(&values);
                ok()
            }
            SUM_ROWS => {
                let src = cursor.u64()?;
                let dst = cursor.u64()?;
                let rows = checked_numel(cursor.u64()?)?;
                let cols = checked_numel(cursor.u64()?)?;
                if !cursor.done() {
                    return Err("tensor_sum_invalid");
                }
                let expected = rows.checked_mul(cols).ok_or("tensor_size_overflow")?;
                let values = self.tensors.get(&src).ok_or("tensor_unknown")?;
                if values.len() != expected {
                    return Err("tensor_sum_shape_mismatch");
                }
                let mut sums = vec![0.0_f32; cols];
                for row in 0..rows {
                    for col in 0..cols {
                        sums[col] += values[row * cols + col];
                    }
                }
                let output = self.tensors.get_mut(&dst).ok_or("tensor_unknown")?;
                if output.len() != cols {
                    return Err("tensor_sum_output_mismatch");
                }
                output.copy_from_slice(&sums);
                ok()
            }
            STATS => {
                if !cursor.done() {
                    return Err("tensor_stats_invalid");
                }
                let mut out = Vec::with_capacity(1 + 8 * 8);
                out.push(0);
                for value in [
                    self.tensors.len() as u64,
                    self.remote_bytes as u64,
                    self.mm_ops,
                    self.add_ops,
                    self.allocations,
                    self.uploads,
                    self.downloads,
                ] {
                    push_u64(&mut out, value);
                }
                out
            }
            _ => return Err("tensor_unknown_opcode"),
        };
        Ok(reply)
    }
}
