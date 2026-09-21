// SPDX-License-Identifier: AGPL-3.0-only

//! SSM (Gated Delta Net) kernel microbenchmarks.
//!
//! Includes causal conv1d update, gated delta rule decode, and the exact
//! Qwen3.8 WY32 prefill parent/candidate parity gate.
//! Shapes match Qwen3-Next/Qwen3.8: d_inner=8192, d_conv=4,
//! gdn_num_k=16, gdn_num_v=32, dim=128.

use std::ffi::c_void;
use std::sync::OnceLock;
use std::time::Duration;

use atlas_core::registry::RawCudaFunc;
use atlas_spark_bench::gpu;
use criterion::{Criterion, criterion_group, criterion_main};

unsafe extern "C" {
    fn cuMemcpyHtoD_v2(dst: u64, src: *const c_void, bytes: usize) -> i32;
    fn cuMemcpyDtoH_v2(dst: *mut c_void, src: u64, bytes: usize) -> i32;
}

const OUTPUT_CANARY_BYTES: usize = 4096;
const OUTPUT_CANARY_PREFIX: u8 = 0xA5;
const OUTPUT_CANARY_SUFFIX: u8 = 0x5A;
const WY32_PARENT_SMEM: u32 = 86_528;
const WY32_DOT_BATCH_SMEM: u32 = 95_232;
const _: () = assert!(WY32_DOT_BATCH_SMEM - WY32_PARENT_SMEM == 8_704);

fn h2d<T: Copy>(dev: u64, host: &[T]) {
    let bytes = std::mem::size_of_val(host);
    let status = unsafe { cuMemcpyHtoD_v2(dev, host.as_ptr().cast(), bytes) };
    assert_eq!(status, 0, "cuMemcpyHtoD failed: {status}");
}

fn d2h<T: Copy>(host: &mut [T], dev: u64) {
    let bytes = std::mem::size_of_val(host);
    let status = unsafe { cuMemcpyDtoH_v2(host.as_mut_ptr().cast(), dev, bytes) };
    assert_eq!(status, 0, "cuMemcpyDtoH failed: {status}");
}

fn guarded_alloc(stream: u64, data_bytes: usize) -> (u64, u64) {
    let total = data_bytes
        .checked_add(2 * OUTPUT_CANARY_BYTES)
        .expect("guarded allocation overflow");
    let raw = gpu::gpu_alloc_zeroed(stream, total).expect("allocate guarded buffer");
    gpu::gpu_sync(stream).expect("finish guarded buffer initialization");
    h2d(raw, &vec![OUTPUT_CANARY_PREFIX; OUTPUT_CANARY_BYTES]);
    h2d(
        raw + (OUTPUT_CANARY_BYTES + data_bytes) as u64,
        &vec![OUTPUT_CANARY_SUFFIX; OUTPUT_CANARY_BYTES],
    );
    (raw, raw + OUTPUT_CANARY_BYTES as u64)
}

fn assert_canaries(raw: u64, data_bytes: usize, label: &str) {
    let mut prefix = vec![0u8; OUTPUT_CANARY_BYTES];
    let mut suffix = vec![0u8; OUTPUT_CANARY_BYTES];
    d2h(&mut prefix, raw);
    d2h(&mut suffix, raw + (OUTPUT_CANARY_BYTES + data_bytes) as u64);
    assert!(
        prefix.iter().all(|&byte| byte == OUTPUT_CANARY_PREFIX),
        "{label} wrote before its buffer"
    );
    assert!(
        suffix.iter().all(|&byte| byte == OUTPUT_CANARY_SUFFIX),
        "{label} wrote after its buffer"
    );
}

fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325_u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    })
}

fn assert_bytes_identical(label: &str, parent: &[u8], candidate: &[u8]) {
    assert_eq!(parent.len(), candidate.len(), "{label} lengths differ");
    if parent != candidate {
        let first = parent
            .iter()
            .zip(candidate)
            .position(|(left, right)| left != right)
            .expect("different byte vectors must contain a mismatch");
        panic!(
            "{label} bitwise mismatch at byte {first}: parent=0x{:02x} candidate=0x{:02x}",
            parent[first], candidate[first]
        );
    }
}

fn deterministic_bf16(elements: usize, salt: usize) -> Vec<u16> {
    (0..elements)
        .map(|index| {
            let magnitude = 0x3d00u16 + ((index * 131 + salt * 17) % 0x180) as u16;
            if (index + salt).is_multiple_of(3) {
                magnitude | 0x8000
            } else {
                magnitude
            }
        })
        .collect()
}

fn bench_wy32_seq_len() -> u32 {
    let seq_len = match std::env::var("ATLAS_BENCH_SEQ") {
        Ok(raw) => raw
            .parse::<u32>()
            .expect("ATLAS_BENCH_SEQ must be an integer"),
        Err(std::env::VarError::NotPresent) => 256,
        Err(error) => panic!("ATLAS_BENCH_SEQ is not valid Unicode: {error}"),
    };
    assert!(
        (32..=32_768).contains(&seq_len) && seq_len.is_multiple_of(32),
        "ATLAS_BENCH_SEQ must be a multiple of 32 in [32, 32768]"
    );
    seq_len
}

static CONV1D_FN: OnceLock<RawCudaFunc> = OnceLock::new();
static GDN_DECODE_FN: OnceLock<RawCudaFunc> = OnceLock::new();
static GDN_WY32_PARENT_FN: OnceLock<RawCudaFunc> = OnceLock::new();
static GDN_WY32_DOT_BATCH_FN: OnceLock<RawCudaFunc> = OnceLock::new();

/// causal_conv1d_update(conv_state, new_input, weight, bias, output, batch, dim, d_conv)
/// Grid: (ceil(dim/256), batch, 1)  Block: (256, 1, 1)
fn bench_conv1d(c: &mut Criterion) {
    let reg = gpu::ensure_registry();
    let stream = reg.raw_stream();
    let kernel = gpu::get_kernel(reg, &CONV1D_FN, "causal_conv1d", "causal_conv1d_update");

    let batch: u32 = 1;
    let d_inner: u32 = 8192;
    let d_conv: u32 = 4;
    let elem_bytes = 2_usize;

    // conv_state: [batch, d_inner, d_conv] FP32 (4 bytes)
    let state_bytes = batch as usize * d_inner as usize * d_conv as usize * 4;
    let input_bytes = batch as usize * d_inner as usize * elem_bytes;
    let weight_bytes = d_inner as usize * d_conv as usize * 4;
    let bias_bytes = d_inner as usize * 4;
    let output_bytes = batch as usize * d_inner as usize * elem_bytes;

    let state_ptr = gpu::gpu_alloc_zeroed(stream, state_bytes).unwrap();
    let input_ptr = gpu::gpu_alloc_zeroed(stream, input_bytes).unwrap();
    let weight_ptr = gpu::gpu_alloc_zeroed(stream, weight_bytes).unwrap();
    let bias_ptr = gpu::gpu_alloc_zeroed(stream, bias_bytes).unwrap();
    let output_ptr = gpu::gpu_alloc_zeroed(stream, output_bytes).unwrap();
    gpu::gpu_sync(stream).unwrap();

    let grid_x = d_inner.div_ceil(256);

    let mut group = c.benchmark_group("conv1d");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));

    let label = format!("decode dim={d_inner}");
    group.bench_function(&label, |b| {
        b.iter_custom(|iters| {
            let ms = gpu::bench_kernel_ms(stream, 50, iters as usize, || {
                let mut params: Vec<*mut c_void> = vec![
                    &state_ptr as *const u64 as *mut c_void,
                    &input_ptr as *const u64 as *mut c_void,
                    &weight_ptr as *const u64 as *mut c_void,
                    &bias_ptr as *const u64 as *mut c_void,
                    &output_ptr as *const u64 as *mut c_void,
                    &batch as *const u32 as *mut c_void,
                    &d_inner as *const u32 as *mut c_void,
                    &d_conv as *const u32 as *mut c_void,
                ];
                unsafe {
                    gpu::launch(
                        reg,
                        kernel,
                        (grid_x, batch, 1),
                        (256, 1, 1),
                        0,
                        stream,
                        &mut params,
                    )
                    .unwrap();
                }
            });
            Duration::from_secs_f64(ms as f64 / 1000.0 * iters as f64)
        });
    });

    gpu::gpu_free(state_ptr);
    gpu::gpu_free(input_ptr);
    gpu::gpu_free(weight_ptr);
    gpu::gpu_free(bias_ptr);
    gpu::gpu_free(output_ptr);

    group.finish();
}

/// gated_delta_rule_decode(h_state, query, key, value, gate, beta, output,
///                         batch_size, num_k_heads, num_v_heads, k_dim, v_dim)
/// Grid: (num_v_heads, batch_size, 1)  Block: (128, 1, 1)
fn bench_gdn(c: &mut Criterion) {
    let reg = gpu::ensure_registry();
    let stream = reg.raw_stream();
    let kernel = gpu::get_kernel(
        reg,
        &GDN_DECODE_FN,
        "gated_delta_rule",
        "gated_delta_rule_decode",
    );

    let batch: u32 = 1;
    let num_k_heads: u32 = 16;
    let num_v_heads: u32 = 32;
    let k_dim: u32 = 128;
    let v_dim: u32 = 128;
    let elem_bytes = 2_usize;

    // h_state: [batch, num_v_heads, k_dim, v_dim] FP32
    let state_bytes = batch as usize * num_v_heads as usize * k_dim as usize * v_dim as usize * 4;
    let q_bytes = batch as usize * num_k_heads as usize * k_dim as usize * elem_bytes;
    let k_bytes = batch as usize * num_k_heads as usize * k_dim as usize * elem_bytes;
    let v_bytes = batch as usize * num_v_heads as usize * v_dim as usize * elem_bytes;
    let gate_bytes = batch as usize * num_v_heads as usize * 4; // FP32
    let beta_bytes = gate_bytes;
    let output_bytes = batch as usize * num_v_heads as usize * v_dim as usize * elem_bytes;

    let h_state_ptr = gpu::gpu_alloc_zeroed(stream, state_bytes).unwrap();
    let q_ptr = gpu::gpu_alloc_zeroed(stream, q_bytes).unwrap();
    let k_ptr = gpu::gpu_alloc_zeroed(stream, k_bytes).unwrap();
    let v_ptr = gpu::gpu_alloc_zeroed(stream, v_bytes).unwrap();
    let gate_ptr = gpu::gpu_alloc_zeroed(stream, gate_bytes).unwrap();
    let beta_ptr = gpu::gpu_alloc_zeroed(stream, beta_bytes).unwrap();
    let output_ptr = gpu::gpu_alloc_zeroed(stream, output_bytes).unwrap();
    gpu::gpu_sync(stream).unwrap();

    let mut group = c.benchmark_group("gdn");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));

    let label = format!("decode {num_v_heads}vh dim={k_dim}");
    group.bench_function(&label, |b| {
        b.iter_custom(|iters| {
            let ms = gpu::bench_kernel_ms(stream, 50, iters as usize, || {
                let mut params: Vec<*mut c_void> = vec![
                    &h_state_ptr as *const u64 as *mut c_void,
                    &q_ptr as *const u64 as *mut c_void,
                    &k_ptr as *const u64 as *mut c_void,
                    &v_ptr as *const u64 as *mut c_void,
                    &gate_ptr as *const u64 as *mut c_void,
                    &beta_ptr as *const u64 as *mut c_void,
                    &output_ptr as *const u64 as *mut c_void,
                    &batch as *const u32 as *mut c_void,
                    &num_k_heads as *const u32 as *mut c_void,
                    &num_v_heads as *const u32 as *mut c_void,
                    &k_dim as *const u32 as *mut c_void,
                    &v_dim as *const u32 as *mut c_void,
                ];
                unsafe {
                    gpu::launch(
                        reg,
                        kernel,
                        (num_v_heads, batch, 1),
                        (128, 1, 1),
                        0,
                        stream,
                        &mut params,
                    )
                    .unwrap();
                }
            });
            Duration::from_secs_f64(ms as f64 / 1000.0 * iters as f64)
        });
    });

    gpu::gpu_free(h_state_ptr);
    gpu::gpu_free(q_ptr);
    gpu::gpu_free(k_ptr);
    gpu::gpu_free(v_ptr);
    gpu::gpu_free(gate_ptr);
    gpu::gpu_free(beta_ptr);
    gpu::gpu_free(output_ptr);

    group.finish();
}

static GDN_CHUNK2_FN: OnceLock<RawCudaFunc> = OnceLock::new();

/// gated_delta_rule_chunk2 vs 2× sequential gated_delta_rule_decode.
/// Measures the kernel-level speedup of fused 2-token processing.
fn bench_gdn_chunk2(c: &mut Criterion) {
    let reg = gpu::ensure_registry();
    let stream = reg.raw_stream();
    let kernel_seq = gpu::get_kernel(
        reg,
        &GDN_DECODE_FN,
        "gated_delta_rule",
        "gated_delta_rule_decode",
    );
    let kernel_chunk2 = gpu::get_kernel(
        reg,
        &GDN_CHUNK2_FN,
        "gated_delta_rule",
        "gated_delta_rule_chunk2",
    );

    let batch: u32 = 1;
    let num_k_heads: u32 = 16;
    let num_v_heads: u32 = 32;
    let k_dim: u32 = 128;
    let v_dim: u32 = 128;
    let bf16 = 2_usize;
    let fp32 = 4_usize;

    // Shared dimensions
    let key_dim = num_k_heads as usize * k_dim as usize; // 2048
    let value_dim = num_v_heads as usize * v_dim as usize; // 4096
    let conv_dim: usize = key_dim * 2 + value_dim; // 8192

    // h_state: [batch, num_v_heads, k_dim, v_dim] FP32
    let state_bytes = batch as usize * num_v_heads as usize * k_dim as usize * v_dim as usize * 4;

    // Chunk2 layout: Q/K/V interleaved per token with stride = conv_dim
    // Layout: [2, conv_dim] = [2, Q(2048) + K(2048) + V(4096)]
    let qkv_buf_bytes = 2 * conv_dim * bf16;
    // gate+beta: [2, nv + nv] FP32
    let gb_buf_bytes = 2 * num_v_heads as usize * 2 * fp32;
    // output: [2, value_dim] BF16
    let out_buf_bytes = 2 * value_dim * bf16;

    let h_state_ptr = gpu::gpu_alloc_zeroed(stream, state_bytes).unwrap();
    let h_state_copy = gpu::gpu_alloc_zeroed(stream, state_bytes).unwrap();
    let h_inter_ptr = gpu::gpu_alloc_zeroed(stream, state_bytes).unwrap();
    let qkv_buf = gpu::gpu_alloc_zeroed(stream, qkv_buf_bytes).unwrap();
    let gb_buf = gpu::gpu_alloc_zeroed(stream, gb_buf_bytes).unwrap();
    let out_buf = gpu::gpu_alloc_zeroed(stream, out_buf_bytes).unwrap();
    gpu::gpu_sync(stream).unwrap();

    // Strides for chunk2 kernel
    let qk_stride: u32 = conv_dim as u32;
    let v_stride_val: u32 = conv_dim as u32;
    let gb_stride: u32 = num_v_heads * 2;

    // Per-token offsets for sequential path
    let q0_offset = 0_u64;
    let k0_offset = (key_dim * bf16) as u64;
    let v0_offset = (key_dim * 2 * bf16) as u64;
    let q1_offset = (conv_dim * bf16) as u64;
    let k1_offset = (conv_dim * bf16 + key_dim * bf16) as u64;
    let v1_offset = (conv_dim * bf16 + key_dim * 2 * bf16) as u64;
    let gate0_offset = 0_u64;
    let beta0_offset = (num_v_heads as usize * fp32) as u64;
    let gate1_offset = (num_v_heads as usize * 2 * fp32) as u64;
    let beta1_offset = (num_v_heads as usize * 3 * fp32) as u64;
    let out0_offset = 0_u64;
    let out1_offset = (value_dim * bf16) as u64;

    let mut group = c.benchmark_group("gdn_chunk2");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));

    // Benchmark: 2× sequential gdn_decode
    let one: u32 = 1;
    group.bench_function("sequential_2x", |b| {
        b.iter_custom(|iters| {
            let ms = gpu::bench_kernel_ms(stream, 50, iters as usize, || {
                // Token 0
                let q0 = qkv_buf + q0_offset;
                let k0 = qkv_buf + k0_offset;
                let v0 = qkv_buf + v0_offset;
                let g0 = gb_buf + gate0_offset;
                let b0 = gb_buf + beta0_offset;
                let o0 = out_buf + out0_offset;
                let mut params0: Vec<*mut c_void> = vec![
                    &h_state_ptr as *const u64 as *mut c_void,
                    &q0 as *const u64 as *mut c_void,
                    &k0 as *const u64 as *mut c_void,
                    &v0 as *const u64 as *mut c_void,
                    &g0 as *const u64 as *mut c_void,
                    &b0 as *const u64 as *mut c_void,
                    &o0 as *const u64 as *mut c_void,
                    &one as *const u32 as *mut c_void,
                    &num_k_heads as *const u32 as *mut c_void,
                    &num_v_heads as *const u32 as *mut c_void,
                    &k_dim as *const u32 as *mut c_void,
                    &v_dim as *const u32 as *mut c_void,
                ];
                unsafe {
                    gpu::launch(
                        reg,
                        kernel_seq,
                        (num_v_heads, one, 1),
                        (128, 1, 1),
                        0,
                        stream,
                        &mut params0,
                    )
                    .unwrap();
                }
                // Token 1
                let q1 = qkv_buf + q1_offset;
                let k1 = qkv_buf + k1_offset;
                let v1 = qkv_buf + v1_offset;
                let g1 = gb_buf + gate1_offset;
                let b1 = gb_buf + beta1_offset;
                let o1 = out_buf + out1_offset;
                let mut params1: Vec<*mut c_void> = vec![
                    &h_state_ptr as *const u64 as *mut c_void,
                    &q1 as *const u64 as *mut c_void,
                    &k1 as *const u64 as *mut c_void,
                    &v1 as *const u64 as *mut c_void,
                    &g1 as *const u64 as *mut c_void,
                    &b1 as *const u64 as *mut c_void,
                    &o1 as *const u64 as *mut c_void,
                    &one as *const u32 as *mut c_void,
                    &num_k_heads as *const u32 as *mut c_void,
                    &num_v_heads as *const u32 as *mut c_void,
                    &k_dim as *const u32 as *mut c_void,
                    &v_dim as *const u32 as *mut c_void,
                ];
                unsafe {
                    gpu::launch(
                        reg,
                        kernel_seq,
                        (num_v_heads, one, 1),
                        (128, 1, 1),
                        0,
                        stream,
                        &mut params1,
                    )
                    .unwrap();
                }
            });
            Duration::from_secs_f64(ms as f64 / 1000.0 * iters as f64)
        });
    });

    // Benchmark: 1× chunk2 gdn_decode
    group.bench_function("chunk2_fused", |b| {
        b.iter_custom(|iters| {
            let ms = gpu::bench_kernel_ms(stream, 50, iters as usize, || {
                let q_base = qkv_buf;
                let k_base = qkv_buf + k0_offset;
                let v_base = qkv_buf + v0_offset;
                let g_base = gb_buf;
                let b_base = gb_buf + beta0_offset;
                let mut params: Vec<*mut c_void> = vec![
                    &h_state_ptr as *const u64 as *mut c_void,
                    &q_base as *const u64 as *mut c_void,
                    &k_base as *const u64 as *mut c_void,
                    &v_base as *const u64 as *mut c_void,
                    &g_base as *const u64 as *mut c_void,
                    &b_base as *const u64 as *mut c_void,
                    &out_buf as *const u64 as *mut c_void,
                    &h_inter_ptr as *const u64 as *mut c_void,
                    &batch as *const u32 as *mut c_void,
                    &num_k_heads as *const u32 as *mut c_void,
                    &num_v_heads as *const u32 as *mut c_void,
                    &k_dim as *const u32 as *mut c_void,
                    &v_dim as *const u32 as *mut c_void,
                    &qk_stride as *const u32 as *mut c_void,
                    &v_stride_val as *const u32 as *mut c_void,
                    &gb_stride as *const u32 as *mut c_void,
                ];
                unsafe {
                    gpu::launch(
                        reg,
                        kernel_chunk2,
                        (num_v_heads, batch, 1),
                        (128, 1, 1),
                        0,
                        stream,
                        &mut params,
                    )
                    .unwrap();
                }
            });
            Duration::from_secs_f64(ms as f64 / 1000.0 * iters as f64)
        });
    });

    gpu::gpu_free(h_state_ptr);
    gpu::gpu_free(h_state_copy);
    gpu::gpu_free(h_inter_ptr);
    gpu::gpu_free(qkv_buf);
    gpu::gpu_free(gb_buf);
    gpu::gpu_free(out_buf);

    group.finish();
}

#[allow(clippy::too_many_arguments)]
fn launch_wy32(
    reg: &atlas_core::registry::AtlasRegistry,
    kernel: RawCudaFunc,
    stream: u64,
    shared_mem: u32,
    h_state: u64,
    query: u64,
    key: u64,
    value: u64,
    gate: u64,
    beta: u64,
    output: u64,
    batch_size: u32,
    seq_len: u32,
    num_k_heads: u32,
    num_v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    qk_stride: u32,
    v_stride: u32,
    gb_stride: u32,
) {
    let mut params: Vec<*mut c_void> = vec![
        &h_state as *const u64 as *mut c_void,
        &query as *const u64 as *mut c_void,
        &key as *const u64 as *mut c_void,
        &value as *const u64 as *mut c_void,
        &gate as *const u64 as *mut c_void,
        &beta as *const u64 as *mut c_void,
        &output as *const u64 as *mut c_void,
        &batch_size as *const u32 as *mut c_void,
        &seq_len as *const u32 as *mut c_void,
        &num_k_heads as *const u32 as *mut c_void,
        &num_v_heads as *const u32 as *mut c_void,
        &k_dim as *const u32 as *mut c_void,
        &v_dim as *const u32 as *mut c_void,
        &qk_stride as *const u32 as *mut c_void,
        &v_stride as *const u32 as *mut c_void,
        &gb_stride as *const u32 as *mut c_void,
    ];
    unsafe {
        gpu::launch(
            reg,
            kernel,
            (num_v_heads, batch_size, 1),
            (128, 1, 1),
            shared_mem,
            stream,
            &mut params,
        )
        .unwrap();
    }
}

/// Exact parent-vs-dot-batched WY32 prefill state/output gate and kernel timer.
///
/// The correctness run uses independent guarded state and output allocations.
/// Timing uses another guarded pair per arm so Criterion cannot alter the
/// already-proven bytes. The recurrent state evolves between timed launches,
/// but the instruction path is value-independent and no reset/copy is included
/// in the reported kernel time.
fn bench_gdn_wy32_prefill(c: &mut Criterion) {
    let reg = gpu::ensure_registry();
    let stream = reg.raw_stream();
    let parent_kernel = gpu::get_kernel(
        reg,
        &GDN_WY32_PARENT_FN,
        "gated_delta_rule_wy64_prefill",
        "gated_delta_rule_prefill_wy64",
    );
    let dot_batch_kernel = gpu::get_kernel(
        reg,
        &GDN_WY32_DOT_BATCH_FN,
        "gated_delta_rule_wy32_gatecache",
        "gated_delta_rule_prefill_wy32_gatecache",
    );

    let batch_size = 1u32;
    let seq_len = bench_wy32_seq_len();
    let num_k_heads = 16u32;
    let num_v_heads = 32u32;
    let k_dim = 128u32;
    let v_dim = 128u32;
    let qk_stride = num_k_heads * k_dim;
    let v_stride = num_v_heads * v_dim;
    let gb_stride = num_v_heads;

    let state_elements = num_v_heads as usize * k_dim as usize * v_dim as usize;
    let qk_elements = seq_len as usize * qk_stride as usize;
    let value_elements = seq_len as usize * v_stride as usize;
    let gate_elements = seq_len as usize * gb_stride as usize;
    let state_bytes = state_elements * 4;
    let output_bytes = value_elements * 2;

    let host_state: Vec<f32> = (0..state_elements)
        .map(|index| ((index * 73 % 2001) as i32 - 1000) as f32 * 0.00001)
        .collect();
    let host_query = deterministic_bf16(qk_elements, 11);
    let host_key = deterministic_bf16(qk_elements, 29);
    let host_value = deterministic_bf16(value_elements, 47);
    let host_gate: Vec<f32> = (0..gate_elements)
        .map(|index| 0.90 + (index % 17) as f32 * 0.001)
        .collect();
    let host_beta: Vec<f32> = (0..gate_elements)
        .map(|index| 0.05 + (index % 13) as f32 * 0.0005)
        .collect();

    let query = gpu::gpu_alloc_zeroed(stream, qk_elements * 2).unwrap();
    let key = gpu::gpu_alloc_zeroed(stream, qk_elements * 2).unwrap();
    let value = gpu::gpu_alloc_zeroed(stream, value_elements * 2).unwrap();
    let gate = gpu::gpu_alloc_zeroed(stream, gate_elements * 4).unwrap();
    let beta = gpu::gpu_alloc_zeroed(stream, gate_elements * 4).unwrap();
    h2d(query, &host_query);
    h2d(key, &host_key);
    h2d(value, &host_value);
    h2d(gate, &host_gate);
    h2d(beta, &host_beta);

    let (parent_state_raw, parent_state) = guarded_alloc(stream, state_bytes);
    let (candidate_state_raw, candidate_state) = guarded_alloc(stream, state_bytes);
    let (parent_output_raw, parent_output) = guarded_alloc(stream, output_bytes);
    let (candidate_output_raw, candidate_output) = guarded_alloc(stream, output_bytes);
    let (timed_parent_state_raw, timed_parent_state) = guarded_alloc(stream, state_bytes);
    let (timed_candidate_state_raw, timed_candidate_state) = guarded_alloc(stream, state_bytes);
    let (timed_parent_output_raw, timed_parent_output) = guarded_alloc(stream, output_bytes);
    let (timed_candidate_output_raw, timed_candidate_output) = guarded_alloc(stream, output_bytes);
    for state in [
        parent_state,
        candidate_state,
        timed_parent_state,
        timed_candidate_state,
    ] {
        h2d(state, &host_state);
    }
    gpu::gpu_sync(stream).unwrap();

    launch_wy32(
        reg,
        parent_kernel,
        stream,
        WY32_PARENT_SMEM,
        parent_state,
        query,
        key,
        value,
        gate,
        beta,
        parent_output,
        batch_size,
        seq_len,
        num_k_heads,
        num_v_heads,
        k_dim,
        v_dim,
        qk_stride,
        v_stride,
        gb_stride,
    );
    launch_wy32(
        reg,
        dot_batch_kernel,
        stream,
        WY32_DOT_BATCH_SMEM,
        candidate_state,
        query,
        key,
        value,
        gate,
        beta,
        candidate_output,
        batch_size,
        seq_len,
        num_k_heads,
        num_v_heads,
        k_dim,
        v_dim,
        qk_stride,
        v_stride,
        gb_stride,
    );
    gpu::gpu_sync(stream).unwrap();

    let mut parent_state_bytes = vec![0u8; state_bytes];
    let mut candidate_state_bytes = vec![0u8; state_bytes];
    let mut parent_output_bytes = vec![0u8; output_bytes];
    let mut candidate_output_bytes = vec![0u8; output_bytes];
    d2h(&mut parent_state_bytes, parent_state);
    d2h(&mut candidate_state_bytes, candidate_state);
    d2h(&mut parent_output_bytes, parent_output);
    d2h(&mut candidate_output_bytes, candidate_output);
    assert_bytes_identical(
        "WY32 final H state",
        &parent_state_bytes,
        &candidate_state_bytes,
    );
    assert_bytes_identical(
        "WY32 BF16 output",
        &parent_output_bytes,
        &candidate_output_bytes,
    );
    for (raw, bytes, label) in [
        (parent_state_raw, state_bytes, "parent state"),
        (candidate_state_raw, state_bytes, "candidate state"),
        (parent_output_raw, output_bytes, "parent output"),
        (candidate_output_raw, output_bytes, "candidate output"),
    ] {
        assert_canaries(raw, bytes, label);
    }
    eprintln!(
        "[gdn_wy32 validation] M={seq_len} state=fnv1a64:{:016x} output=fnv1a64:{:016x} bitwise=PASS canaries=PASS",
        fnv1a(&parent_state_bytes),
        fnv1a(&parent_output_bytes)
    );

    let mut group = c.benchmark_group("gdn_wy32_prefill");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));
    group.bench_function(format!("parent_M{seq_len}"), |b| {
        b.iter_custom(|iters| {
            let ms = gpu::bench_kernel_ms(stream, 1, iters as usize, || {
                launch_wy32(
                    reg,
                    parent_kernel,
                    stream,
                    WY32_PARENT_SMEM,
                    timed_parent_state,
                    query,
                    key,
                    value,
                    gate,
                    beta,
                    timed_parent_output,
                    batch_size,
                    seq_len,
                    num_k_heads,
                    num_v_heads,
                    k_dim,
                    v_dim,
                    qk_stride,
                    v_stride,
                    gb_stride,
                );
            });
            Duration::from_secs_f64(ms as f64 / 1000.0 * iters as f64)
        });
    });
    group.bench_function(format!("dot_batch_M{seq_len}"), |b| {
        b.iter_custom(|iters| {
            let ms = gpu::bench_kernel_ms(stream, 1, iters as usize, || {
                launch_wy32(
                    reg,
                    dot_batch_kernel,
                    stream,
                    WY32_DOT_BATCH_SMEM,
                    timed_candidate_state,
                    query,
                    key,
                    value,
                    gate,
                    beta,
                    timed_candidate_output,
                    batch_size,
                    seq_len,
                    num_k_heads,
                    num_v_heads,
                    k_dim,
                    v_dim,
                    qk_stride,
                    v_stride,
                    gb_stride,
                );
            });
            Duration::from_secs_f64(ms as f64 / 1000.0 * iters as f64)
        });
    });
    group.finish();

    gpu::gpu_sync(stream).unwrap();
    for (raw, bytes, label) in [
        (timed_parent_state_raw, state_bytes, "timed parent state"),
        (
            timed_candidate_state_raw,
            state_bytes,
            "timed candidate state",
        ),
        (timed_parent_output_raw, output_bytes, "timed parent output"),
        (
            timed_candidate_output_raw,
            output_bytes,
            "timed candidate output",
        ),
    ] {
        assert_canaries(raw, bytes, label);
    }

    for raw in [
        parent_state_raw,
        candidate_state_raw,
        parent_output_raw,
        candidate_output_raw,
        timed_parent_state_raw,
        timed_candidate_state_raw,
        timed_parent_output_raw,
        timed_candidate_output_raw,
    ] {
        gpu::gpu_free(raw);
    }
    for input in [query, key, value, gate, beta] {
        gpu::gpu_free(input);
    }
}

criterion_group!(
    benches,
    bench_conv1d,
    bench_gdn,
    bench_gdn_chunk2,
    bench_gdn_wy32_prefill
);
criterion_main!(benches);
