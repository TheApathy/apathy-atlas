// SPDX-License-Identifier: AGPL-3.0-only

//! Isolated throughput probe for the exact NVFP4 attention QKV projections,
//! with the LM-head kernel as an in-process control.
//!
//! WHY THIS EXISTS. The `attn_qkv_proj` profile scope reports 14.2 ms/step
//! against a 2.85 ms DRAM weight floor (5.0x off). The LM-head runs the SAME
//! kernel shape — same 256-thread block, same 4 outputs per block, same 64
//! lanes per output, same K1 lane ownership, same MAX_M=17 accumulator bank,
//! and per ptxas the same register footprint with no spills (76 vs 72) — yet
//! profiles at 1.4x its floor. Every structural explanation for a 3.6x gap
//! has been eliminated on paper: load width (0.61% instruction delta), launch
//! overhead (CUDA graphs save only 2.55 ms across all of verify), wave/tail
//! utilization (12.8 waves for Q), activation re-read traffic (both ~10 GB per
//! step), and register pressure.
//!
//! So either the kernel really is slow in isolation, or the 14.2 ms is an
//! attribution/contention artifact and the projections are already near their
//! true floor. This probe answers that with both kernels timed back to back in
//! one process, on one stream, with no engine around them.
//!
//! READ THE OUTPUT LIKE THIS:
//!   * attention within ~1.5x of the LM-head's bytes/s  -> the kernel is fine;
//!     the 14.2 ms is in-situ contention or misattribution, and the remaining
//!     time is NOT recoverable by rewriting this kernel.
//!   * attention ~3x slower per byte than the LM-head  -> the kernel is
//!     genuinely slow and the gap is real; hunt inside it with a live signal.
//!
//! Run on a GPU box (needs ~250 MB of device memory):
//! ```text
//! ATLAS_TARGET_MODEL=qwen3.8-27b ATLAS_TARGET_QUANT=nvfp4 \
//!   cargo run --release -p spark-model --features cuda \
//!   --example w4a16_attention_qkv_throughput_probe
//! ```

#[allow(dead_code)]
#[path = "w4a16_exact_lm_head_microtest/data.rs"]
mod data;

use std::time::Instant;

use anyhow::{Context, Result};
use data::{Fixture, as_le_bytes, random_fixture};
use spark_model::layers::ops::{self, W4a16ExactAttentionKernels, W4a16ExactLmHeadKernels};
use spark_model::weight_map::QuantizedWeight;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

/// Production Qwen3.8-27B full-attention geometry.
const HIDDEN: usize = 5_120;
const NQ: usize = 24;
const NKV: usize = 4;
const HD: usize = 256;
const Q_PROJ_DIM: usize = 2 * NQ * HD; // 12288, gated
const KV_DIM: usize = NKV * HD; // 1024
const ROWS: usize = 17; // K=17 DFlash verify width

/// LM-head control width. Smaller than the real 248320-row vocabulary so the
/// fixture stays a couple hundred MB; at 16384 blocks it is still ~68 waves,
/// far from tail-limited, so bytes/s remains comparable.
const LM_HEAD_N: usize = 65_536;

const ITERS: usize = 50;
/// 4 output rows per block, matching OUTS_PER_BLOCK in the kernels.
const OUTS_PER_BLOCK: usize = 4;

/// Weight bytes a projection of `n` output rows streams from DRAM: NVFP4
/// packed at K/2 bytes per row plus one FP8 scale per 16-element group.
fn weight_bytes(n: usize) -> usize {
    n * (HIDDEN / 2 + HIDDEN / 16)
}

/// Activation bytes the launch pulls through L2. Every block re-reads the
/// whole [ROWS, HIDDEN] BF16 activation tile once, so this scales with the
/// block count, not with the tile size.
fn activation_bytes(n: usize) -> usize {
    n.div_ceil(OUTS_PER_BLOCK) * ROWS * HIDDEN * 2
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len().max(1))?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}

fn upload_weight(gpu: &dyn GpuBackend, fixture: &Fixture) -> Result<QuantizedWeight> {
    Ok(QuantizedWeight {
        weight: upload(gpu, &fixture.packed)?,
        weight_scale: upload(gpu, &fixture.scales)?,
        weight_scale_2: 1.0,
        input_scale: DevicePtr::NULL,
    })
}

/// Median-free mean over `ITERS` launches after one warm-up. Wall clock around
/// a stream sync — the backend exposes no elapsed-time event API, so the
/// iteration count carries the accuracy.
fn bench(gpu: &dyn GpuBackend, stream: u64, mut launch: impl FnMut() -> Result<()>) -> Result<f64> {
    launch()?;
    gpu.synchronize(stream)?;
    let start = Instant::now();
    for _ in 0..ITERS {
        launch()?;
    }
    gpu.synchronize(stream)?;
    Ok(start.elapsed().as_secs_f64() * 1e3 / ITERS as f64)
}

fn report(label: &str, ms: f64, n: usize) -> f64 {
    let w = weight_bytes(n) as f64;
    let a = activation_bytes(n) as f64;
    let weight_gbs = w / (ms * 1e-3) / 1e9;
    let l2_gbs = a / (ms * 1e-3) / 1e9;
    println!(
        "  {label:<22} N={n:>6}  blocks={:>6}  {ms:>7.3} ms/launch  \
         weights {:>6.1} MB @ {weight_gbs:>6.1} GB/s  \
         activations {:>7.1} MB @ {l2_gbs:>7.1} GB/s (L2)",
        n.div_ceil(OUTS_PER_BLOCK),
        w / 1e6,
        a / 1e6,
    );
    weight_gbs
}

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())
        .context("initialize CUDA backend with compiled Qwen3.8 kernels")?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.create_stream()?;

    let attn = W4a16ExactAttentionKernels::new(
        gpu.kernel("w4a16_gemv_exact_attention", "w4a16_gemv_qg_exact_m17")?,
        gpu.kernel("w4a16_gemv_exact_attention", "w4a16_gemv_dual_kv_exact_m17")?,
    );
    let lm_head = W4a16ExactLmHeadKernels::new(
        gpu.kernel("w4a16_gemv", "w4a16_gemv_batch_logits_exact_m4")?,
        gpu.kernel("w4a16_gemv", "w4a16_gemv_batch_logits_exact_m8")?,
        gpu.kernel("w4a16_gemv", "w4a16_gemv_batch_logits_exact_m17")?,
        gpu.kernel("w4a16_gemv", "w4a16_gemv_batch_logits_exact_m32")?,
    );

    println!(
        "isolated throughput, M={ROWS} K={HIDDEN}, {ITERS} launches each \
         (no engine, one stream)\n"
    );

    let q_fix = random_fixture(ROWS, Q_PROJ_DIM, HIDDEN, 0x51c0_0000_c000_0011);
    let k_fix = random_fixture(1, KV_DIM, HIDDEN, 0x51c0_0000_0400_0012);
    let v_fix = random_fixture(1, KV_DIM, HIDDEN, 0x51c0_0000_0400_0013);
    let input = upload(gpu, &as_le_bytes(&q_fix.activations))?;
    let (q_w, k_w, v_w) = (
        upload_weight(gpu, &q_fix)?,
        upload_weight(gpu, &k_fix)?,
        upload_weight(gpu, &v_fix)?,
    );
    let q_out = gpu.alloc(ROWS * Q_PROJ_DIM * size_of::<u16>())?;
    let k_out = gpu.alloc(ROWS * KV_DIM * size_of::<u16>())?;
    let v_out = gpu.alloc(ROWS * KV_DIM * size_of::<u16>())?;

    let q_ms = bench(gpu, stream, || {
        ops::w4a16_gemv_qg_exact(
            gpu,
            attn.qg_for_rows(ROWS),
            input,
            &q_w,
            q_out,
            ROWS as u32,
            Q_PROJ_DIM as u32,
            HIDDEN as u32,
            NQ as u32,
            HD as u32,
            Q_PROJ_DIM as u32,
            stream,
        )
    })?;
    let q_gbs = report("attention gated-Q", q_ms, Q_PROJ_DIM);

    // The dual kernel does K and V in one launch via grid.z, so it streams two
    // KV_DIM weight matrices; charge it both.
    let kv_ms = bench(gpu, stream, || {
        ops::w4a16_gemv_dual_kv_exact(
            gpu,
            attn.dual_kv_for_rows(ROWS),
            input,
            &k_w,
            k_out,
            &v_w,
            v_out,
            ROWS as u32,
            KV_DIM as u32,
            HIDDEN as u32,
            KV_DIM as u32,
            stream,
        )
    })?;
    let kv_gbs = report("attention dual-K/V", kv_ms, 2 * KV_DIM);

    for ptr in [q_out, k_out, v_out] {
        gpu.free(ptr)?;
    }

    let lm_fix = random_fixture(ROWS, LM_HEAD_N, HIDDEN, 0x51c0_0000_1000_0014);
    let lm_w = upload_weight(gpu, &lm_fix)?;
    let lm_out = gpu.alloc(ROWS * LM_HEAD_N * size_of::<u16>())?;
    let lm_ms = bench(gpu, stream, || {
        ops::w4a16_gemv_batch_logits_exact_with(
            gpu,
            lm_head,
            input,
            &lm_w,
            lm_out,
            ROWS as u32,
            LM_HEAD_N as u32,
            HIDDEN as u32,
            stream,
            false,
        )
    })?;
    let lm_gbs = report("lm_head control", lm_ms, LM_HEAD_N);

    let per_layer_ms = q_ms + kv_ms;
    println!(
        "\nper full-attention layer: {per_layer_ms:.3} ms  \
         x16 layers = {:.2} ms/step (profile attributes 14.2 ms)",
        per_layer_ms * 16.0
    );
    println!(
        "attention-vs-lm_head bytes/s ratio: Q {:.2}x, K/V {:.2}x  \
         (1.0 = same efficiency; ~0.3 = attention genuinely 3x slower)",
        q_gbs / lm_gbs,
        kv_gbs / lm_gbs,
    );
    println!(
        "\nVERDICT: if x16 lands near 14.2 ms the kernel owns the cost and is \
         worth rewriting; if it lands near 4-5 ms the kernel is at its floor \
         and the profile gap is contention or misattribution."
    );
    Ok(())
}
