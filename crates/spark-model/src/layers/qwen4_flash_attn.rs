// SPDX-License-Identifier: AGPL-3.0-only

//! Flash-Attention prefill core for Qwen3.8-Flash-Next (`ATLAS_QWEN4_PREFILL_ATTN_FLASH=1`).
//!
//! The shipping prefill runs the attention core through the PAGED DECODE kernel,
//! once per 16-row tile, presenting each row as its own "sequence" whose length
//! is its causal prefix. That is correct but it is a decode-shaped kernel doing
//! prefill work: 1536 launches per prefill and ~2.3 TFLOP/s of scalar FP32 math.
//!
//! Prefill attention here is plain dense causal: QSA maintains a ring for later
//! decode but does not select keys during prefill, and the block table handed to
//! the paged kernel is the sequential one. So a Flash-Attention-2 kernel computes
//! the same function. `inferspark_prefill` (BF16 tensor cores, HDIM=256, GQA
//! aware, causal) is already compiled into this target and already wrapped by
//! `ops::prefill_attention`; all that is missing is the contiguous layout it
//! wants, which `qwen4_attn_compact_qkv` produces from the fast projection's
//! strided QKV rows.
//!
//! This is a NUMERICS CHANGE and is default off: online (streaming) softmax with
//! tensor-core accumulation replaces the paged kernel's per-row full softmax, so
//! the reduction order differs. It does not change the operand precision — both
//! paths consume the same BF16 Q/K/V — and it is gated by the same teacher-forced
//! comparison as the other fast-arm levers.

use std::sync::{Mutex, OnceLock};

use anyhow::{Context, Result};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

pub const SELECTOR: &str = "ATLAS_QWEN4_PREFILL_ATTN_FLASH";

/// Selected variant: 0 = off, 1 = BR=32 core, 2 = BR=64 core.
///
/// The BR=64 twin halves the CTA count and the number of causal KV iterations
/// per CTA; it is the same algorithm and the same online-softmax order per row,
/// with 256 threads and 90,112 B of static shared memory (GB10's ceiling is
/// 101,376 B).
pub fn level() -> u32 {
    static S: OnceLock<u32> = OnceLock::new();
    *S.get_or_init(|| match std::env::var(SELECTOR).ok().as_deref() {
        Some("1") => 1,
        Some("2") => 2,
        _ => 0,
    })
}

/// True when the flash prefill core is selected.
pub fn selected() -> bool {
    level() > 0
}

fn br64_kernel(gpu: &dyn GpuBackend) -> Result<KernelHandle> {
    static K: OnceLock<Result<KernelHandle, String>> = OnceLock::new();
    K.get_or_init(|| {
        gpu.kernel("inferspark_prefill", "inferspark_prefill_64")
            .map_err(|e| e.to_string())
    })
    .clone()
    .map_err(|e| anyhow::anyhow!("{SELECTOR}: BR=64 flash kernel unavailable: {e}"))
}

fn compact_kernel(gpu: &dyn GpuBackend) -> Result<KernelHandle> {
    static K: OnceLock<Result<KernelHandle, String>> = OnceLock::new();
    K.get_or_init(|| {
        gpu.kernel("qwen4_attn_compact_qkv", "qwen4_attn_compact_qkv")
            .map_err(|e| e.to_string())
    })
    .clone()
    .map_err(|e| anyhow::anyhow!("{SELECTOR}: compaction kernel unavailable: {e}"))
}

/// Grow-only scratch holding contiguous Q, K and V for the whole prefill.
///
/// One allocation serves every attention layer; all callers run on the single
/// prefill stream, so reuse is stream ordered. The old block is leaked on a grow
/// step rather than freed while it may still be in flight.
fn scratch(gpu: &dyn GpuBackend, bytes: usize) -> Result<DevicePtr> {
    static S: Mutex<Option<(DevicePtr, usize)>> = Mutex::new(None);
    let mut guard = S
        .lock()
        .map_err(|_| anyhow::anyhow!("{SELECTOR}: scratch lock poisoned"))?;
    if let Some((ptr, cap)) = *guard
        && cap >= bytes
    {
        return Ok(ptr);
    }
    let cap = bytes.next_multiple_of(1 << 20);
    let ptr = gpu
        .alloc(cap)
        .with_context(|| format!("{SELECTOR}: alloc {cap} B flash scratch"))?;
    tracing::info!("QWEN4_FLASH_ATTN scratch allocated bytes={cap}");
    *guard = Some((ptr, cap));
    Ok(ptr)
}

/// Run the whole prefill attention core in one Flash-Attention launch.
///
/// `qkv` holds RoPE'd Q/gate/K/V for every row at `row_stride` elements per row;
/// the caller has already written K/V into the paged cache and updated QSA.
/// `output` receives `[num_tokens][nq*hd]` BF16 at `out_row_elems` stride, which
/// must equal `nq*hd` (the fast path's raw attention buffer is contiguous).
#[allow(clippy::too_many_arguments)]
pub fn run(
    gpu: &dyn GpuBackend,
    flash_kernel: KernelHandle,
    qkv: DevicePtr,
    row_stride: usize,
    k_offset: usize,
    v_offset: usize,
    output: DevicePtr,
    out_row_elems: usize,
    num_tokens: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    inv_sqrt_d: f32,
    stream: u64,
) -> Result<()> {
    let q_elems = num_q_heads * head_dim;
    let kv_elems = num_kv_heads * head_dim;
    anyhow::ensure!(
        out_row_elems == q_elems,
        "{SELECTOR}: flash core needs a contiguous [tokens, nq*hd] output \
         (row {out_row_elems} elems, expected {q_elems})"
    );
    anyhow::ensure!(
        head_dim == 256,
        "{SELECTOR}: inferspark_prefill in this target is compiled with HDIM=256 \
         (got {head_dim})"
    );
    anyhow::ensure!(
        num_kv_heads > 0 && num_q_heads.is_multiple_of(num_kv_heads),
        "{SELECTOR}: GQA ratio {num_q_heads}/{num_kv_heads} is not integral"
    );

    // Contiguous Q | K | V, laid out back to back in one allocation.
    let q_bytes = num_tokens * q_elems * 2;
    let kv_bytes = num_tokens * kv_elems * 2;
    let base = scratch(gpu, q_bytes + 2 * kv_bytes)?;
    let q_c = base;
    let k_c = base.offset(q_bytes);
    let v_c = k_c.offset(kv_bytes);

    KernelLaunch::new(gpu, compact_kernel(gpu)?)
        .grid([24, num_tokens as u32, 1])
        .block([256, 1, 1])
        .arg_ptr(qkv)
        .arg_ptr(q_c)
        .arg_ptr(k_c)
        .arg_ptr(v_c)
        .arg_u32(num_tokens as u32)
        .arg_u32(row_stride as u32)
        .arg_u32(q_elems as u32)
        .arg_u32(kv_elems as u32)
        .arg_u32(k_offset as u32)
        .arg_u32(v_offset as u32)
        .launch(stream)?;

    if level() == 2 {
        crate::layers::ops::prefill_attention_64(
            gpu,
            br64_kernel(gpu)?,
            q_c,
            k_c,
            v_c,
            output,
            num_tokens as u32,
            1,
            num_q_heads as u32,
            num_kv_heads as u32,
            head_dim as u32,
            inv_sqrt_d,
            true,
            0,
            stream,
        )?;
    } else {
        crate::layers::ops::prefill_attention(
            gpu,
            flash_kernel,
            q_c,
            k_c,
            v_c,
            output,
            num_tokens as u32,
            1,
            num_q_heads as u32,
            num_kv_heads as u32,
            head_dim as u32,
            inv_sqrt_d,
            true,
            0,
            stream,
        )?;
    }

    static LOGGED: OnceLock<()> = OnceLock::new();
    LOGGED.get_or_init(|| {
        tracing::warn!(
            "ATTN_PREFILL_FLASH_ENGAGED selector={SELECTOR} rows={num_tokens} \
             heads={num_q_heads}/{num_kv_heads} hd={head_dim} \
             core=inferspark_prefill_bf16_tensorcore_non_bit_exact level={}", level()
        );
    });
    Ok(())
}
