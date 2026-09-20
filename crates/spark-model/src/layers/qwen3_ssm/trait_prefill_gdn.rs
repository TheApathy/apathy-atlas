// SPDX-License-Identifier: AGPL-3.0-only

//! prefill_gdn_full.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use super::*;

/// Per-call gdn_prefill profile (gated by ATLAS_GDN_PROFILE=1).
/// Aggregates total invocations + total nanoseconds + bucketed total per
/// "size class" (total token count) so we can separate small chunks from
/// the dominating large prefill chunk.
static GDN_CALLS: AtomicU64 = AtomicU64::new(0);
static GDN_TOTAL_NS: AtomicU64 = AtomicU64::new(0);
static GDN_LARGE_CALLS: AtomicU64 = AtomicU64::new(0);
static GDN_LARGE_NS: AtomicU64 = AtomicU64::new(0);
static GDN_MAX_NS: AtomicU64 = AtomicU64::new(0);
static GDN_MAX_TOTAL: AtomicU64 = AtomicU64::new(0);

#[inline]
fn gdn_profile_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| std::env::var("ATLAS_GDN_PROFILE").ok().as_deref() == Some("1"))
}

pub(super) fn use_wy32_gatecache(
    requested: bool,
    has_kernel: bool,
    total_tokens: u32,
    key_dim: usize,
    value_dim: usize,
) -> Result<bool, &'static str> {
    let eligible = requested && total_tokens > 32 && key_dim == 128 && value_dim == 128;
    if eligible && !has_kernel {
        return Err(
            "ATLAS_GDN_PREFILL_GATECACHE=1 requires gated_delta_rule_prefill_wy32_gatecache",
        );
    }
    Ok(eligible)
}

pub(super) fn log_wy32_gatecache_engaged(total_tokens: u32) {
    static GATECACHE_SEEN: std::sync::Once = std::sync::Once::new();
    GATECACHE_SEEN.call_once(|| {
        tracing::info!(
            "ATLAS_GDN_PREFILL_GATECACHE engaged: WY32 cached gate products for M={total_tokens}"
        );
    });
}

const WY32_CHUNK_SIZE: usize = 32;
const WY32_WARP_COUNT: usize = 4;
const WY32_DOT_PAIR_COUNT: usize = WY32_CHUNK_SIZE * (WY32_CHUNK_SIZE - 1) / 2;

pub(super) fn wy32_dynamic_smem_bytes(
    key_dim: usize,
    value_dim: usize,
    dot_batched: bool,
) -> Option<u32> {
    let terms = [
        key_dim.checked_mul(value_dim)?.checked_mul(4)?,
        WY32_CHUNK_SIZE.checked_mul(key_dim)?.checked_mul(2)?,
        WY32_CHUNK_SIZE.checked_mul(key_dim)?.checked_mul(2)?,
        WY32_WARP_COUNT.checked_mul(4)?,
        WY32_CHUNK_SIZE
            .checked_mul(WY32_CHUNK_SIZE)?
            .checked_mul(4)?,
        WY32_CHUNK_SIZE.checked_mul(4)?,
        WY32_CHUNK_SIZE.checked_mul(4)?,
        if dot_batched {
            WY32_DOT_PAIR_COUNT
                .checked_mul(WY32_WARP_COUNT)?
                .checked_mul(4)?
        } else {
            0
        },
        if dot_batched {
            WY32_DOT_PAIR_COUNT.checked_mul(2)?
        } else {
            0
        },
    ];
    let bytes = terms
        .into_iter()
        .try_fold(0usize, |sum, term| sum.checked_add(term))?;
    let aligned = bytes.div_ceil(256).checked_mul(256)?;
    aligned.try_into().ok()
}

/// Dynamic smem for the v2 gate-cache shadow: parent layout + 16 B alignment
/// slack + two FP32 [key_dim][36] transposed chunk copies.
pub(super) fn wy32_gatecache_v2_dynamic_smem_bytes(key_dim: usize, value_dim: usize) -> Option<u32> {
    let base = wy32_dynamic_smem_bytes(key_dim, value_dim, true)? as usize;
    let extra = 16usize.checked_add(2usize.checked_mul(key_dim)?.checked_mul(36)?.checked_mul(4)?)?;
    let bytes = base.checked_add(extra)?;
    (bytes.div_ceil(256) * 256).try_into().ok()
}

/// Selects the v2 kernel handle + smem when requested and available.
pub(super) fn gatecache_kernel_choice(
    v2_requested: bool,
    v1: spark_runtime::gpu::KernelHandle,
    v2: spark_runtime::gpu::KernelHandle,
    smem_v1: u32,
    smem_v2: u32,
) -> Result<(spark_runtime::gpu::KernelHandle, u32, bool), &'static str> {
    if !v2_requested {
        return Ok((v1, smem_v1, false));
    }
    if v2.0 == 0 {
        return Err("ATLAS_GDN_PREFILL_GATECACHE_V2=1 requires gated_delta_rule_prefill_wy32_gatecache_v2");
    }
    Ok((v2, smem_v2, true))
}

/// Public dumper used from the server shutdown / bench script if needed.
#[allow(dead_code)]
pub fn dump_gdn_profile() {
    let n = GDN_CALLS.load(Ordering::Relaxed);
    let ns = GDN_TOTAL_NS.load(Ordering::Relaxed);
    let nl = GDN_LARGE_CALLS.load(Ordering::Relaxed);
    let nls = GDN_LARGE_NS.load(Ordering::Relaxed);
    let mx = GDN_MAX_NS.load(Ordering::Relaxed);
    let mxt = GDN_MAX_TOTAL.load(Ordering::Relaxed);
    if n == 0 {
        return;
    }
    let per = ns / n.max(1);
    let per_l = if nl > 0 { nls / nl } else { 0 };
    tracing::info!(
        "GDN_PROF total_calls={n} total_us={} avg_per_call_us={} large_calls={nl} \
         large_total_us={} large_avg_us={} max_us={} max_total={}",
        ns / 1000,
        per / 1000,
        nls / 1000,
        per_l / 1000,
        mx / 1000,
        mxt
    );
}

impl Qwen3SsmLayer {
    pub(super) fn prefill_gdn_full_inner(
        &self,
        state: &mut dyn LayerState,
        gdn_bufs: &GdnPrefillBuffers,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let ssm_state = state
            .as_any_mut()
            .downcast_mut::<SsmLayerState>()
            .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState"))?;

        let nk = ctx.config.linear_num_key_heads;
        let kd = ctx.config.linear_key_head_dim;
        let nv = ctx.config.linear_num_value_heads;
        let vd = ctx.config.linear_value_head_dim;
        let key_dim = nk * kd;
        let value_dim = nv * vd;
        let conv_dim = key_dim * 2 + value_dim;
        let bf16 = 2usize;
        let fp32 = 4usize;

        let total = gdn_bufs.total_len as u32;
        let gdn_c143 = self.preflight_gdn_c143_prefill(
            ctx,
            gdn_bufs.total_len,
            nk,
            nv,
            kd,
            vd,
            conv_dim,
            nv * 2,
            ctx.config.ssm_qkvz_size(),
            ssm_state.h_state,
            gdn_bufs.qkv,
            gdn_bufs.gate_beta,
            gdn_bufs.output,
            stream,
        )?;

        let __gdn_prof = gdn_profile_enabled();
        let __gdn_t0 = if __gdn_prof {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        // Packed QKV layout: Q at offset 0, K at key_dim, V at key_dim*2
        // Strides: qk_stride = conv_dim, v_stride = conv_dim (elements, not bytes)
        let q_ptr = gdn_bufs.qkv;
        let k_ptr = gdn_bufs.qkv.offset(key_dim * bf16);
        let v_ptr = gdn_bufs.qkv.offset(key_dim * 2 * bf16);

        // Gate/beta: interleaved [total_len, 2*nv] FP32
        let gate_ptr = gdn_bufs.gate_beta;
        let beta_ptr = gdn_bufs.gate_beta.offset(nv * fp32);
        let gb_stride = (nv * 2) as u32;

        // WY32 persistent: processes 32 tokens per WY iteration with H in
        // shared memory (~84KB). ~30× faster than per-token for 14k+ sequences.
        // Falls through to WY4 or sub-chunked persistent for shorter sequences.
        // Silenced per-call info log; instrumentation lives in trait_prefill.rs.
        // SMEM layout: H[kd*vd]FP32 + smem_k[32*kd]BF16 + smem_q[32*kd]BF16
        //   + smem_warp[4]FP32 + smem_kd[32*32]FP32
        //   + smem_g[32]FP32 + smem_bt[32]FP32. The gate-cache shadow adds
        //   dot_partials[496][4]FP32; each route is rounded to 256 B alignment.
        let wy32_smem = wy32_dynamic_smem_bytes(kd, vd, false)
            .ok_or_else(|| anyhow::anyhow!("WY32 dynamic shared-memory size overflow"))?;
        let wy32_gatecache_smem = wy32_dynamic_smem_bytes(kd, vd, true)
            .ok_or_else(|| anyhow::anyhow!("WY32 gate-cache shared-memory size overflow"))?;
        let gatecache = use_wy32_gatecache(
            crate::layers::gdn_prefill_gatecache_enabled(),
            self.gdn_prefill_wy32_gatecache_k.0 != 0,
            total,
            kd,
            vd,
        )
        .map_err(anyhow::Error::msg)?;
        if let Some(gdn_c143) = gdn_c143.as_ref() {
            gdn_c143.launch(ctx.gpu)?;
        } else if gatecache {
            log_wy32_gatecache_engaged(total);
            let smem_v2 = wy32_gatecache_v2_dynamic_smem_bytes(kd, vd)
                .ok_or_else(|| anyhow::anyhow!("WY32 gate-cache v2 shared-memory size overflow"))?;
            let (gatecache_k, wy32_gatecache_smem, is_v2) = gatecache_kernel_choice(
                crate::layers::gdn_prefill_gatecache_v2_enabled(),
                self.gdn_prefill_wy32_gatecache_k,
                self.gdn_prefill_wy32_gatecache_v2_k,
                wy32_gatecache_smem,
                smem_v2,
            )
            .map_err(anyhow::Error::msg)?;
            if is_v2 {
                static V2_SEEN: std::sync::Once = std::sync::Once::new();
                V2_SEEN.call_once(|| tracing::info!("ENGAGED ATLAS_GDN_PREFILL_GATECACHE_V2: M={total} smem={wy32_gatecache_smem}"));
            }
            ops::gdn_prefill_persistent_smem(
                ctx.gpu,
                gatecache_k,
                ssm_state.h_state,
                q_ptr,
                k_ptr,
                v_ptr,
                gate_ptr,
                beta_ptr,
                gdn_bufs.output,
                1,
                total,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                conv_dim as u32,
                conv_dim as u32,
                gb_stride,
                wy32_gatecache_smem,
                stream,
            )?;
        } else if self.gdn_prefill_wy32_k.0 != 0 && total > 32 {
            ops::gdn_prefill_persistent_smem(
                ctx.gpu,
                self.gdn_prefill_wy32_k,
                ssm_state.h_state,
                q_ptr,
                k_ptr,
                v_ptr,
                gate_ptr,
                beta_ptr,
                gdn_bufs.output,
                1,
                total,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                conv_dim as u32,
                conv_dim as u32,
                gb_stride,
                wy32_smem,
                stream,
            )?;
        } else if total > 4096 {
            // Sub-chunk fallback for >4096 tokens when WY32 isn't available.
            let chunk_max = 4096u32;
            let mut offset = 0u32;
            while offset < total {
                let chunk = (total - offset).min(chunk_max);
                let q_chunk = q_ptr.offset(offset as usize * conv_dim * bf16);
                let k_chunk = k_ptr.offset(offset as usize * conv_dim * bf16);
                let v_chunk = v_ptr.offset(offset as usize * conv_dim * bf16);
                let gate_chunk = gate_ptr.offset(offset as usize * gb_stride as usize * fp32);
                let beta_chunk = beta_ptr.offset(offset as usize * gb_stride as usize * fp32);
                let out_chunk = gdn_bufs.output.offset(offset as usize * value_dim * bf16);

                if self.gdn_prefill_persistent_k.0 != 0 && chunk >= 256 {
                    ops::gdn_prefill_persistent(
                        ctx.gpu,
                        self.gdn_prefill_persistent_k,
                        ssm_state.h_state,
                        q_chunk,
                        k_chunk,
                        v_chunk,
                        gate_chunk,
                        beta_chunk,
                        out_chunk,
                        1,
                        chunk,
                        nk as u32,
                        nv as u32,
                        kd as u32,
                        vd as u32,
                        conv_dim as u32,
                        conv_dim as u32,
                        gb_stride,
                        stream,
                    )?;
                } else {
                    ops::gdn_prefill_split4(
                        ctx.gpu,
                        self.gdn_prefill_split4_k,
                        ssm_state.h_state,
                        q_chunk,
                        k_chunk,
                        v_chunk,
                        gate_chunk,
                        beta_chunk,
                        out_chunk,
                        1,
                        chunk,
                        nk as u32,
                        nv as u32,
                        kd as u32,
                        vd as u32,
                        conv_dim as u32,
                        conv_dim as u32,
                        gb_stride,
                        stream,
                    )?;
                }
                offset += chunk;
            }
        } else if self.gdn_prefill_persistent_wy4_k.0 != 0 {
            let smem = (kd * vd * 4 + 8 * kd * 4 + 56) as u32;
            ops::gdn_prefill_persistent_smem(
                ctx.gpu,
                self.gdn_prefill_persistent_wy4_k,
                ssm_state.h_state,
                q_ptr,
                k_ptr,
                v_ptr,
                gate_ptr,
                beta_ptr,
                gdn_bufs.output,
                1,
                total,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                conv_dim as u32,
                conv_dim as u32,
                gb_stride,
                smem,
                stream,
            )?;
        } else if (256..=4096).contains(&total) && self.gdn_prefill_persistent_k.0 != 0 {
            ops::gdn_prefill_persistent(
                ctx.gpu,
                self.gdn_prefill_persistent_k,
                ssm_state.h_state,
                q_ptr,
                k_ptr,
                v_ptr,
                gate_ptr,
                beta_ptr,
                gdn_bufs.output,
                1,
                total,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                conv_dim as u32,
                conv_dim as u32,
                gb_stride,
                stream,
            )?;
        } else {
            ops::gdn_prefill_split4(
                ctx.gpu,
                self.gdn_prefill_split4_k,
                ssm_state.h_state,
                q_ptr,
                k_ptr,
                v_ptr,
                gate_ptr,
                beta_ptr,
                gdn_bufs.output,
                1,
                total,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                conv_dim as u32,
                conv_dim as u32,
                gb_stride,
                stream,
            )?;
        }

        if let Some(t0) = __gdn_t0 {
            ctx.gpu.synchronize(stream)?;
            let ns = t0.elapsed().as_nanos() as u64;
            GDN_CALLS.fetch_add(1, Ordering::Relaxed);
            GDN_TOTAL_NS.fetch_add(ns, Ordering::Relaxed);
            // "large" bucket = total >= 256 (typical chunked-prefill chunk).
            if total >= 256 {
                GDN_LARGE_CALLS.fetch_add(1, Ordering::Relaxed);
                GDN_LARGE_NS.fetch_add(ns, Ordering::Relaxed);
            }
            // Track per-call max so a single very-large call shows up.
            let mut prev = GDN_MAX_NS.load(Ordering::Relaxed);
            while ns > prev {
                match GDN_MAX_NS.compare_exchange_weak(
                    prev,
                    ns,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        GDN_MAX_TOTAL.store(total as u64, Ordering::Relaxed);
                        break;
                    }
                    Err(p) => prev = p,
                }
            }
            tracing::info!(
                "GDN_PROF call total={total} us={} max_us={}",
                ns / 1000,
                GDN_MAX_NS.load(Ordering::Relaxed) / 1000
            );
        }

        Ok(())
    }

    /// Q12 Path B: batched GDN recurrence — mirrors prefill_gdn_full_inner
    /// dispatch ladder but routes to the `*_batched` kernel variants and
    /// passes `h_state_ptrs` (device array of N pointers) instead of a
    /// single h_state device pointer.
    ///
    /// Constraint: scheduler-enforced same-chunk-len across all N streams.
    /// `gdn_bufs.qkv` / `gate_beta` / `output` are stacked
    /// `[batch_size, chunk_len, *]` contiguous in memory. Each batch
    /// element's QKV starts at `b * chunk_len * conv_dim` (BF16).
    ///
    /// Validation status: kernels unvalidated against hardware.
    pub(super) fn prefill_gdn_full_batched_inner(
        &self,
        h_state_ptrs: spark_runtime::gpu::DevicePtr,
        gdn_bufs: &GdnPrefillBuffers,
        batch_size: u32,
        chunk_len: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let nk = ctx.config.linear_num_key_heads;
        let kd = ctx.config.linear_key_head_dim;
        let nv = ctx.config.linear_num_value_heads;
        let vd = ctx.config.linear_value_head_dim;
        let key_dim = nk * kd;
        let value_dim = nv * vd;
        let conv_dim = key_dim * 2 + value_dim;
        let bf16 = 2usize;
        let fp32 = 4usize;

        let q_ptr = gdn_bufs.qkv;
        let k_ptr = gdn_bufs.qkv.offset(key_dim * bf16);
        let v_ptr = gdn_bufs.qkv.offset(key_dim * 2 * bf16);
        let gate_ptr = gdn_bufs.gate_beta;
        let beta_ptr = gdn_bufs.gate_beta.offset(nv * fp32);
        let gb_stride = (nv * 2) as u32;

        // Mirror the single-stream dispatch ladder. Total tokens per stream
        // is `chunk_len`; the kernel internally processes `batch_size` such
        // streams (grid dim Y).
        if self.gdn_prefill_wy32_batched_k.0 != 0 && chunk_len > 32 {
            let smem = (kd * vd * 4 + 32 * kd * 2 + 32 * kd * 2 + 32 * 32 * 4 + 256) as u32;
            ops::gdn_prefill_persistent_smem_batched(
                ctx.gpu,
                self.gdn_prefill_wy32_batched_k,
                h_state_ptrs,
                q_ptr,
                k_ptr,
                v_ptr,
                gate_ptr,
                beta_ptr,
                gdn_bufs.output,
                batch_size,
                chunk_len,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                conv_dim as u32,
                conv_dim as u32,
                gb_stride,
                smem,
                stream,
            )?;
        } else if self.gdn_prefill_persistent_wy4_batched_k.0 != 0 {
            let smem = (kd * vd * 4 + 8 * kd * 4 + 56) as u32;
            ops::gdn_prefill_persistent_smem_batched(
                ctx.gpu,
                self.gdn_prefill_persistent_wy4_batched_k,
                h_state_ptrs,
                q_ptr,
                k_ptr,
                v_ptr,
                gate_ptr,
                beta_ptr,
                gdn_bufs.output,
                batch_size,
                chunk_len,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                conv_dim as u32,
                conv_dim as u32,
                gb_stride,
                smem,
                stream,
            )?;
        } else if (256..=4096).contains(&chunk_len) && self.gdn_prefill_persistent_batched_k.0 != 0
        {
            ops::gdn_prefill_persistent_batched(
                ctx.gpu,
                self.gdn_prefill_persistent_batched_k,
                h_state_ptrs,
                q_ptr,
                k_ptr,
                v_ptr,
                gate_ptr,
                beta_ptr,
                gdn_bufs.output,
                batch_size,
                chunk_len,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                conv_dim as u32,
                conv_dim as u32,
                gb_stride,
                stream,
            )?;
        } else if self.gdn_prefill_split4_batched_k.0 != 0 {
            ops::gdn_prefill_split4_batched(
                ctx.gpu,
                self.gdn_prefill_split4_batched_k,
                h_state_ptrs,
                q_ptr,
                k_ptr,
                v_ptr,
                gate_ptr,
                beta_ptr,
                gdn_bufs.output,
                batch_size,
                chunk_len,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                conv_dim as u32,
                conv_dim as u32,
                gb_stride,
                stream,
            )?;
        } else {
            anyhow::bail!(
                "Qwen3SsmLayer::prefill_gdn_full_batched_inner: no batched GDN \
                 kernel handle is loaded for this target — caller should fall \
                 back to per-stream prefill_gdn_full."
            );
        }

        Ok(())
    }
}
