// SPDX-License-Identifier: AGPL-3.0-only

//! Live raw-BF16 parity oracle for the multi-row NVFP4 attention QKV
//! projections at the production Qwen3.8-27B gated shapes.
//!
//! Oracle: the per-token K1 kernels the serial route uses — `w4a16_gemv_qg`
//! for the gated Q/Gate projection (deinterleaved at the store) and
//! `w4a16_gemv` for K and V. Every candidate must reproduce those raw BF16
//! bytes for all `M` rows, across the FULL `q_proj_dim` width, i.e. the Q half
//! AND the output-gate half.
//!
//! Candidates:
//! 1. `w4a16_gemv_qg_exact` / `w4a16_gemv_dual_kv_exact` — the K1-order exact
//!    multi-row kernels `ms_qkv_exact` dispatches. Held to bit-identity.
//! 2. `w4a16_gemm` at M=rows plus the per-token `deinterleave_qg` fixup — the
//!    arithmetic of `ms_qkv_batched_plain`'s non-transposed branch, gated
//!    behind `ATLAS_ATTN_QKV_BATCHED`. Its result is CLASSIFIED rather than
//!    asserted, because the question this microtest exists to settle is which
//!    kind of difference it is:
//!      * `LAYOUT`   — the values are right but land in the wrong slots
//!                     (the deinterleave stride/offset or head mapping);
//!                     fixable to bit-exact.
//!      * `ROUNDING` — the values land in the right slots but differ, because
//!                     the GEMM partitions K differently from the K1 GEMV
//!                     reduction; NOT fixable to bit-exact.
//!      * `EXACT`    — no difference at this M.
//!
//! Run on a GPU box:
//! ```text
//! ATLAS_TARGET_MODEL=qwen3.8-27b ATLAS_TARGET_QUANT=nvfp4 \
//!   cargo run --release -p spark-model --features cuda \
//!   --example w4a16_exact_attention_qkv_microtest
//! ```

// Shared with the LM-head microtest, which exercises the fixture builders this
// one does not need.
#[allow(dead_code)]
#[path = "w4a16_exact_lm_head_microtest/data.rs"]
mod data;

use anyhow::{Context, Result, bail};
use data::{Fixture, as_le_bytes, fnv1a64, from_le_bytes, random_fixture};
use spark_model::layers::ops::{
    self, ExactAttentionQkvRoute, W4a16ExactAttentionKernels, exact_attention_qkv_route,
};
use spark_model::weight_map::QuantizedWeight;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

/// Production Qwen3.8-27B full-attention geometry (kernels/gb10/qwen3.8-27b).
const HIDDEN: usize = 5_120;
const NQ: usize = 24;
const NKV: usize = 4;
const HD: usize = 256;
const Q_DIM: usize = NQ * HD; // 6144
const Q_PROJ_DIM: usize = 2 * Q_DIM; // 12288 — gated: [Q_h0|G_h0|Q_h1|G_h1|…]
const KV_DIM: usize = NKV * HD; // 1024
const MAX_ROWS: usize = 32;

struct Kernels {
    gemv: KernelHandle,
    gemv_qg: KernelHandle,
    gemm: KernelHandle,
    deinterleave_qg: KernelHandle,
    exact: W4a16ExactAttentionKernels,
}

struct Weights {
    q: QuantizedWeight,
    k: QuantizedWeight,
    v: QuantizedWeight,
}

/// Where output element `out_idx` of the deinterleaved `[Q… | G…]` layout came
/// from in the raw interleaved `[Q_h0|G_h0|Q_h1|G_h1|…]` projection output.
fn interleaved_source(out_idx: usize) -> usize {
    let group_dim = 2 * HD;
    if out_idx < Q_DIM {
        let h = out_idx / HD;
        let d = out_idx % HD;
        h * group_dim + d
    } else {
        let gi = out_idx - Q_DIM;
        let h = gi / HD;
        let d = gi % HD;
        h * group_dim + HD + d
    }
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len().max(1))?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}

fn read_bf16(
    gpu: &dyn GpuBackend,
    ptr: DevicePtr,
    elements: usize,
    stream: u64,
) -> Result<Vec<u16>> {
    let mut bytes = vec![0u8; elements * size_of::<u16>()];
    gpu.copy_d2h_on_stream(ptr, &mut bytes, stream)?;
    Ok(from_le_bytes(&bytes))
}

fn upload_weight(gpu: &dyn GpuBackend, fixture: &Fixture) -> Result<QuantizedWeight> {
    Ok(QuantizedWeight {
        weight: upload(gpu, &fixture.packed)?,
        weight_scale: upload(gpu, &fixture.scales)?,
        weight_scale_2: 1.0,
        input_scale: DevicePtr::NULL,
    })
}

fn load_kernels(gpu: &dyn GpuBackend) -> Result<Kernels> {
    Ok(Kernels {
        gemv: gpu.kernel("w4a16_gemv", "w4a16_gemv")?,
        gemv_qg: gpu.kernel("w4a16_gemv", "w4a16_gemv_qg")?,
        gemm: gpu.kernel("w4a16", "w4a16_gemm")?,
        deinterleave_qg: gpu.kernel("ssm_preprocess", "deinterleave_qg")?,
        exact: W4a16ExactAttentionKernels::new(
            gpu.kernel("w4a16_gemv_exact_attention", "w4a16_gemv_qg_exact_m17")?,
            gpu.kernel("w4a16_gemv_exact_attention", "w4a16_gemv_dual_kv_exact_m17")?,
        )
        .with_m4(
            gpu.kernel("w4a16_gemv_exact_attention", "w4a16_gemv_qg_exact_m4")?,
            gpu.kernel("w4a16_gemv_exact_attention", "w4a16_gemv_dual_kv_exact_m4")?,
        ),
    })
}

/// Per-token K1 reference for `rows` rows: gated Q/Gate + K + V.
fn run_oracle(
    gpu: &dyn GpuBackend,
    stream: u64,
    kernels: &Kernels,
    weights: &Weights,
    input: DevicePtr,
    rows: usize,
    q_out: DevicePtr,
    k_out: DevicePtr,
    v_out: DevicePtr,
) -> Result<()> {
    for row in 0..rows {
        let input_row = input.offset(row * HIDDEN * size_of::<u16>());
        ops::w4a16_gemv_qg(
            gpu,
            kernels.gemv_qg,
            input_row,
            &weights.q,
            q_out.offset(row * Q_PROJ_DIM * size_of::<u16>()),
            Q_PROJ_DIM as u32,
            HIDDEN as u32,
            NQ as u32,
            HD as u32,
            stream,
        )
        .with_context(|| format!("oracle gated-Q row {row}"))?;
        ops::w4a16_gemv(
            gpu,
            kernels.gemv,
            input_row,
            &weights.k,
            k_out.offset(row * KV_DIM * size_of::<u16>()),
            KV_DIM as u32,
            HIDDEN as u32,
            stream,
        )
        .with_context(|| format!("oracle K row {row}"))?;
        ops::w4a16_gemv(
            gpu,
            kernels.gemv,
            input_row,
            &weights.v,
            v_out.offset(row * KV_DIM * size_of::<u16>()),
            KV_DIM as u32,
            HIDDEN as u32,
            stream,
        )
        .with_context(|| format!("oracle V row {row}"))?;
    }
    Ok(())
}

fn first_mismatch(actual: &[u16], oracle: &[u16]) -> Option<usize> {
    actual.iter().zip(oracle).position(|(a, b)| a != b)
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// Largest absolute BF16-bit-pattern distance, a proxy for ULP distance that
/// is meaningful because all values here share a sign and a narrow exponent
/// range. Reported only to separate "1 ULP everywhere" from "wrong number".
fn max_bit_distance(actual: &[u16], oracle: &[u16]) -> u16 {
    actual
        .iter()
        .zip(oracle)
        .map(|(a, b)| a.abs_diff(*b))
        .max()
        .unwrap_or(0)
}

fn assert_exact(
    label: &str,
    what: &str,
    actual: &[u16],
    oracle: &[u16],
    width: usize,
) -> Result<()> {
    if let Some(flat) = first_mismatch(actual, oracle) {
        bail!(
            "{label}: {what} raw BF16 mismatch at flat={flat}, row={}, col={}, \
             actual=0x{:04x}, oracle=0x{:04x} ({} vs {})",
            flat / width,
            flat % width,
            actual[flat],
            oracle[flat],
            bf16_to_f32(actual[flat]),
            bf16_to_f32(oracle[flat]),
        );
    }
    Ok(())
}

/// Candidate 1: the exact K1-order multi-row kernels. Must be bit-identical.
#[allow(clippy::too_many_arguments)]
fn check_exact_route(
    gpu: &dyn GpuBackend,
    stream: u64,
    kernels: &Kernels,
    weights: &Weights,
    input: DevicePtr,
    rows: usize,
    oracle_q: &[u16],
    oracle_k: &[u16],
    oracle_v: &[u16],
) -> Result<bool> {
    let route = exact_attention_qkv_route(rows, true, true, kernels.exact);
    let Some(route) = route else {
        println!("  exact route: none at M={rows} (outside the 4..=17 exact tiers) — skipped");
        return Ok(false);
    };
    if !matches!(
        route,
        ExactAttentionQkvRoute::ExactM4 | ExactAttentionQkvRoute::ExactM17
    ) {
        bail!("M={rows}: exact kernels present but route selected {route:?}");
    }

    let q_out = gpu.alloc(rows * Q_PROJ_DIM * size_of::<u16>())?;
    let k_out = gpu.alloc(rows * KV_DIM * size_of::<u16>())?;
    let v_out = gpu.alloc(rows * KV_DIM * size_of::<u16>())?;

    ops::w4a16_gemv_qg_exact(
        gpu,
        kernels.exact.qg_for_rows(rows),
        input,
        &weights.q,
        q_out,
        rows as u32,
        Q_PROJ_DIM as u32,
        HIDDEN as u32,
        NQ as u32,
        HD as u32,
        Q_PROJ_DIM as u32,
        stream,
    )
    .with_context(|| format!("exact gated-Q at M={rows}"))?;
    ops::w4a16_gemv_dual_kv_exact(
        gpu,
        kernels.exact.dual_kv_for_rows(rows),
        input,
        &weights.k,
        k_out,
        &weights.v,
        v_out,
        rows as u32,
        KV_DIM as u32,
        HIDDEN as u32,
        KV_DIM as u32,
        stream,
    )
    .with_context(|| format!("exact dual-KV at M={rows}"))?;

    let q = read_bf16(gpu, q_out, rows * Q_PROJ_DIM, stream)?;
    let k = read_bf16(gpu, k_out, rows * KV_DIM, stream)?;
    let v = read_bf16(gpu, v_out, rows * KV_DIM, stream)?;

    let label = format!("exact M={rows}");
    // Q half and Gate half are asserted separately so a gate-only regression
    // cannot hide behind a passing Q half.
    for row in 0..rows {
        let base = row * Q_PROJ_DIM;
        assert_exact(
            &label,
            "Q half",
            &q[base..base + Q_DIM],
            &oracle_q[base..base + Q_DIM],
            Q_DIM,
        )?;
        assert_exact(
            &label,
            "GATE half",
            &q[base + Q_DIM..base + Q_PROJ_DIM],
            &oracle_q[base + Q_DIM..base + Q_PROJ_DIM],
            Q_DIM,
        )?;
    }
    assert_exact(&label, "K", &k, oracle_k, KV_DIM)?;
    assert_exact(&label, "V", &v, oracle_v, KV_DIM)?;

    for ptr in [q_out, k_out, v_out] {
        gpu.free(ptr)?;
    }
    println!(
        "  PASS exact {route:?}: Q+GATE+K+V bit-identical to per-token K1, \
         q_fnv1a64={:016x}",
        fnv1a64(&q)
    );
    Ok(true)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Verdict {
    Exact,
    Layout,
    Rounding,
}

/// Candidate 2: the batched `w4a16_gemm` + `deinterleave_qg` arithmetic of
/// `ms_qkv_batched_plain`. Classified, not asserted.
#[allow(clippy::too_many_arguments)]
fn classify_batched_plain(
    gpu: &dyn GpuBackend,
    stream: u64,
    kernels: &Kernels,
    weights: &Weights,
    input: DevicePtr,
    rows: usize,
    oracle_q: &[u16],
    oracle_k: &[u16],
    oracle_v: &[u16],
) -> Result<Verdict> {
    let raw_out = gpu.alloc(rows * Q_PROJ_DIM * size_of::<u16>())?;
    let deint_out = gpu.alloc(rows * Q_PROJ_DIM * size_of::<u16>())?;
    let k_out = gpu.alloc(rows * KV_DIM * size_of::<u16>())?;
    let v_out = gpu.alloc(rows * KV_DIM * size_of::<u16>())?;

    ops::w4a16_gemm(
        gpu,
        kernels.gemm,
        input,
        &weights.q,
        raw_out,
        rows as u32,
        Q_PROJ_DIM as u32,
        HIDDEN as u32,
        stream,
    )
    .with_context(|| format!("batched gated-Q GEMM at M={rows}"))?;
    ops::w4a16_gemm(
        gpu,
        kernels.gemm,
        input,
        &weights.k,
        k_out,
        rows as u32,
        KV_DIM as u32,
        HIDDEN as u32,
        stream,
    )?;
    ops::w4a16_gemm(
        gpu,
        kernels.gemm,
        input,
        &weights.v,
        v_out,
        rows as u32,
        KV_DIM as u32,
        HIDDEN as u32,
        stream,
    )?;

    // Snapshot the raw interleaved GEMM output BEFORE the in-place fixup, then
    // apply `deinterleave_qg` exactly as ms_qkv_batched_plain does: one launch
    // per token, num_tokens=1, stride=q_proj_dim.
    let raw = read_bf16(gpu, raw_out, rows * Q_PROJ_DIM, stream)?;
    gpu.copy_d2d_async(
        raw_out,
        deint_out,
        rows * Q_PROJ_DIM * size_of::<u16>(),
        stream,
    )?;
    for row in 0..rows {
        ops::deinterleave_qg(
            gpu,
            kernels.deinterleave_qg,
            deint_out.offset(row * Q_PROJ_DIM * size_of::<u16>()),
            1,
            NQ as u32,
            HD as u32,
            Q_PROJ_DIM as u32,
            stream,
        )?;
    }
    let deint = read_bf16(gpu, deint_out, rows * Q_PROJ_DIM, stream)?;
    let k = read_bf16(gpu, k_out, rows * KV_DIM, stream)?;
    let v = read_bf16(gpu, v_out, rows * KV_DIM, stream)?;

    // Discriminator. `raw_mapped[i]` is what the deinterleave SHOULD produce
    // given the GEMM's own numbers. If raw_mapped == deint, the fixup is
    // faithful and any residual gap against the oracle is pure arithmetic.
    let raw_mapped: Vec<u16> = (0..rows * Q_PROJ_DIM)
        .map(|flat| {
            let row = flat / Q_PROJ_DIM;
            let out_idx = flat % Q_PROJ_DIM;
            raw[row * Q_PROJ_DIM + interleaved_source(out_idx)]
        })
        .collect();
    let fixup_faithful = raw_mapped == deint;

    let q_mismatches = deint.iter().zip(oracle_q).filter(|(a, b)| a != b).count();
    let q_half_mismatches: usize = (0..rows)
        .map(|row| {
            let base = row * Q_PROJ_DIM;
            deint[base..base + Q_DIM]
                .iter()
                .zip(&oracle_q[base..base + Q_DIM])
                .filter(|(a, b)| a != b)
                .count()
        })
        .sum();
    let gate_mismatches = q_mismatches - q_half_mismatches;
    let kv_mismatches = k.iter().zip(oracle_k).filter(|(a, b)| a != b).count()
        + v.iter().zip(oracle_v).filter(|(a, b)| a != b).count();

    // "Values right, slots wrong" test: does the raw interleaved GEMM output
    // already match the oracle under the identity mapping (i.e. the fixup was
    // applied when it should not have been, or vice versa)?
    let raw_matches_oracle = raw == oracle_q;

    let verdict = if q_mismatches == 0 && kv_mismatches == 0 {
        Verdict::Exact
    } else if raw_matches_oracle || !fixup_faithful {
        Verdict::Layout
    } else {
        Verdict::Rounding
    };

    println!(
        "  batched-plain M={rows}: verdict={verdict:?} \
         q_half_mismatch={q_half_mismatches}/{} gate_mismatch={gate_mismatches}/{} \
         kv_mismatch={kv_mismatches}/{} max_bf16_bit_distance(Q)={} \
         fixup_faithful={fixup_faithful} raw_matches_oracle={raw_matches_oracle}",
        rows * Q_DIM,
        rows * Q_DIM,
        2 * rows * KV_DIM,
        max_bit_distance(&deint, oracle_q),
    );
    if verdict == Verdict::Rounding {
        // K and V share the GEMM but have no gate and no fixup. If they diverge
        // too, the difference cannot be gate-specific.
        println!(
            "    K/V diverge as well: {} — the difference is NOT gate-specific.",
            if kv_mismatches > 0 { "yes" } else { "no" }
        );
    }

    for ptr in [raw_out, deint_out, k_out, v_out] {
        gpu.free(ptr)?;
    }
    Ok(verdict)
}

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())
        .context("initialize CUDA backend with compiled Qwen3.8 kernels")?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.create_stream()?;
    let kernels = load_kernels(gpu).context("resolve attention QKV kernels")?;

    println!(
        "Qwen3.8-27B gated attention QKV parity: hidden={HIDDEN} q={NQ}x{HD} \
         (q_proj_dim={Q_PROJ_DIM}, gated) kv={NKV}x{HD} (kv_dim={KV_DIM}) M=1..={MAX_ROWS}"
    );

    let q_fixture = random_fixture(MAX_ROWS, Q_PROJ_DIM, HIDDEN, 0x9f38_0000_c000_0001);
    let k_fixture = random_fixture(1, KV_DIM, HIDDEN, 0x9f38_0000_0400_0002);
    let v_fixture = random_fixture(1, KV_DIM, HIDDEN, 0x9f38_0000_0400_0003);
    let input = upload(gpu, &as_le_bytes(&q_fixture.activations))?;
    let weights = Weights {
        q: upload_weight(gpu, &q_fixture)?,
        k: upload_weight(gpu, &k_fixture)?,
        v: upload_weight(gpu, &v_fixture)?,
    };

    let oracle_q_buf = gpu.alloc(MAX_ROWS * Q_PROJ_DIM * size_of::<u16>())?;
    let oracle_k_buf = gpu.alloc(MAX_ROWS * KV_DIM * size_of::<u16>())?;
    let oracle_v_buf = gpu.alloc(MAX_ROWS * KV_DIM * size_of::<u16>())?;

    let mut exact_tiers_covered = 0usize;
    let mut batched_exact = Vec::new();
    let mut batched_layout = Vec::new();
    let mut batched_rounding = Vec::new();

    for rows in 1..=MAX_ROWS {
        println!("M={rows}");
        run_oracle(
            gpu,
            stream,
            &kernels,
            &weights,
            input,
            rows,
            oracle_q_buf,
            oracle_k_buf,
            oracle_v_buf,
        )?;
        let oracle_q = read_bf16(gpu, oracle_q_buf, rows * Q_PROJ_DIM, stream)?;
        let oracle_k = read_bf16(gpu, oracle_k_buf, rows * KV_DIM, stream)?;
        let oracle_v = read_bf16(gpu, oracle_v_buf, rows * KV_DIM, stream)?;

        if check_exact_route(
            gpu, stream, &kernels, &weights, input, rows, &oracle_q, &oracle_k, &oracle_v,
        )? {
            exact_tiers_covered += 1;
        }

        match classify_batched_plain(
            gpu, stream, &kernels, &weights, input, rows, &oracle_q, &oracle_k, &oracle_v,
        )? {
            Verdict::Exact => batched_exact.push(rows),
            Verdict::Layout => batched_layout.push(rows),
            Verdict::Rounding => batched_rounding.push(rows),
        }
    }

    for ptr in [oracle_q_buf, oracle_k_buf, oracle_v_buf, input] {
        gpu.free(ptr)?;
    }

    println!(
        "\nexact multi-row route: {exact_tiers_covered}/{MAX_ROWS} widths bit-identical \
         to per-token K1 (Q, GATE, K, V)"
    );
    println!(
        "batched w4a16_gemm + deinterleave_qg: exact={batched_exact:?} \
         layout-divergent={batched_layout:?} rounding-divergent={batched_rounding:?}"
    );
    if !batched_layout.is_empty() {
        bail!(
            "batched plain QKV has a LAYOUT divergence at M={batched_layout:?} — \
             the deinterleave stride/offset or head mapping is wrong and is fixable"
        );
    }
    println!(
        "PASS: exact attention QKV parity matrix complete; batched-plain divergence is \
         reassociation-only and cannot be made bit-exact against the K1 oracle"
    );
    Ok(())
}
