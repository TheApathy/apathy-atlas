// SPDX-License-Identifier: AGPL-3.0-only

//! Raw-BF16 parity tests for the M8/M17 FP32-preactivation down path.

use anyhow::{Context, Result};
use spark_model::layers::ops;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::data::{
    WeightFixture, as_le_bytes, fnv1a64, raw_bf16_equal, read_bf16, upload, upload_weight,
};
use super::kernels::Kernels;

const M8_ROWS: usize = 5;
const M17_ROWS: usize = 16;
const HIDDEN: usize = 5_120;
const INTERMEDIATE: usize = 17_408;

#[allow(clippy::too_many_arguments)]
fn compare_downs(
    gpu: &dyn GpuBackend,
    stream: u64,
    kernels: Kernels,
    label: &str,
    gate: DevicePtr,
    up: DevicePtr,
    oracle_gate: DevicePtr,
    oracle_up: DevicePtr,
    down_fixture: &WeightFixture,
    rows: usize,
    scale2: f32,
) -> Result<()> {
    let n = down_fixture.logical_n;
    let k = down_fixture.k;
    let down_weight = upload_weight(gpu, down_fixture, scale2)?;
    let preactivation = gpu.alloc(rows * k * size_of::<f32>())?;
    let output_bytes = rows * n * size_of::<u16>();
    let materialized_out = gpu.alloc(output_bytes)?;
    let rt2_out = gpu.alloc(output_bytes)?;
    let inline_out = gpu.alloc(output_bytes)?;
    let serial_out = gpu.alloc(output_bytes)?;

    ops::w4a16_gemv_dual_silu_f32_exact(
        gpu,
        kernels.exact,
        gate,
        up,
        preactivation,
        rows as u32,
        k as u32,
        stream,
    )
    .with_context(|| format!("{label}: materialize exact SiLU(gate)*up as FP32"))?;
    ops::w4a16_gemv_f32_input_exact(
        gpu,
        kernels.exact,
        preactivation,
        &down_weight,
        materialized_out,
        rows as u32,
        n as u32,
        k as u32,
        stream,
    )
    .with_context(|| format!("{label}: exact FP32-input down"))?;
    // Register-tiled down projection, held to the same serial-K1 oracle. T=2
    // straddles the odd-N tail, so the second output of the last lane group is
    // out of range and must stay unwritten.
    ops::w4a16_gemv_f32_input_exact_with(
        gpu,
        kernels.exact,
        preactivation,
        &down_weight,
        rt2_out,
        rows as u32,
        n as u32,
        k as u32,
        stream,
        true,
    )
    .with_context(|| format!("{label}: rt2 FP32-input down"))?;
    ops::w4a16_gemv_silu_input_exact(
        gpu,
        kernels.exact,
        gate,
        up,
        &down_weight,
        inline_out,
        rows as u32,
        n as u32,
        k as u32,
        stream,
    )
    .with_context(|| format!("{label}: old inline-SiLU exact down"))?;
    for row in 0..rows {
        ops::w4a16_gemv_silu_input(
            gpu,
            kernels.serial_silu,
            oracle_gate.offset(row * k * size_of::<u16>()),
            oracle_up.offset(row * k * size_of::<u16>()),
            &down_weight,
            serial_out.offset(row * n * size_of::<u16>()),
            n as u32,
            k as u32,
            stream,
        )
        .with_context(|| format!("{label}: ordinary K1 down row {row}"))?;
    }

    let elements = rows * n;
    let materialized_bits = read_bf16(gpu, materialized_out, elements, stream)?;
    let inline_bits = read_bf16(gpu, inline_out, elements, stream)?;
    let serial_bits = read_bf16(gpu, serial_out, elements, stream)?;
    raw_bf16_equal(label, &materialized_bits, &inline_bits, n)?;
    raw_bf16_equal(label, &materialized_bits, &serial_bits, n)?;
    let rt2_bits = read_bf16(gpu, rt2_out, elements, stream)?;
    raw_bf16_equal(&format!("{label}/rt2-down"), &rt2_bits, &serial_bits, n)?;
    gpu.free(rt2_out)?;
    for ptr in [
        down_weight.weight,
        down_weight.weight_scale,
        preactivation,
        materialized_out,
        inline_out,
        serial_out,
    ] {
        gpu.free(ptr)?;
    }
    println!(
        "PASS {label}: materialized-FP32/inline/K1 M={rows} N={n} K={k} out={:016x}",
        fnv1a64(&materialized_bits)
    );
    Ok(())
}

pub(crate) fn run_cancellation(gpu: &dyn GpuBackend, stream: u64, kernels: Kernels) -> Result<()> {
    let fixture = super::data::cancellation_silu(M8_ROWS, 8);
    let gate = upload(gpu, &as_le_bytes(&fixture.gate))?;
    let up = upload(gpu, &as_le_bytes(&fixture.up))?;
    compare_downs(
        gpu,
        stream,
        kernels,
        "materialized association-cancellation",
        gate,
        up,
        gate,
        up,
        &fixture.down,
        M8_ROWS,
        1.0,
    )?;
    gpu.free(gate)?;
    gpu.free(up)?;
    Ok(())
}

fn run_production_rows(
    gpu: &dyn GpuBackend,
    stream: u64,
    kernels: Kernels,
    rows: usize,
    seed: u64,
) -> Result<()> {
    let dual = super::data::random_dual(rows, INTERMEDIATE, HIDDEN, seed);
    let down = super::data::random_silu(rows, HIDDEN, INTERMEDIATE, seed ^ 0xa5a5_a5a5).down;
    let input = upload(gpu, &as_le_bytes(&dual.activations))?;
    let gate_weight = upload_weight(gpu, &dual.gate, 0.003_906_25)?;
    let up_weight = upload_weight(gpu, &dual.up, 0.007_812_5)?;
    let projection_bytes = rows * INTERMEDIATE * size_of::<u16>();
    let gate = gpu.alloc(projection_bytes)?;
    let up = gpu.alloc(projection_bytes)?;
    let serial_gate = gpu.alloc(projection_bytes)?;
    let serial_up = gpu.alloc(projection_bytes)?;
    ops::w4a16_gemv_dual_exact(
        gpu,
        kernels.exact,
        input,
        &gate_weight,
        gate,
        &up_weight,
        up,
        rows as u32,
        INTERMEDIATE as u32,
        HIDDEN as u32,
        stream,
    )?;
    for row in 0..rows {
        ops::w4a16_gemv_dual(
            gpu,
            kernels.serial_dual,
            input.offset(row * HIDDEN * size_of::<u16>()),
            &gate_weight,
            serial_gate.offset(row * INTERMEDIATE * size_of::<u16>()),
            &up_weight,
            serial_up.offset(row * INTERMEDIATE * size_of::<u16>()),
            INTERMEDIATE as u32,
            HIDDEN as u32,
            stream,
        )
        .with_context(|| format!("production materialized: ordinary K1 dual row {row}"))?;
    }
    let exact_gate_bits = read_bf16(gpu, gate, rows * INTERMEDIATE, stream)?;
    let exact_up_bits = read_bf16(gpu, up, rows * INTERMEDIATE, stream)?;
    let serial_gate_bits = read_bf16(gpu, serial_gate, rows * INTERMEDIATE, stream)?;
    let serial_up_bits = read_bf16(gpu, serial_up, rows * INTERMEDIATE, stream)?;
    raw_bf16_equal(
        "production materialized gate",
        &exact_gate_bits,
        &serial_gate_bits,
        INTERMEDIATE,
    )?;
    raw_bf16_equal(
        "production materialized up",
        &exact_up_bits,
        &serial_up_bits,
        INTERMEDIATE,
    )?;
    compare_downs(
        gpu,
        stream,
        kernels,
        &format!("Qwen3.8 production materialized down M={rows}"),
        gate,
        up,
        serial_gate,
        serial_up,
        &down,
        rows,
        0.003_906_25,
    )?;
    for ptr in [
        input,
        gate_weight.weight,
        gate_weight.weight_scale,
        up_weight.weight,
        up_weight.weight_scale,
        gate,
        up,
        serial_gate,
        serial_up,
    ] {
        gpu.free(ptr)?;
    }
    Ok(())
}

/// The fused M17 gate/up materialization is what the default verify FFN
/// actually launches, so its register-tiled twin must reproduce the shipping
/// kernel's FP32 activation bit for bit — including the BF16 gate rounding
/// boundary that feeds the SiLU.
fn run_fused_rt2_parity(gpu: &dyn GpuBackend, stream: u64, kernels: Kernels) -> Result<()> {
    let rows = M17_ROWS;
    let dual = super::data::random_dual(rows, INTERMEDIATE, HIDDEN, 0xf05e_d170_0000_0011u64);
    let input = upload(gpu, &as_le_bytes(&dual.activations))?;
    let gate_weight = upload_weight(gpu, &dual.gate, 0.003_906_25)?;
    let up_weight = upload_weight(gpu, &dual.up, 0.007_812_5)?;
    let bytes = rows * INTERMEDIATE * size_of::<f32>();
    let base_out = gpu.alloc(bytes)?;
    let rt2_out = gpu.alloc(bytes)?;

    for (out, use_rt2) in [(base_out, false), (rt2_out, true)] {
        ops::w4a16_gemv_dual_materialize_f32_exact_m17_with(
            gpu,
            kernels.exact,
            input,
            &gate_weight,
            &up_weight,
            out,
            rows as u32,
            INTERMEDIATE as u32,
            HIDDEN as u32,
            stream,
            use_rt2,
        )
        .with_context(|| format!("fused materialize m17 (rt2={use_rt2})"))?;
    }

    let mut base = vec![0u8; bytes];
    let mut rt2 = vec![0u8; bytes];
    gpu.copy_d2h_on_stream(base_out, &mut base, stream)?;
    gpu.copy_d2h_on_stream(rt2_out, &mut rt2, stream)?;
    if base != rt2 {
        let first = base
            .chunks_exact(4)
            .zip(rt2.chunks_exact(4))
            .position(|(a, b)| a != b)
            .expect("different buffers have a differing element");
        anyhow::bail!(
            "fused materialize m17: rt2 FP32 mismatch at flat={first}, row={}, n={}",
            first / INTERMEDIATE,
            first % INTERMEDIATE
        );
    }
    println!("PASS fused materialize m17 rt2 == shipping: M={rows} N={INTERMEDIATE} K={HIDDEN}");

    for ptr in [
        input,
        gate_weight.weight,
        gate_weight.weight_scale,
        up_weight.weight,
        up_weight.weight_scale,
        base_out,
        rt2_out,
    ] {
        gpu.free(ptr)?;
    }
    Ok(())
}

pub(crate) fn run_production(gpu: &dyn GpuBackend, stream: u64, kernels: Kernels) -> Result<()> {
    run_production_rows(gpu, stream, kernels, M8_ROWS, 0xf320_5120_1740_8005)?;
    run_production_rows(gpu, stream, kernels, M17_ROWS, 0xf320_5120_1740_8010)?;
    run_fused_rt2_parity(gpu, stream, kernels)
}
