// SPDX-License-Identifier: AGPL-3.0-only

//! Device microtest: exact dynamic-M dense FFN must equal independent K1 rows.

#[path = "w4a16_exact_ffn_microtest/data.rs"]
mod data;
#[path = "w4a16_exact_ffn_microtest/kernels.rs"]
mod kernels;
#[path = "w4a16_exact_ffn_microtest/materialized.rs"]
mod materialized;

use anyhow::{Context, Result, ensure};
use data::{
    DualFixture, SiluFixture, as_le_bytes, fnv1a64, raw_bf16_equal, read_bf16, upload,
    upload_weight,
};
use kernels::{Kernels, load_kernels};
use spark_model::layers::ops;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::GpuBackend;

const ROW_TIER_BOUNDARIES: [usize; 8] = [2, 4, 5, 8, 9, 17, 18, 32];
// K1's tail return precedes a CTA barrier, so its oracle requires N % 4 == 0.
const SMALL_WIDTHS: [usize; 8] = [4, 8, 12, 16, 20, 24, 28, 32];
const HIDDEN: usize = 5_120;
const INTERMEDIATE: usize = 17_408;
const PRODUCTION_ROWS: usize = 5;

fn run_dual(
    gpu: &dyn GpuBackend,
    stream: u64,
    kernels: Kernels,
    label: &str,
    fixture: &DualFixture,
    gate_scale2: f32,
    up_scale2: f32,
) -> Result<()> {
    ensure!(
        fixture.gate.logical_n == fixture.up.logical_n && fixture.gate.k == fixture.up.k,
        "{label}: inconsistent dual fixture"
    );
    let rows = fixture.rows;
    let n = fixture.gate.logical_n;
    let k = fixture.gate.k;
    let input = upload(gpu, &as_le_bytes(&fixture.activations))?;
    let gate_weight = upload_weight(gpu, &fixture.gate, gate_scale2)?;
    let up_weight = upload_weight(gpu, &fixture.up, up_scale2)?;
    let outputs = rows * n;
    let exact_gate = gpu.alloc(outputs * size_of::<u16>())?;
    let exact_up = gpu.alloc(outputs * size_of::<u16>())?;
    let serial_gate = gpu.alloc(outputs * size_of::<u16>())?;
    let serial_up = gpu.alloc(outputs * size_of::<u16>())?;
    ops::w4a16_gemv_dual_exact(
        gpu,
        kernels.exact,
        input,
        &gate_weight,
        exact_gate,
        &up_weight,
        exact_up,
        rows as u32,
        n as u32,
        k as u32,
        stream,
    )
    .with_context(|| format!("{label}: exact dual launch"))?;
    for row in 0..rows {
        ops::w4a16_gemv_dual(
            gpu,
            kernels.serial_dual,
            input.offset(row * k * size_of::<u16>()),
            &gate_weight,
            serial_gate.offset(row * n * size_of::<u16>()),
            &up_weight,
            serial_up.offset(row * n * size_of::<u16>()),
            n as u32,
            k as u32,
            stream,
        )
        .with_context(|| format!("{label}: K1 dual row {row}"))?;
    }
    let exact_gate_bits = read_bf16(gpu, exact_gate, outputs, stream)?;
    let exact_up_bits = read_bf16(gpu, exact_up, outputs, stream)?;
    let serial_gate_bits = read_bf16(gpu, serial_gate, outputs, stream)?;
    let serial_up_bits = read_bf16(gpu, serial_up, outputs, stream)?;
    raw_bf16_equal(label, &exact_gate_bits, &serial_gate_bits, n)?;
    raw_bf16_equal(label, &exact_up_bits, &serial_up_bits, n)?;
    for ptr in [
        input,
        gate_weight.weight,
        gate_weight.weight_scale,
        up_weight.weight,
        up_weight.weight_scale,
        exact_gate,
        exact_up,
        serial_gate,
        serial_up,
    ] {
        gpu.free(ptr)?;
    }
    println!(
        "PASS {label}: dual M={rows} N={n} K={k} gate={:016x} up={:016x}",
        fnv1a64(&exact_gate_bits),
        fnv1a64(&exact_up_bits)
    );
    Ok(())
}

fn run_silu(
    gpu: &dyn GpuBackend,
    stream: u64,
    kernels: Kernels,
    label: &str,
    fixture: &SiluFixture,
    scale2: f32,
) -> Result<()> {
    let rows = fixture.rows;
    let n = fixture.down.logical_n;
    let k = fixture.down.k;
    let gate = upload(gpu, &as_le_bytes(&fixture.gate))?;
    let up = upload(gpu, &as_le_bytes(&fixture.up))?;
    let weight = upload_weight(gpu, &fixture.down, scale2)?;
    let outputs = rows * n;
    let exact_out = gpu.alloc(outputs * size_of::<u16>())?;
    let serial_out = gpu.alloc(outputs * size_of::<u16>())?;
    ops::w4a16_gemv_silu_input_exact(
        gpu,
        kernels.exact,
        gate,
        up,
        &weight,
        exact_out,
        rows as u32,
        n as u32,
        k as u32,
        stream,
    )
    .with_context(|| format!("{label}: exact SiLU-input launch"))?;
    for row in 0..rows {
        ops::w4a16_gemv_silu_input(
            gpu,
            kernels.serial_silu,
            gate.offset(row * k * size_of::<u16>()),
            up.offset(row * k * size_of::<u16>()),
            &weight,
            serial_out.offset(row * n * size_of::<u16>()),
            n as u32,
            k as u32,
            stream,
        )
        .with_context(|| format!("{label}: K1 SiLU-input row {row}"))?;
    }

    let exact_bits = read_bf16(gpu, exact_out, outputs, stream)?;
    let serial_bits = read_bf16(gpu, serial_out, outputs, stream)?;
    raw_bf16_equal(label, &exact_bits, &serial_bits, n)?;
    for ptr in [
        gate,
        up,
        weight.weight,
        weight.weight_scale,
        exact_out,
        serial_out,
    ] {
        gpu.free(ptr)?;
    }
    println!(
        "PASS {label}: SiLU-down M={rows} N={n} K={k} out={:016x}",
        fnv1a64(&exact_bits)
    );
    Ok(())
}

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())
        .context("initialize CUDA backend with compiled Qwen3.8 kernels")?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.create_stream()?;
    let kernels = load_kernels(gpu).context("resolve exact FFN and ordinary K1 kernels")?;
    for (index, (&rows, &n)) in ROW_TIER_BOUNDARIES.iter().zip(&SMALL_WIDTHS).enumerate() {
        let dual = data::random_dual(rows, n, 512, 0xe8af_1000_0000_0000 | index as u64);
        run_dual(gpu, stream, kernels, "tier-boundary", &dual, 1.0, 0.5)?;
        let silu = data::random_silu(rows, n, 512, 0xe8af_2000_0000_0000 | index as u64);
        run_silu(gpu, stream, kernels, "tier-boundary", &silu, 0.25)?;
    }
    let cancellation_dual = data::cancellation_dual(17, 8);
    run_dual(
        gpu,
        stream,
        kernels,
        "association-cancellation",
        &cancellation_dual,
        1.0,
        1.0,
    )?;
    let cancellation_silu = data::cancellation_silu(17, 8);
    run_silu(
        gpu,
        stream,
        kernels,
        "inline-SiLU association-cancellation",
        &cancellation_silu,
        1.0,
    )?;
    materialized::run_cancellation(gpu, stream, kernels)?;
    let production_dual =
        data::random_dual(PRODUCTION_ROWS, INTERMEDIATE, HIDDEN, 0xe8af_5120_1740_8005);
    run_dual(
        gpu,
        stream,
        kernels,
        "Qwen3.8 production gate/up",
        &production_dual,
        0.003_906_25,
        0.007_812_5,
    )?;
    drop(production_dual);
    let production_silu =
        data::random_silu(PRODUCTION_ROWS, HIDDEN, INTERMEDIATE, 0xe8af_1740_8512_0005);
    run_silu(
        gpu,
        stream,
        kernels,
        "Qwen3.8 production inline-SiLU down",
        &production_silu,
        0.003_906_25,
    )?;
    materialized::run_production(gpu, stream, kernels)?;
    println!("PASS: exact dynamic-M dense FFN raw-BF16 K1 parity matrix complete");
    Ok(())
}
