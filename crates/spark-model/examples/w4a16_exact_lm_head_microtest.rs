// SPDX-License-Identifier: AGPL-3.0-only

//! Live raw-BF16 parity oracle for the exact multi-row NVFP4 LM-head GEMV.

#[path = "w4a16_exact_lm_head_microtest/data.rs"]
mod data;

use anyhow::{Context, Result, bail, ensure};
use data::{Fixture, as_le_bytes, fnv1a64, from_le_bytes};
use spark_model::layers::ops::{self, W4a16ExactLmHeadKernels};
use spark_model::weight_map::QuantizedWeight;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

const REQUIRED_ROWS: [usize; 9] = [2, 3, 4, 5, 8, 16, 17, 31, 32];
const LOGICAL_WIDTHS: [usize; 9] = [12, 13, 7, 11, 16, 17, 19, 23, 29];
const K: usize = 2_048;

#[derive(Clone, Copy)]
struct Kernels {
    serial: KernelHandle,
    exact: W4a16ExactLmHeadKernels,
    legacy_batch3: KernelHandle,
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

fn load_kernels(gpu: &dyn GpuBackend) -> Result<Kernels> {
    let exact = W4a16ExactLmHeadKernels::new(
        gpu.kernel("w4a16_gemv", "w4a16_gemv_batch_logits_exact_m4")?,
        gpu.kernel("w4a16_gemv", "w4a16_gemv_batch_logits_exact_m8")?,
        gpu.kernel("w4a16_gemv", "w4a16_gemv_batch_logits_exact_m17")?,
        gpu.kernel("w4a16_gemv", "w4a16_gemv_batch_logits_exact_m32")?,
    )
    .with_rt2(
        spark_model::layers::try_kernel(
            gpu,
            "w4a16_gemv_rt",
            "w4a16_gemv_batch_logits_exact_rt2_m4",
        ),
        spark_model::layers::try_kernel(
            gpu,
            "w4a16_gemv_rt",
            "w4a16_gemv_batch_logits_exact_rt2_m8",
        ),
        spark_model::layers::try_kernel(
            gpu,
            "w4a16_gemv_rt",
            "w4a16_gemv_batch_logits_exact_rt2_m17",
        ),
        spark_model::layers::try_kernel(
            gpu,
            "w4a16_gemv_rt",
            "w4a16_gemv_batch_logits_exact_rt2_m32",
        ),
    );
    Ok(Kernels {
        serial: gpu.kernel("w4a16_gemv", "w4a16_gemv")?,
        exact,
        legacy_batch3: gpu.kernel("w4a16_gemv", "w4a16_gemv_batch3")?,
    })
}

fn run_exact_case(
    gpu: &dyn GpuBackend,
    stream: u64,
    kernels: Kernels,
    label: &str,
    fixture: &Fixture,
    scale2: f32,
) -> Result<Vec<u16>> {
    let input = upload(gpu, &as_le_bytes(&fixture.activations))?;
    let packed = upload(gpu, &fixture.packed)?;
    let scales = upload(gpu, &fixture.scales)?;
    let exact_out = gpu.alloc(fixture.rows * fixture.logical_n * size_of::<u16>())?;
    let serial_out = gpu.alloc(fixture.rows * fixture.logical_n * size_of::<u16>())?;
    let weight = QuantizedWeight {
        weight: packed,
        weight_scale: scales,
        weight_scale_2: scale2,
        input_scale: DevicePtr::NULL,
    };

    ops::w4a16_gemv_batch_logits_exact_with(
        gpu,
        kernels.exact,
        input,
        &weight,
        exact_out,
        fixture.rows as u32,
        fixture.logical_n as u32,
        fixture.k as u32,
        stream,
        false,
    )
    .with_context(|| format!("{label}: exact launch"))?;
    for row in 0..fixture.rows {
        ops::w4a16_gemv(
            gpu,
            kernels.serial,
            input.offset(row * fixture.k * size_of::<u16>()),
            &weight,
            serial_out.offset(row * fixture.logical_n * size_of::<u16>()),
            fixture.logical_n as u32,
            fixture.k as u32,
            stream,
        )
        .with_context(|| format!("{label}: serial K1 row {row}"))?;
    }

    // Register-tiled twin, held to the same oracle. A T=2 lane group covers
    // two adjacent outputs, so this also exercises the odd-N tail where the
    // second output of the last group is out of range.
    let rt2_tier = ops::exact_lm_head_tier_for_rows(fixture.rows as u32)
        .expect("microtest rows are all within 2..=32");
    let rt2_present = kernels.exact.rt2_for_tier(rt2_tier).0 != 0;
    let rt2_out = gpu.alloc(fixture.rows * fixture.logical_n * size_of::<u16>())?;
    if rt2_present {
        ops::w4a16_gemv_batch_logits_exact_with(
            gpu,
            kernels.exact,
            input,
            &weight,
            rt2_out,
            fixture.rows as u32,
            fixture.logical_n as u32,
            fixture.k as u32,
            stream,
            true,
        )
        .with_context(|| format!("{label}: rt2 launch"))?;
    }

    let exact = read_bf16(gpu, exact_out, fixture.rows * fixture.logical_n, stream)?;
    let serial = read_bf16(gpu, serial_out, fixture.rows * fixture.logical_n, stream)?;
    if rt2_present {
        let rt2 = read_bf16(gpu, rt2_out, fixture.rows * fixture.logical_n, stream)?;
        if rt2 != serial {
            let first = rt2
                .iter()
                .zip(&serial)
                .position(|(actual, oracle)| actual != oracle)
                .expect("different vectors have a differing element");
            bail!(
                "{label}: rt2 raw BF16 mismatch at flat={first}, row={}, n={}, rt2=0x{:04x}, serial=0x{:04x}",
                first / fixture.logical_n,
                first % fixture.logical_n,
                rt2[first],
                serial[first]
            );
        }
    } else {
        bail!(
            "{label}: rt2 kernel {} absent from the PTX cache",
            rt2_tier.symbol_rt2()
        );
    }
    if exact != serial {
        let first = exact
            .iter()
            .zip(&serial)
            .position(|(actual, oracle)| actual != oracle)
            .expect("different vectors have a differing element");
        bail!(
            "{label}: raw BF16 mismatch at flat={first}, row={}, n={}, exact=0x{:04x}, serial=0x{:04x}",
            first / fixture.logical_n,
            first % fixture.logical_n,
            exact[first],
            serial[first]
        );
    }

    for ptr in [input, packed, scales, exact_out, serial_out, rt2_out] {
        gpu.free(ptr)?;
    }
    println!(
        "PASS {label} (exact+rt2 == serial-K1): M={} N={} weight_rows={} K={} raw_bf16_fnv1a64={:016x}",
        fixture.rows,
        fixture.logical_n,
        fixture.physical_n,
        fixture.k,
        fnv1a64(&exact)
    );
    Ok(exact)
}

fn run_legacy_negative(gpu: &dyn GpuBackend, stream: u64, kernels: Kernels) -> Result<()> {
    let fixture = data::association_negative_fixture();
    let exact = run_exact_case(
        gpu,
        stream,
        kernels,
        "negative-control exact M3",
        &fixture,
        1.0,
    )?;
    let input = upload(gpu, &as_le_bytes(&fixture.activations))?;
    let packed = upload(gpu, &fixture.packed)?;
    let scales = upload(gpu, &fixture.scales)?;
    let legacy_out = gpu.alloc(fixture.rows * fixture.physical_n * size_of::<u16>())?;
    let weight = QuantizedWeight {
        weight: packed,
        weight_scale: scales,
        weight_scale_2: 1.0,
        input_scale: DevicePtr::NULL,
    };
    ops::w4a16_gemv_batch3(
        gpu,
        kernels.legacy_batch3,
        input,
        &weight,
        legacy_out,
        fixture.physical_n as u32,
        fixture.k as u32,
        stream,
    )?;
    let legacy = read_bf16(gpu, legacy_out, fixture.rows * fixture.physical_n, stream)?;
    let mismatches = exact.iter().zip(&legacy).filter(|(a, b)| a != b).count();
    ensure!(
        mismatches > 0,
        "legacy batch3 negative control passed vacuously"
    );
    ensure!(
        exact[0] == 0 && legacy[0] == 0x3f80,
        "negative witness drifted: serial/exact=0x{:04x}, legacy=0x{:04x}",
        exact[0],
        legacy[0]
    );
    for ptr in [input, packed, scales, legacy_out] {
        gpu.free(ptr)?;
    }
    println!(
        "PASS legacy batch3 negative control: mismatched_raw_bf16={mismatches}/{}",
        exact.len()
    );
    Ok(())
}

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())
        .context("initialize CUDA backend with compiled Qwen3.8 kernels")?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.create_stream()?;
    let kernels = load_kernels(gpu).context("resolve exact, serial, and negative kernels")?;

    for (index, (&rows, &logical_n)) in REQUIRED_ROWS.iter().zip(&LOGICAL_WIDTHS).enumerate() {
        let fixture =
            data::random_fixture(rows, logical_n, K, 0xe8ac_7000_0000_0000 | index as u64);
        run_exact_case(gpu, stream, kernels, "random tier boundary", &fixture, 1.0)?;
    }
    let production_m8 = data::random_fixture(5, 19, 5_120, 0xe8ac_7000_5120_0005);
    run_exact_case(
        gpu,
        stream,
        kernels,
        "production M8 hidden width",
        &production_m8,
        1.0,
    )?;
    let production_full = data::random_fixture(5, 248_077, 5_120, 0xe8ac_7000_5120_0248);
    run_exact_case(
        gpu,
        stream,
        kernels,
        "production M8 full vocabulary",
        &production_full,
        0.003_906_25,
    )?;
    let cancellation = data::cancellation_fixture(17, 9, 4_096);
    run_exact_case(
        gpu,
        stream,
        kernels,
        "adversarial cancellation",
        &cancellation,
        1.0,
    )?;
    run_legacy_negative(gpu, stream, kernels)?;
    println!("PASS: exact multi-row LM-head raw BF16 parity matrix complete");
    Ok(())
}
