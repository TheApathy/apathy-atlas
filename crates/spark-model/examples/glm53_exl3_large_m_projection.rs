// SPDX-License-Identifier: AGPL-3.0-only

//! Real-checkpoint large-M execution and timing probe for one GLM-5.3 EXL3 projection.

use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use half::bf16;
use spark_model::layers::ops::{GLM53_EXL3_LOCK_BYTES, Glm53Exl3Buffer};
use spark_model::weight_loader::{
    Glm53Exl3Bf16LinearBuffers, Glm53Exl3Linear, admit_glm53_exl3_files, load_glm53_exl3_store,
};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

const RESERVE_BYTES: usize = 16 * 1024 * 1024 * 1024;
const DEFAULT_PROJECTION: &str = "model.language_model.layers.0.self_attn.qkv_proj";
const ROWS: [u32; 4] = [128, 512, 1024, 2048];
const WARMUPS: usize = 2;
const SAMPLES: usize = 7;

struct Allocations<'a> {
    gpu: &'a dyn GpuBackend,
    pointers: Vec<DevicePtr>,
}

impl<'a> Allocations<'a> {
    fn new(gpu: &'a dyn GpuBackend) -> Self {
        Self {
            gpu,
            pointers: Vec::new(),
        }
    }

    fn alloc(&mut self, bytes: usize) -> Result<Glm53Exl3Buffer> {
        let ptr = self.gpu.alloc(bytes)?;
        self.pointers.push(ptr);
        Ok(Glm53Exl3Buffer { ptr, bytes })
    }

    fn zeroed(&mut self, bytes: usize) -> Result<Glm53Exl3Buffer> {
        let buffer = self.alloc(bytes)?;
        self.gpu.memset(buffer.ptr, 0, bytes)?;
        Ok(buffer)
    }

    fn free_all(mut self) -> Result<()> {
        let mut first_error = None;
        for ptr in self.pointers.drain(..).rev() {
            if let Err(error) = self.gpu.free(ptr) {
                first_error.get_or_insert(error);
            }
        }
        match first_error {
            Some(error) => Err(error).context("free large-M projection probe buffers"),
            None => Ok(()),
        }
    }
}

#[derive(Debug)]
struct ShapeResult {
    rows: u32,
    median_ms: f64,
    rows_per_second: f64,
    hash: u64,
    mismatches: usize,
    max_abs_error: f32,
    rmse: f64,
}

fn input_bytes(rows: u32, width: u32) -> Vec<u8> {
    (0..rows)
        .flat_map(|row| {
            (0..width).flat_map(move |column| {
                let numerator = ((column as u64 + u64::from(row) * 17) % 61) as f32 - 30.0;
                bf16::from_f32(numerator / 64.0).to_bits().to_le_bytes()
            })
        })
        .collect()
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    })
}

fn prefix(buffer: Glm53Exl3Buffer, bytes: usize) -> Glm53Exl3Buffer {
    assert!(buffer.bytes >= bytes);
    Glm53Exl3Buffer {
        ptr: buffer.ptr,
        bytes,
    }
}

fn row(buffer: Glm53Exl3Buffer, index: usize, row_bytes: usize) -> Glm53Exl3Buffer {
    assert!(buffer.bytes >= (index + 1) * row_bytes);
    Glm53Exl3Buffer {
        ptr: buffer.ptr.offset(index * row_bytes),
        bytes: row_bytes,
    }
}

fn compare_bf16(reference: &[u8], candidate: &[u8]) -> Result<(usize, f32, f64)> {
    ensure!(
        reference.len() == candidate.len() && reference.len() % 2 == 0,
        "large-M comparison extent drift"
    );
    let mut mismatches = 0usize;
    let mut max_abs_error = 0.0f32;
    let mut squared_error = 0.0f64;
    for (expected, actual) in reference.chunks_exact(2).zip(candidate.chunks_exact(2)) {
        let expected = bf16::from_bits(u16::from_le_bytes([expected[0], expected[1]])).to_f32();
        let actual = bf16::from_bits(u16::from_le_bytes([actual[0], actual[1]])).to_f32();
        ensure!(
            expected.is_finite() && actual.is_finite(),
            "large-M projection produced a non-finite value"
        );
        let error = (actual - expected).abs();
        if error != 0.0 {
            mismatches += 1;
        }
        max_abs_error = max_abs_error.max(error);
        squared_error += f64::from(error) * f64::from(error);
    }
    let elements = reference.len() / 2;
    let rmse = (squared_error / elements as f64).sqrt();
    Ok((mismatches, max_abs_error, rmse))
}

fn execute_shape(gpu: &dyn GpuBackend, linear: &Glm53Exl3Linear, rows: u32) -> Result<ShapeResult> {
    let batch = linear.prepare_bf16(gpu, rows)?;
    let one = linear.prepare_bf16(gpu, 1)?;
    let batch_plan = batch.plan();
    let one_plan = one.plan();
    ensure!(
        batch_plan.size_k == one_plan.size_k
            && batch_plan.size_n == one_plan.size_n
            && batch_plan.input_bytes == rows as usize * one_plan.input_bytes
            && batch_plan.output_bytes == rows as usize * one_plan.output_bytes,
        "large-M and M1 projection geometry drift"
    );

    let stream = gpu.create_stream()?;
    let mut allocations = Allocations::new(gpu);
    let result = (|| {
        let input_bf16 = allocations.alloc(batch_plan.input_bytes)?;
        gpu.copy_h2d(&input_bytes(rows, linear.size_k()), input_bf16.ptr)?;
        let batch_output_bf16 = allocations.zeroed(batch_plan.output_bytes)?;
        let serial_output_bf16 = allocations.zeroed(batch_plan.output_bytes)?;
        let batch_input_f16 = allocations.zeroed(batch_plan.input_bytes)?;
        let batch_output_f16 = allocations.zeroed(batch_plan.output_bytes)?;
        let batch_hadamard_f16 = allocations.zeroed(batch_plan.input_bytes)?;
        let one_input_f16 = allocations.zeroed(one_plan.input_bytes)?;
        let one_output_f16 = allocations.zeroed(one_plan.output_bytes)?;
        let one_hadamard_f16 = allocations.zeroed(one_plan.input_bytes)?;
        let locks_i32 = allocations.zeroed(GLM53_EXL3_LOCK_BYTES)?;

        for index in 0..rows as usize {
            one.launch(
                gpu,
                Glm53Exl3Bf16LinearBuffers {
                    input_bf16: row(input_bf16, index, one_plan.input_bytes),
                    output_bf16: row(serial_output_bf16, index, one_plan.output_bytes),
                    input_f16: one_input_f16,
                    output_f16: one_output_f16,
                    locks_i32,
                    input_hadamard_f16: one_hadamard_f16,
                },
                stream,
            )?;
        }
        gpu.synchronize(stream)?;

        let batch_buffers = Glm53Exl3Bf16LinearBuffers {
            input_bf16,
            output_bf16: batch_output_bf16,
            input_f16: prefix(batch_input_f16, batch_plan.input_bytes),
            output_f16: prefix(batch_output_f16, batch_plan.output_bytes),
            locks_i32,
            input_hadamard_f16: prefix(batch_hadamard_f16, batch_plan.input_bytes),
        };
        gpu.memset(locks_i32.ptr, 0, locks_i32.bytes)?;
        batch.launch(gpu, batch_buffers, stream)?;
        gpu.synchronize(stream)?;

        let mut reference = vec![0u8; batch_plan.output_bytes];
        let mut candidate = vec![0u8; batch_plan.output_bytes];
        gpu.copy_d2h(serial_output_bf16.ptr, &mut reference)?;
        gpu.copy_d2h(batch_output_bf16.ptr, &mut candidate)?;
        let (mismatches, max_abs_error, rmse) = compare_bf16(&reference, &candidate)?;
        let initial_hash = fnv1a64(&candidate);
        ensure!(
            max_abs_error <= 0.125 && rmse <= 0.01,
            "large-M projection exceeds M1 error bounds: max_abs={max_abs_error}, rmse={rmse}"
        );

        for _ in 0..WARMUPS {
            gpu.memset(locks_i32.ptr, 0, locks_i32.bytes)?;
            batch.launch(gpu, batch_buffers, stream)?;
            gpu.synchronize(stream)?;
        }
        let mut samples_ms = Vec::with_capacity(SAMPLES);
        for _ in 0..SAMPLES {
            gpu.memset(locks_i32.ptr, 0, locks_i32.bytes)?;
            let started = Instant::now();
            batch.launch(gpu, batch_buffers, stream)?;
            gpu.synchronize(stream)?;
            samples_ms.push(started.elapsed().as_secs_f64() * 1_000.0);
        }
        samples_ms.sort_by(f64::total_cmp);
        let median_ms = samples_ms[SAMPLES / 2];
        gpu.copy_d2h(batch_output_bf16.ptr, &mut candidate)?;
        let repeated_hash = fnv1a64(&candidate);
        ensure!(
            repeated_hash == initial_hash,
            "large-M output hash is unstable"
        );
        Ok(ShapeResult {
            rows,
            median_ms,
            rows_per_second: f64::from(rows) * 1_000.0 / median_ms,
            hash: repeated_hash,
            mismatches,
            max_abs_error,
            rmse,
        })
    })();
    let cleanup = allocations.free_all();
    match (result, cleanup) {
        (Ok(value), Ok(())) => Ok(value),
        (Err(error), Ok(())) => Err(error),
        (Ok(_), Err(cleanup)) => Err(cleanup),
        (Err(error), Err(cleanup)) => Err(error).context(format!(
            "large-M projection probe cleanup also failed: {cleanup:#}"
        )),
    }
}

fn main() -> Result<()> {
    let mut arguments = std::env::args_os().skip(1);
    let root = arguments.next().map(std::path::PathBuf::from).context(
            "usage: glm53_exl3_large_m_projection <exact-checkpoint-directory> [projection[,projection...]]",
    )?;
    let projection_argument = arguments
        .next()
        .map(|value| {
            value
                .into_string()
                .map_err(|_| anyhow::anyhow!("projection path must be UTF-8"))
        })
        .transpose()?
        .unwrap_or_else(|| DEFAULT_PROJECTION.to_owned());
    let projections = projection_argument
        .split(',')
        .map(|projection| {
            ensure!(!projection.is_empty(), "projection path must not be empty");
            Ok(projection.to_owned())
        })
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        arguments.next().is_none(),
        "large-M projection probe accepts at most one projection path"
    );
    let files = admit_glm53_exl3_files(&root)?;
    let rows = match std::env::var("ATLAS_GLM53_EXL3_LARGE_M_ROWS") {
        Ok(value) => value
            .split(',')
            .map(|raw| {
                let rows = raw
                    .parse::<u32>()
                    .with_context(|| format!("invalid large-M row count {raw:?}"))?;
                ensure!(
                    (16..=2_048).contains(&rows),
                    "large-M rows must be 16..=2048"
                );
                Ok(rows)
            })
            .collect::<Result<Vec<_>>>()?,
        Err(std::env::VarError::NotPresent) => ROWS.to_vec(),
        Err(error) => return Err(error).context("read large-M row selection"),
    };
    ensure!(!rows.is_empty(), "large-M row selection must not be empty");
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let store = match load_glm53_exl3_store(&files, gpu, RESERVE_BYTES) {
        Ok(store) => store,
        Err(error) => match error.retry_cleanup(gpu) {
            Ok(primary) => return Err(primary),
            Err(retained) => bail!("EXL3 load and cleanup failed: {retained}"),
        },
    };
    let result = (|| {
        let mut results = Vec::new();
        for projection in projections {
            let linear = Glm53Exl3Linear::bind(&store, &projection)?;
            ensure!(
                linear.bits() == 4
                    && matches!(
                        (linear.size_k(), linear.size_n()),
                        (4_096, 24_576)
                            | (4_096, 2_048)
                            | (8_192, 4_096)
                            | (4_096, 1_536)
                            | (4_096, 512)
                    ),
                "unsupported large-M diagnostic projection geometry"
            );
            for &row_count in &rows {
                results.push((projection.clone(), execute_shape(gpu, &linear, row_count)?));
            }
        }
        Ok::<_, anyhow::Error>(results)
    })();
    let cleanup = match store.free(gpu) {
        Ok(()) => Ok(()),
        Err(error) => error.retry(gpu).map_err(|retry| {
            anyhow::anyhow!(
                "EXL3 store cleanup failed after retry for {} slabs: {:#}",
                retry.failed_slab_count(),
                retry.failure()
            )
        }),
    };
    let results = match (result, cleanup) {
        (Ok(results), Ok(())) => results,
        (Err(error), Ok(())) => return Err(error),
        (Ok(_), Err(cleanup)) => return Err(cleanup),
        (Err(error), Err(cleanup)) => {
            return Err(error).context(format!("EXL3 store cleanup also failed: {cleanup:#}"));
        }
    };
    for (projection, result) in results {
        println!(
            "LARGE_M: PASS projection={projection} rows={} median_ms={:.3} rows_per_second={:.2} mismatches={} max_abs_error={:.6} rmse={:.8} fnv1a64={:016x} warmups={WARMUPS} samples={SAMPLES}",
            result.rows,
            result.median_ms,
            result.rows_per_second,
            result.mismatches,
            result.max_abs_error,
            result.rmse,
            result.hash,
        );
    }
    Ok(())
}
