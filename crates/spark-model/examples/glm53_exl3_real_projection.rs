// SPDX-License-Identifier: AGPL-3.0-only

//! Real-checkpoint execution gate for one GLM-5.3 EXL3 projection.

use std::collections::BTreeMap;

use anyhow::{Context, Result, bail, ensure};
use half::{bf16, f16};
use spark_model::layers::ops::{
    GLM53_EXL3_LOCK_BYTES, GLM53_EXL3_MAX_INPUT_F16_BYTES, GLM53_EXL3_MAX_OUTPUT_F16_BYTES,
    Glm53Exl3Buffer, Glm53Exl3Output, Glm53Exl3Projection, Glm53Exl3ProjectionBuffers,
    Glm53Exl3ProjectionScratch,
};
use spark_model::weight_loader::{
    GLM53_EXL3_TARGET_LINEAR_COUNT, GLM53_EXL3_TARGET_RAW_COUNT, GLM53_EXL3_VISION_LINEAR_COUNT,
    GLM53_EXL3_VISION_RAW_COUNT, Glm53Exl3Bf16LinearBuffers, Glm53Exl3Linear,
    Glm53Exl3LinearBuffers, Glm53Exl3TargetCatalog, Glm53Exl3VisionCatalog, admit_glm53_exl3_files,
    load_glm53_exl3_store, materialize_glm53_exl3_native,
};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

const RESERVE_BYTES: usize = 16 * 1024 * 1024 * 1024;
const PROJECTION: &str = "model.language_model.layers.0.self_attn.qkv_proj";
const VISION_PROJECTION: &str = "model.visual.blocks.0.attn.q_proj";
const LANGUAGE_PROJECTIONS: usize = 37_419;

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
            Some(error) => Err(error).context("free real EXL3 projection scratch"),
            None => Ok(()),
        }
    }
}

fn input_bytes(size_k: usize) -> Vec<u8> {
    (0..size_k)
        .flat_map(|index| {
            let value = ((index % 31) as f32 - 15.0) / 32.0;
            f16::from_f32(value).to_bits().to_le_bytes()
        })
        .collect()
}

fn input_bf16_bytes(size_k: usize) -> Vec<u8> {
    (0..size_k)
        .flat_map(|index| {
            let value = ((index % 31) as f32 - 15.0) / 32.0;
            bf16::from_f32(value).to_bits().to_le_bytes()
        })
        .collect()
}

fn input_bf16_bytes_rows(rows: usize, size_k: usize) -> Vec<u8> {
    (0..rows)
        .flat_map(|row| {
            (0..size_k).flat_map(move |index| {
                let value = (((index + row * 7) % 31) as f32 - 15.0) / 32.0;
                bf16::from_f32(value).to_bits().to_le_bytes()
            })
        })
        .collect()
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    })
}

fn execute(gpu: &dyn GpuBackend, linear: &Glm53Exl3Linear) -> Result<(Vec<u8>, usize)> {
    let prepared = linear.prepare(gpu, 1, Glm53Exl3Output::F16)?;
    let plan = prepared.plan();
    let stream = gpu.create_stream()?;
    let mut allocations = Allocations::new(gpu);
    let result = (|| {
        let input_f16 = allocations.alloc(plan.input_bytes)?;
        gpu.copy_h2d(&input_bytes(linear.size_k() as usize), input_f16.ptr)?;
        let output = allocations.alloc(plan.output_bytes)?;
        gpu.copy_h2d(&vec![0x5au8; plan.output_bytes], output.ptr)?;
        let locks_i32 = allocations.zeroed(1024 * 1024 * size_of::<i32>())?;
        let input_hadamard_f16 = allocations.zeroed(plan.input_bytes)?;
        prepared.launch(
            gpu,
            Glm53Exl3LinearBuffers {
                input_f16,
                output,
                locks_i32,
                input_hadamard_f16,
            },
            stream,
        )?;
        gpu.synchronize(stream)?;
        let mut raw = vec![0u8; plan.output_bytes];
        gpu.copy_d2h(output.ptr, &mut raw)?;
        let values = raw
            .chunks_exact(2)
            .map(|pair| f16::from_bits(u16::from_le_bytes([pair[0], pair[1]])))
            .collect::<Vec<_>>();
        ensure!(
            values.iter().all(|value| value.is_finite()),
            "real EXL3 projection produced a non-finite F16 value"
        );
        let nonzero = values
            .iter()
            .filter(|value| value.to_bits() & 0x7fff != 0)
            .count();
        ensure!(
            nonzero != 0,
            "real EXL3 projection produced only zero values"
        );
        Ok((raw, nonzero))
    })();
    let cleanup = allocations.free_all();
    match (result, cleanup) {
        (Ok(value), Ok(())) => Ok(value),
        (Err(error), Ok(())) => Err(error),
        (Ok(_), Err(cleanup)) => Err(cleanup),
        (Err(error), Err(cleanup)) => {
            Err(error).context(format!("EXL3 scratch cleanup also failed: {cleanup:#}"))
        }
    }
}

fn execute_bf16(gpu: &dyn GpuBackend, linear: &Glm53Exl3Linear) -> Result<Vec<u8>> {
    let prepared = linear.prepare_bf16(gpu, 1)?;
    let plan = prepared.plan();
    let stream = gpu.create_stream()?;
    let mut allocations = Allocations::new(gpu);
    let result = (|| {
        let input_bf16 = allocations.alloc(plan.input_bytes)?;
        gpu.copy_h2d(&input_bf16_bytes(linear.size_k() as usize), input_bf16.ptr)?;
        let output_bf16 = allocations.alloc(plan.output_bytes)?;
        let input_f16 = allocations.zeroed(plan.input_bytes)?;
        let output_f16 = allocations.zeroed(plan.output_bytes)?;
        let locks_i32 = allocations.zeroed(1024 * 1024 * size_of::<i32>())?;
        let input_hadamard_f16 = allocations.zeroed(plan.input_bytes)?;
        prepared.launch(
            gpu,
            Glm53Exl3Bf16LinearBuffers {
                input_bf16,
                output_bf16,
                input_f16,
                output_f16,
                locks_i32,
                input_hadamard_f16,
            },
            stream,
        )?;
        gpu.synchronize(stream)?;
        let mut raw = vec![0u8; plan.output_bytes];
        gpu.copy_d2h(output_bf16.ptr, &mut raw)?;
        ensure!(
            raw.chunks_exact(2).all(|pair| {
                bf16::from_bits(u16::from_le_bytes([pair[0], pair[1]])).is_finite()
            }),
            "BF16-bridged EXL3 projection produced a non-finite value"
        );
        Ok(raw)
    })();
    let cleanup = allocations.free_all();
    match (result, cleanup) {
        (Ok(value), Ok(())) => Ok(value),
        (Err(error), Ok(())) => Err(error),
        (Ok(_), Err(cleanup)) => Err(cleanup),
        (Err(error), Err(cleanup)) => Err(error).context(format!(
            "BF16 EXL3 scratch cleanup also failed: {cleanup:#}"
        )),
    }
}

fn execute_wide_bf16(
    gpu: &dyn GpuBackend,
    linear: &Glm53Exl3Linear,
    rows: u32,
    row_exact: bool,
) -> Result<Vec<u8>> {
    let wide = linear.prepare_bf16_row_exact(gpu, rows)?;
    let plan = wide.plan();
    let stream = gpu.create_stream()?;
    let mut allocations = Allocations::new(gpu);
    let result = (|| {
        let input_bf16 = allocations.alloc(plan.input_bytes)?;
        gpu.copy_h2d(
            &input_bf16_bytes_rows(rows as usize, linear.size_k() as usize),
            input_bf16.ptr,
        )?;
        let output_bf16 = allocations.alloc(plan.output_bytes)?;
        let input_f16 = allocations.zeroed(plan.input_bytes)?;
        let output_f16 = allocations.zeroed(plan.output_bytes)?;
        let locks_i32 = allocations.zeroed(GLM53_EXL3_LOCK_BYTES)?;
        let input_hadamard_f16 = allocations.zeroed(plan.input_bytes)?;
        let buffers = Glm53Exl3Bf16LinearBuffers {
            input_bf16,
            output_bf16,
            input_f16,
            output_f16,
            locks_i32,
            input_hadamard_f16,
        };
        if row_exact {
            wide.launch(gpu, buffers, stream)?;
        } else {
            let one = linear.prepare_bf16(gpu, 1)?;
            let one_plan = one.plan();
            for row in 0..rows as usize {
                one.launch(
                    gpu,
                    Glm53Exl3Bf16LinearBuffers {
                        input_bf16: Glm53Exl3Buffer {
                            ptr: input_bf16.ptr.offset(row * one_plan.input_bytes),
                            bytes: one_plan.input_bytes,
                        },
                        output_bf16: Glm53Exl3Buffer {
                            ptr: output_bf16.ptr.offset(row * one_plan.output_bytes),
                            bytes: one_plan.output_bytes,
                        },
                        input_f16: Glm53Exl3Buffer {
                            ptr: buffers.input_f16.ptr,
                            bytes: one_plan.input_bytes,
                        },
                        output_f16: Glm53Exl3Buffer {
                            ptr: buffers.output_f16.ptr,
                            bytes: one_plan.output_bytes,
                        },
                        locks_i32: buffers.locks_i32,
                        input_hadamard_f16: Glm53Exl3Buffer {
                            ptr: buffers.input_hadamard_f16.ptr,
                            bytes: one_plan.input_bytes,
                        },
                    },
                    stream,
                )?;
            }
        }
        gpu.synchronize(stream)?;
        let mut raw = vec![0u8; plan.output_bytes];
        gpu.copy_d2h(output_bf16.ptr, &mut raw)?;
        Ok(raw)
    })();
    let cleanup = allocations.free_all();
    match (result, cleanup) {
        (Ok(value), Ok(())) => Ok(value),
        (Err(error), Ok(())) => Err(error),
        (Ok(_), Err(cleanup)) => Err(cleanup),
        (Err(error), Err(cleanup)) => Err(error).context(format!(
            "wide BF16 EXL3 scratch cleanup also failed: {cleanup:#}"
        )),
    }
}

fn execute_unified(gpu: &dyn GpuBackend, projection: Glm53Exl3Projection<'_>) -> Result<Vec<u8>> {
    let plan = projection.plan(1)?;
    let stream = gpu.create_stream()?;
    let mut allocations = Allocations::new(gpu);
    let result = (|| {
        let input_bf16 = allocations.alloc(plan.input_bytes)?;
        gpu.copy_h2d(&input_bf16_bytes(plan.input as usize), input_bf16.ptr)?;
        let output_bf16 = allocations.alloc(plan.output_bytes)?;
        gpu.copy_h2d(&vec![0x5au8; plan.output_bytes], output_bf16.ptr)?;
        let scratch = Glm53Exl3ProjectionScratch {
            input_f16: allocations.zeroed(GLM53_EXL3_MAX_INPUT_F16_BYTES)?,
            output_f16: allocations.zeroed(GLM53_EXL3_MAX_OUTPUT_F16_BYTES)?,
            locks_i32: allocations.zeroed(GLM53_EXL3_LOCK_BYTES)?,
            input_hadamard_f16: allocations.zeroed(GLM53_EXL3_MAX_INPUT_F16_BYTES)?,
            // Decode-shape probe: no prompt-scope reconstruct staging.
            reconstruct_f16: Glm53Exl3Buffer {
                ptr: spark_runtime::gpu::DevicePtr::NULL,
                bytes: 0,
            },
            reconstruct_f16_b: Glm53Exl3Buffer {
                ptr: spark_runtime::gpu::DevicePtr::NULL,
                bytes: 0,
            },
        };
        projection.launch(
            gpu,
            plan,
            Glm53Exl3ProjectionBuffers {
                input_bf16,
                output_bf16,
                scratch,
            },
            stream,
        )?;
        gpu.synchronize(stream)?;
        let mut raw = vec![0u8; plan.output_bytes];
        gpu.copy_d2h(output_bf16.ptr, &mut raw)?;
        ensure!(
            raw.chunks_exact(2).all(|pair| {
                bf16::from_bits(u16::from_le_bytes([pair[0], pair[1]])).is_finite()
            }),
            "unified EXL3 projection produced a non-finite BF16 value"
        );
        ensure!(
            raw.chunks_exact(2)
                .any(|pair| u16::from_le_bytes([pair[0], pair[1]]) & 0x7fff != 0),
            "unified EXL3 projection produced only zero values"
        );
        Ok(raw)
    })();
    let cleanup = allocations.free_all();
    match (result, cleanup) {
        (Ok(value), Ok(())) => Ok(value),
        (Err(error), Ok(())) => Err(error),
        (Ok(_), Err(cleanup)) => Err(cleanup),
        (Err(error), Err(cleanup)) => Err(error).context(format!(
            "unified EXL3 projection scratch cleanup also failed: {cleanup:#}"
        )),
    }
}

fn f16_output_as_bf16(raw: &[u8]) -> Vec<u8> {
    raw.chunks_exact(2)
        .flat_map(|pair| {
            let value = f16::from_bits(u16::from_le_bytes([pair[0], pair[1]]));
            bf16::from_f32(value.to_f32()).to_bits().to_le_bytes()
        })
        .collect()
}

fn bind_language_census(
    store: &spark_model::weight_loader::Glm53Exl3DeviceStore,
) -> Result<[usize; 6]> {
    let logical_names = store
        .names()
        .filter_map(|name| name.strip_suffix(".trellis"))
        .filter(|name| name.starts_with("model.language_model.") || *name == "lm_head")
        .collect::<Vec<_>>();
    ensure!(
        logical_names.len() == LANGUAGE_PROJECTIONS + 1,
        "real EXL3 language projection census drift: got {}",
        logical_names.len()
    );
    let mut bits = [0usize; 6];
    for name in logical_names {
        let linear = Glm53Exl3Linear::bind(store, name)
            .with_context(|| format!("bind admitted language projection {name}"))?;
        bits[usize::from(linear.bits())] += 1;
    }
    Ok(bits)
}

fn rowexact_geometry_census(
    gpu: &dyn GpuBackend,
    store: &spark_model::weight_loader::Glm53Exl3DeviceStore,
) -> Result<(usize, u64)> {
    let mut representatives = BTreeMap::new();
    for name in store
        .names()
        .filter_map(|name| name.strip_suffix(".trellis"))
        .filter(|name| name.starts_with("model.language_model.") || *name == "lm_head")
    {
        let linear = Glm53Exl3Linear::bind(store, name)?;
        representatives
            .entry((linear.bits(), linear.size_k(), linear.size_n()))
            .or_insert_with(|| name.to_owned());
    }

    let mut combined_hash = 0u64;
    for ((bits, size_k, size_n), name) in &representatives {
        let linear = Glm53Exl3Linear::bind(store, name)?;
        let serial = execute_wide_bf16(gpu, &linear, 8, false)?;
        let row_exact = execute_wide_bf16(gpu, &linear, 8, true)?;
        ensure!(
            row_exact == serial,
            "row-exact output drift for {name} (K{bits}, {size_k}x{size_n})"
        );
        let hash = fnv1a64(&row_exact);
        combined_hash ^= hash.rotate_left(u32::from(*bits));
        println!(
            "ROWEXACT: PASS bits={bits} k={size_k} n={size_n} fnv1a64={hash:016x} projection={name}"
        );
    }
    Ok((representatives.len(), combined_hash))
}

fn main() -> Result<()> {
    let root = std::env::args_os()
        .nth(1)
        .map(std::path::PathBuf::from)
        .context("usage: glm53_exl3_real_projection <exact-checkpoint-directory>")?;
    let files = admit_glm53_exl3_files(&root)?;
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let store = match load_glm53_exl3_store(&files, gpu, RESERVE_BYTES) {
        Ok(store) => store,
        Err(error) => match error.retry_cleanup(gpu) {
            Ok(primary) => return Err(primary),
            Err(retained) => bail!("EXL3 load and cleanup failed: {retained}"),
        },
    };
    let allocated_bytes = store.allocated_bytes();
    let payload_bytes = store.payload_bytes();
    let mut native_store = None;
    let result = (|| {
        let catalog = Glm53Exl3TargetCatalog::new(&files, &store)?;
        let vision_catalog = Glm53Exl3VisionCatalog::new(&files, &store)?;
        ensure!(
            catalog.linear_count() == GLM53_EXL3_TARGET_LINEAR_COUNT
                && catalog.raw_count() == GLM53_EXL3_TARGET_RAW_COUNT,
            "real EXL3 target catalog count drift"
        );
        ensure!(
            vision_catalog.linear_count() == GLM53_EXL3_VISION_LINEAR_COUNT
                && vision_catalog.raw_count() == GLM53_EXL3_VISION_RAW_COUNT,
            "real EXL3 vision catalog count drift"
        );
        let native = materialize_glm53_exl3_native(&catalog, gpu).map_err(|error| {
            let retained = error.retained_bytes();
            match error.retry_cleanup(gpu) {
                Ok(primary) => primary.context(format!(
                    "materialize graph-native target operands; retained before cleanup={retained}"
                )),
                Err(error) => anyhow::anyhow!("{error}; retained={}", error.retained_bytes()),
            }
        })?;
        ensure!(
            native.len() == GLM53_EXL3_TARGET_RAW_COUNT && !native.is_empty(),
            "real EXL3 graph-native store count drift"
        );
        let native_count = native.materialized_count();
        let native_bytes = native.materialized_bytes();
        let bits = bind_language_census(&store)?;
        let (rowexact_geometries, rowexact_census_hash) = rowexact_geometry_census(gpu, &store)?;
        let linear = Glm53Exl3Linear::bind(&store, PROJECTION)?;
        let wide_serial = execute_wide_bf16(gpu, &linear, 8, false)?;
        let wide_row_exact = execute_wide_bf16(gpu, &linear, 8, true)?;
        ensure!(
            wide_row_exact == wide_serial,
            "row-exact cooperative output differs from ordered M=1 launches"
        );
        let wide_row_exact_hash = fnv1a64(&wide_row_exact);
        let direct = execute(gpu, &linear)?;
        let expected_bf16 = f16_output_as_bf16(&direct.0);
        let first = execute_bf16(gpu, &linear)?;
        let second = execute_bf16(gpu, &linear)?;
        ensure!(
            first == second,
            "BF16-bridged EXL3 projection is not byte-deterministic across fresh scratch"
        );
        ensure!(
            first == expected_bf16,
            "BF16 EXL3 bridge differs from exact CPU rounding of direct F16 output"
        );
        let unified_compressed = execute_unified(
            gpu,
            Glm53Exl3Projection::Compressed(
                catalog
                    .linear(PROJECTION)
                    .context("catalog lost layer0 QKV projection")?,
            ),
        )?;
        ensure!(
            unified_compressed == first,
            "unified compressed projection differs from qualified BF16 bridge"
        );
        let native_f_a_name = "model.language_model.layers.0.self_attn.f_a_proj.weight";
        let native_f_a = native
            .get(native_f_a_name)
            .context("native store lost layer0 f_a projection")?;
        let unified_native = execute_unified(gpu, Glm53Exl3Projection::NativeBf16(native_f_a))?;
        let unified_native_second =
            execute_unified(gpu, Glm53Exl3Projection::NativeBf16(native_f_a))?;
        ensure!(
            unified_native == unified_native_second,
            "unified native projection is not deterministic"
        );
        let unified_native_hash = fnv1a64(&unified_native);
        let vision_q = vision_catalog
            .linear(VISION_PROJECTION)
            .context("vision catalog lost block0 Q projection")?;
        let vision_first = execute_unified(gpu, Glm53Exl3Projection::Compressed(vision_q))?;
        let vision_second = execute_unified(gpu, Glm53Exl3Projection::Compressed(vision_q))?;
        ensure!(
            vision_first == vision_second,
            "unified GLM vision projection is not deterministic"
        );
        native_store = Some(native);
        Ok((
            fnv1a64(&direct.0),
            fnv1a64(&first),
            direct.1,
            bits,
            catalog.linear_count(),
            catalog.raw_count(),
            native_count,
            native_bytes,
            fnv1a64(&unified_compressed),
            unified_native_hash,
            vision_catalog.linear_count(),
            vision_catalog.raw_count(),
            fnv1a64(&vision_first),
            wide_row_exact_hash,
            rowexact_geometries,
            rowexact_census_hash,
        ))
    })();
    let native_cleanup = match native_store.take() {
        Some(native) => native.free(gpu),
        None => Ok(()),
    };
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
    let (
        f16_hash,
        bf16_hash,
        nonzero,
        bits,
        target_linears,
        target_raw,
        native_count,
        native_bytes,
        unified_compressed_hash,
        unified_native_hash,
        vision_linears,
        vision_raw,
        vision_projection_hash,
        wide_row_exact_hash,
        rowexact_geometries,
        rowexact_census_hash,
    ) = match (result, native_cleanup, cleanup) {
        (Ok(value), Ok(()), Ok(())) => value,
        (Err(error), Ok(()), Ok(())) => return Err(error),
        (Ok(_), Err(cleanup), Ok(())) | (Ok(_), Ok(()), Err(cleanup)) => {
            return Err(cleanup);
        }
        (Err(error), Err(native), Ok(())) => {
            return Err(error).context(format!("native cleanup also failed: {native:#}"));
        }
        (Err(error), Ok(()), Err(store)) => {
            return Err(error).context(format!("EXL3 store cleanup also failed: {store:#}"));
        }
        (Ok(_), Err(native), Err(store)) => {
            return Err(native).context(format!("EXL3 store cleanup also failed: {store:#}"));
        }
        (Err(error), Err(native), Err(store)) => {
            return Err(error).context(format!(
                "native cleanup failed: {native:#}; EXL3 store cleanup failed: {store:#}"
            ));
        }
    };
    println!(
        "RESULT: PASS projection={PROJECTION} target_linears={target_linears} target_raw={target_raw} vision_linears={vision_linears} vision_raw={vision_raw} native_materialized={native_count} native_bytes={native_bytes} language_projections={} lm_head=1 bit_census=k2:{},k3:{},k4:{},k5:{} payload_bytes={payload_bytes} store_bytes={allocated_bytes} k=4096 n=24576 bits=4 bf16_repeats=2 rowexact_rows=8 rowexact_geometries={rowexact_geometries} rowexact_census_hash={rowexact_census_hash:016x} nonzero={nonzero} f16_fnv1a64={f16_hash:016x} bf16_fnv1a64={bf16_hash:016x} wide_rowexact_fnv1a64={wide_row_exact_hash:016x} unified_compressed_fnv1a64={unified_compressed_hash:016x} unified_native_fa_fnv1a64={unified_native_hash:016x} vision_q_fnv1a64={vision_projection_hash:016x}",
        LANGUAGE_PROJECTIONS, bits[2], bits[3], bits[4], bits[5]
    );
    Ok(())
}
