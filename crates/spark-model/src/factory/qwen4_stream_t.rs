// SPDX-License-Identifier: AGPL-3.0-only

//! Post-load allocation of one shared Qwen4 routed-expert transpose arena.

use anyhow::{Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use crate::layer::{MoeStreamTransposeScratch, TransformerLayer};

const EXPERTS: usize = 512;
const INTER: usize = 640;
const HIDDEN: usize = 2560;
const PROJECTIONS: usize = 3;
const PACKED_PER_EXPERT: usize = INTER * HIDDEN / 2;
const SCALE_PER_EXPERT: usize = INTER * HIDDEN / 16;
const PACKED_BYTES: usize = EXPERTS * PACKED_PER_EXPERT;
const SCALE_BYTES: usize = EXPERTS * SCALE_PER_EXPERT;
const SCRATCH_BYTES: usize = PROJECTIONS * (PACKED_BYTES + SCALE_BYTES);
const EXPECTED_SCRATCH_BYTES: usize = 1_415_577_600;
const SAFETY_BYTES: usize = 4 * 1024 * 1024 * 1024;

fn pointer_table(gpu: &dyn GpuBackend, base: DevicePtr, stride: usize) -> Result<DevicePtr> {
    ensure!(
        !base.is_null(),
        "Qwen4 streaming transpose allocation is null"
    );
    let mut bytes = Vec::with_capacity(EXPERTS * 8);
    for expert in 0..EXPERTS {
        let offset = expert
            .checked_mul(stride)
            .ok_or_else(|| anyhow::anyhow!("Qwen4 stream pointer offset overflow"))?;
        bytes.extend_from_slice(&base.offset(offset).0.to_le_bytes());
    }
    let table = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(&bytes, table)?;
    Ok(table)
}

pub(super) fn maybe_setup_qwen4_stream_t(
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    layers: &mut [Box<dyn TransformerLayer>],
) -> Result<()> {
    if !crate::layers::moe::qwen4_prefill_compact::stream_selected()? {
        return Ok(());
    }
    ensure!(
        config.is_qwen4_exp()
            && config.num_experts == EXPERTS
            && config.moe_intermediate_size == INTER
            && config.hidden_size == HIDDEN
            && config.num_hidden_layers == 48,
        "Qwen4 streaming transpose requires canonical Flash-Next geometry"
    );
    ensure!(
        SCRATCH_BYTES == EXPECTED_SCRATCH_BYTES,
        "Qwen4 streaming transpose byte contract changed"
    );
    let free = gpu.free_memory()?;
    ensure!(
        free >= SCRATCH_BYTES + SAFETY_BYTES,
        "Qwen4 streaming transpose needs {} scratch bytes plus safety, only {} free",
        SCRATCH_BYTES,
        free
    );
    let packed = [
        gpu.alloc(PACKED_BYTES)?,
        gpu.alloc(PACKED_BYTES)?,
        gpu.alloc(PACKED_BYTES)?,
    ];
    let scale = [
        gpu.alloc(SCALE_BYTES)?,
        gpu.alloc(SCALE_BYTES)?,
        gpu.alloc(SCALE_BYTES)?,
    ];
    let packed_tables = [
        pointer_table(gpu, packed[0], PACKED_PER_EXPERT)?,
        pointer_table(gpu, packed[1], PACKED_PER_EXPERT)?,
        pointer_table(gpu, packed[2], PACKED_PER_EXPERT)?,
    ];
    let scale_tables = [
        pointer_table(gpu, scale[0], SCALE_PER_EXPERT)?,
        pointer_table(gpu, scale[1], SCALE_PER_EXPERT)?,
        pointer_table(gpu, scale[2], SCALE_PER_EXPERT)?,
    ];
    let scratch = MoeStreamTransposeScratch {
        packed,
        scale,
        packed_tables,
        scale_tables,
    };
    for layer in layers.iter_mut() {
        layer.set_moe_stream_transpose_scratch(scratch);
    }
    tracing::info!(
        bytes = SCRATCH_BYTES,
        free_after = gpu.free_memory()?,
        "Qwen4 one-layer streaming transpose scratch enabled"
    );
    Ok(())
}
