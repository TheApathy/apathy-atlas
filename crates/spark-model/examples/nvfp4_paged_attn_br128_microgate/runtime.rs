// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Context, Result, bail, ensure};
use spark_model::layers::ops::{
    prefill_attention_paged_nvfp4_64, prefill_attention_paged_nvfp4_128,
};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::contract::{CACHE_BLOCK, Case, HD, NKV, NQ};
use super::fixtures::{as_bytes_u32, cache_fixture, q_fixture};
use super::guarded::Guarded;

#[allow(clippy::too_many_arguments)]
pub(super) fn launch(
    gpu: &dyn GpuBackend,
    stream: u64,
    kernel: KernelHandle,
    br128: bool,
    q: DevicePtr,
    k: DevicePtr,
    v: DevicePtr,
    output: DevicePtr,
    table: DevicePtr,
    case: Case,
    block_stride: u64,
    data_bytes: u64,
) -> Result<()> {
    let args = (
        gpu,
        kernel,
        q,
        k,
        v,
        output,
        table,
        case.q_len,
        case.kv_len(),
        case.q_offset,
        NQ,
        NKV,
        HD,
        CACHE_BLOCK,
        case.sliding_window,
        1.0 / (HD as f32).sqrt(),
        block_stride,
        data_bytes,
        stream,
    );
    if br128 {
        prefill_attention_paged_nvfp4_128(
            args.0, args.1, args.2, args.3, args.4, args.5, args.6, args.7, args.8, args.9,
            args.10, args.11, args.12, args.13, args.14, args.15, args.16, args.17, args.18,
        )
    } else {
        prefill_attention_paged_nvfp4_64(
            args.0, args.1, args.2, args.3, args.4, args.5, args.6, args.7, args.8, args.9,
            args.10, args.11, args.12, args.13, args.14, args.15, args.16, args.17, args.18,
        )
    }
}

pub(super) struct Buffers {
    pub(super) q: Guarded,
    pub(super) k: Guarded,
    pub(super) v: Guarded,
    pub(super) table: Guarded,
    pub(super) block_stride: u64,
    pub(super) data_bytes: u64,
}

pub(super) fn buffers(gpu: &dyn GpuBackend, case: Case) -> Result<Buffers> {
    let q = Guarded::input(gpu, &q_fixture(case))?;
    let (k_bytes, table_values, block_stride, data_bytes) =
        cache_fixture(case, 0x0123_4567_89ab_cdef);
    let (v_bytes, v_table, v_stride, v_data) = cache_fixture(case, 0xfedc_ba98_7654_3210);
    ensure!(
        table_values == v_table,
        "{}: K/V page tables differ",
        case.label()
    );
    ensure!(
        block_stride == v_stride && data_bytes == v_data,
        "{}: K/V layouts differ",
        case.label()
    );
    Ok(Buffers {
        q,
        k: Guarded::input(gpu, &k_bytes)?,
        v: Guarded::input(gpu, &v_bytes)?,
        table: Guarded::input(gpu, &as_bytes_u32(&table_values))?,
        block_stride,
        data_bytes,
    })
}

impl Buffers {
    pub(super) fn verify(&self, gpu: &dyn GpuBackend, label: &str) -> Result<()> {
        self.q.verify_immutable(gpu, &format!("{label}/Q"))?;
        self.k.verify_immutable(gpu, &format!("{label}/K-cache"))?;
        self.v.verify_immutable(gpu, &format!("{label}/V-cache"))?;
        self.table
            .verify_immutable(gpu, &format!("{label}/block-table"))
    }

    pub(super) fn free(self, gpu: &dyn GpuBackend) -> Result<()> {
        self.q.free(gpu)?;
        self.k.free(gpu)?;
        self.v.free(gpu)?;
        self.table.free(gpu)
    }
}

pub(super) fn run_case(
    gpu: &dyn GpuBackend,
    stream: u64,
    parent: KernelHandle,
    candidate: KernelHandle,
    case: Case,
) -> Result<()> {
    let label = case.label();
    let buffers = buffers(gpu, case)?;
    let output_bytes = case.q_len as usize * NQ as usize * HD as usize * 2;
    let parent_out = Guarded::output(gpu, output_bytes, 0x3a)?;
    let candidate_out = Guarded::output(gpu, output_bytes, 0xc5)?;
    for (kernel, br128, output) in [
        (parent, false, &parent_out),
        (candidate, true, &candidate_out),
    ] {
        launch(
            gpu,
            stream,
            kernel,
            br128,
            buffers.q.payload_ptr(),
            buffers.k.payload_ptr(),
            buffers.v.payload_ptr(),
            output.payload_ptr(),
            buffers.table.payload_ptr(),
            case,
            buffers.block_stride,
            buffers.data_bytes,
        )?;
    }
    gpu.synchronize(stream)?;
    let parent_bytes = parent_out.output_payload(gpu, &format!("{label}/parent"))?;
    let candidate_bytes = candidate_out.output_payload(gpu, &format!("{label}/candidate"))?;
    if parent_bytes != candidate_bytes {
        let first = parent_bytes
            .iter()
            .zip(&candidate_bytes)
            .position(|(a, b)| a != b)
            .context("mismatch index")?;
        bail!(
            "{label}: BR128 differs at byte {first}: parent=0x{:02x}, candidate=0x{:02x}",
            parent_bytes[first],
            candidate_bytes[first]
        );
    }
    buffers.verify(gpu, &label)?;
    println!(
        "PARITY {label} bytes={} exact=true redzones=true immutable=true",
        parent_bytes.len()
    );
    parent_out.free(gpu)?;
    candidate_out.free(gpu)?;
    buffers.free(gpu)
}
