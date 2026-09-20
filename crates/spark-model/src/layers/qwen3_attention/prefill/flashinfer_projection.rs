// SPDX-License-Identifier: AGPL-3.0-only

//! Fully admitted, default-off FlashInfer projection route for the exact
//! Qwen3.8 C=1 M2079 prefill shape retained after bounded paired endpoint
//! output equality. M8192 remains on the BF16-activation W4A16 parent because
//! the earlier isolated receipt compared only prequantized W4A4 peers and
//! endpoint replay exposed prompt-dependent output drift. Preparation owns all
//! transformed operands; dispatch performs no allocation and falls back for
//! every other shape.

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;
use spark_runtime::weights::WeightDtype;
use std::sync::{Mutex, MutexGuard};

use super::{Qwen38FlashinferProjectionRoute, qwen38_flashinfer_projection_route};
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::layers::qwen3_attention::Qwen3AttentionLayer;
use crate::weight_map::flashinfer_projection_admission::{
    AdmittedQwen38Projection, QWEN38_ATTN_QGKV, QWEN38_ATTN_VALUE, QWEN38_HIDDEN, QWEN38_KV,
    QWEN38_QG, Qwen38AttentionMergedQkvCache, Qwen38AttentionProjectionFamily,
    Qwen38CheckpointProjection, Qwen38Projection, Qwen38ProjectionOperandCache,
    Qwen38ProjectionSource, admit_qwen38_projection, finalize_qwen38_attention_qkv_merged_cache,
    finalize_qwen38_projection_cache, merge_qwen38_attention_qkv,
    select_qwen38_attention_projection_launch,
};
use crate::weight_map::modelopt_scale_admission::{
    ModeloptScaleProjection, ModeloptScaleSource, admit_modelopt_checkpoint_scale,
};
use crate::weight_map::{Nvfp4Variant, QuantWeight, QuantizedWeight};

const QUALIFIED_FLASHINFER_SM121_SHA256: [u8; 32] = [
    0xa0, 0x07, 0xa8, 0x25, 0x66, 0xca, 0x3d, 0x31, 0x15, 0xc8, 0xcc, 0x0e, 0x73, 0xe2, 0xbb, 0xfc,
    0x0b, 0xd1, 0xc7, 0x6b, 0x63, 0x42, 0x31, 0x3f, 0xba, 0x7e, 0xe5, 0x48, 0x3e, 0x49, 0xc0, 0x20,
];
const MAX_M: usize = 2_079;
const MAX_M_PADDED: usize = 2_176;
const MAX_K: usize = QWEN38_ATTN_VALUE;

pub(in crate::layers::qwen3_attention) struct FlashinferAttentionProjectionPrefill {
    layer: usize,
    library: ops::flashinfer_sm121::FlashInferSm121,
    qgkv: Qwen38AttentionMergedQkvCache,
    output: Qwen38ProjectionOperandCache,
    quantize_atlas_128x4_k: KernelHandle,
    split_qgkv_k: KernelHandle,
    activation_fp4: DevicePtr,
    activation_fp4_bytes: usize,
    activation_scales: DevicePtr,
    activation_scale_bytes: usize,
    activation_use_stream: Mutex<Option<u64>>,
}

pub(super) fn bind_activation_stream(admitted_stream: &mut Option<u64>, stream: u64) -> Result<()> {
    match *admitted_stream {
        Some(existing) => ensure!(
            existing == stream,
            "FlashInfer attention activation scratch is bound to CUDA stream {existing:#x}; rejected concurrent/reordered use on {stream:#x}"
        ),
        None => *admitted_stream = Some(stream),
    }
    Ok(())
}

impl FlashinferAttentionProjectionPrefill {
    fn lock_activation_use(&self, stream: u64) -> Result<MutexGuard<'_, Option<u64>>> {
        let mut admitted_stream = self.activation_use_stream.lock().map_err(|_| {
            anyhow::anyhow!("FlashInfer attention activation scratch lock poisoned")
        })?;
        bind_activation_stream(&mut admitted_stream, stream)?;
        Ok(admitted_stream)
    }
}

fn scale_projection(projection: Qwen38Projection) -> ModeloptScaleProjection {
    match projection {
        Qwen38Projection::AttentionQueryGate => ModeloptScaleProjection::AttentionQuery,
        Qwen38Projection::AttentionKey => ModeloptScaleProjection::AttentionKey,
        Qwen38Projection::AttentionValue => ModeloptScaleProjection::AttentionValue,
        Qwen38Projection::AttentionOutput => ModeloptScaleProjection::AttentionOutput,
        Qwen38Projection::SsmQkvz | Qwen38Projection::SsmOutput => {
            unreachable!("attention preparation never admits an SSM projection")
        }
    }
}

fn read_admitted_projection(
    gpu: &dyn GpuBackend,
    layer: usize,
    projection: Qwen38Projection,
    weight: &QuantizedWeight,
    n: usize,
    k: usize,
) -> Result<(AdmittedQwen38Projection, Vec<u8>)> {
    ensure!(
        !weight.weight.is_null() && !weight.weight_scale.is_null() && !weight.input_scale.is_null(),
        "FlashInfer attention {projection:?} requires original non-NULL ModelOpt operands"
    );
    let packed_len = n
        .checked_mul(k / 2)
        .context("FlashInfer attention packed-weight extent overflow")?;
    let scale_len = n
        .checked_mul(k / 16)
        .context("FlashInfer attention block-scale extent overflow")?;
    let mut packed = vec![0_u8; packed_len];
    let mut logical_scales = vec![0_u8; scale_len];
    let mut input_scale_bytes = [0_u8; 4];
    gpu.copy_d2h(weight.weight, &mut packed)
        .with_context(|| format!("read original ModelOpt {projection:?} packed weight"))?;
    gpu.copy_d2h(weight.weight_scale, &mut logical_scales)
        .with_context(|| format!("read original ModelOpt {projection:?} block scales"))?;
    gpu.copy_d2h(weight.input_scale, &mut input_scale_bytes)
        .with_context(|| format!("read original ModelOpt {projection:?} input scale"))?;
    let input_scale = admit_modelopt_checkpoint_scale(
        ModeloptScaleSource {
            layer,
            projection: scale_projection(projection),
        },
        WeightDtype::FP32,
        &[],
        weight.input_scale,
        input_scale_bytes,
    )?;
    let admitted = admit_qwen38_projection(
        layer,
        projection,
        Qwen38CheckpointProjection {
            source: Qwen38ProjectionSource { layer, projection },
            packed_weight: weight.weight,
            packed_weight_bytes: packed_len,
            logical_weight_scales: &logical_scales,
            input_scale,
            weight_scale_2_le_bytes: weight.weight_scale_2.to_le_bytes(),
        },
    )?;
    Ok((admitted, packed))
}

fn fingerprint(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn checked_extent(rows: usize, cols: usize, bytes: usize, label: &str) -> Result<usize> {
    rows.checked_mul(cols)
        .and_then(|elements| elements.checked_mul(bytes))
        .with_context(|| format!("{label} extent overflow"))
}

fn ensure_disjoint(
    left: DevicePtr,
    left_len: usize,
    right: DevicePtr,
    right_len: usize,
) -> Result<()> {
    let left_end = left
        .0
        .checked_add(u64::try_from(left_len).context("left pointer extent exceeds u64")?)
        .context("left pointer range overflow")?;
    let right_end = right
        .0
        .checked_add(u64::try_from(right_len).context("right pointer extent exceeds u64")?)
        .context("right pointer range overflow")?;
    ensure!(
        left_end <= right.0 || right_end <= left.0,
        "FlashInfer projection buffers overlap"
    );
    Ok(())
}

fn launch_shape(
    family: Qwen38AttentionProjectionFamily,
    m: usize,
) -> Result<ops::flashinfer_sm121::FlashInferSm121Shape> {
    let plan = select_qwen38_attention_projection_launch(family, m)?;
    ensure!(
        plan.real_checkpoint_qualified && plan.workspace_bytes == 0,
        "FlashInfer attention projection plan is not production-qualified zero-workspace"
    );
    ops::flashinfer_sm121::FlashInferSm121Shape::new(
        u8::try_from(plan.tactic).context("FlashInfer attention tactic exceeds u8")?,
        plan.m,
        plan.n,
        plan.k,
        1,
    )
}

impl Qwen3AttentionLayer {
    /// Build immutable merged-QGKV and retained-O operands for one exact
    /// Standard ModelOpt Qwen3.8 attention layer. The caller retains the
    /// original Q/K/V/O allocations for this layer's lifetime.
    pub fn prepare_flashinfer_projection_prefill(
        &mut self,
        gpu: &dyn GpuBackend,
        layer: usize,
        variant: Nvfp4Variant,
        config: &atlas_core::config::ModelConfig,
    ) -> Result<()> {
        ensure!(
            crate::layers::prefill_proj_flashinfer_enabled()?,
            "prepare_flashinfer_projection_prefill called while its route is disabled"
        );
        ensure!(
            self.flashinfer_projection_prefill.is_none(),
            "FlashInfer attention projection operands are already prepared"
        );
        ensure!(
            variant == Nvfp4Variant::Standard,
            "FlashInfer attention projections require the Standard ModelOpt checkpoint layout"
        );
        ensure!(
            config.model_type == "qwen3_5"
                && config.num_experts == 0
                && config.hidden_size == QWEN38_HIDDEN
                && config.num_attention_heads == 24
                && config.num_key_value_heads == 4
                && config.head_dim == 256
                && config.attn_gated
                && config.tp_world_size.max(1) == 1
                && config.mrope_section == [11, 11, 10]
                && self.gated
                && self.mla.is_none(),
            "FlashInfer attention projections require exact dense Qwen3.8 Standard geometry"
        );
        ensure!(
            self.o_weight.is_none() && self.o_dense_bf16.is_none(),
            "FlashInfer attention output requires the original ModelOpt O operand"
        );

        let q = match self.q_weight {
            Some(QuantWeight::Nvfp4(weight)) => weight,
            _ => anyhow::bail!("FlashInfer attention requires original NVFP4 Q/G weight"),
        };
        let k = match self.k_weight {
            Some(QuantWeight::Nvfp4(weight)) => weight,
            _ => anyhow::bail!("FlashInfer attention requires original NVFP4 K weight"),
        };
        let v = match self.v_weight {
            Some(QuantWeight::Nvfp4(weight)) => weight,
            _ => anyhow::bail!("FlashInfer attention requires original NVFP4 V weight"),
        };
        let o = self.attn.o_proj;

        let library_path = std::env::var_os("ATLAS_FLASHINFER_SM121_LIB").ok_or_else(|| {
            anyhow::anyhow!("ATLAS_PREFILL_PROJ_FLASHINFER=1 requires ATLAS_FLASHINFER_SM121_LIB")
        })?;
        let library = ops::flashinfer_sm121::FlashInferSm121::open_with_sha256(
            std::path::Path::new(&library_path),
            QUALIFIED_FLASHINFER_SM121_SHA256,
        )?;
        let quantize_atlas_128x4_k = gpu.kernel(
            "quantize_bf16_to_nvfp4_cutlass",
            "quantize_bf16_to_nvfp4_atlas_128x4",
        )?;
        let split_qgkv_k = gpu.kernel(
            "flashinfer_projection_split",
            "flashinfer_projection_split_qgkv",
        )?;
        ensure!(
            quantize_atlas_128x4_k.0 != 0 && split_qgkv_k.0 != 0,
            "FlashInfer attention preparation resolved a NULL required kernel"
        );

        // Freeze both exact M2079 zero-workspace ABI shapes before retaining
        // or uploading any route operand. M8192 is intentionally ineligible:
        // its W4A4 activation conversion failed endpoint output parity.
        let stream = gpu.default_stream();
        for family in [
            Qwen38AttentionProjectionFamily::MergedQkv,
            Qwen38AttentionProjectionFamily::Output,
        ] {
            let shape = launch_shape(family, 2_079)?;
            let prepared = library.prepare_borrowed_zero_workspace(gpu, shape, stream)?;
            ensure!(
                prepared.shape() == shape,
                "FlashInfer preparation changed frozen shape"
            );
        }

        let (q_admitted, q_packed) = read_admitted_projection(
            gpu,
            layer,
            Qwen38Projection::AttentionQueryGate,
            &q,
            QWEN38_QG,
            QWEN38_HIDDEN,
        )?;
        let (k_admitted, k_packed) = read_admitted_projection(
            gpu,
            layer,
            Qwen38Projection::AttentionKey,
            &k,
            QWEN38_KV,
            QWEN38_HIDDEN,
        )?;
        let (v_admitted, v_packed) = read_admitted_projection(
            gpu,
            layer,
            Qwen38Projection::AttentionValue,
            &v,
            QWEN38_KV,
            QWEN38_HIDDEN,
        )?;
        let merged = merge_qwen38_attention_qkv(
            layer,
            &q_admitted,
            &q_packed,
            &k_admitted,
            &k_packed,
            &v_admitted,
            &v_packed,
        )?;
        let (o_admitted, _) = read_admitted_projection(
            gpu,
            layer,
            Qwen38Projection::AttentionOutput,
            &o,
            QWEN38_HIDDEN,
            QWEN38_ATTN_VALUE,
        )?;

        let qgkv_weight = gpu.alloc(merged.packed_weight().len())?;
        gpu.copy_h2d(merged.packed_weight(), qgkv_weight)
            .context("upload immutable merged QGKV packed weight")?;
        let qgkv_scales = gpu.alloc(merged.physical_weight_scales().len())?;
        gpu.copy_h2d(merged.physical_weight_scales(), qgkv_scales)
            .context("upload immutable merged QGKV physical scales")?;
        let qgkv_alpha = gpu.alloc(std::mem::size_of::<f32>())?;
        gpu.copy_h2d(&merged.alpha_bits().to_le_bytes(), qgkv_alpha)
            .context("upload immutable merged QGKV alpha")?;
        let qgkv = finalize_qwen38_attention_qkv_merged_cache(
            &merged,
            qgkv_weight,
            qgkv_scales,
            qgkv_alpha,
            merged.alpha_bits(),
        )?;

        let output_scales = gpu.alloc(o_admitted.physical_weight_scales().len())?;
        gpu.copy_h2d(o_admitted.physical_weight_scales(), output_scales)
            .context("upload immutable attention O physical scales")?;
        let output_alpha = gpu.alloc(std::mem::size_of::<f32>())?;
        gpu.copy_h2d(&o_admitted.alpha().value_bits().to_le_bytes(), output_alpha)
            .context("upload immutable attention O alpha")?;
        let output = finalize_qwen38_projection_cache(
            &o_admitted,
            output_scales,
            output_alpha,
            o_admitted.alpha().value_bits(),
        )?;

        let activation_fp4_bytes = checked_extent(MAX_M, MAX_K, 1, "activation FP4")? / 2;
        let activation_scale_bytes =
            checked_extent(MAX_M_PADDED, MAX_K / 16, 1, "activation scales")?;
        let activation_fp4 = gpu.alloc(activation_fp4_bytes)?;
        let activation_scales = gpu.alloc(activation_scale_bytes)?;

        let (library_device, library_inode) = library.file_identity();
        tracing::info!(
            layer,
            library = %library.path().display(),
            library_sha256 = %library.sha256_hex(),
            library_device,
            library_inode,
            qgkv_weight_hash = format_args!("{:016x}", fingerprint(merged.packed_weight())),
            qgkv_scale_hash = format_args!("{:016x}", fingerprint(merged.physical_weight_scales())),
            qgkv_input_scale_bits = format_args!("{:08x}", qgkv.input_scale_bits()),
            qgkv_alpha_bits = format_args!("{:08x}", qgkv.alpha_bits()),
            output_weight = format_args!("{:#x}", output.packed_weight().0),
            output_scale_hash = format_args!("{:016x}", fingerprint(o_admitted.physical_weight_scales())),
            output_input_scale_bits = format_args!("{:08x}", output.input_scale_bits()),
            output_alpha_bits = format_args!("{:08x}", output.alpha_bits()),
            "admitted FlashInfer SM121 Qwen3.8 attention prefill operands"
        );
        self.flashinfer_projection_prefill = Some(FlashinferAttentionProjectionPrefill {
            layer,
            library,
            qgkv,
            output,
            quantize_atlas_128x4_k,
            split_qgkv_k,
            activation_fp4,
            activation_fp4_bytes,
            activation_scales,
            activation_scale_bytes,
            activation_use_stream: Mutex::new(None),
        });
        Ok(())
    }

    fn flashinfer_projection_route(
        &self,
        n: u32,
        h: u32,
        qg: usize,
        kv: usize,
        ctx: &ForwardContext,
    ) -> Result<Qwen38FlashinferProjectionRoute> {
        let exact_qwen38_dense = ctx.config.model_type == "qwen3_5"
            && ctx.config.num_experts == 0
            && ctx.config.hidden_size == QWEN38_HIDDEN
            && ctx.config.tp_world_size.max(1) == 1
            && ctx.config.mrope_section == [11, 11, 10];
        Ok(qwen38_flashinfer_projection_route(
            crate::layers::prefill_proj_flashinfer_enabled()?,
            exact_qwen38_dense,
            ctx.attn_metadata
                .is_some_and(|metadata| metadata.num_seqs == 1),
            self.mla.is_none(),
            self.gated,
            n == 2_079
                || crate::weight_map::flashinfer_projection_admission::
                    select_qwen38_attention_projection_launch(
                        crate::weight_map::flashinfer_projection_admission::
                            Qwen38AttentionProjectionFamily::MergedQkv,
                        n as usize,
                    )
                    .is_ok(),
            h,
            qg,
            kv,
            self.flashinfer_projection_prefill.is_some(),
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn try_flashinfer_projection_qgkv(
        &self,
        normed: DevicePtr,
        n: u32,
        h: u32,
        qg: usize,
        kv: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        match self.flashinfer_projection_route(n, h, qg, kv, ctx)? {
            Qwen38FlashinferProjectionRoute::Disabled
            | Qwen38FlashinferProjectionRoute::Ineligible => return Ok(false),
            Qwen38FlashinferProjectionRoute::Missing => anyhow::bail!(
                "ATLAS_PREFILL_PROJ_FLASHINFER=1 requires prepared attention operands for exact M={n}"
            ),
            Qwen38FlashinferProjectionRoute::Complete => {}
        }
        let route = self.flashinfer_projection_prefill.as_ref().unwrap();
        ensure!(
            route.qgkv.layer() == route.layer,
            "FlashInfer merged QGKV layer changed"
        );
        let m = n as usize;
        let merged_bytes = checked_extent(m, QWEN38_ATTN_QGKV, 2, "merged QGKV output")?;
        let qg_bytes = checked_extent(m, QWEN38_QG, 2, "Q/G output")?;
        let kv_bytes = checked_extent(m, QWEN38_KV, 2, "K/V output")?;
        let activation_fp4_bytes = checked_extent(m, QWEN38_HIDDEN, 1, "QGKV activation FP4")? / 2;
        let activation_scale_bytes = checked_extent(
            m.div_ceil(128) * 128,
            QWEN38_HIDDEN / 16,
            1,
            "QGKV activation scales",
        )?;
        ensure!(
            activation_fp4_bytes <= route.activation_fp4_bytes
                && activation_scale_bytes <= route.activation_scale_bytes,
            "FlashInfer QGKV activation scratch is undersized"
        );
        ensure!(
            ctx.buffers.sizes().ssm_conv_out_f32 >= merged_bytes,
            "merged QGKV scratch is undersized"
        );
        ensure!(
            ctx.buffers.sizes().qkv_output >= qg_bytes,
            "Q/G output buffer is undersized"
        );
        ensure!(
            ctx.buffers.sizes().ssm_qkvz >= 2 * kv_bytes,
            "K/V output buffer is undersized"
        );
        ensure!(
            ctx.buffers.sizes().norm_output >= checked_extent(m, QWEN38_HIDDEN, 2, "O output")?,
            "O output buffer is undersized"
        );

        let merged_out = ctx.buffers.ssm_conv_out_f32();
        let qg_out = ctx.buffers.qkv_output();
        let k_out = ctx.buffers.ssm_qkvz();
        let v_out = k_out.offset(kv_bytes);
        for pointer in [merged_out, qg_out, k_out, v_out, normed] {
            ensure!(
                !pointer.is_null() && pointer.0.is_multiple_of(16),
                "FlashInfer projection pointer is NULL or misaligned"
            );
        }
        ensure_disjoint(merged_out, merged_bytes, qg_out, qg_bytes)?;
        ensure_disjoint(merged_out, merged_bytes, k_out, 2 * kv_bytes)?;
        ensure_disjoint(qg_out, qg_bytes, k_out, 2 * kv_bytes)?;

        // Preflight both projection families before the first quantization or
        // FlashInfer output. This prevents an incomplete explicit route from
        // committing Q/K/V and failing only at O later in the layer.
        let qgkv_shape = launch_shape(Qwen38AttentionProjectionFamily::MergedQkv, m)?;
        let output_shape = launch_shape(Qwen38AttentionProjectionFamily::Output, m)?;
        let mut qgkv_launch = route
            .library
            .prepare_borrowed_zero_workspace(ctx.gpu, qgkv_shape, stream)?;
        let _output_preflight =
            route
                .library
                .prepare_borrowed_zero_workspace(ctx.gpu, output_shape, stream)?;
        let _activation_use = route.lock_activation_use(stream)?;

        ops::quantize_bf16_to_nvfp4_atlas_128x4(
            ctx.gpu,
            route.quantize_atlas_128x4_k,
            normed,
            route.activation_fp4,
            route.activation_scales,
            f32::from_bits(route.qgkv.input_scale_bits()),
            n,
            h,
            stream,
        )?;
        qgkv_launch.launch_eager(
            ops::flashinfer_sm121::FlashInferSm121Buffers {
                output_bf16: merged_out,
                activation_fp4: route.activation_fp4,
                weight_fp4: route.qgkv.packed_weight(),
                activation_scales: route.activation_scales,
                weight_scales: route.qgkv.physical_weight_scales(),
                global_scale_f32: route.qgkv.alpha(),
            },
            stream,
        )?;
        KernelLaunch::new(ctx.gpu, route.split_qgkv_k)
            .grid([n, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(merged_out)
            .arg_ptr(qg_out)
            .arg_ptr(k_out)
            .arg_ptr(v_out)
            .arg_u32(n)
            .launch(stream)?;
        log_success(route.layer, Qwen38AttentionProjectionFamily::MergedQkv, m);
        Ok(true)
    }

    pub(super) fn try_flashinfer_projection_output(
        &self,
        attn_out: DevicePtr,
        n: u32,
        h: u32,
        nq: u32,
        hd: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<Option<DevicePtr>> {
        let exact_kv = if nq == 24 && hd == 256 { QWEN38_KV } else { 0 };
        match self.flashinfer_projection_route(n, h, QWEN38_QG, exact_kv, ctx)? {
            Qwen38FlashinferProjectionRoute::Disabled
            | Qwen38FlashinferProjectionRoute::Ineligible => return Ok(None),
            Qwen38FlashinferProjectionRoute::Missing => anyhow::bail!(
                "ATLAS_PREFILL_PROJ_FLASHINFER=1 requires prepared attention operands for exact M={n}"
            ),
            Qwen38FlashinferProjectionRoute::Complete => {}
        }
        ensure!(
            nq == 24 && hd == 256,
            "FlashInfer attention O runtime geometry changed"
        );
        let route = self.flashinfer_projection_prefill.as_ref().unwrap();
        ensure!(
            route.output.source()
                == (Qwen38ProjectionSource {
                    layer: route.layer,
                    projection: Qwen38Projection::AttentionOutput,
                }),
            "FlashInfer attention O source changed after preparation"
        );
        let m = n as usize;
        let activation_fp4_bytes = checked_extent(m, QWEN38_ATTN_VALUE, 1, "O activation FP4")? / 2;
        let activation_scale_bytes = checked_extent(
            m.div_ceil(128) * 128,
            QWEN38_ATTN_VALUE / 16,
            1,
            "O activation scales",
        )?;
        let output_bytes = checked_extent(m, QWEN38_HIDDEN, 2, "O output")?;
        ensure!(
            activation_fp4_bytes <= route.activation_fp4_bytes
                && activation_scale_bytes <= route.activation_scale_bytes,
            "FlashInfer O activation scratch is undersized"
        );
        ensure!(
            ctx.buffers.sizes().norm_output >= output_bytes,
            "O output buffer is undersized"
        );
        let output = ctx.buffers.norm_output();
        for pointer in [attn_out, output] {
            ensure!(
                !pointer.is_null() && pointer.0.is_multiple_of(16),
                "FlashInfer O pointer is NULL or misaligned"
            );
        }
        ensure_disjoint(
            attn_out,
            checked_extent(m, QWEN38_ATTN_VALUE, 2, "O input")?,
            output,
            output_bytes,
        )?;
        let shape = launch_shape(Qwen38AttentionProjectionFamily::Output, m)?;
        let mut launch = route
            .library
            .prepare_borrowed_zero_workspace(ctx.gpu, shape, stream)?;
        let _activation_use = route.lock_activation_use(stream)?;
        ops::quantize_bf16_to_nvfp4_atlas_128x4(
            ctx.gpu,
            route.quantize_atlas_128x4_k,
            attn_out,
            route.activation_fp4,
            route.activation_scales,
            f32::from_bits(route.output.input_scale_bits()),
            n,
            nq * hd,
            stream,
        )?;
        launch.launch_eager(
            ops::flashinfer_sm121::FlashInferSm121Buffers {
                output_bf16: output,
                activation_fp4: route.activation_fp4,
                weight_fp4: route.output.packed_weight(),
                activation_scales: route.activation_scales,
                weight_scales: route.output.physical_weight_scales(),
                global_scale_f32: route.output.alpha(),
            },
            stream,
        )?;
        log_success(route.layer, Qwen38AttentionProjectionFamily::Output, m);
        Ok(Some(output))
    }
}

fn log_success(layer: usize, family: Qwen38AttentionProjectionFamily, m: usize) {
    static RECEIPTS: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
    let family_bit = match family {
        Qwen38AttentionProjectionFamily::MergedQkv => 0,
        Qwen38AttentionProjectionFamily::Output => 1,
    };
    let bit = 1_u8 << family_bit;
    if RECEIPTS.fetch_or(bit, std::sync::atomic::Ordering::Relaxed) & bit != 0 {
        return;
    }
    let plan = select_qwen38_attention_projection_launch(family, m)
        .expect("a successful route already selected an exact frozen plan");
    let family_name = match family {
        Qwen38AttentionProjectionFamily::MergedQkv => "attention_qgkv",
        Qwen38AttentionProjectionFamily::Output => "attention_o",
    };
    tracing::info!(
        layer,
        m,
        tactic = plan.tactic,
        family = family_name,
        "routed Qwen3.8 prefill projection through FlashInfer SM121"
    );
    eprintln!(
        "ATLAS_PREFILL_PROJ_FLASHINFER ENGAGED family={family_name} layer={layer} M={m} tactic={}",
        plan.tactic
    );
}
