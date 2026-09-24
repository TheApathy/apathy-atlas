// SPDX-License-Identifier: AGPL-3.0-only

//! Experimental serial-core/grouped-MoE prefill. No numerical qualification implied.

use anyhow::{Context, Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::DevicePtr;

use super::{FfnComponent, Qwen4HyperConnection};
use crate::layer::ForwardContext;
use plan::{Limits, Plan};

pub(crate) mod attn16;
mod hyper;
mod plan;
pub(crate) const SELECTOR: &str = "ATLAS_QWEN4_PREFILL_MOE_BATCH";
pub(crate) const HYPER_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_HC_EXACT";
pub(crate) const HYPER_CHECK_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_HC_CHECK";
const PLE_SELECTOR: &str = "ATLAS_QWEN4_PLE_PREFILL_BATCH";
const PLE_COMPOSITION_SELECTOR: &str = "ATLAS_QWEN4_PLE_PREFILL_COMBINE";
pub(crate) const EXACT_QKV16_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_ATTN_QKV16";
pub(crate) const EXACT_O16_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_ATTN_O16";
const EXACT_K16_SELECTOR: &str = "ATLAS_QWEN4_K16_EXACT";

thread_local! {
    static FAMILY_SUPPRESSED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// True while [`attn16::admit_or_suppress`] has turned the whole fast-prefill
/// family (F8 MoE batch, HC/SSM exact, compact MoE, QKV16/O16, CORE16 and its
/// dependents) off for the prefill running on this thread. Every family
/// selector then reads as absent, so the request takes the ordinary prefill
/// path instead of failing admission.
pub(crate) fn family_suppressed() -> bool {
    FAMILY_SUPPRESSED.with(std::cell::Cell::get)
}

fn set_family_suppressed(on: bool) -> bool {
    FAMILY_SUPPRESSED.with(|s| s.replace(on))
}

pub(crate) fn hyper_check_selected() -> Result<bool> {
    plan::parse_selector(env_value(HYPER_CHECK_SELECTOR)?.as_deref())
        .map_err(|error| anyhow::anyhow!("{HYPER_CHECK_SELECTOR}: {error}"))
}

pub(crate) fn hyper_selected() -> Result<bool> {
    plan::parse_selector(env_value(HYPER_SELECTOR)?.as_deref())
        .map_err(|error| anyhow::anyhow!("{HYPER_SELECTOR}: {error}"))
}

fn env_value(name: &str) -> Result<Option<String>> {
    // The PLE whole-prompt batch handles continuation chunks itself; only its
    // composition with the F8 path (PLE_COMPOSITION_SELECTOR) is family-scoped.
    if family_suppressed() && name != PLE_SELECTOR {
        return Ok(None);
    }
    match std::env::var(name) {
        Ok(value) => Ok(Some(value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(error).with_context(|| format!("invalid {name}")),
    }
}

pub(crate) fn selected() -> Result<bool> {
    plan::parse_selector(env_value(SELECTOR)?.as_deref())
        .map_err(|error| anyhow::anyhow!("{SELECTOR}: {error}"))
}

pub(crate) fn exact_qkv16_selected() -> Result<bool> {
    plan::parse_selector(env_value(EXACT_QKV16_SELECTOR)?.as_deref())
        .map_err(|error| anyhow::anyhow!("{EXACT_QKV16_SELECTOR}: {error}"))
}

pub(crate) fn exact_o16_selected() -> Result<bool> {
    plan::parse_selector(env_value(EXACT_O16_SELECTOR)?.as_deref())
        .map_err(|error| anyhow::anyhow!("{EXACT_O16_SELECTOR}: {error}"))
}

/// Called before embedding, buffer writes, or recurrent-state changes.
pub(crate) fn admit_request(config: &ModelConfig, rows: usize, start: usize) -> Result<()> {
    // The compact MoE GEMM is row-independent and stays on under suppression;
    // each <=2048-row chunk is still admitted by its own validate_context.
    if !family_suppressed() {
        crate::layers::moe::qwen4_prefill_compact::admit_request(config, rows, start)?;
    }
    crate::layers::qwen3_ssm::qwen4_prefill_exact::admit_request(config, rows, start)?;
    let hyper = hyper_selected()?;
    ensure!(
        !hyper_check_selected()? || hyper,
        "{HYPER_CHECK_SELECTOR} requires {HYPER_SELECTOR}=1"
    );
    let selected = selected()?;
    let exact_qkv16 = exact_qkv16_selected()?;
    let exact_o16 = exact_o16_selected()?;
    let ple_composition = plan::parse_selector(env_value(PLE_COMPOSITION_SELECTOR)?.as_deref())
        .map_err(|error| anyhow::anyhow!("{PLE_COMPOSITION_SELECTOR}: {error}"))?;
    attn16::admit_request(config, rows, start, selected, hyper, exact_qkv16, exact_o16)?;
    if !selected {
        ensure!(!hyper, "{HYPER_SELECTOR} requires {SELECTOR}=1");
        ensure!(!exact_qkv16, "{EXACT_QKV16_SELECTOR} requires {SELECTOR}=1");
        ensure!(!exact_o16, "{EXACT_O16_SELECTOR} requires {SELECTOR}=1");
        ensure!(
            !ple_composition,
            "{PLE_COMPOSITION_SELECTOR} requires {SELECTOR}=1"
        );
        return Ok(());
    }
    ensure!(
        config.is_qwen4_exp()
            && config.hidden_size == 2560
            && config.residual_width() == 10240
            && config.num_hidden_layers == 48
            && config.num_attention_layers() == 12
            && config.num_ssm_layers() == 36
            && config.num_experts == 512
            && config.num_experts_per_tok == 10
            && config.moe_intermediate_size == 640
            && config.shared_expert_intermediate_size == 640
            && !config.use_fp32_residual()
            && config.ep_world_size == 1,
        "{SELECTOR} requires the canonical single-GPU BF16-residual Flash-Next geometry"
    );
    plan::validate_exact_qkv16_composition(
        env_value(EXACT_QKV16_SELECTOR)?.as_deref(),
        env_value(EXACT_K16_SELECTOR)?.as_deref(),
        selected,
        hyper,
        start,
        rows,
    )
    .map_err(|error| anyhow::anyhow!("{EXACT_QKV16_SELECTOR}: {error}"))?;
    plan::validate_exact_o16_composition(env_value(EXACT_O16_SELECTOR)?.as_deref(), exact_qkv16)
        .map_err(|error| anyhow::anyhow!("{EXACT_O16_SELECTOR}: {error}"))?;
    ensure!(
        rows > 0 && start.checked_add(rows).is_some_and(|end| end <= 2048),
        "{SELECTOR} currently requires a nonempty prompt within the first 2048 tokens"
    );
    for name in [
        "ATLAS_QWEN4_ATTN_PREFILL_BATCH",
        "ATLAS_QWEN4_SSM_PREFILL_BATCH",
        "ATLAS_QWEN4_SSM_PREFILL_ROWWISE",
        "ATLAS_QWEN4_SSM_PREFILL_FP32",
        "ATLAS_QWEN4_SSM_PREFILL_CONVSEQ",
        "ATLAS_QWEN4_HYPER_PREFILL_GEMM",
        "ATLAS_QWEN4_QSA_PREFILL_GEMM",
        "ATLAS_NVFP4_MOE_WORKLIST",
        "ATLAS_SSM_H_FP16",
    ] {
        plan::validate_conflict(env_value(name)?.as_deref(), false)
            .map_err(|error| anyhow::anyhow!("{SELECTOR}: {name}: {error}"))?;
    }
    plan::validate_conflict(env_value("ATLAS_QWEN4_SSM_PREFILL_TILE")?.as_deref(), true)
        .map_err(|error| anyhow::anyhow!("{SELECTOR}: ATLAS_QWEN4_SSM_PREFILL_TILE: {error}"))?;
    plan::validate_ple_composition(
        env_value(PLE_SELECTOR)?.as_deref(),
        env_value(PLE_COMPOSITION_SELECTOR)?.as_deref(),
    )
    .map_err(|error| {
        anyhow::anyhow!("{SELECTOR}: {PLE_SELECTOR}+{PLE_COMPOSITION_SELECTOR}: {error}")
    })
}

fn layout(ctx: &ForwardContext, rows: usize, start: usize) -> Result<Plan> {
    let sizes = ctx.buffers.sizes();
    Plan::new(
        rows,
        start,
        ctx.config.hidden_size,
        ctx.config.residual_width(),
        Limits {
            capacity: ctx.buffers.max_batch_tokens(),
            qkv: sizes.qkv_output,
            norm: sizes.norm_output,
            hidden: sizes.hidden_states,
            residual: sizes.residual,
            output: sizes.moe_output,
        },
    )
    .map_err(anyhow::Error::msg)
}

/// Layer admission must precede the first serial core call.
pub(crate) fn validate(ctx: &ForwardContext, rows: usize, start: usize) -> Result<()> {
    admit_request(ctx.config, rows, start)?;
    crate::layers::moe::qwen4_prefill_compact::validate_context(ctx, rows)?;
    ensure!(
        ctx.comm.is_none()
            && !ctx.graph_capture
            && ctx.ddtree_parent_ids_dev.is_none()
            && ctx.tree_aware_attn.is_none()
            && ctx.ssm_multi_seq_ptr_table_override.is_none()
            && ctx.self_spec_sparse_draft.is_none()
            && ctx.ffn_defer.is_none(),
        "{SELECTOR} requires eager single-sequence prefill without verify overrides"
    );
    let meta = ctx
        .attn_metadata
        .context("MoE-only prefill requires paged metadata")?;
    ensure!(
        meta.num_seqs == 1
            && meta.positions != DevicePtr::NULL
            && meta.slot != DevicePtr::NULL
            && meta.seq_len != DevicePtr::NULL
            && meta.block_table != DevicePtr::NULL,
        "{SELECTOR} requires live single-sequence paged metadata"
    );
    ensure!(
        ctx.buffers.sizes().ssm_ba >= 2056,
        "MoE-only HC scratch is undersized"
    );
    layout(ctx, rows, start)?;
    Ok(())
}

pub(crate) fn validate_ffn(ffn: &FfnComponent) -> Result<()> {
    match ffn {
        FfnComponent::Moe(moe) => moe.validate_qwen4_prefill_moe(),
        _ => anyhow::bail!("{SELECTOR} requires grouped original-layout NVFP4 experts"),
    }
}

/// All serial core rows have finished before entering this function.
/// Keep both HC operations identical to decode; change only the FFN dispatch.
pub(crate) fn finish(
    mlp: &Qwen4HyperConnection,
    ffn: &FfnComponent,
    hidden: DevicePtr,
    residual: DevicePtr,
    rows: usize,
    ctx: &ForwardContext,
    stream: u64,
) -> Result<()> {
    let p = layout(ctx, rows, 0)?;
    validate_ffn(ffn)?;
    ensure!(
        mlp.inject.is_some() && matches!(ffn, FfnComponent::Moe(_)),
        "MoE-only prefill requires an injecting MLP hyperconnection and routed MoE"
    );
    if hyper_selected()? {
        return hyper::finish(mlp, ffn, hidden, residual, rows, ctx, stream);
    }
    let staging = ctx.buffers.qkv_output().offset(p.staging_offset);
    for row in 0..rows {
        let (mixed, _) = mlp.prepare_decode(
            hidden.offset(row * p.row_bytes),
            residual.offset(row * p.row_bytes),
            ctx.buffers,
            ctx.gpu,
            ctx.config.rms_norm_eps as f32,
            stream,
        )?;
        ctx.gpu.copy_d2d_async(
            mixed,
            staging.offset(row * p.core_bytes),
            p.core_bytes,
            stream,
        )?;
    }
    // prepare_decode overwrites norm_output row zero each time. Pack only
    // after every row has been preserved outside its QKV projection scratch.
    let input = ctx.buffers.norm_output();
    ctx.gpu
        .copy_d2d_async(staging, input, p.input_bytes, stream)?;
    ffn.forward_prefill(input, rows, ctx, stream)?;
    for row in 0..rows {
        mlp.inject_decode(
            hidden.offset(row * p.row_bytes),
            ctx.buffers.moe_output().offset(row * p.core_bytes),
            mlp.saved_inject(residual.offset(row * p.row_bytes)),
            ctx.gpu,
            stream,
        )?;
    }
    Ok(())
}
