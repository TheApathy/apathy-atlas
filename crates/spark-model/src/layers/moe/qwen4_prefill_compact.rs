// SPDX-License-Identifier: AGPL-3.0-only

//! Default-off F8-only original-layout BF16-MMA compact scheduling.

use anyhow::{Context, Result, bail, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{GpuBackend, KernelHandle};

use super::qwen4_compact_contract as abi;
use crate::layer::ForwardContext;

pub(crate) const SELECTOR: &str = "ATLAS_QWEN4_PREFILL_MOE_COMPACT";
pub(crate) const CHECK_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_MOE_COMPACT_CHECK";
pub(crate) const K32_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_MOE_COMPACT_K32";
pub(crate) const TRANSPOSED_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_MOE_COMPACT_T";
pub(crate) const STREAM_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_MOE_STREAM_T";
pub(crate) const V2_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_MOE_COMPACT_V2";

/// 0 = shipping kernel, 1 = pipelined v2 (64x64), 2 = v3 (64x128, 256 threads),
/// 3 = v3 down + fused gate/up/silu (64x64, 128 threads),
/// 4 = v4 (v3 tile, single-buffered sBf: 3 CTAs/SM = 24 warps, bit-exact with v3),
/// 5 = v5 (v2 tile, single-buffered sBf: 4 CTAs/SM = 16 warps, bit-exact with v2),
/// 6 = v6 (v4 with the MMAs of all-padding 16-row warp slices skipped; stored
///     outputs run v4's exact instruction sequence).
pub(crate) fn v2_level() -> Result<u32> {
    match std::env::var(V2_SELECTOR) {
        Err(std::env::VarError::NotPresent) => Ok(0),
        Ok(v) if v == "0" => Ok(0),
        Ok(v) if v == "1" => Ok(1),
        Ok(v) if v == "2" => Ok(2),
        Ok(v) if v == "3" => Ok(3),
        Ok(v) if v == "4" => Ok(4),
        Ok(v) if v == "5" => Ok(5),
        Ok(v) if v == "6" => Ok(6),
        Ok(v) => bail!("{V2_SELECTOR} must be absent, 0, 1, 2, 3, 4, 5 or 6; got {v:?}"),
        Err(e) => Err(e).with_context(|| format!("invalid {V2_SELECTOR}")),
    }
}

fn flag(name: &str) -> Result<bool> {
    let value = match std::env::var(name) {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(error) => return Err(error).with_context(|| format!("invalid {name}")),
    };
    abi::parse(value.as_deref()).map_err(|error| anyhow::anyhow!("{name}: {error}"))
}

pub(crate) fn selected() -> Result<bool> {
    flag(SELECTOR)
}

pub(crate) fn check_selected() -> Result<bool> {
    let check = flag(CHECK_SELECTOR)?;
    if check {
        super::qwen4_compact_compare::admit(
            check,
            selected()?,
            flag("ATLAS_QWEN4_PREFILL_MOE_BATCH")?,
        )
        .map_err(anyhow::Error::msg)?;
    }
    Ok(check)
}

pub(crate) fn stream_selected() -> Result<bool> {
    flag(STREAM_SELECTOR)
}

pub(crate) fn transposed_selected() -> Result<bool> {
    let persistent = flag(TRANSPOSED_SELECTOR)?;
    let streaming = stream_selected()?;
    ensure!(
        !(persistent && streaming),
        "{TRANSPOSED_SELECTOR} conflicts with {STREAM_SELECTOR}"
    );
    Ok(persistent || streaming)
}

fn selected_step_k() -> Result<u32> {
    Ok(if flag(K32_SELECTOR)? { 32 } else { 16 })
}

/// Safe to call before embedding or any model/layer effects; no recursion into F8.
pub(crate) fn admit_request(config: &ModelConfig, rows: usize, start: usize) -> Result<()> {
    check_selected()?;
    let step_k = selected_step_k()?;
    let transposed = transposed_selected()?;
    let streaming = stream_selected()?;
    ensure!(
        step_k == 16 || selected()?,
        "K32 compact schedule requires {SELECTOR}=1"
    );
    ensure!(
        !transposed || selected()?,
        "{TRANSPOSED_SELECTOR} requires {SELECTOR}=1"
    );
    ensure!(
        !transposed || step_k == 32,
        "{TRANSPOSED_SELECTOR} requires {K32_SELECTOR}=1"
    );
    ensure!(
        !transposed || !check_selected()?,
        "{TRANSPOSED_SELECTOR} rejects replay after original retirement"
    );
    let unified = flag("ATLAS_UNIFIED_MOE_LAYOUT")?;
    ensure!(
        !transposed || streaming || unified,
        "{TRANSPOSED_SELECTOR} requires ATLAS_UNIFIED_MOE_LAYOUT=1"
    );
    ensure!(
        !streaming || !unified,
        "{STREAM_SELECTOR} conflicts with ATLAS_UNIFIED_MOE_LAYOUT"
    );
    if !selected()? {
        return Ok(());
    }
    ensure!(
        flag("ATLAS_QWEN4_PREFILL_MOE_BATCH")?,
        "{SELECTOR} requires the F8 grouped-FFN selector"
    );
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
            && config.ep_world_size == 1
            && config.tp_world_size == 1,
        "{SELECTOR} requires canonical C1 single-GPU original-layout Flash-Next"
    );
    // Singleton chunks retain shipping decode and do not engage compact GEMM.
    ensure!(
        rows > 0 && start.checked_add(rows).is_some_and(|end| end <= 2048),
        "{SELECTOR} requires a nonempty dense-window prompt within 2048 tokens"
    );
    for name in [
        "ATLAS_NVFP4_MOE_WORKLIST",
        "ATLAS_MOE_EXACT_PREFILL_GRID",
        "ATLAS_NVFP4_GATE_UP_M128",
        "ATLAS_HYBRID_MOE_LAYOUT",
    ] {
        ensure!(!flag(name)?, "{SELECTOR} conflicts with {name}");
    }
    ensure!(
        transposed || !unified,
        "{SELECTOR} original layout conflicts with ATLAS_UNIFIED_MOE_LAYOUT"
    );
    Ok(())
}

#[derive(Clone, Copy)]
pub(super) struct Kernels {
    pub plan: KernelHandle,
    pub gemm: KernelHandle,
    pub block: u32,
    pub gateup: Option<KernelHandle>,
    pub step_k: u32,
    pub transposed: bool,
    pub streamed: bool,
}

pub(super) fn validate_handles(kernels: Option<Kernels>) -> Result<()> {
    let step_k = selected_step_k()?;
    let transposed = transposed_selected()?;
    let streamed = stream_selected()?;
    ensure!(
        abi::bundle_matches(selected()?, kernels.map(|k| (k.plan.0, k.gemm.0))),
        "{SELECTOR} requires an unchanged selector and both private kernels"
    );
    ensure!(
        kernels.is_none_or(|kernels| kernels.step_k == step_k),
        "compact STEP_K selector changed after kernel load"
    );
    ensure!(
        kernels.is_none_or(|kernels| kernels.transposed == transposed),
        "compact layout selector changed after kernel load"
    );
    ensure!(
        kernels.is_none_or(|kernels| kernels.streamed == streamed),
        "compact stream selector changed after kernel load"
    );
    Ok(())
}

pub(super) fn load(gpu: &dyn GpuBackend, config: &ModelConfig) -> Result<Option<Kernels>> {
    admit_request(config, 2, 0)?;
    if !selected()? {
        return Ok(None);
    }
    let step_k = selected_step_k()?;
    let transposed = transposed_selected()?;
    let streamed = stream_selected()?;
    let (module, plan, gemm) = match (transposed, step_k) {
        (false, 16) => (
            "qwen4_moe_compact",
            "qwen4_moe_compact_plan",
            "qwen4_moe_compact_gemm",
        ),
        (false, 32) => (
            "qwen4_moe_compact_k32",
            "qwen4_moe_compact_plan_k32",
            "qwen4_moe_compact_gemm_k32",
        ),
        (true, 32) => (
            "qwen4_moe_compact_t_k32",
            "qwen4_moe_compact_t_plan_k32",
            "qwen4_moe_compact_t_gemm_k32",
        ),
        _ => unreachable!("validated compact layout and STEP_K"),
    };
    // Default-off V2 GEMM (same ABI/plan/workspace, pipelined kernel body).
    let v2 = transposed && step_k == 32 && v2_level()? > 0;
    let (plan_handle, gemm_handle, block, gateup) = if v2 {
        let level = v2_level()?;
        let name = match level {
            6 => "qwen4_moe_compact_t_gemm_k32_v6",
            5 => "qwen4_moe_compact_t_gemm_k32_v5",
            4 => "qwen4_moe_compact_t_gemm_k32_v4",
            2 | 3 => "qwen4_moe_compact_t_gemm_k32_v3",
            _ => "qwen4_moe_compact_t_gemm_k32_v2",
        };
        // 256-thread CTAs only for the 64x128 tile (v3/v4); v2 and v5 are 64x64/128.
        let block = if matches!(level, 2 | 3 | 4 | 6) { 256 } else { 128 };
        let gemm = gpu.kernel("qwen4_moe_compact_t_k32_v2", name)?;
        let plan = gpu.kernel("qwen4_moe_compact_t_k32_v2", "qwen4_moe_compact_t_plan_k32_v2")?;
        let gateup = if level == 3 {
            Some(gpu.kernel("qwen4_moe_compact_t_k32_v2", "qwen4_moe_compact_t_gemm_gateup_k32")?)
        } else {
            None
        };
        tracing::info!("MOE_PREFILL_COMPACT_V2_SELECTED selector={V2_SELECTOR} kernel={name} fused_gateup={}", gateup.is_some());
        (plan, gemm, block, gateup)
    } else {
        (gpu.kernel(module, plan)?, gpu.kernel(module, gemm)?, 128, None)
    };
    let kernels = Some(Kernels {
        plan: plan_handle,
        gemm: gemm_handle,
        block,
        gateup,
        step_k,
        transposed,
        streamed,
    });
    validate_handles(kernels)?;
    Ok(kernels)
}

/// Called by F8 before serial core effects, and defensively at the GEMM boundary.
pub(crate) fn validate_context(ctx: &ForwardContext, rows: usize) -> Result<()> {
    admit_request(ctx.config, rows, 0)?;
    if !selected()? {
        return Ok(());
    }
    ensure!(
        ctx.comm.is_none() && !ctx.graph_capture,
        "{SELECTOR} requires eager execution without a communicator"
    );
    if rows == 1 {
        return Ok(());
    }
    abi::contract(rows).map_err(anyhow::Error::msg)?;
    let sizes = ctx.buffers.sizes();
    let expanded = rows * 10;
    ensure!(
        ctx.buffers.max_batch_tokens() >= rows
            && sizes.moe_worklist >= abi::ARENA_BYTES
            && sizes.moe_worklist_total >= 4
            && sizes.expert_gate_out >= expanded * 640 * 2
            && sizes.expert_up_out >= expanded * 640 * 2
            && sizes.expert_down_out >= expanded * 2560 * 2
            && sizes.norm_output >= rows * 2560 * 2
            && sizes.attn_output >= rows * 2560 * 2
            && sizes.gate_logits >= (expanded * 3 + 513) * 4
            && sizes.scratch >= expanded * 8,
        "{SELECTOR} persistent workspace/status/expert arenas are undersized"
    );
    let buffers = [
        (ctx.buffers.moe_worklist(), abi::ARENA_BYTES, 16),
        (ctx.buffers.moe_worklist_total(), 4, 4),
        (ctx.buffers.expert_gate_out(), expanded * 640 * 2, 16),
        (ctx.buffers.expert_up_out(), expanded * 640 * 2, 16),
        (ctx.buffers.expert_down_out(), expanded * 2560 * 2, 16),
        (ctx.buffers.norm_output(), sizes.norm_output, 16),
        (ctx.buffers.attn_output(), sizes.attn_output, 16),
        (ctx.buffers.gate_logits(), sizes.gate_logits, 16),
        (ctx.buffers.scratch(), sizes.scratch, 16),
    ];
    let regions = buffers.map(|(ptr, bytes, align)| abi::region(ptr.0, bytes, align));
    for (i, region) in regions.iter().enumerate() {
        let region = (*region).map_err(anyhow::Error::msg)?;
        for other in &regions[..i] {
            ensure!(
                abi::disjoint(region, (*other).map_err(anyhow::Error::msg)?),
                "{SELECTOR} live arena alias"
            );
        }
    }
    Ok(())
}
