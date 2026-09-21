// SPDX-License-Identifier: AGPL-3.0-only

//! EXL3 trellis (K2/K3, 2.0/3.0 bpw) routed-expert DECODE dispatch (M=1).
//!
//! Reference tp1 checkpoint (`quant_method: "exl3"`, 216 routed experts/layer).
//! The routed FFN is THREE launches — fused gate+up over all slots, one flat
//! SwiGLU, fused down over all slots — plus the FP8→NVFP4 shared expert on the
//! existing `w4a16_gemv` path. The blend downstream (`moe_weighted_sum_blend`)
//! is unchanged: outputs land in exactly the layout the fused NVFP4 decode
//! kernels produce.
//!
//! ## Launch budget per layer (top_k = 8)
//!
//! | stage                | bring-up (per-slot) | fused |
//! |----------------------|--------------------:|------:|
//! | routed gate          | top_k = 8           | 1     |
//! | routed up            | top_k = 8           | (same launch) |
//! | routed SwiGLU        | top_k = 8           | 1     |
//! | routed down          | top_k = 8           | 1     |
//! | shared (NVFP4)       | 4                   | 4     |
//! | **total**            | **36**              | **7** |
//!
//! At 43 layers that is 301 launches/token instead of 1548. The bring-up arm
//! is retained behind `ATLAS_EXL3_FUSED=0` for A/B — it is BIT-IDENTICAL
//! (same body, same K-slice per split, same fixed-order combines; fusion moves
//! only WHICH CTA owns which (slot, strip, split) triple — gated by GATE8 of
//! `exl3_gemv_microtest`).
//!
//! Launch sequence is identical every step (expert ids are read ON DEVICE from
//! the routed `indices` buffer inside the kernels), so the arm is
//! CUDA-graph-safe and needs no D2H of the routing.
//!
//! Split-K scratch: the per-slot chain serialized on the stream and could
//! share ONE `ws`/`counters` region. Fused slots run CONCURRENTLY, so the
//! kernels carve a private region per launch group (`group·gridDim.y·N` floats
//! of `ws`, `group·N/128` ints of `counters`) and the host sizes both for the
//! widest group count it will ever launch (`2·top_k`, from gate+up).
//!
//! Prefill uses `forward_prefill_exl3.rs`; speculative wide verify uses the
//! deduplicated m-row ladder in this module. Every legacy NVFP4/E8M0 decode
//! site fails loudly via the `Exl3Trellis` format tag.

use anyhow::{Context, Result, ensure};
use spark_runtime::kernel_args::KernelLaunch;

use super::persistent_work::{
    EXL3_M16_MAX_ROWS, EXL3_M16_WORK_CAPACITY, EXL3_M16_WORK_RECORD_BYTES,
};
use super::*;
use crate::weight_map::Exl3ExpertWeight;

/// Widest split the scratch is sized for (matches the microtest sweep).
const EXL3_MAX_SPLIT: u32 = 12;

/// CTAs in one full GB10 wave for `exl3_gemv_m1*`: 48 SMs × 4 CTAs/SM
/// (`__launch_bounds__(256, 4)`; 64 regs and 20,996 B smem both admit 4).
const EXL3_WAVE_CTAS: u32 = 192;
/// Largest N across the expert shapes (w2: N = hidden = 4096).
const EXL3_MAX_N: u32 = 4096;
/// Hard cap on routed slots the split-K scratch will be sized for. Guards the
/// `2·top_k` group count against a pathological config before it can silently
/// overrun `ws`/`counters`.
const EXL3_MAX_TOP_K: u32 = 32;
const EXL3_PERSISTENT_ROWS: [u32; 2] = [6, EXL3_M16_MAX_ROWS as u32];
const EXL3_PERSISTENT_TOP_K: u32 = 6;
const EXL3_PERSISTENT_WORK_CAPACITIES: [u32; 2] =
    [6 * EXL3_PERSISTENT_TOP_K, EXL3_M16_WORK_CAPACITY as u32];
const EXL3_PERSISTENT_WORK_BYTES: usize = EXL3_M16_WORK_CAPACITY * EXL3_M16_WORK_RECORD_BYTES;
const EXL3_PERSISTENT_STATE_BYTES: usize = 16;
const EXL3_PERSISTENT_CTAS: u32 = 96;

/// Compiled `exl3_gemv_mrow_fused_*` ladder rungs, ascending. An `MROW = R`
/// entry is correct for any `num_tokens <= R`; the host picks the SMALLEST rung
/// `>= num_tokens` so the accumulator array is never over-provisioned, and
/// declines past the last rung (a gather wider than MROW silently drops rows).
/// EXL3 can serve the full 16-row DFlash2 verify block. Keep this separate
/// from the MXFP4 `_t` ladder, which is still proven only through MROW=8.
pub(super) const EXL3_MROW_MAX_ROWS: u32 = 16;
const EXL3_MROW_ARMS: [u32; 6] = [1, 2, 4, 6, 8, EXL3_MROW_MAX_ROWS];

fn exl3_shared_batch_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_VERIFY_GEMV_V2").as_deref() == Ok("1"))
}

/// Device pointer tables for one EXL3 projection across all routed experts.
pub(crate) struct Exl3ProjTable {
    /// `[num_experts]` u64 → each expert's I16 trellis payload.
    pub(crate) trellis_tab: DevicePtr,
    /// `[num_experts]` u64 → each expert's F16 `suh` (length K).
    pub(crate) suh_tab: DevicePtr,
    /// `[num_experts]` u64 → each expert's F16 `svh` (length N).
    pub(crate) svh_tab: DevicePtr,
    /// Output rows of every matrix in this table.
    pub(crate) n: u32,
    /// Input columns of every matrix in this table.
    pub(crate) k: u32,
    /// Homogeneous trellis bitrate for this table (K2 or K3).
    pub(crate) bits: u32,
}

/// EXL3 decode state hung off `MoeLayer` (None on non-EXL3 checkpoints).
pub(crate) struct Exl3MoeState {
    pub(crate) gate: Exl3ProjTable,     // checkpoint w1: [h -> inter]
    pub(crate) up: Exl3ProjTable,       // checkpoint w3: [h -> inter]
    pub(crate) down: Exl3ProjTable,     // checkpoint w2: [inter -> h]
    gemv_idx_k: KernelHandle,           // bring-up: one launch per (slot, proj)
    gemv_fused_gate_up_k: KernelHandle, // fused: all slots × {gate, up}
    gemv_fused_down_k: KernelHandle,    // fused: all slots
    /// `exl3_gemv_mrow_fused_gate_up_m{1,2,4,6,8}` — the dedup'd wide-verify
    /// twin, indexed alongside [`EXL3_MROW_ARMS`].
    mrow_gate_up_k: [KernelHandle; EXL3_MROW_ARMS.len()],
    /// `exl3_gemv_mrow_fused_down_m{1,2,4,6,8}`.
    mrow_down_k: [KernelHandle; EXL3_MROW_ARMS.len()],
    persistent_verify: bool,
    persistent_worklist: DevicePtr,
    persistent_state: DevicePtr,
    persistent_build_k: [KernelHandle; EXL3_PERSISTENT_ROWS.len()],
    persistent_gate_up_k: [KernelHandle; EXL3_PERSISTENT_ROWS.len()],
    persistent_down_k: [KernelHandle; EXL3_PERSISTENT_ROWS.len()],
    num_experts: u32,
    silu_mul_clamped_k: KernelHandle, // routed: DeepSeek-V4 swiglu_limit=10
    silu_mul_noclamp_k: KernelHandle, // shared expert: unclamped
    /// f32 split-K partial scratch. The M=1 fused pair addresses it as
    /// `ws + group·gridDim.y·N` (one region per launch group); the m-row pair
    /// addresses it as `ws + (ws_row·S + split)·N` where `ws_row` is keyed by
    /// the flat routed SLOT (`2·slot + proj` for gate+up, `slot` for down) —
    /// slots are unique across leaders, so leaders never collide. Sized by
    /// [`Self::ws_floats_needed`] for the widest of the two.
    ws: DevicePtr,
    /// i32 `[groups, N/128]` split-election counters (self-resetting; both
    /// families address them as `counters + group·N/128`, so the sizing is
    /// simply the widest group count either family launches).
    counters: DevicePtr,
    /// Widest launch-group count the scratch is sized for: `2·top_k` at M=1,
    /// `2·MOE_VERIFY_MAX_ROWS·top_k` for the m-row verify.
    groups: u32,
    /// Split override from `ATLAS_EXL3_SPLIT` (0 = auto).
    split_override: u32,
    /// `ATLAS_EXL3_FUSED=0` falls back to the per-slot bring-up chain (A/B;
    /// bit-identical, ~5x the launches).
    fused: bool,
    /// P1 prefill state (scratch dequant + H128 activation passes).
    pub(crate) prefill: Exl3PrefillState,
}

impl Exl3MoeState {
    fn persistent_verify_enabled(&self, num_tokens: u32) -> bool {
        self.persistent_verify && EXL3_PERSISTENT_ROWS.contains(&num_tokens)
    }

    fn persistent_arm(&self, num_tokens: u32) -> Option<usize> {
        EXL3_PERSISTENT_ROWS
            .iter()
            .position(|&rows| rows == num_tokens)
    }
}

/// P1 prefill (M>1) state: fixed-size expert-chunk scratch ring for the
/// dequant-to-BF16 path feeding `moe_bf16_grouped_gemm` (plan §3 "P1").
///
/// All three expert projections have the same element count
/// (`inter×h == h×inter`), so ONE slot size and ONE static pointer table
/// serve gate, up and down: slot `z` of every chunk lives at
/// `scratch + z·slot_elems·2`, and the grouped GEMM is launched per chunk
/// with `weight_ptrs = slot_tab`, `expert_offsets + e0`, `num_experts =
/// chunk_len` (offsets are absolute rows, so sub-range launches read and
/// write the correct global rows).
pub(crate) struct Exl3PrefillState {
    /// Mode is frozen at model load so allocation and dispatch cannot drift.
    pub(crate) direct: bool,
    pub(crate) direct_m128: bool,
    pub(crate) direct_k64: bool,
    pub(crate) direct_n128: bool,
    pub(crate) direct_n256: bool,
    pub(crate) fixed_k2: bool,
    /// Exact DeepSeek-V4 K2 persistent shapes use projection-specific kernels.
    pub(crate) fixed_shape: bool,
    pub(crate) persistent: bool,
    pub(crate) fused_post: bool,
    /// Opt-in exact H4096 gate/up pre-rotation sharing one expanded input read.
    pub(crate) dual_pre: bool,
    /// Explicit request for the exact DeepSeek K2 W2A8 prefill chain.
    pub(crate) w2a8_requested: bool,
    /// Fail closed for the exact N=2410 max-prefill receipt instead of
    /// silently selecting any subordinate fallback arm.
    pub(crate) prefill_max_require_arms: bool,
    /// Subordinate opt-in for the fused N128 gate/up-to-down-A8 stage.
    pub(crate) w2a8_fused_gu_down_requested: bool,
    pub(crate) w2a8_pre_dual_emit_k: KernelHandle,
    pub(crate) w2a8_post_silu_pre_emit_k: KernelHandle,
    pub(crate) w2a8_grouped_gu_k: KernelHandle,
    pub(crate) w2a8_grouped_down_k: KernelHandle,
    pub(crate) w2a8_fused_gu_down_emit_n128_k: KernelHandle,
    /// Exact-2410 subordinate rung widening the fused gate/up stage to N256.
    pub(crate) w2a8_fused_gu_down_n256_requested: bool,
    pub(crate) w2a8_fused_gu_down_emit_n256_k: KernelHandle,
    /// Independent subordinate opt-in for the exact N256 down projection.
    pub(crate) w2a8_n256_down_requested: bool,
    pub(crate) w2a8_grouped_n256_down_k: KernelHandle,
    pub(crate) fused_unpermute: bool,
    /// Exact DeepSeek-V4 K2 H4096 uses the fixed post-unpermute kernel.
    pub(crate) hrow_fixed_shape: bool,
    /// Opt-in exact H4096 routed-unpermute + shared-expert blend tail.
    pub(crate) fused_blend: bool,
    /// Preserve the literal request separately from shape eligibility so the
    /// strict max-profile contract can report a declined combined tail.
    pub(crate) fused_blend_requested: bool,
    /// BF16 `[chunk, n·k]` slot-major dequant scratch; null in direct mode.
    pub(crate) scratch: DevicePtr,
    /// `[chunk]` u64 device table → scratch slots; null in direct mode.
    pub(crate) slot_tab: DevicePtr,
    /// Experts dequanted per chunk (`ATLAS_EXL3_PREFILL_CHUNK`, default 8
    /// → 8 × 16.8 MB = 134 MB scratch at the V4 expert shapes).
    pub(crate) chunk: u32,
    pub(crate) dequant_chunk_k: KernelHandle,
    pub(crate) h128_pre_k: KernelHandle,
    pub(crate) h128_pre_dual_h4096_k: KernelHandle,
    pub(crate) h128_post_k: KernelHandle,
    pub(crate) h128_post_silu_pre_k: KernelHandle,
    pub(crate) h128_post_unpermute_k: KernelHandle,
    pub(crate) h128_post_unpermute_h4096_k: KernelHandle,
    pub(crate) h128_post_unpermute_blend_h4096_k: KernelHandle,
    /// Direct trellis -> BF16 register fragments -> tensor-core grouped GEMM (P2).
    pub(crate) grouped_direct_k: KernelHandle,
    pub(crate) grouped_direct_k2_k: KernelHandle,
    pub(crate) grouped_direct_k64_k: KernelHandle,
    pub(crate) grouped_direct_k64_k2_k: KernelHandle,
    pub(crate) grouped_direct_k2_gu_k: KernelHandle,
    pub(crate) grouped_direct_k2_down_k: KernelHandle,
    pub(crate) grouped_direct_k64_k2_gu_k: KernelHandle,
    pub(crate) grouped_direct_k64_k2_down_k: KernelHandle,
    pub(crate) grouped_direct_k64_n128_k2_gu_k: KernelHandle,
    pub(crate) grouped_direct_k64_n128_k2_down_k: KernelHandle,
    pub(crate) grouped_direct_k64_n256_k2_gu_k: KernelHandle,
    pub(crate) grouped_direct_k64_n256_k2_down_k: KernelHandle,
    pub(crate) grouped_direct_m128_k: KernelHandle,
}

const fn exl3_dual_pre_enabled(fixed_shape: bool, fused_post: bool, requested: bool) -> bool {
    fixed_shape && fused_post && requested
}

impl Exl3MoeState {
    /// SPLIT_K policy: ~96 CTAs per launch GROUP (2 CTAs/SM on GB10's 48 SMs),
    /// then rounded UP to the first split that makes the whole fused grid an
    /// exact multiple of a 192-CTA wave.
    ///
    /// The kernel is `__launch_bounds__(256, 4)` at 64 registers / 20,996 B
    /// smem, so a full wave is 4 CTAs/SM × 48 SMs = **192 CTAs**. A fused
    /// launch runs `strips · split · groups` CTAs, and any remainder leaves the
    /// tail wave partly empty — SMs idle for the whole tail.
    ///
    /// At the V4 shapes the base target already lands wave-exact for gate/up:
    /// `inter = 2048` → 16 strips, base split 6 → `16·6·2·top_k = 192·top_k`
    /// for ANY `top_k`. `down` is the exposed one: `h = 4096` → 32 strips,
    /// base split 3 → `32·3·top_k = 96·top_k`, which is a whole wave only when
    /// `top_k` is EVEN. With adaptive top-K merged, an odd routed width would
    /// run the last `down` wave half empty; walking the split up to 6 restores
    /// `192·top_k`. For even `top_k` (the measured configuration) this returns
    /// the same value as before — the guard is a no-op on the hot path.
    ///
    /// `groups` MUST be the FUSED group count (`2·top_k` for gate+up, `top_k`
    /// for down) even when the per-slot fallback arm runs, so that both arms
    /// use the same SPLIT_K and stay bit-identical (microtest GATE8).
    ///
    /// NOTE (unmeasured, deliberately NOT the default): `ATLAS_EXL3_SPLIT=1`
    /// drops the whole `ws` round-trip, the election atomics and the partial
    /// re-read, and gives each CTA the full K-slice (one long trellis stream
    /// instead of `S` islands) — but it re-slices K, so it is NOT bit-identical
    /// to the default, and at 16·1·2·top_k = 32·top_k CTAs it underfills.
    fn split_for(&self, n: u32, groups: u32) -> u32 {
        if self.split_override > 0 {
            return self.split_override.min(EXL3_MAX_SPLIT);
        }
        let strips = (n / 128).max(1);
        let base = (96 / strips).clamp(1, EXL3_MAX_SPLIT);
        let groups = groups.max(1);
        for s in base..=EXL3_MAX_SPLIT {
            if (strips * s * groups) % EXL3_WAVE_CTAS == 0 {
                return s;
            }
        }
        base
    }

    /// The m-row ladder rung that serves `num_tokens`: the smallest compiled
    /// `MROW >= num_tokens`. `None` past the widest arm — going wider would
    /// silently drop the gathered rows past MROW, which is a correctness bug,
    /// not a slowdown.
    fn mrow_arm(&self, num_tokens: u32) -> Option<(usize, u32)> {
        EXL3_MROW_ARMS
            .iter()
            .position(|&r| r >= num_tokens)
            .map(|i| (i, EXL3_MROW_ARMS[i]))
    }
}

/// [`Exl3MoeState::split_for`] as a free function so the constructor can size
/// the split-K scratch by the SAME splits the dispatch will launch with.
fn exl3_split_for(n: u32, split_override: u32, groups: u32) -> u32 {
    if split_override > 0 {
        return split_override.min(EXL3_MAX_SPLIT);
    }
    let strips = (n / 128).max(1);
    let base = (96 / strips).clamp(1, EXL3_MAX_SPLIT);
    let groups = groups.max(1);
    for s in base..=EXL3_MAX_SPLIT {
        if (strips * s * groups) % EXL3_WAVE_CTAS == 0 {
            return s;
        }
    }
    base
}

/// f32 slots the split-K partial scratch must hold to serve every launch this
/// state can make.
///
/// Four claims, take the max:
///   * M=1 gate+up — `2·top_k` groups × `[split_gu, inter]`
///   * M=1 down    — `top_k` groups × `[split_dn, h]`
///   * m-row gate+up — `ws_row` runs to `2·max_rows·top_k`, each `[split_gu, inter]`
///   * m-row down    — `ws_row` runs to `max_rows·top_k`, each `[split_dn, h]`
///
/// Sized from the ACTUAL splits (not `EXL3_MAX_SPLIT`), so the override arm is
/// covered without paying for it by default. At the DeepSeek-V4 shapes
/// (inter 2048, h 4096, top_k 8, max_rows 8) the binding term is the m-row
/// gate+up: 2·64·6·2048 f32 = 6.3 MB per layer.
fn exl3_ws_floats(
    gate_n: u32,
    down_n: u32,
    top_k: u32,
    split_gu: u32,
    split_dn: u32,
    max_rows: u32,
) -> usize {
    let gu = |groups: u32| groups as usize * split_gu as usize * gate_n as usize;
    let dn = |groups: u32| groups as usize * split_dn as usize * down_n as usize;
    gu(2 * top_k)
        .max(dn(top_k))
        .max(gu(2 * max_rows * top_k))
        .max(dn(max_rows * top_k))
}

fn build_proj_table(
    experts: &[Exl3ExpertWeight],
    proj: impl Fn(&Exl3ExpertWeight) -> &crate::weight_map::Exl3Weight,
    gpu: &dyn GpuBackend,
) -> Result<Exl3ProjTable> {
    let n_exp = experts.len();
    ensure!(n_exp > 0, "EXL3: empty expert list");
    let (n, k) = {
        let w = proj(&experts[0]);
        (w.n, w.k)
    };
    for (i, e) in experts.iter().enumerate() {
        let w = proj(e);
        ensure!(
            w.n == n && w.k == k && w.bits == proj(&experts[0]).bits,
            "EXL3 expert {i}: shape/bits [{}x{}, K{}] != expert 0 [{n}x{k}, K{}]",
            w.n,
            w.k,
            w.bits,
            proj(&experts[0]).bits,
        );
    }
    let ptr_bytes = |f: &dyn Fn(&crate::weight_map::Exl3Weight) -> DevicePtr| -> Vec<u8> {
        experts
            .iter()
            .flat_map(|e| f(proj(e)).0.to_le_bytes())
            .collect()
    };
    let up = |bytes: Vec<u8>| -> Result<DevicePtr> {
        let p = gpu.alloc(bytes.len())?;
        gpu.copy_h2d(&bytes, p)?;
        Ok(p)
    };
    Ok(Exl3ProjTable {
        trellis_tab: up(ptr_bytes(&|w| w.trellis))?,
        suh_tab: up(ptr_bytes(&|w| w.suh))?,
        svh_tab: up(ptr_bytes(&|w| w.svh))?,
        n,
        k,
        bits: proj(&experts[0]).bits,
    })
}

/// Resolve the `EXL3_MROW_ARMS` ladder for one projection family
/// (`prefix` + the rung's MROW).
fn mrow_handles(
    gpu: &dyn GpuBackend,
    module: &str,
    prefix: &str,
) -> Result<[KernelHandle; EXL3_MROW_ARMS.len()]> {
    let mut out = [KernelHandle(0); EXL3_MROW_ARMS.len()];
    for (slot, &mrow) in out.iter_mut().zip(EXL3_MROW_ARMS.iter()) {
        let name = format!("{prefix}{mrow}");
        *slot = gpu
            .kernel(module, &name)
            .with_context(|| format!("EXL3 wide verify needs the {name} kernel"))?;
    }
    Ok(out)
}

/// Resolve one exact-width persistent worklist family for M6 and M16.  These
/// are equality arms rather than ladder rungs because their wire records have
/// different strides and fixed slot counts.
fn persistent_handles(
    gpu: &dyn GpuBackend,
    module: &str,
    prefix: &str,
) -> Result<[KernelHandle; EXL3_PERSISTENT_ROWS.len()]> {
    let mut out = [KernelHandle(0); EXL3_PERSISTENT_ROWS.len()];
    for (slot, &rows) in out.iter_mut().zip(EXL3_PERSISTENT_ROWS.iter()) {
        let name = format!("{prefix}{rows}");
        *slot = gpu
            .kernel(module, &name)
            .with_context(|| format!("EXL3 persistent verify needs the {name} kernel"))?;
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn launch_gemv_idx(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    tab: &Exl3ProjTable,
    indices: DevicePtr,
    slot: u32,
    output: DevicePtr,
    ws: DevicePtr,
    counters: DevicePtr,
    split: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([tab.n / 128, split, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(tab.trellis_tab)
        .arg_ptr(tab.suh_tab)
        .arg_ptr(tab.svh_tab)
        .arg_ptr(indices)
        .arg_u32(slot)
        .arg_ptr(output)
        .arg_ptr(ws)
        .arg_ptr(counters)
        .arg_u32(tab.n)
        .arg_u32(tab.k)
        .arg_u32(tab.bits)
        .launch(stream)
}

impl MoeLayer {
    /// Attach EXL3 routed experts (called by the DeepSeek-V4 loader after
    /// construction, mirroring the `experts_scale_kind` override pattern).
    /// Builds the device pointer tables and the split-K scratch, resolves the
    /// kernel handles, and tags the routed format `Exl3Trellis` so every
    /// legacy NVFP4/E8M0 dispatch site fails loudly instead of dereferencing
    /// the null `gate_ptrs` tables.
    pub fn set_exl3_experts(
        &mut self,
        experts: &[Exl3ExpertWeight],
        top_k: u32,
        gpu: &dyn GpuBackend,
    ) -> Result<()> {
        let gate = build_proj_table(experts, |e| &e.gate_proj, gpu)?;
        let up = build_proj_table(experts, |e| &e.up_proj, gpu)?;
        let down = build_proj_table(experts, |e| &e.down_proj, gpu)?;
        ensure!(
            gate.bits == up.bits && gate.bits == down.bits,
            "EXL3 fused dispatch requires one bitrate across gate/up/down; got K{}/K{}/K{}",
            gate.bits,
            up.bits,
            down.bits
        );
        ensure!(
            gate.n <= EXL3_MAX_N && down.n <= EXL3_MAX_N,
            "EXL3 scratch sized for N <= {EXL3_MAX_N}, got gate N={} down N={}",
            gate.n,
            down.n
        );
        ensure!(
            top_k > 0 && top_k <= EXL3_MAX_TOP_K,
            "EXL3 decode scratch sized for top_k in 1..={EXL3_MAX_TOP_K}, got {top_k}"
        );
        let decode_module =
            if gate.bits == 2 && std::env::var("ATLAS_EXL3_FIXED_K2").as_deref() != Ok("0") {
                "exl3_gemv_k2"
            } else {
                "exl3_gemv"
            };
        let split_override = std::env::var("ATLAS_EXL3_SPLIT")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(0);
        // Widest launch-group count. M=1 fused gate+up takes `2·top_k`; the
        // m-row verify takes `2·max_rows·top_k` (one CTA-set per flat routed
        // slot per projection, duplicates exiting at the leader election).
        // Every group needs its own [N/128] election counters, and every
        // OUTPUT ROW its own [SPLIT_K, N] fp32 partial region, since fused /
        // dedup'd groups run concurrently.
        let max_rows = *EXL3_MROW_ARMS.last().unwrap();
        let groups = 2 * max_rows * top_k;
        // MUST match what the dispatch launches with, for EVERY arm: the
        // wave-walk in `split_for` can round the split UP, and the partial
        // region is [split, N] per row — sizing with a smaller split would
        // undersize the scratch (memory corruption, not a slowdown). Take the
        // max over the m=1 fused group counts (2*top_k / top_k) and the m-row
        // ones (2*max_rows*top_k / max_rows*top_k).
        let split_gu = exl3_split_for(gate.n, split_override, 2 * top_k).max(exl3_split_for(
            gate.n,
            split_override,
            groups,
        ));
        let split_dn = exl3_split_for(down.n, split_override, top_k).max(exl3_split_for(
            down.n,
            split_override,
            max_rows * top_k,
        ));
        let ws_floats = exl3_ws_floats(gate.n, down.n, top_k, split_gu, split_dn, max_rows);
        let counter_bytes = groups as usize * (EXL3_MAX_N as usize / 128) * 4;
        let ws = gpu.alloc(ws_floats * 4)?;
        let counters = gpu.alloc(counter_bytes)?;
        // Counters must be zero before the FIRST launch; the kernel re-arms
        // them to zero on completion, so this is the only memset ever needed.
        // (Groups never touch each other's counters, so the invariant "all
        // zero at rest" survives fusion, dedup and CUDA-graph replay
        // unchanged — a duplicate slot's group returns before the atomic.)
        gpu.memset(counters, 0, counter_bytes)?;

        // Experimental compact M6/M16 verify: a one-CTA device builder emits
        // one record per distinct routed expert, then fixed 96-CTA kernels
        // consume the exact same logical (row, N-strip, split) work as the
        // corresponding m-row arm. Keep the default path unchanged until GB10
        // byte-parity and timing qualify these exact DeepSeek K2/top-6 shapes.
        let persistent_verify = std::env::var("ATLAS_EXL3_VERIFY_WORKLIST").as_deref() == Ok("1");
        let persistent_shape = gate.bits == 2
            && up.bits == 2
            && down.bits == 2
            && top_k == EXL3_PERSISTENT_TOP_K
            && gate.n == 2048
            && gate.k == 4096
            && up.n == 2048
            && up.k == 4096
            && down.n == 4096
            && down.k == 2048;
        ensure!(
            !persistent_verify || persistent_shape,
            "ATLAS_EXL3_VERIFY_WORKLIST=1 requires exact DeepSeek K2 shapes and top_k=6"
        );
        let (
            persistent_worklist,
            persistent_state,
            persistent_build_k,
            persistent_gate_up_k,
            persistent_down_k,
        ) = if persistent_verify {
            let build = persistent_handles(gpu, decode_module, "exl3_build_m")?;
            let gate_up =
                persistent_handles(gpu, decode_module, "exl3_gemv_mrow_persistent_gate_up_m")?;
            let down = persistent_handles(gpu, decode_module, "exl3_gemv_mrow_persistent_down_m")?;
            let worklist = gpu.alloc(EXL3_PERSISTENT_WORK_BYTES)?;
            let state = match gpu.alloc(EXL3_PERSISTENT_STATE_BYTES) {
                Ok(state) => state,
                Err(error) => {
                    let _ = gpu.free(worklist);
                    return Err(error);
                }
            };
            (worklist, state, build, gate_up, down)
        } else {
            (
                DevicePtr(0),
                DevicePtr(0),
                [KernelHandle(0); EXL3_PERSISTENT_ROWS.len()],
                [KernelHandle(0); EXL3_PERSISTENT_ROWS.len()],
                [KernelHandle(0); EXL3_PERSISTENT_ROWS.len()],
            )
        };

        // ── P1 prefill scratch (see Exl3PrefillState) ──
        // One slot size serves gate/up/down: all three are inter×h elements.
        ensure!(
            gate.n as u64 * gate.k as u64 == down.n as u64 * down.k as u64
                && up.n == gate.n
                && up.k == gate.k,
            "EXL3 prefill scratch assumes equal-element projections \
             (gate {}x{}, up {}x{}, down {}x{})",
            gate.n,
            gate.k,
            up.n,
            up.k,
            down.n,
            down.k
        );
        let direct = std::env::var("ATLAS_EXL3_PREFILL_DIRECT").as_deref() == Ok("1");
        let direct_m128 = direct && std::env::var("ATLAS_EXL3_PREFILL_M128").as_deref() == Ok("1");
        let direct_k64 = direct && std::env::var("ATLAS_EXL3_PREFILL_K64").as_deref() == Ok("1");
        let fixed_k2 = direct && std::env::var("ATLAS_EXL3_PREFILL_FIXED_K2").as_deref() != Ok("0");
        ensure!(
            !(direct_m128 && direct_k64),
            "ATLAS_EXL3_PREFILL_M128 and ATLAS_EXL3_PREFILL_K64 are separate experimental arms"
        );
        let persistent =
            direct && std::env::var("ATLAS_EXL3_PREFILL_PERSISTENT").as_deref() != Ok("0");
        let fixed_shape = direct
            && persistent
            && !direct_m128
            && fixed_k2
            && gate.bits == 2
            && up.bits == 2
            && down.bits == 2
            && gate.n == 2048
            && gate.k == 4096
            && up.n == 2048
            && up.k == 4096
            && down.n == 4096
            && down.k == 2048
            && std::env::var("ATLAS_EXL3_PREFILL_FIXED_SHAPE").as_deref() != Ok("0");
        let direct_n128 = fixed_shape
            && direct_k64
            && std::env::var("ATLAS_EXL3_PREFILL_N128").as_deref() == Ok("1");
        let direct_n256 = fixed_shape
            && direct_k64
            && std::env::var("ATLAS_EXL3_PREFILL_N256").as_deref() == Ok("1");
        ensure!(
            !(direct_n128 && direct_n256),
            "ATLAS_EXL3_PREFILL_N128 and ATLAS_EXL3_PREFILL_N256 are separate experimental arms"
        );
        let fused_post = std::env::var("ATLAS_EXL3_PREFILL_FUSED_POST").as_deref() == Ok("1");
        let dual_pre = exl3_dual_pre_enabled(
            fixed_shape,
            fused_post,
            std::env::var("ATLAS_EXL3_PREFILL_DUAL_PRE").as_deref() == Ok("1"),
        );
        let fused_unpermute =
            std::env::var("ATLAS_EXL3_PREFILL_FUSED_UNPERMUTE").as_deref() == Ok("1");
        let hrow_fixed_shape = fused_unpermute
            && gate.bits == 2
            && up.bits == 2
            && down.bits == 2
            && gate.n == 2048
            && gate.k == 4096
            && up.n == 2048
            && up.k == 4096
            && down.n == 4096
            && down.k == 2048
            && std::env::var("ATLAS_EXL3_HROW_FIXED_SHAPE").as_deref() != Ok("0");
        let fused_blend_requested =
            std::env::var("ATLAS_EXL3_PREFILL_FUSED_BLEND").as_deref() == Ok("1");
        let fused_blend = fixed_shape && hrow_fixed_shape && fused_blend_requested;
        let w2a8_requested = std::env::var("ATLAS_EXL3_PREFILL_W2A8").as_deref() == Ok("1");
        let prefill_max_require_arms =
            std::env::var("ATLAS_PREFILL_MAX_REQUIRE_ARMS").as_deref() == Ok("1");
        let (
            w2a8_pre_dual_emit_k,
            w2a8_post_silu_pre_emit_k,
            w2a8_grouped_gu_k,
            w2a8_grouped_down_k,
        ) = if w2a8_requested {
            (
                super::super::try_kernel(
                    gpu,
                    "exl3_w2a8_h128_emit",
                    "exl3_w2a8_h128_pre_dual_emit_h4096",
                ),
                super::super::try_kernel(
                    gpu,
                    "exl3_w2a8_h128_emit",
                    "exl3_w2a8_h128_post_silu_pre_emit_h2048",
                ),
                super::super::try_kernel(
                    gpu,
                    "exl3_w2a8_grouped_prefill_k2_gu",
                    "exl3_w2a8_grouped_prefill_k2_gu",
                ),
                super::super::try_kernel(
                    gpu,
                    "exl3_w2a8_grouped_prefill_k2_down",
                    "exl3_w2a8_grouped_prefill_k2_down",
                ),
            )
        } else {
            (
                KernelHandle(0),
                KernelHandle(0),
                KernelHandle(0),
                KernelHandle(0),
            )
        };
        let w2a8_fused_gu_down_requested = w2a8_requested
            && std::env::var("ATLAS_EXL3_PREFILL_W2A8_FUSED_GU_DOWN").as_deref() == Ok("1");
        let w2a8_fused_gu_down_emit_n128_k = if w2a8_fused_gu_down_requested {
            super::super::try_kernel(
                gpu,
                "exl3_w2a8_fused_gu_down_emit_n128",
                "exl3_w2a8_fused_gu_down_emit_n128",
            )
        } else {
            KernelHandle(0)
        };
        let w2a8_fused_gu_down_n256_requested = w2a8_fused_gu_down_requested
            && std::env::var("ATLAS_EXL3_PREFILL_W2A8_FUSED_GU_DOWN_N256").as_deref() == Ok("1");
        let w2a8_fused_gu_down_emit_n256_k = if w2a8_fused_gu_down_n256_requested {
            super::super::try_kernel(
                gpu,
                "exl3_w2a8_fused_gu_down_emit_n256",
                "exl3_w2a8_fused_gu_down_emit_n256",
            )
        } else {
            KernelHandle(0)
        };
        let w2a8_n256_down_requested = w2a8_requested
            && std::env::var("ATLAS_EXL3_PREFILL_W2A8_N256_DOWN").as_deref() == Ok("1");
        let w2a8_grouped_n256_down_k = if w2a8_n256_down_requested {
            super::super::try_kernel(
                gpu,
                "exl3_w2a8_grouped_prefill_n256_k2_down",
                "exl3_w2a8_grouped_prefill_n256_k2_down",
            )
        } else {
            KernelHandle(0)
        };
        if prefill_max_require_arms {
            ensure!(
                w2a8_requested,
                "ATLAS_PREFILL_MAX_REQUIRE_ARMS=1 requires ATLAS_EXL3_PREFILL_W2A8=1"
            );
            ensure!(
                direct
                    && persistent
                    && fixed_shape
                    && fixed_k2
                    && !direct_m128
                    && !direct_n128
                    && !direct_n256
                    && dual_pre
                    && fused_post,
                "ATLAS_PREFILL_MAX_REQUIRE_ARMS=1 requires the exact direct persistent K2 W2A8 prerequisites"
            );
            ensure!(
                w2a8_pre_dual_emit_k.0 != 0
                    && w2a8_post_silu_pre_emit_k.0 != 0
                    && w2a8_grouped_gu_k.0 != 0
                    && w2a8_grouped_down_k.0 != 0,
                "ATLAS_PREFILL_MAX_REQUIRE_ARMS=1 cannot load the base W2A8 kernels"
            );
            ensure!(
                w2a8_fused_gu_down_requested && w2a8_fused_gu_down_emit_n128_k.0 != 0,
                "ATLAS_PREFILL_MAX_REQUIRE_ARMS=1 requires the fused GU N128 arm"
            );
            ensure!(
                !w2a8_fused_gu_down_n256_requested,
                "ATLAS_PREFILL_MAX_REQUIRE_ARMS=1 rejects the native-losing fused GU N256 arm"
            );
            ensure!(
                w2a8_n256_down_requested && w2a8_grouped_n256_down_k.0 != 0,
                "ATLAS_PREFILL_MAX_REQUIRE_ARMS=1 requires the N256 down arm"
            );
            ensure!(
                !fused_unpermute || hrow_fixed_shape,
                "ATLAS_PREFILL_MAX_REQUIRE_ARMS=1 requested fused unpermute tail is not exact-H4096 eligible"
            );
            ensure!(
                !fused_blend_requested || fused_blend,
                "ATLAS_PREFILL_MAX_REQUIRE_ARMS=1 requested fused blend tail is not exact-H4096 eligible"
            );
        }
        let configured_chunk = std::env::var("ATLAS_EXL3_PREFILL_CHUNK")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .filter(|&c| c > 0)
            .unwrap_or(8)
            .min(experts.len() as u32);
        let slot_bytes = gate.n as usize * gate.k as usize * 2;
        let (scratch, slot_tab, chunk) = if direct {
            (DevicePtr(0), DevicePtr(0), 0)
        } else {
            let scratch = gpu.alloc(configured_chunk as usize * slot_bytes)?;
            let slot_ptr_bytes: Vec<u8> = (0..configured_chunk as usize)
                .flat_map(|z| (scratch.0 + (z * slot_bytes) as u64).to_le_bytes())
                .collect();
            let slot_tab = gpu.alloc(slot_ptr_bytes.len())?;
            gpu.copy_h2d(&slot_ptr_bytes, slot_tab)?;
            (scratch, slot_tab, configured_chunk)
        };
        // Do not expand the required kernel-module ABI for a default-off
        // experiment. Older bundles can keep using the incumbent/N128 paths;
        // the N256 symbols become mandatory only when explicitly selected.
        let (grouped_direct_k64_n256_k2_gu_k, grouped_direct_k64_n256_k2_down_k) = if direct_n256 {
            (
                gpu.kernel(
                    "exl3_grouped_prefill_k64_n256_k2_gu",
                    "exl3_grouped_prefill_k64_n256_k2_gu",
                )?,
                gpu.kernel(
                    "exl3_grouped_prefill_k64_n256_k2_down",
                    "exl3_grouped_prefill_k64_n256_k2_down",
                )?,
            )
        } else {
            (KernelHandle(0), KernelHandle(0))
        };
        let prefill = Exl3PrefillState {
            direct,
            direct_m128,
            direct_k64,
            direct_n128,
            direct_n256,
            fixed_k2,
            fixed_shape,
            persistent,
            fused_post,
            dual_pre,
            w2a8_requested,
            prefill_max_require_arms,
            w2a8_fused_gu_down_requested,
            w2a8_pre_dual_emit_k,
            w2a8_post_silu_pre_emit_k,
            w2a8_grouped_gu_k,
            w2a8_grouped_down_k,
            w2a8_fused_gu_down_emit_n128_k,
            w2a8_fused_gu_down_n256_requested,
            w2a8_fused_gu_down_emit_n256_k,
            w2a8_n256_down_requested,
            w2a8_grouped_n256_down_k,
            fused_unpermute,
            hrow_fixed_shape,
            fused_blend,
            fused_blend_requested,
            scratch,
            slot_tab,
            chunk,
            dequant_chunk_k: gpu.kernel("exl3_gemv", "exl3_dequant_chunk_bf16")?,
            h128_pre_k: gpu.kernel("exl3_gemv", "exl3_h128_pre_rows")?,
            h128_pre_dual_h4096_k: gpu.kernel("exl3_gemv_k2", "exl3_h128_pre_dual_rows_h4096")?,
            h128_post_k: gpu.kernel("exl3_gemv", "exl3_h128_post_rows")?,
            h128_post_silu_pre_k: gpu.kernel("exl3_gemv", "exl3_h128_post_silu_pre_rows")?,
            h128_post_unpermute_k: gpu.kernel("exl3_gemv", "exl3_h128_post_unpermute_rows")?,
            h128_post_unpermute_h4096_k: gpu
                .kernel("exl3_gemv_k2", "exl3_h128_post_unpermute_rows_h4096")?,
            h128_post_unpermute_blend_h4096_k: gpu
                .kernel("exl3_gemv_k2", "exl3_h128_post_unpermute_blend_h4096")?,
            grouped_direct_k: gpu.kernel("exl3_grouped_prefill", "exl3_grouped_prefill")?,
            grouped_direct_k2_k: gpu
                .kernel("exl3_grouped_prefill_k2", "exl3_grouped_prefill_k2")?,
            grouped_direct_k64_k: gpu
                .kernel("exl3_grouped_prefill_k64", "exl3_grouped_prefill_k64")?,
            grouped_direct_k64_k2_k: gpu
                .kernel("exl3_grouped_prefill_k64_k2", "exl3_grouped_prefill_k64_k2")?,
            grouped_direct_k2_gu_k: gpu
                .kernel("exl3_grouped_prefill_k2_gu", "exl3_grouped_prefill_k2_gu")?,
            grouped_direct_k2_down_k: gpu.kernel(
                "exl3_grouped_prefill_k2_down",
                "exl3_grouped_prefill_k2_down",
            )?,
            grouped_direct_k64_k2_gu_k: gpu.kernel(
                "exl3_grouped_prefill_k64_k2_gu",
                "exl3_grouped_prefill_k64_k2_gu",
            )?,
            grouped_direct_k64_k2_down_k: gpu.kernel(
                "exl3_grouped_prefill_k64_k2_down",
                "exl3_grouped_prefill_k64_k2_down",
            )?,
            grouped_direct_k64_n128_k2_gu_k: gpu.kernel(
                "exl3_grouped_prefill_k64_n128_k2_gu",
                "exl3_grouped_prefill_k64_n128_k2_gu",
            )?,
            grouped_direct_k64_n128_k2_down_k: gpu.kernel(
                "exl3_grouped_prefill_k64_n128_k2_down",
                "exl3_grouped_prefill_k64_n128_k2_down",
            )?,
            grouped_direct_k64_n256_k2_gu_k,
            grouped_direct_k64_n256_k2_down_k,
            grouped_direct_m128_k: gpu
                .kernel("exl3_grouped_prefill_m128", "exl3_grouped_prefill_m128")?,
        };

        self.exl3 = Some(Exl3MoeState {
            gate,
            up,
            down,
            gemv_idx_k: gpu
                .kernel(decode_module, "exl3_gemv_m1_idx")
                .context("EXL3 checkpoint needs the exl3_gemv kernel module")?,
            gemv_fused_gate_up_k: gpu.kernel(decode_module, "exl3_gemv_m1_fused_gate_up")?,
            gemv_fused_down_k: gpu.kernel(decode_module, "exl3_gemv_m1_fused_down")?,
            mrow_gate_up_k: mrow_handles(gpu, decode_module, "exl3_gemv_mrow_fused_gate_up_m")?,
            mrow_down_k: mrow_handles(gpu, decode_module, "exl3_gemv_mrow_fused_down_m")?,
            persistent_verify,
            persistent_worklist,
            persistent_state,
            persistent_build_k,
            persistent_gate_up_k,
            persistent_down_k,
            num_experts: experts.len() as u32,
            silu_mul_clamped_k: gpu.kernel("moe_silu_mul", "moe_silu_mul")?,
            silu_mul_noclamp_k: gpu.kernel("moe_silu_mul", "silu_mul_noclamp")?,
            ws,
            counters,
            groups,
            split_override,
            fused: std::env::var("ATLAS_EXL3_FUSED").as_deref() != Ok("0"),
            prefill,
        });
        self.experts_scale_kind = crate::weight_map::WeightQuantFormat::Exl3Trellis;
        Ok(())
    }

    /// M=1 decode expert FFN over EXL3 routed experts + NVFP4 shared expert.
    /// Output layout is identical to the fused NVFP4 decode kernels: routed
    /// slot s in `expert_{gate,up,down}_out[s]`, shared rows in the shared
    /// scratch buffers — the downstream blend is untouched.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn dispatch_exl3_decode(
        &self,
        ctx: &ForwardContext,
        expert_input: DevicePtr,
        expert_gate_out: DevicePtr,
        expert_up_out: DevicePtr,
        expert_down_out: DevicePtr,
        shared_gate_scratch: DevicePtr,
        shared_up_scratch: DevicePtr,
        shared_out: DevicePtr,
        indices_dev: DevicePtr,
        h: u32,
        inter: u32,
        top_k: u32,
        stream: u64,
    ) -> Result<()> {
        let st = self
            .exl3
            .as_ref()
            .expect("dispatch_exl3_decode without state");
        ensure!(
            st.gate.n == inter && st.gate.k == h && st.down.n == h && st.down.k == inter,
            "EXL3 dims mismatch: gate [{}x{}] down [{}x{}] vs h={h} inter={inter}",
            st.gate.n,
            st.gate.k,
            st.down.n,
            st.down.k
        );
        ensure!(
            2 * top_k <= st.groups,
            "EXL3 decode scratch sized for {} launch groups, need {} (top_k={top_k})",
            st.groups,
            2 * top_k
        );
        let gpu = ctx.gpu;
        // Group counts are the FUSED ones for BOTH arms, so the per-slot
        // fallback keeps the same SPLIT_K and stays bit-identical (GATE8).
        let split_gu = st.split_for(inter, 2 * top_k);
        let split_dn = st.split_for(h, top_k);

        if st.fused {
            // ── 3 launches for the whole routed FFN. ──
            // (1) gate+up over all slots: grid.z = 2·top_k, z = 2·slot + proj.
            KernelLaunch::new(gpu, st.gemv_fused_gate_up_k)
                .grid([inter / 128, split_gu, 2 * top_k])
                .block([256, 1, 1])
                .arg_ptr(expert_input)
                .arg_ptr(st.gate.trellis_tab)
                .arg_ptr(st.gate.suh_tab)
                .arg_ptr(st.gate.svh_tab)
                .arg_ptr(st.up.trellis_tab)
                .arg_ptr(st.up.suh_tab)
                .arg_ptr(st.up.svh_tab)
                .arg_ptr(indices_dev)
                .arg_ptr(expert_gate_out)
                .arg_ptr(expert_up_out)
                .arg_ptr(st.ws)
                .arg_ptr(st.counters)
                .arg_u32(inter)
                .arg_u32(h)
                .arg_u32(st.gate.bits)
                .launch(stream)?;
            // (2) ONE flat clamped SwiGLU over all slots: gate/up are
            // contiguous `[top_k, inter]`, the kernel is a pure elementwise
            // map over `total_elements`, and the write is same-index in place
            // over the gate rows — bit-identical to the per-slot calls.
            ops::moe_silu_mul(
                gpu,
                st.silu_mul_clamped_k,
                expert_gate_out,
                expert_up_out,
                expert_gate_out,
                top_k * inter,
                stream,
            )?;
            // (3) down over all slots: grid.z = top_k, A row = act + slot·inter.
            KernelLaunch::new(gpu, st.gemv_fused_down_k)
                .grid([h / 128, split_dn, top_k])
                .block([256, 1, 1])
                .arg_ptr(expert_gate_out)
                .arg_ptr(st.down.trellis_tab)
                .arg_ptr(st.down.suh_tab)
                .arg_ptr(st.down.svh_tab)
                .arg_ptr(indices_dev)
                .arg_ptr(expert_down_out)
                .arg_ptr(st.ws)
                .arg_ptr(st.counters)
                .arg_u32(h)
                .arg_u32(inter)
                .arg_u32(st.down.bits)
                .launch(stream)?;
        } else {
            // ── A/B fallback (`ATLAS_EXL3_FUSED=0`): the per-slot bring-up
            // chain, 4·top_k launches. Same-stream launches serialize, so
            // group 0 of the ws/counters scratch serves every slot.
            for slot in 0..top_k {
                let gate_row = expert_gate_out.offset(slot as usize * inter as usize * 2);
                let up_row = expert_up_out.offset(slot as usize * inter as usize * 2);
                let down_row = expert_down_out.offset(slot as usize * h as usize * 2);
                launch_gemv_idx(
                    gpu,
                    st.gemv_idx_k,
                    expert_input,
                    &st.gate,
                    indices_dev,
                    slot,
                    gate_row,
                    st.ws,
                    st.counters,
                    split_gu,
                    stream,
                )?;
                launch_gemv_idx(
                    gpu,
                    st.gemv_idx_k,
                    expert_input,
                    &st.up,
                    indices_dev,
                    slot,
                    up_row,
                    st.ws,
                    st.counters,
                    split_gu,
                    stream,
                )?;
                // act = clamped_swiglu(gate, up), in place over the gate row
                // (elementwise same-index read/write — safe).
                ops::moe_silu_mul(
                    gpu,
                    st.silu_mul_clamped_k,
                    gate_row,
                    up_row,
                    gate_row,
                    inter,
                    stream,
                )?;
                launch_gemv_idx(
                    gpu,
                    st.gemv_idx_k,
                    gate_row,
                    &st.down,
                    indices_dev,
                    slot,
                    down_row,
                    st.ws,
                    st.counters,
                    split_dn,
                    stream,
                )?;
            }
        }

        if self.run_native_fp8_shared_expert(
            expert_input,
            1,
            h,
            ctx.config.shared_expert_intermediate_size as u32,
            shared_gate_scratch,
            shared_up_scratch,
            shared_out,
            ctx,
            stream,
        )? {
            return Ok(());
        }

        // ── Shared expert: FP8-block-scaled on disk → NVFP4 at load (the
        // heterogeneous-checkpoint arm of `load_expert_proj`). UNCLAMPED
        // SwiGLU (DeepseekV4MLP — see moe_shared_expert_fused.cu).
        //
        // NOT EXL3, so it keeps its own `w4a16_gemv` kernels. It is data-
        // independent of the routed chain (same `expert_input`, disjoint
        // outputs), but shares this stream, so it runs after. Overlapping it
        // would need a side stream + event join around the fused pair — worth
        // measuring, but it is 4 small NVFP4 GEMVs against 3 large trellis
        // launches, so the routed side is what the launch collapse targets. ──
        self.shared_experts_scale_kind.expect(
            crate::weight_map::WeightQuantFormat::Nvfp4,
            "EXL3 decode arm computes the shared expert via w4a16_gemv (NVFP4)",
        );
        let shared_inter = ctx.config.shared_expert_intermediate_size as u32;
        let sh = &self.weights.shared_expert;
        ops::w4a16_gemv(
            gpu,
            self.w4a16_gemv,
            expert_input,
            &sh.gate_proj,
            shared_gate_scratch,
            shared_inter,
            h,
            stream,
        )?;
        ops::w4a16_gemv(
            gpu,
            self.w4a16_gemv,
            expert_input,
            &sh.up_proj,
            shared_up_scratch,
            shared_inter,
            h,
            stream,
        )?;
        ops::moe_silu_mul(
            gpu,
            st.silu_mul_noclamp_k,
            shared_gate_scratch,
            shared_up_scratch,
            shared_gate_scratch,
            shared_inter,
            stream,
        )?;
        ops::w4a16_gemv(
            gpu,
            self.w4a16_gemv,
            shared_gate_scratch,
            &sh.down_proj,
            shared_out,
            h,
            shared_inter,
            stream,
        )?;
        Ok(())
    }

    /// True when a `num_tokens`-row verify would take the dedup'd EXL3 m-row
    /// kernels. The twin of [`Self::verify_ffn_is_batched`] for the trellis
    /// arm — callers use it to decide whether batching the verify MoE is worth
    /// it at all, since the fallback (`forward_batched`) re-streams the whole
    /// routed expert set once PER ROW.
    pub(super) fn exl3_verify_is_batched(&self, ctx: &ForwardContext, num_tokens: u32) -> bool {
        let Some(st) = self.exl3.as_ref() else {
            return false;
        };
        if st.mrow_arm(num_tokens).is_none() {
            return false;
        }
        if st
            .mrow_gate_up_k
            .iter()
            .chain(&st.mrow_down_k)
            .any(|k| k.0 == 0)
        {
            return false;
        }
        // The shared expert uses either the exact grouped-batch NVFP4 chain or
        // its exact per-row fallback. Both need the shared scratch to hold
        // `num_tokens` rows at the MXFP4 verify stride.
        (ctx.config.shared_expert_intermediate_size as u32)
            <= ctx.config.moe_intermediate_size as u32
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_exl3_persistent_verify(
        &self,
        gpu: &dyn GpuBackend,
        st: &Exl3MoeState,
        expert_input: DevicePtr,
        expert_gate_out: DevicePtr,
        expert_up_out: DevicePtr,
        expert_down_out: DevicePtr,
        indices_dev: DevicePtr,
        h: u32,
        inter: u32,
        top_k: u32,
        num_tokens: u32,
        split_gu: u32,
        split_dn: u32,
        stream: u64,
    ) -> Result<()> {
        let arm = st
            .persistent_arm(num_tokens)
            .context("EXL3 persistent verify requires exactly 6 or 16 rows")?;
        ensure!(
            st.persistent_verify_enabled(num_tokens)
                && top_k == EXL3_PERSISTENT_TOP_K
                && st.persistent_build_k[arm].0 != 0
                && st.persistent_gate_up_k[arm].0 != 0
                && st.persistent_down_k[arm].0 != 0
                && st.persistent_worklist.0 != 0
                && st.persistent_state.0 != 0,
            "EXL3 persistent verify called without an eligible initialized M6/M16 worklist"
        );
        KernelLaunch::new(gpu, st.persistent_build_k[arm])
            .grid([1, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(indices_dev)
            .arg_ptr(st.persistent_worklist)
            .arg_ptr(st.persistent_state)
            .arg_u32(num_tokens)
            .arg_u32(top_k)
            .arg_u32(st.num_experts)
            .arg_u32(EXL3_PERSISTENT_WORK_CAPACITIES[arm])
            .launch(stream)?;
        KernelLaunch::new(gpu, st.persistent_gate_up_k[arm])
            .grid([EXL3_PERSISTENT_CTAS, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(expert_input)
            .arg_ptr(st.gate.trellis_tab)
            .arg_ptr(st.gate.suh_tab)
            .arg_ptr(st.gate.svh_tab)
            .arg_ptr(st.up.trellis_tab)
            .arg_ptr(st.up.suh_tab)
            .arg_ptr(st.up.svh_tab)
            .arg_ptr(st.persistent_worklist)
            .arg_ptr(st.persistent_state)
            .arg_ptr(expert_gate_out)
            .arg_ptr(expert_up_out)
            .arg_ptr(st.ws)
            .arg_ptr(st.counters)
            .arg_u32(inter)
            .arg_u32(h)
            .arg_u32(top_k)
            .arg_u32(split_gu)
            .arg_u32(st.gate.bits)
            .launch(stream)?;
        ops::moe_silu_mul(
            gpu,
            st.silu_mul_clamped_k,
            expert_gate_out,
            expert_up_out,
            expert_gate_out,
            num_tokens * top_k * inter,
            stream,
        )?;
        KernelLaunch::new(gpu, st.persistent_down_k[arm])
            .grid([EXL3_PERSISTENT_CTAS, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(expert_gate_out)
            .arg_ptr(st.down.trellis_tab)
            .arg_ptr(st.down.suh_tab)
            .arg_ptr(st.down.svh_tab)
            .arg_ptr(st.persistent_worklist)
            .arg_ptr(st.persistent_state)
            .arg_ptr(expert_down_out)
            .arg_ptr(st.ws)
            .arg_ptr(st.counters)
            .arg_u32(h)
            .arg_u32(inter)
            .arg_u32(top_k)
            .arg_u32(split_dn)
            .arg_u32(st.down.bits)
            .launch(stream)
    }

    /// `num_tokens`-row speculative-verify expert FFN over EXL3 routed experts
    /// + NVFP4 or native FP8 shared expert, through the dedup'd m-row kernels.
    ///
    /// Output layout is EXACTLY what `dispatch_splitk_m_t` leaves behind —
    /// routed slots flat in `expert_{gate,up,down}_out`, one shared row per
    /// token in the shared scratch — so `moe_weighted_sum_blend_batchn`
    /// downstream is untouched.
    ///
    /// ## The exact-GEMV law
    ///
    /// Every row's routed output is BIT-IDENTICAL to what
    /// [`Self::dispatch_exl3_decode`] computes for that token, because:
    ///   * the m-row kernels run the same per-row op sequence as
    ///     `exl3_gemv_m1_body` (see the kernel header's structural argument),
    ///   * this dispatch passes the SAME `split_for(N)` the M=1 path passes, so
    ///     the K-slice per split is identical,
    ///   * the SwiGLU is the same elementwise `moe_silu_mul` kernel over a
    ///     wider flat extent, and
    ///   * with `ATLAS_VERIFY_GEMV_V2=1`, the shared expert uses the grouped
    ///     batch kernel whose per-row FMA and reduction order is copied from
    ///     `w4a16_gemv`; otherwise it uses that single-row kernel directly.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn dispatch_exl3_verify(
        &self,
        ctx: &ForwardContext,
        expert_input: DevicePtr, // [num_tokens, h] BF16
        expert_gate_out: DevicePtr,
        expert_up_out: DevicePtr,
        expert_down_out: DevicePtr,
        shared_gate_scratch: DevicePtr,
        shared_up_scratch: DevicePtr,
        shared_out: DevicePtr,
        indices_dev: DevicePtr, // [num_tokens*top_k] u32
        h: u32,
        inter: u32,
        top_k: u32,
        num_tokens: u32,
        stream: u64,
    ) -> Result<bool> {
        let st = self
            .exl3
            .as_ref()
            .expect("dispatch_exl3_verify without state");
        if !self.exl3_verify_is_batched(ctx, num_tokens) {
            return Ok(false);
        }
        ensure!(
            st.gate.n == inter && st.gate.k == h && st.down.n == h && st.down.k == inter,
            "EXL3 dims mismatch: gate [{}x{}] down [{}x{}] vs h={h} inter={inter}",
            st.gate.n,
            st.gate.k,
            st.down.n,
            st.down.k
        );
        let (arm, _mrow) = st.mrow_arm(num_tokens).expect("checked above");
        let gpu = ctx.gpu;
        // SAME splits as the M=1 decode path — this is half the bit-identity
        // argument, so it must not be re-tuned independently.
        // Pass the M=1 FUSED group counts (2*top_k for gate+up, top_k for
        // down), NOT the m-row ones: split_for's wave-walk is a function of
        // `groups`, so feeding the wider m-row count could round the split to
        // a different value and silently break per-row bit-identity with the
        // M=1 path — which is the property GATE9 exists to prove.
        let split_gu = st.split_for(inter, 2 * top_k as u32);
        let split_dn = st.split_for(h, top_k as u32);
        let total_routed = num_tokens * top_k;
        // `set_exl3_experts` sized ws/counters for `2·max_rows·top_k` groups;
        // re-check before any launch can walk off the end.
        ensure!(
            2 * total_routed <= st.groups,
            "EXL3 verify scratch sized for {} launch groups, need {} \
             (num_tokens={num_tokens} top_k={top_k})",
            st.groups,
            2 * total_routed
        );

        if st.persistent_verify_enabled(num_tokens) {
            self.dispatch_exl3_persistent_verify(
                gpu,
                st,
                expert_input,
                expert_gate_out,
                expert_up_out,
                expert_down_out,
                indices_dev,
                h,
                inter,
                top_k,
                num_tokens,
                split_gu,
                split_dn,
                stream,
            )?;
        } else {
            // (1) gate+up over every routed slot of every row: grid.z = 2·slots,
            //     z = 2·slot + proj. Duplicate expert ids exit at the leader
            //     election, so each distinct expert's trellis is streamed ONCE for
            //     the whole verify block.
            KernelLaunch::new(gpu, st.mrow_gate_up_k[arm])
                .grid([inter / 128, split_gu, 2 * total_routed])
                .block([256, 1, 1])
                .arg_ptr(expert_input)
                .arg_ptr(st.gate.trellis_tab)
                .arg_ptr(st.gate.suh_tab)
                .arg_ptr(st.gate.svh_tab)
                .arg_ptr(st.up.trellis_tab)
                .arg_ptr(st.up.suh_tab)
                .arg_ptr(st.up.svh_tab)
                .arg_ptr(indices_dev)
                .arg_ptr(expert_gate_out)
                .arg_ptr(expert_up_out)
                .arg_ptr(st.ws)
                .arg_ptr(st.counters)
                .arg_u32(inter)
                .arg_u32(h)
                .arg_u32(top_k)
                .arg_u32(num_tokens)
                .arg_u32(st.gate.bits)
                .launch(stream)?;
            // (2) ONE flat clamped SwiGLU over every slot of every row — same
            //     kernel, same same-index in-place map, just a wider extent.
            ops::moe_silu_mul(
                gpu,
                st.silu_mul_clamped_k,
                expert_gate_out,
                expert_up_out,
                expert_gate_out,
                total_routed * inter,
                stream,
            )?;
            // (3) down over every routed slot: grid.z = slots, A row = act + slot·inter.
            KernelLaunch::new(gpu, st.mrow_down_k[arm])
                .grid([h / 128, split_dn, total_routed])
                .block([256, 1, 1])
                .arg_ptr(expert_gate_out)
                .arg_ptr(st.down.trellis_tab)
                .arg_ptr(st.down.suh_tab)
                .arg_ptr(st.down.svh_tab)
                .arg_ptr(indices_dev)
                .arg_ptr(expert_down_out)
                .arg_ptr(st.ws)
                .arg_ptr(st.counters)
                .arg_u32(h)
                .arg_u32(inter)
                .arg_u32(top_k)
                .arg_u32(num_tokens)
                .arg_u32(st.down.bits)
                .launch(stream)?;
        }

        // Actual Vision keeps checkpoint-native FP8 shared weights and clamp10.
        // Reuse serial GEMVs per row before entering the legacy NVFP4 chain.
        if self.run_native_fp8_shared_verify(
            expert_input,
            num_tokens,
            h,
            ctx.config.shared_expert_intermediate_size as u32,
            shared_gate_scratch,
            shared_up_scratch,
            shared_out,
            ctx,
            stream,
        )? {
            return Ok(true);
        }

        // ── Shared expert. The grouped-batch kernel preserves the exact
        //    single-row K order while streaming each NVFP4 weight once. ──
        self.shared_experts_scale_kind.expect(
            crate::weight_map::WeightQuantFormat::Nvfp4,
            "EXL3 verify arm computes the shared expert via w4a16_gemv (NVFP4)",
        );
        let shared_inter = ctx.config.shared_expert_intermediate_size as u32;
        let sh = &self.weights.shared_expert;
        let exact_batch = exl3_shared_batch_enabled();
        let v2_kernel = if h % 32 == 0 && shared_inter % 32 == 0 {
            match num_tokens {
                4 => self.w4a16_gemv_grouped_batchm_v2_k[0],
                5 => self.w4a16_gemv_grouped_batchm_v2_k[1],
                6 => self.w4a16_gemv_grouped_batchm_v2_k[2],
                8 => self.w4a16_gemv_grouped_batchm_v2_k[3],
                16 => self.w4a16_gemv_grouped_batchm_v2_k[4],
                _ => KernelHandle(0),
            }
        } else {
            KernelHandle(0)
        };
        let batch_kernel = if exact_batch && v2_kernel.0 != 0 {
            Some((v2_kernel, true))
        } else if exact_batch && num_tokens <= 8 {
            Some((self.w4a16_gemv_grouped_batchm_k, false))
        } else {
            None
        }
        .filter(|(kernel, _)| kernel.0 != 0);
        if let Some((kernel, v2)) = batch_kernel {
            let run = |input: DevicePtr,
                       weight: &crate::weight_map::QuantizedWeight,
                       output: DevicePtr,
                       n: u32,
                       k: u32,
                       lda: u32,
                       ldc: u32|
             -> Result<()> {
                if v2 {
                    ops::w4a16_gemv_grouped_batchm_v2(
                        gpu, kernel, input, weight, output, num_tokens, n, k, lda, ldc, n, stream,
                    )
                } else {
                    ops::w4a16_gemv_grouped_batchm(
                        gpu, kernel, input, weight, output, num_tokens, n, k, lda, ldc, n, stream,
                    )
                }
            };
            run(
                expert_input,
                &sh.gate_proj,
                shared_gate_scratch,
                shared_inter,
                h,
                h,
                shared_inter,
            )?;
            run(
                expert_input,
                &sh.up_proj,
                shared_up_scratch,
                shared_inter,
                h,
                h,
                shared_inter,
            )?;
            ops::moe_silu_mul(
                gpu,
                st.silu_mul_noclamp_k,
                shared_gate_scratch,
                shared_up_scratch,
                shared_gate_scratch,
                num_tokens * shared_inter,
                stream,
            )?;
            run(
                shared_gate_scratch,
                &sh.down_proj,
                shared_out,
                h,
                shared_inter,
                shared_inter,
                h,
            )?;
        } else {
            for t in 0..num_tokens as usize {
                let a_row = expert_input.offset(t * h as usize * 2);
                let g_row = shared_gate_scratch.offset(t * shared_inter as usize * 2);
                let u_row = shared_up_scratch.offset(t * shared_inter as usize * 2);
                let o_row = shared_out.offset(t * h as usize * 2);
                ops::w4a16_gemv(
                    gpu,
                    self.w4a16_gemv,
                    a_row,
                    &sh.gate_proj,
                    g_row,
                    shared_inter,
                    h,
                    stream,
                )?;
                ops::w4a16_gemv(
                    gpu,
                    self.w4a16_gemv,
                    a_row,
                    &sh.up_proj,
                    u_row,
                    shared_inter,
                    h,
                    stream,
                )?;
                ops::moe_silu_mul(
                    gpu,
                    st.silu_mul_noclamp_k,
                    g_row,
                    u_row,
                    g_row,
                    shared_inter,
                    stream,
                )?;
                ops::w4a16_gemv(
                    gpu,
                    self.w4a16_gemv,
                    g_row,
                    &sh.down_proj,
                    o_row,
                    h,
                    shared_inter,
                    stream,
                )?;
            }
        }
        Ok(true)
    }
}
