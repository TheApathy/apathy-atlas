// SPDX-License-Identifier: AGPL-3.0-only

//! EXL3 trellis (3.0 bpw) routed-expert DECODE dispatch (M=1).
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
//! NOT yet wired: prefill / wide-verify (M>1) go through
//! `forward_prefill_exl3.rs`; every legacy NVFP4/E8M0 decode site fails loudly
//! via the `Exl3Trellis` format tag.

use anyhow::{Context, Result, ensure};
use spark_runtime::kernel_args::KernelLaunch;

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
}

/// EXL3 decode state hung off `MoeLayer` (None on non-EXL3 checkpoints).
pub(crate) struct Exl3MoeState {
    pub(crate) gate: Exl3ProjTable, // checkpoint w1: [h -> inter]
    pub(crate) up: Exl3ProjTable,   // checkpoint w3: [h -> inter]
    pub(crate) down: Exl3ProjTable, // checkpoint w2: [inter -> h]
    gemv_idx_k: KernelHandle,           // bring-up: one launch per (slot, proj)
    gemv_fused_gate_up_k: KernelHandle, // fused: all slots × {gate, up}
    gemv_fused_down_k: KernelHandle,    // fused: all slots
    silu_mul_clamped_k: KernelHandle,   // routed: DeepSeek-V4 swiglu_limit=10
    silu_mul_noclamp_k: KernelHandle,   // shared expert: unclamped
    /// f32 `[groups, EXL3_MAX_SPLIT, EXL3_MAX_N]` split-K partial scratch —
    /// one private region per fused launch group (see module docs). The
    /// kernels address it as `ws + group·gridDim.y·N`, which is inside this
    /// allocation for every `(groups ≤ self.groups, split ≤ EXL3_MAX_SPLIT,
    /// N ≤ EXL3_MAX_N)` launch.
    ws: DevicePtr,
    /// i32 `[groups, EXL3_MAX_N/128]` split-election counters (self-resetting;
    /// addressed as `counters + group·N/128`).
    counters: DevicePtr,
    /// Widest launch-group count the scratch is sized for: `2·top_k` (gate+up).
    groups: u32,
    /// Split override from `ATLAS_EXL3_SPLIT` (0 = auto).
    split_override: u32,
    /// `ATLAS_EXL3_FUSED=0` falls back to the per-slot bring-up chain (A/B;
    /// bit-identical, ~5x the launches).
    fused: bool,
    /// P1 prefill state (scratch dequant + H128 activation passes).
    pub(crate) prefill: Exl3PrefillState,
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
    /// BF16 `[chunk, n·k]` slot-major dequant scratch.
    pub(crate) scratch: DevicePtr,
    /// `[chunk]` u64 device table → the scratch slots (static across chunks).
    pub(crate) slot_tab: DevicePtr,
    /// Experts dequanted per chunk (`ATLAS_EXL3_PREFILL_CHUNK`, default 8
    /// → 8 × 16.8 MB = 134 MB scratch at the V4 expert shapes).
    pub(crate) chunk: u32,
    pub(crate) dequant_chunk_k: KernelHandle,
    pub(crate) h128_pre_k: KernelHandle,
    pub(crate) h128_post_k: KernelHandle,
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
            w.n == n && w.k == k,
            "EXL3 expert {i}: shape [{}, {}] != expert 0 [{n}, {k}]",
            w.n,
            w.k
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
    })
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
            gate.n <= EXL3_MAX_N && down.n <= EXL3_MAX_N,
            "EXL3 scratch sized for N <= {EXL3_MAX_N}, got gate N={} down N={}",
            gate.n,
            down.n
        );
        ensure!(
            top_k > 0 && top_k <= EXL3_MAX_TOP_K,
            "EXL3 decode scratch sized for top_k in 1..={EXL3_MAX_TOP_K}, got {top_k}"
        );
        // Widest fused launch: gate+up = 2·top_k groups (down uses top_k).
        // Every group needs its own [SPLIT_K, N] fp32 partial region and its
        // own [N/128] election counters, since fused groups run concurrently.
        let groups = 2 * top_k;
        let ws_floats =
            groups as usize * EXL3_MAX_SPLIT as usize * EXL3_MAX_N as usize;
        let counter_bytes = groups as usize * (EXL3_MAX_N as usize / 128) * 4;
        let ws = gpu.alloc(ws_floats * 4)?;
        let counters = gpu.alloc(counter_bytes)?;
        // Counters must be zero before the FIRST launch; the kernel re-arms
        // them to zero on completion, so this is the only memset ever needed.
        // (Groups never touch each other's counters, so the invariant "all
        // zero at rest" survives fusion and CUDA-graph replay unchanged.)
        gpu.memset(counters, 0, counter_bytes)?;
        let split_override = std::env::var("ATLAS_EXL3_SPLIT")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(0);

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
        let chunk = std::env::var("ATLAS_EXL3_PREFILL_CHUNK")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .filter(|&c| c > 0)
            .unwrap_or(8)
            .min(experts.len() as u32);
        let slot_bytes = gate.n as usize * gate.k as usize * 2;
        let scratch = gpu.alloc(chunk as usize * slot_bytes)?;
        let slot_ptr_bytes: Vec<u8> = (0..chunk as usize)
            .flat_map(|z| (scratch.0 + (z * slot_bytes) as u64).to_le_bytes())
            .collect();
        let slot_tab = gpu.alloc(slot_ptr_bytes.len())?;
        gpu.copy_h2d(&slot_ptr_bytes, slot_tab)?;
        let prefill = Exl3PrefillState {
            scratch,
            slot_tab,
            chunk,
            dequant_chunk_k: gpu.kernel("exl3_gemv", "exl3_dequant_chunk_bf16")?,
            h128_pre_k: gpu.kernel("exl3_gemv", "exl3_h128_pre_rows")?,
            h128_post_k: gpu.kernel("exl3_gemv", "exl3_h128_post_rows")?,
        };

        self.exl3 = Some(Exl3MoeState {
            gate,
            up,
            down,
            gemv_idx_k: gpu
                .kernel("exl3_gemv", "exl3_gemv_m1_idx")
                .context("EXL3 checkpoint needs the exl3_gemv kernel module")?,
            gemv_fused_gate_up_k: gpu.kernel("exl3_gemv", "exl3_gemv_m1_fused_gate_up")?,
            gemv_fused_down_k: gpu.kernel("exl3_gemv", "exl3_gemv_m1_fused_down")?,
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
        let st = self.exl3.as_ref().expect("dispatch_exl3_decode without state");
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
                    gpu, st.gemv_idx_k, expert_input, &st.gate, indices_dev, slot, gate_row,
                    st.ws, st.counters, split_gu, stream,
                )?;
                launch_gemv_idx(
                    gpu, st.gemv_idx_k, expert_input, &st.up, indices_dev, slot, up_row,
                    st.ws, st.counters, split_gu, stream,
                )?;
                // act = clamped_swiglu(gate, up), in place over the gate row
                // (elementwise same-index read/write — safe).
                ops::moe_silu_mul(
                    gpu, st.silu_mul_clamped_k, gate_row, up_row, gate_row, inter, stream,
                )?;
                launch_gemv_idx(
                    gpu, st.gemv_idx_k, gate_row, &st.down, indices_dev, slot, down_row,
                    st.ws, st.counters, split_dn, stream,
                )?;
            }
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
}
