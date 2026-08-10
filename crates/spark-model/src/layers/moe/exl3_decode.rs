// SPDX-License-Identifier: AGPL-3.0-only

//! EXL3 trellis (3.0 bpw) routed-expert DECODE dispatch (M=1).
//!
//! Bring-up arm for the reference tp1 checkpoint (`quant_method: "exl3"`,
//! 216 routed experts/layer): per routed slot, three `exl3_gemv_m1_idx`
//! launches (gate/up/down) + the clamped SwiGLU elementwise kernel; the
//! FP8→NVFP4 shared expert rides on the existing `w4a16_gemv` path. The
//! blend downstream (`moe_weighted_sum_blend`) is unchanged — outputs land
//! in exactly the layout the fused NVFP4 decode kernels produce.
//!
//! Launch sequence is identical every step (the expert id is read ON DEVICE
//! from the routed `indices` buffer by `exl3_gemv_m1_idx`), so the arm is
//! CUDA-graph-safe and needs no D2H of the routing.
//!
//! Known bring-up costs vs the end-state kernel (plan §3 / exl3-gemv.md §6):
//!   - 3·top_k + 3 GEMV launches + 7 elementwise/blend per layer instead of
//!     2 fused kernels — launch overhead only, the weight stream is optimal
//!     (each expert matrix read exactly once, 3.012 bpw).
//!   - routed slots are serialized on the stream; the split-K grid fills the
//!     48 SMs per launch (split chosen to land ~96 CTAs), so the serialization
//!     costs latency, not bandwidth.
//!   - NOT yet wired: prefill / wide-verify (M>1) — those paths fail loudly
//!     via the `Exl3Trellis` format tag (see plan §3 "P1 scratch dequant").

use anyhow::{Context, Result, ensure};
use spark_runtime::kernel_args::KernelLaunch;

use super::*;
use crate::weight_map::Exl3ExpertWeight;

/// Widest split the scratch is sized for (matches the microtest sweep).
const EXL3_MAX_SPLIT: u32 = 12;
/// Largest N across the expert shapes (w2: N = hidden = 4096).
const EXL3_MAX_N: u32 = 4096;

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
    gemv_idx_k: KernelHandle,
    silu_mul_clamped_k: KernelHandle, // routed: DeepSeek-V4 swiglu_limit=10
    silu_mul_noclamp_k: KernelHandle, // shared expert: unclamped
    /// f32 `[EXL3_MAX_SPLIT, EXL3_MAX_N]` split-K partial scratch.
    ws: DevicePtr,
    /// i32 `[EXL3_MAX_N/128]` split-election counters (self-resetting).
    counters: DevicePtr,
    /// Split override from `ATLAS_EXL3_SPLIT` (0 = auto).
    split_override: u32,
}

impl Exl3MoeState {
    /// SPLIT_K policy: fill ~2 CTAs/SM (96 slots on GB10's 48 SMs) — the
    /// microtest measured splits 1-3 underfilled at N=2048 (16 strips).
    fn split_for(&self, n: u32) -> u32 {
        if self.split_override > 0 {
            return self.split_override.min(EXL3_MAX_SPLIT);
        }
        let strips = (n / 128).max(1);
        (96 / strips).clamp(1, EXL3_MAX_SPLIT)
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
        let ws = gpu.alloc((EXL3_MAX_SPLIT as usize) * (EXL3_MAX_N as usize) * 4)?;
        let counters = gpu.alloc((EXL3_MAX_N as usize / 128) * 4)?;
        // Counters must be zero before the FIRST launch; the kernel re-arms
        // them to zero on completion, so this is the only memset ever needed.
        gpu.memset(counters, 0, EXL3_MAX_N as usize / 128 * 4)?;
        let split_override = std::env::var("ATLAS_EXL3_SPLIT")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(0);
        self.exl3 = Some(Exl3MoeState {
            gate,
            up,
            down,
            gemv_idx_k: gpu
                .kernel("exl3_gemv", "exl3_gemv_m1_idx")
                .context("EXL3 checkpoint needs the exl3_gemv kernel module")?,
            silu_mul_clamped_k: gpu.kernel("moe_silu_mul", "moe_silu_mul")?,
            silu_mul_noclamp_k: gpu.kernel("moe_silu_mul", "silu_mul_noclamp")?,
            ws,
            counters,
            split_override,
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
        let gpu = ctx.gpu;
        let split_gu = st.split_for(inter);
        let split_dn = st.split_for(h);

        // ── Routed slots: gate → up → swiglu(clamped) → down, per slot. ──
        // Same-stream launches serialize, so one ws/counters scratch is safe.
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
            ops::moe_silu_mul(gpu, st.silu_mul_clamped_k, gate_row, up_row, gate_row, inter, stream)?;
            launch_gemv_idx(
                gpu, st.gemv_idx_k, gate_row, &st.down, indices_dev, slot, down_row,
                st.ws, st.counters, split_dn, stream,
            )?;
        }

        // ── Shared expert: FP8-block-scaled on disk → NVFP4 at load (the
        // heterogeneous-checkpoint arm of `load_expert_proj`). UNCLAMPED
        // SwiGLU (DeepseekV4MLP — see moe_shared_expert_fused.cu). ──
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
