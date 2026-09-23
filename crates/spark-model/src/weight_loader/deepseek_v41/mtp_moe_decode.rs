// SPDX-License-Identifier: AGPL-3.0-only

//! The DSpark (MTP block) routed MoE for the draft block: 128 FP4 experts, top-3, T <= 8 rows
//! (`kernels/gb10/deepseek-v4.1/cb3/mtp_moe_decode.cu`). Five launches, no host sync:
//!
//! ```text
//! router logits (bf16 weight, fp32 GEMV) -> route (top-3 of 128, group by expert)
//!   -> gate+up+SwiGLU (h, bf16) -> down (fp32 per pick) -> sum over the 3 picks (bf16)
//! ```
//!
//! Experts are addressed through a per-block device table of the store-resident tensors (`ptrs`
//! u64 [128][6] = w1, s1, w3, s3, w2, s2), so nothing is copied into an arena. Implements
//! [`V41RoutedMoe`] for layer ids 40 + k (the draft block k), which is how `fwd::block` names them.

use anyhow::{Context, Result, ensure};

use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::device_allocs::{DeviceAllocs, SharedGpu};
use super::fwd::V41RoutedMoe;
use super::mtp::{DsparkWeights, Fp4Linear};
use super::ops::Ops;

pub const MTP_MODULE: &str = "mtp_moe_decode";
pub const MTP_EXPERTS: usize = 128;
pub const MTP_TOP_K: usize = 3;
pub const MTP_MAX_T: usize = 8;
/// Layer id of draft block 0 (`fwd::block` is called with 40 + k).
pub const MTP_LAYER0: usize = 40;
const WARPS: u32 = 8;
const GROUP_INTS: usize = 2 + MTP_MAX_T;
const MAX_ROWS: usize = MTP_MAX_T * MTP_TOP_K;
const HIDDEN: usize = 5120;
const INTER: usize = 2304;

struct Kernels {
    router: KernelHandle,
    route: KernelHandle,
    gateup_r1: KernelHandle,
    gateup_r8: KernelHandle,
    down_r1: KernelHandle,
    down_r8: KernelHandle,
    sum: KernelHandle,
}

struct BlockTables {
    /// u64 [128][6] expert tensor addresses.
    ptrs: DevicePtr,
    gate: DevicePtr,
    bias: DevicePtr,
}

pub struct MtpMoeDecode {
    gpu: SharedGpu,
    k: Kernels,
    blocks: Vec<BlockTables>,
    logits: DevicePtr,
    groups: DevicePtr,
    row_w: DevicePtr,
    sel: DevicePtr,
    h: DevicePtr,
    down: DevicePtr,
    limit: f32,
    route_scale: f32,
    _allocs: DeviceAllocs,
}

fn check(w: &Fp4Linear, n: usize, k: usize, what: &str) -> Result<()> {
    ensure!(w.n == n && w.k == k, "DSpark expert {what} is [{}, {}], expected [{n}, {k}]", w.n, w.k);
    Ok(())
}

impl MtpMoeDecode {
    pub fn new(shared: &SharedGpu, w: &DsparkWeights, limit: f32, route_scale: f32) -> Result<Self> {
        let gpu: &dyn GpuBackend = shared.as_ref();
        let k = |name: &str| {
            gpu.kernel(MTP_MODULE, name)
                .with_context(|| format!("{MTP_MODULE}::{name} is not in the compiled PTX (new .cu: touch crates/atlas-kernels/build.rs)"))
        };
        let k = Kernels {
            router: k("dsv41_mtp_router_logits")?,
            route: k("dsv41_mtp_route")?,
            gateup_r1: k("dsv41_mtp_gateup_r1")?,
            gateup_r8: k("dsv41_mtp_gateup_r8")?,
            down_r1: k("dsv41_mtp_down_r1")?,
            down_r8: k("dsv41_mtp_down_r8")?,
            sum: k("dsv41_mtp_sum")?,
        };
        let mut allocs = DeviceAllocs::owned(shared.clone());
        let mut blocks = Vec::with_capacity(w.blocks.len());
        for b in &w.blocks {
            ensure!(b.experts.len() == MTP_EXPERTS, "DSpark block {}: {} experts", b.k, b.experts.len());
            let mut table: Vec<u64> = Vec::with_capacity(MTP_EXPERTS * 6);
            for e in &b.experts {
                check(&e.w1, INTER, HIDDEN, "w1")?;
                check(&e.w3, INTER, HIDDEN, "w3")?;
                check(&e.w2, HIDDEN, INTER, "w2")?;
                for p in [e.w1.weight, e.w1.scale, e.w3.weight, e.w3.scale, e.w2.weight, e.w2.scale] {
                    // 16-byte row loads: every row start must stay aligned.
                    ensure!(p.0 % 16 == 0, "DSpark expert tensor at {:#x} is not 16-byte aligned", p.0);
                    table.push(p.0);
                }
            }
            let bytes: Vec<u8> = table.iter().flat_map(|v| v.to_le_bytes()).collect();
            let ptrs = allocs.alloc(gpu, bytes.len())?;
            gpu.copy_h2d(&bytes, ptrs)?;
            blocks.push(BlockTables { ptrs, gate: b.router.weight, bias: b.router.bias });
        }
        let mut a = |bytes: usize| -> Result<DevicePtr> {
            let p = allocs.alloc(gpu, bytes.max(256))?;
            gpu.memset(p, 0, bytes.max(256))?;
            Ok(p)
        };
        Ok(Self {
            gpu: shared.clone(),
            k,
            blocks,
            logits: a(MTP_MAX_T * MTP_EXPERTS * 4)?,
            groups: a((1 + MAX_ROWS * GROUP_INTS) * 4)?,
            row_w: a(MAX_ROWS * 4)?,
            sel: a(MAX_ROWS * 4)?,
            h: a(MAX_ROWS * INTER * 2)?,
            down: a(MAX_ROWS * HIDDEN * 4)?,
            limit,
            route_scale,
            _allocs: allocs,
        })
    }

    /// Draft block `blk`: `y [t, 5120] -> out [t, 5120]` (routed experts only; the shared expert
    /// is `SharedExpert::forward`, as for the main layers).
    pub fn forward_block(&self, blk: usize, y: DevicePtr, out: DevicePtr, t: usize, stream: u64) -> Result<()> {
        ensure!(t >= 1 && t <= MTP_MAX_T, "DSpark MoE: t = {t} (max {MTP_MAX_T})");
        let b = self.blocks.get(blk).with_context(|| format!("DSpark MoE: no block {blk}"))?;
        let gpu = self.gpu.as_ref();
        KernelLaunch::new(gpu, self.k.router)
            .grid([(MTP_EXPERTS as u32).div_ceil(WARPS), t as u32, 1])
            .block([32 * WARPS, 1, 1])
            .arg_ptr(y).arg_ptr(b.gate).arg_ptr(self.logits)
            .launch(stream)?;
        KernelLaunch::new(gpu, self.k.route)
            .grid([1, 1, 1])
            .block([32 * t as u32, 1, 1])
            .arg_ptr(self.logits).arg_ptr(b.bias).arg_ptr(self.groups).arg_ptr(self.row_w).arg_ptr(self.sel)
            .arg_i32(t as i32).arg_f32(self.route_scale)
            .launch(stream)?;
        let max_groups = (t * MTP_TOP_K) as u32;
        let (gateup, down) = if t == 1 { (self.k.gateup_r1, self.k.down_r1) } else { (self.k.gateup_r8, self.k.down_r8) };
        KernelLaunch::new(gpu, gateup)
            .grid([(INTER as u32).div_ceil(WARPS), max_groups, 1])
            .block([32 * WARPS, 1, 1])
            .arg_ptr(b.ptrs).arg_ptr(y).arg_ptr(self.groups).arg_ptr(self.row_w).arg_ptr(self.h)
            .arg_f32(self.limit)
            .launch(stream)?;
        KernelLaunch::new(gpu, down)
            .grid([(HIDDEN as u32).div_ceil(WARPS), max_groups, 1])
            .block([32 * WARPS, 1, 1])
            .arg_ptr(b.ptrs).arg_ptr(self.h).arg_ptr(self.groups).arg_ptr(self.down)
            .launch(stream)?;
        KernelLaunch::new(gpu, self.k.sum)
            .grid([(HIDDEN as u32).div_ceil(256), t as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(self.down).arg_ptr(out)
            .launch(stream)
    }

    /// The last call's routing (ids [t, 3], weights [t, 3]); synchronous, for gates only.
    pub fn read_routing(&self, t: usize, stream: u64) -> Result<(Vec<i32>, Vec<f32>)> {
        let gpu = self.gpu.as_ref();
        gpu.synchronize(stream)?;
        let mut s = vec![0u8; t * MTP_TOP_K * 4];
        let mut w = vec![0u8; t * MTP_TOP_K * 4];
        gpu.copy_d2h(self.sel, &mut s)?;
        gpu.copy_d2h(self.row_w, &mut w)?;
        Ok((
            s.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            w.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        ))
    }
}

impl V41RoutedMoe for MtpMoeDecode {
    fn forward(&self, ops: &Ops, layer: usize, y: DevicePtr, out: DevicePtr, t: usize) -> Result<()> {
        ensure!(layer >= MTP_LAYER0, "DSpark MoE called for main layer {layer}");
        self.forward_block(layer - MTP_LAYER0, y, out, t, ops.stream)
    }
}
