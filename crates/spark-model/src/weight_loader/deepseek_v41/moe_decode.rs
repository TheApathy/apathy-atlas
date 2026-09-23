// SPDX-License-Identifier: AGPL-3.0-only

//! The routed MoE at **decode size** (T <= [`MAX_DECODE_T`] tokens): GPU routing and a CB3
//! grouped GEMV (`kernels/gb10/deepseek-v4.1/cb3/cb3_moe_decode.cu`) that decodes the 3-bit
//! codes in registers. No host sync, no bf16 expert copy, five launches per layer:
//!
//! ```text
//! router logits (fp32 GEMV) -> route (top-6, mask, text/vision bias, group by slot)
//!   -> gate+up+SwiGLU (h, bf16) -> down (fp32 per pick) -> sum over picks (bf16)
//! ```
//!
//! Same rounding points as `Cb3RoutedMoe`'s prefill path (h once, the pick sum once); only the
//! fp32 accumulation order differs. See the kernel file for the argument.
//!
//! Graph-capturable: every launch reads its extents from device memory or from `t`, and the
//! pass's token ids are uploaded once per pass by [`MoeDecode::set_ids`], outside the layers.

use anyhow::{Context, Result, ensure};

use atlas_core::config::ROUTED_EXPERTS;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::cb3_arena::{Cb3ExpertArena, Cb3LayerResidency};
use super::device_allocs::{DeviceAllocs, SharedGpu};
use super::moe::Cb3Matrix;
use super::moe_forward::{RouterF32, TOP_K};
use super::routing::Routing;

/// Module compiled from `cb3_moe_decode.cu` (file stem).
pub const DECODE_MODULE: &str = "cb3_moe_decode";
/// Largest pass the decode path serves (the kernels' MAXR / MAXT).
pub const MAX_DECODE_T: usize = 8;
/// ON by default (gated end to end at keep=124: 48/48 greedy tokens identical to the prefill
/// path's base arm). `ATLAS_DSV41_MOE_DECODE=0` turns it off.
pub const MOE_DECODE_ENV: &str = "ATLAS_DSV41_MOE_DECODE";
const WARPS: u32 = 8;
const GROUP_INTS: usize = 2 + MAX_DECODE_T;
const MAX_ROWS: usize = MAX_DECODE_T * TOP_K;
const HIDDEN: usize = 5120;
const INTER: usize = 2304;

pub fn enabled() -> bool {
    std::env::var(MOE_DECODE_ENV).as_deref() != Ok("0")
}

#[derive(Clone, Copy)]
struct Kernels {
    router: KernelHandle,
    router_bf16w: KernelHandle,
    route: KernelHandle,
    /// Row-count variants R = 1, 2, 4, 6, 8 (a group holds at most `t` rows, since a token's
    /// top-6 experts are distinct): the smallest R >= t runs. Same per-row arithmetic for every R,
    /// fewer registers for small R (down: 73 regs at R=8 vs 64 at R=6 -> 24 vs 32 warps/SM).
    gateup: [KernelHandle; 5],
    down: [KernelHandle; 5],
    sum: KernelHandle,
}

/// Per-layer routing tables on the device.
struct LayerTables {
    layer: usize,
    bias: DevicePtr,
    bias_vl: DevicePtr,
    /// u8 [384] residency mask.
    mask: DevicePtr,
    /// i32 [384] routed id -> slot, -1.
    slot_of: DevicePtr,
}

pub struct MoeDecode {
    k: Kernels,
    tables: Vec<LayerTables>,
    ids: DevicePtr,
    logits: DevicePtr,
    groups: DevicePtr,
    row_w: DevicePtr,
    sel: DevicePtr,
    err: DevicePtr,
    h: DevicePtr,
    down: DevicePtr,
    /// Every device buffer above, freed when this drops (engine2's owned-allocation pattern).
    _allocs: DeviceAllocs,
}

fn bytes_of<T: Copy>(v: &[T]) -> &[u8] {
    // SAFETY: plain-old-data slices (u8/i32/f32/u32), read-only, lifetime tied to v.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

impl MoeDecode {
    pub fn new(shared: &SharedGpu, arena: &Cb3ExpertArena, routers: &[(usize, RouterF32)]) -> Result<Self> {
        let gpu: &dyn GpuBackend = shared.as_ref();
        // Recorded as they are made, so a failure half-way frees what was already taken.
        let allocs = std::cell::RefCell::new(DeviceAllocs::owned(shared.clone()));
        let k = |name: &str| {
            gpu.kernel(DECODE_MODULE, name).with_context(|| {
                format!("{DECODE_MODULE}::{name} is not in the compiled PTX (new .cu: touch crates/atlas-kernels/build.rs)")
            })
        };
        let k = Kernels {
            router: k("dsv41_router_logits_decode")?,
            router_bf16w: k("dsv41_router_logits_decode_bf16w")?,
            route: k("dsv41_route_decode")?,
            gateup: [
                k("dsv41_cb3_gateup_decode_r1")?,
                k("dsv41_cb3_gateup_decode_r2")?,
                k("dsv41_cb3_gateup_decode_r4")?,
                k("dsv41_cb3_gateup_decode_r6")?,
                k("dsv41_cb3_gateup_decode_r8")?,
            ],
            down: [
                k("dsv41_cb3_down_decode_r1")?,
                k("dsv41_cb3_down_decode_r2")?,
                k("dsv41_cb3_down_decode_r4")?,
                k("dsv41_cb3_down_decode_r6")?,
                k("dsv41_cb3_down_decode_r8")?,
            ],
            sum: k("dsv41_moe_sum_decode")?,
        };
        let upload = |bytes: &[u8]| -> Result<DevicePtr> {
            let p = allocs.borrow_mut().alloc(gpu, bytes.len().max(256))?;
            gpu.copy_h2d(bytes, p)?;
            Ok(p)
        };
        let mut tables = Vec::with_capacity(routers.len());
        for (layer, r) in routers {
            let mask: Vec<u8> = arena.routing_mask(*layer)?.iter().map(|&m| u8::from(m)).collect();
            ensure!(mask.len() == ROUTED_EXPERTS, "layer {layer}: mask is {} wide", mask.len());
            let slots = (0..ROUTED_EXPERTS as u32).map(|e| arena.slot_of(*layer, e)).collect::<Result<Vec<i32>>>()?;
            tables.push(LayerTables {
                layer: *layer,
                bias: upload(bytes_of(&r.bias))?,
                bias_vl: upload(bytes_of(&r.bias_vl))?,
                mask: upload(&mask)?,
                slot_of: upload(bytes_of(&slots))?,
            });
        }
        let a = |bytes: usize| -> Result<DevicePtr> {
            let p = allocs.borrow_mut().alloc(gpu, bytes.max(256))?;
            gpu.memset(p, 0, bytes.max(256))?;
            Ok(p)
        };
        let (ids, logits, groups) = (a(MAX_DECODE_T * 4)?, a(MAX_DECODE_T * ROUTED_EXPERTS * 4)?, a((1 + MAX_ROWS * GROUP_INTS) * 4)?);
        let (row_w, sel, err) = (a(MAX_ROWS * 4)?, a(MAX_ROWS * 4)?, a(4)?);
        let (h, down) = (a(MAX_ROWS * INTER * 2)?, a(MAX_ROWS * HIDDEN * 4)?);
        Ok(Self { k, tables, ids, logits, groups, row_w, sel, err, h, down, _allocs: allocs.into_inner() })
    }

    fn table(&self, layer: usize) -> Result<&LayerTables> {
        self.tables.iter().find(|t| t.layer == layer).with_context(|| format!("decode MoE: no tables for layer {layer}"))
    }

    /// The pass's token ids (image rows route with `bias_vl`). Once per pass, before the layers.
    pub fn set_ids(&self, gpu: &dyn GpuBackend, ids: &[u32]) -> Result<()> {
        ensure!(!ids.is_empty() && ids.len() <= MAX_DECODE_T, "decode MoE: {} ids", ids.len());
        gpu.copy_h2d(bytes_of(ids), self.ids)
    }

    /// Router logits + top-6 + grouping, all on the device.
    pub fn route(&self, gpu: &dyn GpuBackend, layer: usize, router: &RouterF32, y: DevicePtr, t: usize, route_scale: f32, stream: u64) -> Result<()> {
        ensure!(t >= 1 && t <= MAX_DECODE_T, "decode MoE: t = {t}");
        let tb = self.table(layer)?;
        // The original bf16 weight when we have it (half the bytes, bit-identical logits);
        // `ATLAS_DSV41_ROUTER_BF16=0` forces the widened fp32 copy (A/B only).
        let (kernel, w) = match router.gate_w_bf16 {
            Some(b) if std::env::var("ATLAS_DSV41_ROUTER_BF16").as_deref() != Ok("0") => (self.k.router_bf16w, b),
            _ => (self.k.router, router.gate_w),
        };
        KernelLaunch::new(gpu, kernel)
            .grid([(ROUTED_EXPERTS as u32).div_ceil(WARPS), t as u32, 1])
            .block([32 * WARPS, 1, 1])
            .arg_ptr(y)
            .arg_ptr(w)
            .arg_ptr(self.logits)
            .launch(stream)?;
        KernelLaunch::new(gpu, self.k.route)
            .grid([1, 1, 1])
            .block([32 * t as u32, 1, 1])
            .arg_ptr(self.logits)
            .arg_ptr(tb.bias)
            .arg_ptr(tb.bias_vl)
            .arg_ptr(self.ids)
            .arg_ptr(tb.mask)
            .arg_ptr(tb.slot_of)
            .arg_ptr(self.groups)
            .arg_ptr(self.row_w)
            .arg_ptr(self.sel)
            .arg_ptr(self.err)
            .arg_i32(t as i32)
            .arg_f32(route_scale)
            .launch(stream)
    }

    /// Install a HOST routing decision (engine-fed routing in gates, and negative controls).
    /// Synchronous. `slot_of` maps a routed id to its resident slot.
    pub fn upload_routing(&self, gpu: &dyn GpuBackend, routing: &Routing, t: usize, slot_of: impl Fn(u32) -> Result<i32>) -> Result<()> {
        ensure!(routing.k == TOP_K && routing.indices.len() == t * TOP_K, "routing extent");
        let mut groups = vec![0i32; 1 + MAX_ROWS * GROUP_INTS];
        let mut n = 0usize;
        for (p, &id) in routing.indices.iter().enumerate() {
            let slot = slot_of(u32::try_from(id).context("expert id")?)?;
            ensure!(slot >= 0, "expert {id} is not resident");
            let g = (0..n).find(|&g| groups[1 + g * GROUP_INTS] == slot).unwrap_or(n);
            let base = 1 + g * GROUP_INTS;
            if g == n {
                groups[base] = slot;
                groups[base + 1] = 0;
                n += 1;
            }
            let r = groups[base + 1] as usize;
            groups[base + 2 + r] = p as i32;
            groups[base + 1] += 1;
        }
        groups[0] = n as i32;
        let sel: Vec<i32> = routing.indices.iter().map(|&i| i as i32).collect();
        gpu.copy_h2d(bytes_of(&groups), self.groups)?;
        gpu.copy_h2d(bytes_of(&routing.weights), self.row_w)?;
        gpu.copy_h2d(bytes_of(&sel), self.sel)
    }

    /// The device routing of the last [`Self::route`] as a host `Routing` (synchronous; gates
    /// and controls only).
    pub fn read_routing(&self, gpu: &dyn GpuBackend, t: usize, stream: u64) -> Result<Routing> {
        gpu.synchronize(stream)?;
        let mut sel = vec![0u8; t * TOP_K * 4];
        let mut w = vec![0u8; t * TOP_K * 4];
        gpu.copy_d2h(self.sel, &mut sel)?;
        gpu.copy_d2h(self.row_w, &mut w)?;
        Ok(Routing {
            indices: sel.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64).collect(),
            weights: w.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            k: TOP_K,
        })
    }

    /// Nonzero if any routing kernel since construction picked a masked / non-resident expert.
    /// Synchronous; read it at the end of a run, not per layer.
    pub fn check_error(&self, gpu: &dyn GpuBackend, stream: u64) -> Result<()> {
        gpu.synchronize(stream)?;
        let mut e = [0u8; 4];
        gpu.copy_d2h(self.err, &mut e)?;
        let code = i32::from_le_bytes(e);
        ensure!(code == 0, "decode MoE routing error {code} (1 = fewer than 6 resident, 2 = routed id with no slot)");
        Ok(())
    }

    /// The expert half, from the routing currently on the device: `y [t, 5120] -> out [t, 5120]`.
    #[allow(clippy::too_many_arguments)]
    pub fn experts(
        &self,
        gpu: &dyn GpuBackend,
        residency: &Cb3LayerResidency,
        keep: usize,
        m: &[Cb3Matrix; 3],
        y: DevicePtr,
        out: DevicePtr,
        t: usize,
        limit: f32,
        stream: u64,
    ) -> Result<()> {
        ensure!(t >= 1 && t <= MAX_DECODE_T, "decode MoE: t = {t}");
        let [w1, w3, w2] = m;
        ensure!(
            (w1.rows, w1.cols, w3.rows, w3.cols, w2.rows, w2.cols) == (INTER, HIDDEN, INTER, HIDDEN, HIDDEN, INTER),
            "decode MoE kernels are compiled for 5120 x 2304 experts"
        );
        // slot-0 plane address and the per-slot stride, checked against the arena's own.
        let plane = |tensor, rows: usize, bytes_per_row: usize| -> Result<(DevicePtr, u64)> {
            let p0 = residency.plane_ptr(tensor, 0, keep)?;
            let p1 = residency.plane_ptr(tensor, 1, keep)?;
            let stride = (rows * bytes_per_row) as u64;
            ensure!(p1.0 - p0.0 == stride, "CB3 plane stride {} != rows x bytes_per_row {stride}", p1.0 - p0.0);
            Ok((p0, stride))
        };
        let (w1lo, lo_s) = plane(w1.lo, w1.rows, w1.cols / 4)?;
        let (w1hi, hi_s) = plane(w1.hi, w1.rows, w1.cols / 8)?;
        let (w1cb, cb_s) = plane(w1.cb, w1.rows, 8)?;
        let (s1, sc_s) = plane(w1.scale, w1.rows, w1.cols / 32)?;
        let (w3lo, _) = plane(w3.lo, w3.rows, w3.cols / 4)?;
        let (w3hi, _) = plane(w3.hi, w3.rows, w3.cols / 8)?;
        let (w3cb, _) = plane(w3.cb, w3.rows, 8)?;
        let (s3, _) = plane(w3.scale, w3.rows, w3.cols / 32)?;
        let (w2lo, lo2_s) = plane(w2.lo, w2.rows, w2.cols / 4)?;
        let (w2hi, hi2_s) = plane(w2.hi, w2.rows, w2.cols / 8)?;
        let (w2cb, cb2_s) = plane(w2.cb, w2.rows, 8)?;
        let (s2, sc2_s) = plane(w2.scale, w2.rows, w2.cols / 32)?;

        let max_groups = (t * TOP_K) as u32;
        // `ATLAS_DSV41_MOE_R8=1`: every t > 1 on the R=8 kernels (the pre-variant path, A/B only).
        let v = match t {
            1 => 0,
            _ if r8_only() => 4,
            2 => 1,
            3 | 4 => 2,
            5 | 6 => 3,
            _ => 4,
        };
        let (gateup, down) = (self.k.gateup[v], self.k.down[v]);
        KernelLaunch::new(gpu, gateup)
            .grid([(INTER as u32).div_ceil(WARPS), max_groups, 1])
            .block([32 * WARPS, 1, 1])
            .arg_ptr(w1lo).arg_ptr(w1hi).arg_ptr(w1cb).arg_ptr(s1)
            .arg_ptr(w3lo).arg_ptr(w3hi).arg_ptr(w3cb).arg_ptr(s3)
            .arg_u64(lo_s).arg_u64(hi_s).arg_u64(cb_s).arg_u64(sc_s)
            .arg_ptr(y).arg_ptr(self.groups).arg_ptr(self.row_w).arg_ptr(self.h)
            .arg_f32(limit)
            .launch(stream)?;
        KernelLaunch::new(gpu, down)
            .grid([(HIDDEN as u32).div_ceil(WARPS), max_groups, 1])
            .block([32 * WARPS, 1, 1])
            .arg_ptr(w2lo).arg_ptr(w2hi).arg_ptr(w2cb).arg_ptr(s2)
            .arg_u64(lo2_s).arg_u64(hi2_s).arg_u64(cb2_s).arg_u64(sc2_s)
            .arg_ptr(self.h).arg_ptr(self.groups).arg_ptr(self.down)
            .launch(stream)?;
        KernelLaunch::new(gpu, self.k.sum)
            .grid([(HIDDEN as u32).div_ceil(256), t as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(self.down)
            .arg_ptr(out)
            .launch(stream)?;
        if t > 1 && groups_log() {
            // Diagnostic (synchronous): how many DISTINCT experts this multi-row pass read.
            gpu.synchronize(stream)?;
            let mut n = [0u8; 4];
            gpu.copy_d2h(self.groups, &mut n)?;
            eprintln!("moe_groups t={t} picks={} distinct={}", t * TOP_K, i32::from_le_bytes(n));
        }
        Ok(())
    }
}

fn r8_only() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_DSV41_MOE_R8").as_deref() == Ok("1"))
}

/// `ATLAS_DSV41_MOE_GROUPS_LOG=1`: print the distinct-expert count of every multi-row MoE pass.
fn groups_log() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_DSV41_MOE_GROUPS_LOG").as_deref() == Ok("1"))
}
