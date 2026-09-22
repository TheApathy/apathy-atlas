// SPDX-License-Identifier: AGPL-3.0-only

//! The attention lane's implementation of `weight_loader::deepseek_v41::seams::Dsv41Attention`.
//!
//! ## What is wired, and what hard-stops
//! * `sparse_attention` — WIRED: launches `dsv41_sparse_attn` (`attn/sparse_attn.cu`), the
//!   production entry of the gated one-pass streaming gather (real-tensor rel_l2 3.47e-07
//!   vs fp32; production entry bit-identical to the gated kernel, SPEC 8d).
//! * `sparse_index_select` — the device half ([`super::index`]) is validated EXACT against
//!   the engine's taps (SPEC 8e), but the q/wts projections that feed it, the score/top-k
//!   scratch sizing and the compressor that must run first are not agreed with the
//!   integrator yet. Until they are it REFUSES, naming what is missing. A selection built on
//!   guessed buffers would attend to the wrong rows and nothing would error.

use anyhow::{Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use crate::layer::ForwardContext;
use crate::weight_loader::deepseek_v41::seams::{Dsv41Attention, INDEX_TOPK, SparseShared};

/// Kernel module compiled from `sparse_attn.cu`.
pub const ATTN_MODULE: &str = "dsv41_sparse_attn";
/// Heads per block and threads per block, compiled into the kernel.
const HEADS_PER_BLOCK: usize = 8;
const THREADS: u32 = 256;
/// `head_dim` and `window_size`: compiled into the kernel / the reference's `wpos` width.
pub const HEAD_DIM: usize = 512;
pub const WINDOW: usize = 128;

pub struct Dsv41SparseAttention {
    pub layer: usize,
    pub num_heads: usize,
    kernel: KernelHandle,
}

impl Dsv41SparseAttention {
    pub fn new(gpu: &dyn GpuBackend, layer: usize, num_heads: usize) -> Result<Self> {
        ensure!(
            num_heads % HEADS_PER_BLOCK == 0,
            "dsv41_sparse_attn handles heads in groups of {HEADS_PER_BLOCK}; got {num_heads}"
        );
        let kernel = gpu.kernel(ATTN_MODULE, "dsv41_sparse_attn").map_err(|e| {
            anyhow::anyhow!(
                "{ATTN_MODULE}::dsv41_sparse_attn is not in the compiled PTX ({e}). It is built \
                 only for the (gb10, deepseek-v4.1, cb3) target."
            )
        })?;
        Ok(Self { layer, num_heads, kernel })
    }
}

impl Dsv41Attention for Dsv41SparseAttention {
    fn sparse_index_select(
        &self,
        _x: DevicePtr,
        _qr: DevicePtr,
        layer: usize,
        _num_tokens: usize,
        _chunk_start: usize,
        _chunk_len: usize,
        _shared: &mut SparseShared,
        _ctx: &ForwardContext,
        _stream: u64,
    ) -> Result<()> {
        bail!(
            "DeepSeek-V4.1 sparse_index_select (layer {layer}) is not wired yet. The indexer \
             KERNELS are validated exact (kernels/gb10/deepseek-v4.1/attn/SPEC.md 8e) and \
             their launches are in layers::deepseek_v41_attn::index; still open with the \
             integrator: the compressor that writes ckv/ik before this call, the q/wts \
             projections (recipe in SPEC 8e.1), and scratch for the [T, n_pad] fp32 score. \
             OWNER: dsv41-attention. Refusing rather than selecting from guessed buffers."
        )
    }

    fn sparse_attention(
        &self,
        q: DevicePtr,
        ring: DevicePtr,
        ring_len: usize,
        wpos: DevicePtr,
        win_lo: usize,
        ckv: Option<DevicePtr>,
        ckv_rows: usize,
        cidx: Option<DevicePtr>,
        sink: DevicePtr,
        scale: f32,
        num_tokens: usize,
        out: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        // ckv and cidx travel together: rows without a selection (or a selection without
        // rows) is a caller bug that would otherwise read garbage or silently drop keys.
        ensure!(
            ckv.is_some() == cidx.is_some(),
            "layer {}: ckv {:?} and cidx {:?} must both be present (ratio != 0) or both absent \
             (layers 0/1, window only)",
            self.layer,
            ckv.map(|_| ()),
            cidx.map(|_| ())
        );
        ensure!(ckv.is_none() || ckv_rows > 0, "layer {}: compressed cache with 0 rows", self.layer);
        ensure!(ring_len > 0, "layer {}: empty window ring", self.layer);
        if num_tokens == 0 {
            return Ok(());
        }
        KernelLaunch::new(ctx.gpu, self.kernel)
            .grid([num_tokens as u32, (self.num_heads / HEADS_PER_BLOCK) as u32, 1])
            .block([THREADS, 1, 1])
            .arg_ptr(q)
            .arg_ptr(ring)
            .arg_ptr(wpos)
            .arg_ptr(ckv.unwrap_or(DevicePtr::NULL))
            .arg_ptr(cidx.unwrap_or(DevicePtr::NULL))
            .arg_ptr(sink)
            .arg_ptr(out)
            .arg_i32(num_tokens as i32)
            .arg_i32(self.num_heads as i32)
            .arg_i32(HEAD_DIM as i32)
            .arg_i32(WINDOW as i32)
            .arg_i32(INDEX_TOPK as i32)
            .arg_i32(ring_len as i32)
            .arg_i32(win_lo as i32)
            .arg_f32(scale)
            .launch(stream)
    }
}
