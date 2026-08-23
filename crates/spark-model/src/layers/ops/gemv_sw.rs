// SPDX-License-Identifier: AGPL-3.0-only

//! Decode W4A16 GEMV launchers whose grid is coupled to the CUDA
//! `N_PER_BLOCK` / `N_PER_BLOCK_SW` defines.
//!
//! Ported from upstream `crates/spark-model/src/layers/ops/gemv_sw.rs`
//! (`7761b965`). The single-warp kernels (`w4a16_gemv_sw`,
//! `w4a16_gemv_dual_sw`, `w4a16_gemv_silu_input_sw`) are bit-identical to
//! their 64-thread bases: same per-lane K-chunk association, same shuffle
//! reduction tree, and the final `acc_a + acc_b` is the base kernel's
//! `smem[0] + smem[1]` operand-for-operand. See the derivation comment above
//! `w4a16_gemv_sw` in `kernels/gb10/common/w4a16_gemv.cu`.
//!
//! Shipping them as the default decode GEMV is a free occupancy win — 8
//! outputs per 256-thread block instead of 4, and no `__syncthreads()` — **if
//! and only if** the launch grid stays coupled. Swapping the kernel without
//! swapping the grid writes the wrong outputs, which is why every launcher
//! here derives its `grid.x` from the SSOT constants below and
//! `sw_grid_covers_every_output_and_is_half_base` pins the relationship.
//!
//! ## Divergence from upstream
//!
//! Upstream re-associated its base `w4a16_gemv` / `w4a16_gemv_dual` into a
//! 2-chunk pipelined K16 loop (`k16 = orig_lane * 2`, stride 128) in the same
//! commit. **We did not port that re-association** — our bases keep the
//! sequential `acc += a*w` loop (`k16 = orig_lane`, stride 64), so our SW
//! kernels are derived against *our* bases. Bit-parity is therefore against
//! the token stream this tree already produces, not against upstream's.
//! Upstream's own `w4a16_gemv_silu_input_sw` is derived the same way, against
//! its still-sequential K8 silu base.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::QuantizedWeight;

/// Base `w4a16_gemv` / `w4a16_gemv_dual` / `w4a16_gemv_silu_input`:
/// 4 outputs / 256-thread block.
/// SSOT with `kernels/**/w4a16_gemv{,_fused}.cu` `#define N_PER_BLOCK 4`.
pub const W4A16_GEMV_OUTS_PER_BLOCK: u32 = 4;

/// Single-warp `*_sw` kernels: 8 outputs / 256-thread block.
/// SSOT with `#define N_PER_BLOCK_SW 8`.
pub const W4A16_GEMV_SW_OUTS_PER_BLOCK: u32 = 8;

pub fn w4a16_gemv_grid_x(n: u32) -> u32 {
    div_ceil(n, W4A16_GEMV_OUTS_PER_BLOCK)
}

pub fn w4a16_gemv_sw_grid_x(n: u32) -> u32 {
    div_ceil(n, W4A16_GEMV_SW_OUTS_PER_BLOCK)
}

/// Kill-switch polarity for the lossless SW GEMV. ON unless `ATLAS_NO_GEMV_SW`
/// is exactly `"1"`.
///
/// Note the polarity deliberately: `ATLAS_NO_GEMV_SW=0` does **not** disable
/// it. That is upstream's reading and it is also the trap recorded as LESSON 9
/// in the GB10 concurrency campaign — `ATLAS_*=0` on a presence-checked flag
/// enables rather than disables. This one is value-checked (`== "1"`), so `=0`
/// leaves the SW path on, which is the intended default anyway.
pub fn gemv_sw_from(no_gemv_sw: Option<&str>) -> bool {
    no_gemv_sw != Some("1")
}

/// Resolved once per process, like the other kernel-path levers in
/// [`crate::layers`]. `std::env::var` is too expensive to call per decode hop.
pub fn gemv_sw_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| gemv_sw_from(std::env::var("ATLAS_NO_GEMV_SW").ok().as_deref()))
}

/// SW kernel when the lever is on **and** the handle resolved.
///
/// The second half matters: `try_kernel` returns `KernelHandle(0)` on a miss
/// and does not log, so a stale PTX cache without `w4a16_gemv_sw` in it would
/// otherwise launch a null handle. Falling back to the base kernel is correct
/// and silent-but-lossless; `--check-kernels` is the tool that makes the miss
/// visible.
pub fn use_gemv_sw(lever: bool, sw_handle: KernelHandle) -> bool {
    lever && sw_handle.0 != 0
}

/// Single-warp-per-output W4A16 GEMV (M=1). Grid: `(ceil(N/8), 1, 1)`.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_sw(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([w4a16_gemv_sw_grid_x(n), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// Decode GEMV: single-warp kernel when the lever and handle agree, else the
/// 64-thread base. Bit-identical either way.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_decode_gemv(
    gpu: &dyn GpuBackend,
    gemv: KernelHandle,
    gemv_sw: KernelHandle,
    use_sw: bool,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    if use_gemv_sw(use_sw, gemv_sw) {
        w4a16_gemv_sw(gpu, gemv_sw, input, weight, output, n, k, stream)
    } else {
        super::quant_dispatch::w4a16_gemv(gpu, gemv, input, weight, output, n, k, stream)
    }
}

/// Single-warp dual-projection GEMV. Grid: `(ceil(N/8), 1, 2)` — `gridDim.z`
/// still selects projection 0 vs 1, exactly as in the base kernel.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_dual_sw(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight1: &QuantizedWeight,
    output1: DevicePtr,
    weight2: &QuantizedWeight,
    output2: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([w4a16_gemv_sw_grid_x(n), 1, 2])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight1.weight)
        .arg_ptr(weight1.weight_scale)
        .arg_f32(weight1.weight_scale_2)
        .arg_ptr(output1)
        .arg_ptr(weight2.weight)
        .arg_ptr(weight2.weight_scale)
        .arg_f32(weight2.weight_scale_2)
        .arg_ptr(output2)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// Decode dual GEMV (FFN gate+up): SW when the lever and handle agree.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_decode_gemv_dual(
    gpu: &dyn GpuBackend,
    dual: KernelHandle,
    dual_sw: KernelHandle,
    use_sw: bool,
    input: DevicePtr,
    weight1: &QuantizedWeight,
    output1: DevicePtr,
    weight2: &QuantizedWeight,
    output2: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    if use_gemv_sw(use_sw, dual_sw) {
        w4a16_gemv_dual_sw(
            gpu, dual_sw, input, weight1, output1, weight2, output2, n, k, stream,
        )
    } else {
        super::moe_prefill::w4a16_gemv_dual(
            gpu, dual, input, weight1, output1, weight2, output2, n, k, stream,
        )
    }
}

/// Single-warp SiLU-fused-input GEMV. Grid: `(ceil(N/8), 1, 1)`.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_silu_input_sw(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate_out: DevicePtr,
    up_out: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([w4a16_gemv_sw_grid_x(n), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(gate_out)
        .arg_ptr(up_out)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// Decode down-projection GEMV with fused SiLU input: SW when the lever and
/// handle agree.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_decode_gemv_silu_input(
    gpu: &dyn GpuBackend,
    silu_input: KernelHandle,
    silu_input_sw: KernelHandle,
    use_sw: bool,
    gate_out: DevicePtr,
    up_out: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    if use_gemv_sw(use_sw, silu_input_sw) {
        w4a16_gemv_silu_input_sw(
            gpu,
            silu_input_sw,
            gate_out,
            up_out,
            weight,
            output,
            n,
            k,
            stream,
        )
    } else {
        super::moe_prefill::w4a16_gemv_silu_input(
            gpu, silu_input, gate_out, up_out, weight, output, n, k, stream,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use spark_runtime::gpu::KernelHandle;
    use std::fs;
    use std::path::{Path, PathBuf};

    #[test]
    fn gemv_sw_ships_on_and_only_the_one_value_kills() {
        assert!(gemv_sw_from(None), "unset → ON");
        assert!(gemv_sw_from(Some("0")), "`=0` is NOT off");
        assert!(gemv_sw_from(Some("")), "empty is NOT off");
        assert!(!gemv_sw_from(Some("1")), "`=1` is the kill");
    }

    #[test]
    fn sw_requires_both_the_lever_and_a_live_handle() {
        assert!(use_gemv_sw(true, KernelHandle(1)));
        assert!(
            !use_gemv_sw(true, KernelHandle(0)),
            "missing kernel falls back"
        );
        assert!(!use_gemv_sw(false, KernelHandle(1)), "kill switch wins");
        assert!(!use_gemv_sw(false, KernelHandle(0)));
    }

    #[test]
    fn sw_grid_covers_every_output_and_is_half_base_when_n_divisible_by_8() {
        for n in 1..=64 {
            assert!(w4a16_gemv_sw_grid_x(n) * W4A16_GEMV_SW_OUTS_PER_BLOCK >= n);
            assert!(w4a16_gemv_grid_x(n) * W4A16_GEMV_OUTS_PER_BLOCK >= n);
        }
        // Our live decode shapes: hidden 5120, FFN intermediate, SSM qkvz
        // 12288, attn qkv 14336, SSM out 6144.
        for n in [8u32, 16, 256, 5120, 6144, 12288, 14336] {
            assert_eq!(
                w4a16_gemv_sw_grid_x(n) * 2,
                w4a16_gemv_grid_x(n),
                "N={n}: SW is 8 outs/block, base is 4 — grid_x must be half"
            );
        }
    }

    fn kernel_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../kernels")
    }

    fn named_cu(file_name: &str) -> Vec<PathBuf> {
        fn visit(d: &Path, name: &str, out: &mut Vec<PathBuf>) {
            let Ok(rd) = std::fs::read_dir(d) else { return };
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() {
                    visit(&p, name, out);
                } else if p.file_name().is_some_and(|n| n == name) {
                    out.push(p);
                }
            }
        }
        let root = kernel_root();
        let mut files = Vec::new();
        visit(&root, file_name, &mut files);
        files.sort();
        files
    }

    /// POSITIVE: every copy of the GEMV sources pins the same occupancy
    /// constants the Rust launchers use. A new backend copy that changes
    /// `N_PER_BLOCK_SW` without updating the launcher writes the wrong N
    /// slice — silent, not a CUDA error.
    ///
    /// PROVEN BY: changing either `#define` in one `.cu` copy turns this red.
    #[test]
    fn cuda_n_per_block_matches_rust_ssot() {
        let want_base = format!("#define N_PER_BLOCK {W4A16_GEMV_OUTS_PER_BLOCK}");
        let want_sw = format!("#define N_PER_BLOCK_SW {W4A16_GEMV_SW_OUTS_PER_BLOCK}");

        let gemv = named_cu("w4a16_gemv.cu");
        assert!(!gemv.is_empty(), "no w4a16_gemv.cu found under kernels/");
        for p in &gemv {
            let src = fs::read_to_string(p).unwrap();
            assert!(
                src.contains(&want_base),
                "{} missing {want_base}",
                p.display()
            );
            assert!(src.contains(&want_sw), "{} missing {want_sw}", p.display());
        }

        let fused = named_cu("w4a16_gemv_fused.cu");
        assert!(
            !fused.is_empty(),
            "dual_sw / silu_input_sw live in w4a16_gemv_fused.cu"
        );
        for p in &fused {
            let src = fs::read_to_string(p).unwrap();
            assert!(src.contains(&want_sw), "{} missing {want_sw}", p.display());
        }
    }

    /// POSITIVE: each SW kernel must share its base kernel's per-lane K-chunk
    /// association, via a `*_partial` helper both call. A hand-copied SW body
    /// that drifts off the base association is 1 ULP lossy and changes the
    /// committed token stream — upstream shipped exactly that bug once
    /// (`upstream-latest/kernels/gb10/common/w4a16_gemv.cu:64-68`).
    ///
    /// The association pinned here is OURS: `k16/k8 = orig_lane`, stride 64.
    /// Upstream's equivalent test pins `orig_lane * 2u` / stride 128 because
    /// upstream re-associated its bases; we did not. If this tree ever adopts
    /// the pipelined base, this test is the thing that must change with it.
    ///
    /// PROVEN BY: dropping `orig_lane` from a `*_partial` signature, or
    /// changing a stride, turns this red.
    #[test]
    fn sw_partials_share_the_base_k_association() {
        for p in named_cu("w4a16_gemv.cu") {
            let src = fs::read_to_string(&p).unwrap();
            assert!(
                src.contains("w4a16_gemv_partial"),
                "{}: w4a16_gemv and w4a16_gemv_sw must share w4a16_gemv_partial",
                p.display()
            );
            assert!(
                src.contains("k16 = orig_lane; k16 < K16; k16 += 64u"),
                "{}: w4a16_gemv_partial must keep the base stride-64 association",
                p.display()
            );
        }
        for p in named_cu("w4a16_gemv_fused.cu") {
            let src = fs::read_to_string(&p).unwrap();
            for helper in ["w4a16_dual_partial", "w4a16_silu_partial"] {
                assert!(
                    src.contains(helper),
                    "{}: base and _sw must share {helper}",
                    p.display()
                );
            }
            assert_eq!(
                src.matches("k8 = orig_lane; k8 < K8; k8 += 64u").count(),
                2,
                "{}: both fused partials must keep the base stride-64 association",
                p.display()
            );
        }
    }

    /// NEGATIVE: the single-token FFN decode path must not launch the base
    /// GEMVs directly. It is 66.9% of the bytes in a decode sweep — a new
    /// `ops::w4a16_gemv_dual(` / `ops::w4a16_gemv_silu_input(` there ships the
    /// 64-thread kernel on the default path even though the `_sw` handles are
    /// resolved on the struct.
    ///
    /// PROVEN BY: restoring either pre-port call site turns this red.
    #[test]
    fn dense_ffn_decode_does_not_call_base_dual_or_silu_input() {
        let src = fs::read_to_string(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("src/layers/dense_ffn.rs"),
        )
        .unwrap();
        for banned in ["ops::w4a16_gemv_dual(", "ops::w4a16_gemv_silu_input("] {
            assert!(
                !src.contains(banned),
                "dense_ffn.rs: use the w4a16_decode_gemv_* dispatchers, not {banned}"
            );
        }
    }
}
