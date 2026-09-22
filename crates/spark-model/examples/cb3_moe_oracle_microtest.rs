// SPDX-License-Identifier: AGPL-3.0-only
//! **The first comparison of an actual MoE OUTPUT against the serving engine.**
//!
//! Everything validated so far has been inputs, selections and bytes: routing reproduces
//! `route_idx` exactly, the arena reads back byte-exact. This runs the whole routed-expert
//! path on the GPU and compares the RESULT to the engine's `moe_routed` tap.
//!
//! ```text
//!   moe_in [T, 5120] bf16   (from the oracle — not recomputed, so attention is not needed)
//!     -> permute to expert-major
//!     -> per expert: CB3 reconstruct w1/w3 -> 2 GEMMs -> moe_silu_mul -> reconstruct w2 -> GEMM
//!     -> moe_unpermute_reduce_indexed with the engine's route_w
//!   == moe_routed [T, 5120] f32
//! ```
//!
//! ## Why this does not need attention, engram, or the full arena
//! `moe_in` is a captured tap, so the input is given rather than computed — the attention
//! seam is out of the picture entirely. `route_idx`/`route_w` are taken from the capture
//! too, ON PURPOSE: routing is already validated exactly over 8488 token-rows elsewhere,
//! and feeding the engine's own selection here isolates the GEMM path. If this disagrees,
//! the bug is in reconstruct, permutation or GEMM — not in routing.
//!
//! And it needs only ONE layer resident (1.79 GB at the served keep), not the full 71.7 GB,
//! because `moe_routed` is a per-layer tap.
//!
//! ## Tolerance — PRE-REGISTERED before the first run: rel_l2 in [1e-3, 8e-3]
//! CB3 itself contributes ZERO error (e2m1 x power-of-two scale is exact in bf16). What
//! differs from the engine is WHERE bf16 rounding happens. The engine's prefill path
//! (`DSV41_CB3_PREFILL=fp4`, `tools/fp4_moe.py`) keeps gate/up in fp32 through the clamp
//! and SwiGLU, multiplies by the route weight, rounds `h` to bf16 ONCE, keeps the down
//! projection in fp32 per (k, token), sums the six in fp32, rounds once. This path rounds
//! gate, up, act and each expert's down output to bf16 (cuBLASLt is bf16-out) — three
//! extra roundings of ~2^-9 relative each, independent across elements, so an expected
//! rel_l2 of a few 1e-3. Above 8e-3 is a bug; below 1e-3 would mean the engine is not
//! doing what its source says and is ALSO worth reporting.
//!
//! The gate is rel_l2 < [`TOL`] AND both controls must land at least [`MIN_SEPARATION`]x
//! further out.
//!
//! ```text
//! cargo run -p spark-model --release --example cb3_moe_oracle_microtest \
//!   --features cuda,gpu-examples -- --run runA --layer 0 [--dump out.bin]
//! ```
//! `--control weights` reverses each token's route weights across its picks;
//! `--control expert` feeds every group the NEXT resident slot's weights. Both are finite,
//! correctly shaped and wrong, and both must FAIL.

use anyhow::{Context, Result, bail, ensure};
use std::path::{Path, PathBuf};

use atlas_core::config::{ExpertPack, SERVED_PACKED_KEEP, parse_config};
use spark_model::weight_loader::deepseek_v41::cb3_arena::Cb3ExpertArena;
use spark_model::weight_loader::deepseek_v41::moe::{
    COMBINE_MODULE, Cb3Permutation, Cb3Reconstruct, MOE_PERMUTE_MODULE, PERMUTE_KERNEL,
    SILU_MUL_MODULE, SWIGLU_WEIGHTED_FN, UNPERMUTE_KERNEL, UNPERMUTE_SUM_FN, expert_matrices,
    gemm_weight_t, gemm_weight_t_f32out, group_by_expert,
};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

const MODEL_DIR: &str = "/home/flocka/models/DeepSeek-V4.1-Flash-Next-DGX-Spark-512K";
const REF_ROOT: &str = "/home/flocka/atlas/DSV41_PORT/oracle/ref";
const K: usize = 6;
/// Pre-registered upper bound; see the module note.
const TOL: f64 = 8e-3;
/// A control must land at least this many times further out than the gate.
const MIN_SEPARATION: f64 = 10.0;

#[derive(Clone, Copy, PartialEq)]
enum Numerics {
    /// Round to bf16 where the engine does: at `h` and at the final sum.
    Engine,
    /// The generic common kernels: bf16 after every GEMM. Measured ~3.8e-3.
    Generic,
}

#[derive(Clone, Copy, PartialEq)]
enum Control {
    None,
    /// Reverse each token's route weights across its picks.
    Weights,
    /// Reconstruct each group from the NEXT resident slot — a real expert, the wrong one.
    Expert,
}

fn read_bytes(path: &Path) -> Result<Vec<u8>> {
    std::fs::read(path).with_context(|| format!("reading {}", path.display()))
}

fn read_bf16_as_f32(path: &Path) -> Result<Vec<f32>> {
    Ok(read_bytes(path)?
        .chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect())
}

fn read_f32(path: &Path) -> Result<Vec<f32>> {
    Ok(read_bytes(path)?
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn read_i64(path: &Path) -> Result<Vec<i64>> {
    Ok(read_bytes(path)?
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect())
}

/// f32 -> bf16 bits, round-to-nearest-even, matching what the engine feeds its GEMM.
fn to_bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounding = 0x7fff + ((bits >> 16) & 1);
    ((bits.wrapping_add(rounding)) >> 16) as u16
}

fn main() -> Result<()> {
    let mut run = "runA".to_string();
    let mut layer = 0usize;
    let mut occurrence = 0usize;
    let mut control = Control::None;
    let mut dump: Option<PathBuf> = None;
    let mut numerics = Numerics::Engine;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--run" => run = args.next().context("--run needs a value")?,
            "--layer" => layer = args.next().context("--layer needs a value")?.parse()?,
            "--occurrence" => {
                occurrence = args.next().context("--occurrence needs a value")?.parse()?
            }
            "--numerics" => {
                numerics = match args.next().context("--numerics needs engine|generic")?.as_str() {
                    "engine" => Numerics::Engine,
                    "generic" => Numerics::Generic,
                    other => bail!("unknown numerics {other}"),
                }
            }
            "--dump" => dump = Some(args.next().context("--dump needs a path")?.into()),
            "--control" => {
                control = match args.next().context("--control needs weights|expert")?.as_str() {
                    "weights" => Control::Weights,
                    "expert" => Control::Expert,
                    other => bail!("unknown control {other}"),
                }
            }
            other => bail!("unknown argument {other}"),
        }
    }
    let dir = PathBuf::from(REF_ROOT).join(&run);
    ensure!(dir.join("manifest.json").is_file(), "{} has no manifest — capture incomplete", dir.display());
    let tag = format!("L{layer:02}");

    let config = parse_config(&std::fs::read_to_string(format!("{MODEL_DIR}/config.json"))?)?;
    let hidden = config.hidden_size;
    let inter = config.moe_intermediate_size;

    let tap = |name: &str| dir.join(format!("{tag}.{name}.{occurrence:03}.bin"));
    let moe_in = read_bf16_as_f32(&tap("moe_in"))?;
    let route_idx = read_i64(&tap("route_idx"))?;
    let mut route_w = read_f32(&tap("route_w"))?;
    let want = read_f32(&tap("moe_routed"))?;
    let tokens = moe_in.len() / hidden;
    ensure!(route_idx.len() == tokens * K, "route_idx is {} for {tokens} tokens", route_idx.len());
    ensure!(want.len() == tokens * hidden, "moe_routed is {} for {tokens} tokens", want.len());
    println!("{run}/{tag}.{occurrence:03}: {tokens} tokens, hidden {hidden}, inter {inter}");

    if control == Control::Weights {
        // NEGATIVE CONTROL: reverse each token's weights across its 6 picks. Same values,
        // same sum, attached to the WRONG experts — finite, correctly shaped, meaningless.
        // This is what a permutation bug looks like, and the gate must reject it.
        for token in 0..tokens {
            route_w[token * K..(token + 1) * K].reverse();
        }
        println!("  [control weights] reversed each token's route_w across its picks — MUST FAIL");
    }

    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let pack_dir = PathBuf::from(MODEL_DIR).join("k154-cb3");
    let pack = ExpertPack::parse(&std::fs::read_to_string(pack_dir.join("manifest.json"))?, SERVED_PACKED_KEEP)?;
    let arena = Cb3ExpertArena::load_one_layer(&pack_dir, &pack, layer, &gpu)?;
    println!("  layer {layer} resident: {:.2} GB", arena.resident_bytes() as f64 / 1e9);

    // Group by expert and build the expert-major permutation.
    let mut groups =
        group_by_expert(&route_idx, &route_w, tokens, K, |id| arena.slot_of(layer, id))?;
    if control == Control::Expert {
        let keep = arena.packed_keep();
        for group in &mut groups {
            group.slot = (group.slot + 1) % keep;
        }
        println!("  [control expert] every group reconstructs the NEXT resident slot — MUST FAIL");
    }
    let plan = Cb3Permutation::build(&groups, tokens, K)?;
    let expanded = plan.total_expanded;
    println!("  {} experts touched, {expanded} expanded rows", groups.len());

    // ---- device buffers
    let matrices = expert_matrices(&config)?;
    let reconstruct = Cb3Reconstruct::new(&gpu)?;
    let stream = gpu.default_stream();

    let d_in = gpu.alloc(tokens * hidden * 2)?;
    let d_perm = gpu.alloc(expanded * hidden * 2)?;
    let d_out = gpu.alloc(tokens * hidden * 2)?;
    let d_w1 = gpu.alloc(inter * hidden * 2)?;
    let d_w3 = gpu.alloc(inter * hidden * 2)?;
    let d_w2 = gpu.alloc(hidden * inter * 2)?;
    let d_sorted = gpu.alloc(expanded * 4)?;
    let d_tok2perm = gpu.alloc(tokens * K * 4)?;

    let in_bits: Vec<u8> = moe_in.iter().flat_map(|v| to_bf16_bits(*v).to_le_bytes()).collect();
    gpu.copy_h2d(&in_bits, d_in)?;
    gpu.copy_h2d(bytemuck_i32(&plan.sorted_token_ids), d_sorted)?;
    gpu.copy_h2d(bytemuck_i32(&plan.token_to_perm), d_tok2perm)?;

    let k_permute = gpu.kernel(MOE_PERMUTE_MODULE, PERMUTE_KERNEL)?;

    // ---- gather into expert-major order
    launch(&gpu, k_permute, [expanded as u32, 1, 1], [256, 1, 1], stream,
        &mut [arg(d_in.0), arg(d_perm.0), arg(d_sorted.0), arg32(hidden as u32), arg32(expanded as u32)])?;

    let residency = arena.layer(layer)?;
    let keep = arena.packed_keep();
    // Not on ModelConfig; read from config.json (flattened under text_config or not).
    let raw: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{MODEL_DIR}/config.json"))?)?;
    let limit = raw
        .get("swiglu_limit")
        .or_else(|| raw["text_config"].get("swiglu_limit"))
        .and_then(serde_json::Value::as_f64)
        .context("V4.1 config must carry swiglu_limit")? as f32;
    ensure!(limit == 10.0, "swiglu_limit {limit}, the engine ran with 10.0");
    println!("  numerics: {}", match numerics {
        Numerics::Engine => "ENGINE (fp32 gate/up/down, bf16 at h and at the sum only)",
        Numerics::Generic => "GENERIC (bf16 at gate, up, act, each down, and the sum)",
    });

    match numerics {
        Numerics::Engine => {
            // Route weight per PERMUTED row: it multiplies h, as the engine's up kernel does.
            let mut row_weight = vec![0.0f32; expanded];
            for (flat, row) in plan.token_to_perm.iter().enumerate() {
                row_weight[*row as usize] = plan.weights[flat];
            }
            let d_row_w = gpu.alloc(expanded * 4)?;
            gpu.copy_h2d(bytemuck_f32(&row_weight), d_row_w)?;
            let d_gate = gpu.alloc(expanded * inter * 4)?;
            let d_up = gpu.alloc(expanded * inter * 4)?;
            let d_h = gpu.alloc(expanded * inter * 2)?;
            let d_down = gpu.alloc(expanded * hidden * 4)?;
            let k_swiglu = gpu.kernel(COMBINE_MODULE, SWIGLU_WEIGHTED_FN)?;
            let k_sum = gpu.kernel(COMBINE_MODULE, UNPERMUTE_SUM_FN)?;

            for (group, (begin, end)) in groups.iter().zip(&plan.group_rows) {
                let rows = end - begin;
                if rows == 0 { continue; }
                reconstruct.run(residency, matrices[0], group.slot, keep, d_w1, &gpu, stream)?;
                reconstruct.run(residency, matrices[1], group.slot, keep, d_w3, &gpu, stream)?;
                reconstruct.run(residency, matrices[2], group.slot, keep, d_w2, &gpu, stream)?;

                let act = DevicePtr(d_perm.0 + (begin * hidden * 2) as u64);
                let gate = DevicePtr(d_gate.0 + (begin * inter * 4) as u64);
                let up = DevicePtr(d_up.0 + (begin * inter * 4) as u64);
                let h = DevicePtr(d_h.0 + (begin * inter * 2) as u64);
                let w = DevicePtr(d_row_w.0 + (begin * 4) as u64);
                let down = DevicePtr(d_down.0 + (begin * hidden * 4) as u64);

                gemm_weight_t_f32out(act, d_w1, gate, rows, inter, hidden, stream)?;
                gemm_weight_t_f32out(act, d_w3, up, rows, inter, hidden, stream)?;
                let total = (rows * inter) as u32;
                launch(&gpu, k_swiglu, [total.div_ceil(256), 1, 1], [256, 1, 1], stream,
                    &mut [arg(gate.0), arg(up.0), arg(w.0), arg(h.0),
                          arg32(rows as u32), arg32(inter as u32), argf(limit)])?;
                gemm_weight_t_f32out(h, d_w2, down, rows, hidden, inter, stream)?;
            }
            launch(&gpu, k_sum, [tokens as u32, 1, 1], [256, 1, 1], stream,
                &mut [arg(d_down.0), arg(d_out.0), arg(d_tok2perm.0),
                      arg32(hidden as u32), arg32(tokens as u32), arg32(K as u32)])?;
        }
        Numerics::Generic => {
            let d_gate = gpu.alloc(expanded * inter * 2)?;
            let d_up = gpu.alloc(expanded * inter * 2)?;
            let d_act = gpu.alloc(expanded * inter * 2)?;
            let d_expert_out = gpu.alloc(expanded * hidden * 2)?;
            let d_weights = gpu.alloc(tokens * K * 4)?;
            gpu.copy_h2d(bytemuck_f32(&plan.weights), d_weights)?;
            let k_silu = gpu.kernel(SILU_MUL_MODULE, "moe_silu_mul")?;
            let k_unperm = gpu.kernel(MOE_PERMUTE_MODULE, UNPERMUTE_KERNEL)?;

            for (group, (begin, end)) in groups.iter().zip(&plan.group_rows) {
                let rows = end - begin;
                if rows == 0 { continue; }
                reconstruct.run(residency, matrices[0], group.slot, keep, d_w1, &gpu, stream)?;
                reconstruct.run(residency, matrices[1], group.slot, keep, d_w3, &gpu, stream)?;
                reconstruct.run(residency, matrices[2], group.slot, keep, d_w2, &gpu, stream)?;

                let act = DevicePtr(d_perm.0 + (begin * hidden * 2) as u64);
                let gate = DevicePtr(d_gate.0 + (begin * inter * 2) as u64);
                let up = DevicePtr(d_up.0 + (begin * inter * 2) as u64);
                let acted = DevicePtr(d_act.0 + (begin * inter * 2) as u64);
                let outp = DevicePtr(d_expert_out.0 + (begin * hidden * 2) as u64);

                gemm_weight_t(act, d_w1, gate, rows, inter, hidden, stream)?;
                gemm_weight_t(act, d_w3, up, rows, inter, hidden, stream)?;
                let total = (rows * inter) as u32;
                launch(&gpu, k_silu, [total.div_ceil(256), 1, 1], [256, 1, 1], stream,
                    &mut [arg(gate.0), arg(up.0), arg(acted.0), arg32(total)])?;
                gemm_weight_t(acted, d_w2, outp, rows, hidden, inter, stream)?;
            }
            launch(&gpu, k_unperm, [tokens as u32, 1, 1], [256, 1, 1], stream,
                &mut [arg(d_expert_out.0), arg(d_out.0), arg(d_tok2perm.0), arg(d_weights.0),
                      arg32(hidden as u32), arg32(tokens as u32), arg32(K as u32)])?;
        }
    }
    gpu.synchronize(stream)?;

    // ---- compare
    let mut out_bytes = vec![0u8; tokens * hidden * 2];
    gpu.copy_d2h(d_out, &mut out_bytes)?;
    let got: Vec<f32> = out_bytes.chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect();

    let (mut num, mut den, mut worst) = (0.0f64, 0.0f64, 0.0f64);
    for (a, b) in got.iter().zip(&want) {
        let d = (*a as f64) - (*b as f64);
        num += d * d;
        den += (*b as f64) * (*b as f64);
        worst = worst.max(d.abs());
    }
    let rel = (num / den).sqrt();
    ensure!(got.iter().all(|v| v.is_finite()), "candidate has non-finite values");
    println!("  rel_l2 = {rel:.3e}   worst_abs = {worst:.3e}   over {tokens} x {hidden}");

    if let Some(path) = &dump {
        // f32, the reference tap's dtype, so compare.py can read both with one --dtype.
        let bytes: Vec<u8> = got.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(path, bytes).with_context(|| format!("writing {}", path.display()))?;
        println!("  candidate dumped to {} (float32, [{tokens}, {hidden}])", path.display());
    }

    if control != Control::None {
        let floor = TOL * MIN_SEPARATION;
        if rel > floor {
            println!("\nCONTROL FAILED AS REQUIRED: rel_l2 {rel:.3e} > {floor:.0e} ({MIN_SEPARATION}x the gate).");
            return Ok(());
        }
        bail!(
            "CONTROL DID NOT FIRE: a wrong-but-plausible MoE gave rel_l2 {rel:.3e}, inside \
             {floor:.0e}. This gate cannot separate right from wrong and proves nothing."
        );
    }
    ensure!(rel < TOL, "FAIL: rel_l2 {rel:.3e} exceeds the pre-registered {TOL:.0e}");
    let band = if rel < 1e-3 { "BELOW the pre-registered band — investigate" } else { "inside the pre-registered band" };
    println!("\nPASS: the MoE routed output matches the engine (rel_l2 {rel:.3e}, {band}).");
    Ok(())
}

fn bytemuck_i32(v: &[i32]) -> &[u8] {
    // SAFETY: i32 has no padding and no invalid bit patterns; the slice is read-only and
    // the lifetime is tied to `v`.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}
fn bytemuck_f32(v: &[f32]) -> &[u8] {
    // SAFETY: as above, for f32.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

enum Arg { U64(u64), U32(u32) }
fn arg(v: u64) -> Arg { Arg::U64(v) }
fn arg32(v: u32) -> Arg { Arg::U32(v) }
/// An f32 kernel argument, passed as its bit pattern in a 4-byte slot.
fn argf(v: f32) -> Arg { Arg::U32(v.to_bits()) }

fn launch(
    gpu: &dyn GpuBackend,
    kernel: spark_runtime::gpu::KernelHandle,
    grid: [u32; 3],
    block: [u32; 3],
    stream: u64,
    args: &mut [Arg],
) -> Result<()> {
    let mut slots: Vec<u64> = Vec::with_capacity(args.len());
    let mut smalls: Vec<u32> = Vec::with_capacity(args.len());
    for a in args.iter() {
        match a {
            Arg::U64(v) => { slots.push(*v); smalls.push(0); }
            Arg::U32(v) => { slots.push(0); smalls.push(*v); }
        }
    }
    let mut params: Vec<*mut std::ffi::c_void> = Vec::with_capacity(args.len());
    for (index, a) in args.iter().enumerate() {
        params.push(match a {
            Arg::U64(_) => &mut slots[index] as *mut u64 as *mut _,
            Arg::U32(_) => &mut smalls[index] as *mut u32 as *mut _,
        });
    }
    gpu.launch(kernel, grid, block, 0, stream, &mut params)
}
