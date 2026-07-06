// SPDX-License-Identifier: AGPL-3.0-only

//! Host-side launch wrappers for the sparsity-drafted self-speculation
//! kernels (TEAL-style FFN activation column skipping).
//!
//! Two families:
//!   1. `ffn_sparsity_measure` — the feasibility/observer harness. Counts
//!      below-threshold activations per FFN site into a device histogram.
//!      Pure READER: never touches weights or the token stream.
//!   2. `ffn_build_keep_chunks` + `w4a16_gemv_sparse_cols` — the DRAFT path
//!      column-sparse GEMV. Thresholds the activation into a surviving
//!      k8-chunk index list, then runs a GEMV that only reads the surviving
//!      weight columns. APPROXIMATE (drops small-activation contributions) —
//!      only ever used to PROPOSE draft tokens the dense verify re-checks.

#![allow(unused_imports, clippy::too_many_arguments)]

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::QuantizedWeight;

use super::*;

/// Number of thresholds tracked by `ffn_sparsity_measure` — MUST match
/// `SPARSITY_NUM_THRESH` in `ffn_sparsity_measure.cu`. Thresholds are
/// {0.5%, 1%, 2%, 5%} of the per-row max-abs.
pub const SPARSITY_NUM_THRESH: usize = 4;

/// The fixed threshold set, as fractions of per-row max-abs. Mirrors
/// `SPARSITY_TAU` in the kernel; used only for host-side log formatting.
pub const SPARSITY_TAU: [f32; SPARSITY_NUM_THRESH] = [0.005, 0.010, 0.020, 0.050];

/// Launch the activation-sparsity measurement kernel for ONE decode row.
///
/// One CTA measures the row `input[0..K]`, atomically accumulating the
/// below-threshold counts into `hist_out[0..NUM_THRESH]` and bumping
/// `count_out[0]` (rows seen) + `count_out[1]` (elements seen). The buffers
/// are caller-owned and persist across steps so the host can average.
///
/// This is a PURE OBSERVER — it reads `input` and writes only into the
/// dedicated counter buffers. It never mutates `input`, weights, or any
/// buffer on the token-producing path, so enabling the measurement gate
/// cannot perturb the greedy token stream.
///
/// Kernel: `ffn_sparsity_measure(input, hist_out, count_out, K)`
/// Grid: (1, 1, 1)  Block: (256, 1, 1)
pub fn ffn_sparsity_measure(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    hist_out: DevicePtr,
    count_out: DevicePtr,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([1, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(hist_out)
        .arg_ptr(count_out)
        .arg_u32(k)
        .launch(stream)
}

/// Launch the on-device keep-chunk selector: threshold `input[0..K]` at
/// `tau * rowmax` and emit the surviving k8-chunk index list into
/// `keep_idx[0..K/8]` + `keep_len[0]`.
///
/// A k8 chunk (8 contiguous activations) survives iff its max-abs is
/// `>= tau * rowmax`. Runs in ONE CTA so the DRAFT path can chain it
/// back-to-back with `w4a16_gemv_sparse_cols` on the same stream with no
/// host round-trip. `keep_idx` order is NOT sorted (the sparse GEMV
/// random-accesses per chunk).
///
/// Kernel: `ffn_build_keep_chunks(input, tau, keep_idx, keep_len, K)`
/// Grid: (1, 1, 1)  Block: (256, 1, 1)
pub fn ffn_build_keep_chunks(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    tau: f32,
    keep_idx: DevicePtr,
    keep_len: DevicePtr,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([1, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_f32(tau)
        .arg_ptr(keep_idx)
        .arg_ptr(keep_len)
        .arg_u32(k)
        .launch(stream)
}

/// Launch the column-sparse W4A16 GEMV: `C[0..N] = A[keep] · dequant(B)`.
///
/// Iterates only over the surviving k8 chunks in `keep_idx[0..keep_len]`,
/// eliding the packed weight read for every skipped chunk. APPROXIMATE —
/// drops the below-threshold activation contributions. `keep_len` is a
/// SCALAR passed by value (the host reads it back from `ffn_build_keep_chunks`
/// via a D2H copy before this launch, OR the draft path passes a fixed
/// upper bound — see the draft seam).
///
/// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1) — mirrors `w4a16_gemv`.
///
/// Kernel: `w4a16_gemv_sparse_cols(A, B_packed, B_scale, scale2,
///          keep_idx, keep_len, C, N, K)`
pub fn w4a16_gemv_sparse_cols(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    keep_idx: DevicePtr,
    keep_len: u32,
    output: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(keep_idx)
        .arg_u32(keep_len)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// Host-side REFERENCE for `ffn_build_keep_chunks` (the on-device
/// keep-chunk selector), used for unit-testing the threshold logic without a
/// GPU. Given an activation row `a[0..K]` and a threshold fraction `tau` (of
/// the per-row max-abs), returns the set of surviving k8-chunk indices.
///
/// This MUST match the kernel predicate exactly:
///   * `rowmax = max_j |a[j]|`
///   * `cut = tau * rowmax`
///   * a k8 chunk (8 contiguous activations `a[8*c .. 8*c+8]`) survives iff
///     its per-chunk max-abs `>= cut` — i.e. "ANY of the 8 in the chunk
///     survives" granularity.
///
/// `K` must be a multiple of 8 (the packed NVFP4 chunk width); a non-multiple
/// tail is ignored (the kernel iterates `k8 < K/8`). A degenerate all-zero row
/// (rowmax == 0) yields NO survivors (cut == 0, and `>= 0` would keep all, but
/// the kernel's `w4a16_gemv_sparse_cols` on an all-zero row is a no-op — we
/// return empty here to match the "nothing worth reading" intent and keep the
/// reference total-order deterministic). Returned indices are SORTED ascending
/// for test determinism; the kernel does not guarantee order (the sparse GEMV
/// random-accesses per chunk), so tests compare as sets / sorted vectors.
pub fn keep_chunks_reference(a: &[f32], tau: f32) -> Vec<u32> {
    let k = a.len();
    let k8 = k / 8;
    let rowmax = a.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    if rowmax <= 0.0 {
        return Vec::new();
    }
    let cut = tau * rowmax;
    let mut out = Vec::new();
    for c in 0..k8 {
        let base = c * 8;
        let cmax = a[base..base + 8]
            .iter()
            .fold(0.0f32, |m, &v| m.max(v.abs()));
        if cmax >= cut {
            out.push(c as u32);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::keep_chunks_reference;

    /// Build a K-length activation row from a list of (index, value) spikes,
    /// zero elsewhere. `k` must be a multiple of 8.
    fn row(k: usize, spikes: &[(usize, f32)]) -> Vec<f32> {
        let mut v = vec![0.0f32; k];
        for &(i, x) in spikes {
            v[i] = x;
        }
        v
    }

    #[test]
    fn all_zero_row_keeps_nothing() {
        let a = vec![0.0f32; 32];
        assert!(keep_chunks_reference(&a, 0.01).is_empty());
    }

    #[test]
    fn single_spike_keeps_only_its_chunk() {
        // K=32 → 4 chunks. One big spike at index 10 (chunk 1). With any
        // reasonable tau, ONLY chunk 1 survives.
        let a = row(32, &[(10, 100.0)]);
        assert_eq!(keep_chunks_reference(&a, 0.01), vec![1]);
        assert_eq!(keep_chunks_reference(&a, 0.5), vec![1]);
    }

    #[test]
    fn any_of_eight_survives_granularity() {
        // Chunk 2 spans indices 16..24. A single above-cut element ANYWHERE in
        // the chunk keeps the WHOLE chunk (8-activation granularity). Put the
        // rowmax spike in chunk 0, and a smaller-but-above-cut value at index
        // 23 (last slot of chunk 2). tau=0.1 → cut=10. value 20 at idx 23
        // survives → chunk 2 kept even though its other 7 slots are 0.
        let a = row(32, &[(0, 100.0), (23, 20.0)]);
        assert_eq!(keep_chunks_reference(&a, 0.1), vec![0, 2]);
    }

    #[test]
    fn below_cut_chunk_dropped() {
        // rowmax=100 (chunk 0). tau=0.1 → cut=10. A 5.0 spike in chunk 3 is
        // BELOW cut → chunk 3 dropped. Only chunk 0 survives.
        let a = row(32, &[(0, 100.0), (25, 5.0)]);
        assert_eq!(keep_chunks_reference(&a, 0.1), vec![0]);
    }

    #[test]
    fn exact_cut_boundary_is_inclusive() {
        // Predicate is `>=` cut (matches kernel `cmax >= cut`). rowmax=100,
        // tau=0.1 → cut=10.0. A value EXACTLY 10.0 must survive.
        let a = row(16, &[(0, 100.0), (8, 10.0)]);
        assert_eq!(keep_chunks_reference(&a, 0.1), vec![0, 1]);
    }

    #[test]
    fn tau_zero_keeps_all_nonempty() {
        // cut = 0 → every chunk with any nonzero (or even all-zero, since
        // 0 >= 0) survives EXCEPT when rowmax==0. Here rowmax>0 so cut=0 and
        // every chunk's cmax (>=0) >= 0 → all 4 chunks kept.
        let a = row(32, &[(0, 1.0)]);
        assert_eq!(keep_chunks_reference(&a, 0.0), vec![0, 1, 2, 3]);
    }

    #[test]
    fn negative_activations_use_abs() {
        // The predicate is on |a[j]|. A large NEGATIVE spike sets rowmax and
        // keeps its chunk; a small negative in another chunk is compared by
        // magnitude. rowmax=|-100|=100, tau=0.1 → cut=10. -50 at idx 9
        // (chunk 1) survives.
        let a = row(24, &[(0, -100.0), (9, -50.0)]);
        assert_eq!(keep_chunks_reference(&a, 0.1), vec![0, 1]);
    }

    #[test]
    fn higher_tau_keeps_fewer_chunks_monotonic() {
        // Monotonicity: raising tau can only shrink (never grow) the keep set.
        let a = row(64, &[(0, 100.0), (10, 40.0), (20, 12.0), (35, 6.0)]);
        let lo = keep_chunks_reference(&a, 0.05); // cut=5 → chunks 0,1,2,4
        let hi = keep_chunks_reference(&a, 0.2); // cut=20 → chunks 0,1
        assert!(hi.iter().all(|c| lo.contains(c)), "hi ⊆ lo");
        assert!(hi.len() <= lo.len());
        assert_eq!(hi, vec![0, 1]);
    }

    #[test]
    fn savings_fraction_matches_skipped_chunks() {
        // The weight-byte savings equal 1 - keep_len/K8. A sparse row that
        // keeps 2 of 8 chunks skips 75% of the down_proj weight reads.
        let a = row(64, &[(0, 100.0), (10, 90.0)]); // chunks 0,1 dominate
        let keep = keep_chunks_reference(&a, 0.5); // cut=50 → chunks 0,1
        let k8 = a.len() / 8;
        assert_eq!(keep.len(), 2);
        let skipped_frac = 1.0 - keep.len() as f32 / k8 as f32;
        assert!((skipped_frac - 0.75).abs() < 1e-6);
    }
}
