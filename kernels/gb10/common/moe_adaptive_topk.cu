// SPDX-License-Identifier: AGPL-3.0-only

// Atlas adaptive top-K prune — ATLAS_MOE_ADAPTIVE_TOPK.
//
// QUALITY-AFFECTING. Default OFF (the Rust side never launches this kernel
// unless the env knob is set); see docs/ADAPTIVE-TOPK.md.
//
// Runs immediately after the router (`moe_topk_sqrtsoftplus` /
// `moe_topk_sigmoid` / `moe_topk_softmax`) and rewrites the routing decision
// IN PLACE, on device:
//
//   mass[t] = w[t] / sum_j w[j]            (gate-mass fraction of slot t)
//   drop  t  if mass[t] < threshold        (never the arg-max slot)
//   dropped: indices[t] = skip_index, weights[t] = 0
//   kept:    weights[t] *= sum_all / sum_kept    (iff renormalize)
//
// `skip_index` is the SENTINEL expert id `num_experts` — one past the last
// real expert. The expert pointer tables carry one extra, all-NULL entry at
// that index (`ptr_table_build.rs`), so the expert GEMVs' existing EP-remote
// guard (`if (B_packed == 0) { emit_zero(); return; }`) fires: the slot's
// output row is zeroed and the block RETURNS BEFORE THE K LOOP — i.e. before
// a single weight byte is streamed. That early-out is the entire point of the
// feature; the byte saving comes from there, not from here.
//
// GRAPH SAFETY. The launch geometry is a compile-time constant — grid (1,1,1),
// block (32,1,1) — and every input is read from device memory. Nothing about
// the launch depends on the data, so this kernel captures into a CUDA graph
// and replays like any other node. The downstream expert GEMVs keep their
// static `grid.y = top_k + 1`; the expert COUNT never changes, only which
// slots do work. No device-side count, no dynamic parallelism, no host
// readback, no re-capture.
//
// Grid: (1, 1, 1)   Block: (32, 1, 1)

#define MAX_TOP_K 32

extern "C" __global__ void moe_adaptive_topk_prune(
    unsigned int* __restrict__ expert_indices,  // [top_k] in/out
    float* __restrict__ expert_weights,         // [top_k] in/out
    unsigned int top_k,
    unsigned int skip_index,   // sentinel expert id (== num_experts)
    float threshold,           // drop slots with gate-mass fraction < threshold
    unsigned int renormalize   // 1 = rescale survivors so the total is preserved
) {
    if (threadIdx.x != 0) return;
    if (top_k == 0 || top_k > MAX_TOP_K) return;
    // threshold <= 0 is the disabled contract: leave the routing untouched.
    if (!(threshold > 0.0f)) return;

    float w[MAX_TOP_K];
    float sum_all = 0.0f;
    // The router emits weights in SELECTION order (descending score+bias), which
    // is NOT descending weight — the correction bias reorders them. So the
    // arg-max must be found explicitly; slot 0 is not guaranteed to be it.
    unsigned int argmax = 0;
    float wmax = -1.0f;
    for (unsigned int t = 0; t < top_k; t++) {
        float v = expert_weights[t];
        w[t] = v;
        sum_all += v;
        if (v > wmax) { wmax = v; argmax = t; }
    }
    if (!(sum_all > 1e-20f)) return;  // degenerate router output — leave alone

    float sum_kept = 0.0f;
    unsigned int kept = 0;
    for (unsigned int t = 0; t < top_k; t++) {
        // Keep the arg-max unconditionally: a token must always reach at least
        // one routed expert, whatever the threshold.
        const bool keep = (t == argmax) || ((w[t] / sum_all) >= threshold);
        if (keep) {
            sum_kept += w[t];
            kept++;
        } else {
            expert_indices[t] = skip_index;
            expert_weights[t] = 0.0f;
            w[t] = 0.0f;
        }
    }

    if (kept == top_k) return;  // nothing dropped — weights untouched

    // Renormalize so the surviving weights still sum to what the full top-K
    // summed to. Correct ONLY when the router normalized over the selected set
    // in the first place (`norm_topk_prob`); the caller passes 0 otherwise.
    if (renormalize && sum_kept > 1e-20f) {
        const float rescale = sum_all / sum_kept;
        for (unsigned int t = 0; t < top_k; t++) {
            if (w[t] > 0.0f) expert_weights[t] = w[t] * rescale;
        }
    }
}
