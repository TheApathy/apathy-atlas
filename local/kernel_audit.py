#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""
kernel_audit.py — layer-by-layer, kernel-by-kernel optimization audit for the
Atlas dense-27B (AEON-Q36) DFlash verify path on GB10.

Fuses three data sources from a `spark serve` run:
  1. Kernel-load signal   — the "Selected kernel target: ... (N modules)" line
                            (+ per-kernel `try_kernel` handles that surface as
                            has_*/kernel= booleans in the dispatch log).
  2. Per-kernel timing     — `KPROF kernel=X calls=C total_us=T per_call_us=P`
                            lines emitted every 150 verify steps when the serve
                            ran with ATLAS_FULL_PROFILE=1.
  3. Fast-path flag status — the `dense_ffn forward_prefill dispatch` /
                            `[atlas-prefill-ffn]` line exposing which faster
                            kernels are PRESENT-but-DISABLED (e2m1_kernel=true
                            but e2m1_gate=false, etc.).

Two output modes:
  --mode full    (INTERNAL, default) full ranked audit + opportunity list with
                 exact ATLAS_* levers. NEVER PUBLISH.
  --mode public  (PUBLIC-SAFE) sanitized JSON health signal: embedded/used/
                 missing counts, generic phase shape, fast_paths_healthy bool.
                 Leaks no proprietary flags, kernel names, or timings.

Stdlib only. Read-only: never drives a restart, never changes serve output.

Usage:
  # audit an existing FULL_PROFILE serve log
  python3 kernel_audit.py --log /tmp/profile_serve.log

  # public health snippet a challenge contributor attaches to result.json
  python3 kernel_audit.py --log serve.log --mode public

  # tail a remote log over ssh (log stays on the box; this reads a local copy)
  scp box:/tmp/profile_serve.log . && python3 kernel_audit.py --log profile_serve.log
"""

import argparse
import json
import re
import sys
from collections import OrderedDict


# ─────────────────────────────────────────────────────────────────────────────
# Knowledge base: kprof kernel label -> classification metadata.
#
# Derived from the dense_ffn / qwen3_attention / qwen3_ssm dispatch code. Each
# entry maps a KPROF label to:
#   phase   : coarse GENERIC phase (used by public mode; no config fingerprint)
#   group   : layer family (SSM x48 / ATTN x16 / SHARED) for full-mode grouping
#   status  : default optimization verdict for the CHAMPION config
#             OPTIMIZED / DISABLED / FALLBACK / FLOOR
#   flag    : the controlling ATLAS_* env var (or "-" if unconditional)
#   note    : opportunity / banked note for full mode
#
# status can be refined at runtime from the dispatch line (e.g. a hot kernel
# whose faster variant is present-but-gated-off is downgraded to DISABLED).
# ─────────────────────────────────────────────────────────────────────────────
KB = {
    # ---- SSM (x48 layers) ----
    "ssm_ffn_kgamma_dense": dict(phase="ffn", group="SSM",
        status="OPTIMIZED", flag="ATLAS_FFN_KGAMMA_M16=1",
        note="SSM-block FFN gate/up+down (forward_kgamma). DOMINANT kernel. Routes fused m32 (FUSED_GATEUP) + down split-K. "
             "Body still BF16-heavy per SSM-weight profile; FP8/NVFP4 SSM = open lever #66 (pass@1-gated)."),
    "ssm_ffn_kgamma_resid": dict(phase="norm", group="SSM",
        status="FLOOR", flag="-", note="residual add after SSM FFN"),
    "ssm_ffn_forward_k3": dict(phase="ffn", group="SSM",
        status="OPTIMIZED", flag="ATLAS_DFLASH_FFN_KGAMMA=1", note="batched-3 SSM FFN forward"),
    "ssm_ffn_per_token_loop_n17": dict(phase="ffn", group="SSM",
        status="FLOOR", flag="-", note="per-token loop residue; recurrence, not batchable"),
    "ssm_conv_gdn_combined": dict(phase="ssm", group="SSM",
        status="FLOOR", flag="ATLAS_WY17_LAZY=16",
        note="GDN conv recurrence; wy17-lazy skips 13/16 state writes. Per-token, NaN at K>4 if batched. Banked."),
    "ssm_compute_gdn_gates_batched": dict(phase="ssm", group="SSM",
        status="OPTIMIZED", flag="-", note="batched gate compute"),
    "ssm_qkvz_w4a16_gemv_batch3": dict(phase="ssm", group="SSM",
        status="OPTIMIZED", flag="ATLAS_SSM_QKVZ_SPLITK=4", note="qkvz proj split-K"),
    "ssm_out_proj": dict(phase="ssm", group="SSM",
        status="OPTIMIZED", flag="ATLAS_SSM_OUT_SPLITK=4",
        note="out_proj split-K=4 (was slow BF16 dense_gemm; +9.4% when enabled). Banked."),
    "ssm_ba_proj_loop": dict(phase="ssm", group="SSM",
        status="OPTIMIZED", flag="ATLAS_SSM_BA_BATCH=1",
        note="BA-proj branch; label wraps the whole branch and BA_BATCH (fused dense_gemv_bf16_batchn) engages inside when set. Champion sets it. Only 0.6% — banked."),
    "ssm_gated_rms_norm_batched": dict(phase="norm", group="SSM",
        status="OPTIMIZED", flag="-", note="batched gated rmsnorm"),
    "ssm_rms_norm_residual": dict(phase="norm", group="SSM", status="FLOOR", flag="-", note="norm floor"),
    "ssm_post_attn_resid_norm": dict(phase="norm", group="SSM", status="FLOOR", flag="-", note="norm floor"),
    "ssm_ffn_residual_add": dict(phase="norm", group="SSM", status="FLOOR", flag="-", note="elementwise floor"),

    # ---- Attention (x16 layers) ----
    "attn_qkv_proj": dict(phase="attention", group="ATTN",
        status="OPTIMIZED", flag="ATLAS_ATTN_QKV_SPLITK=4",
        note="QKV split-K=4 (16->64 CTAs at N=1024) + ATLAS_ATTN_QKV_BATCHED=1."),
    "attn_ffn_kgamma_dense": dict(phase="ffn", group="ATTN",
        status="OPTIMIZED", flag="ATLAS_FFN_KGAMMA_M16=1",
        note="attn-block FFN gate/up+down (forward_kgamma). Routes fused m32 + down split-K."),
    "attn_ffn_kgamma_resid": dict(phase="ffn", group="ATTN",
        status="OPTIMIZED", flag="ATLAS_DFLASH_ATTN_KGAMMA=1", note="attn-block FFN fused kgamma"),
    "attn_ffn": dict(phase="ffn", group="ATTN",
        status="OPTIMIZED", flag="ATLAS_DFLASH_ATTN_KGAMMA=1", note="attn-block FFN"),
    "attn_ffn_kgamma_norm": dict(phase="norm", group="ATTN", status="OPTIMIZED", flag="-", note="fused norm"),
    "attn_ffn_per_token_loop_n17": dict(phase="ffn", group="ATTN",
        status="FLOOR", flag="-", note="per-token residue"),
    "attn_paged_decode": dict(phase="attention", group="ATTN",
        status="OPTIMIZED", flag="ATLAS_PAGED_DECODE_SPLITK=1",
        note="paged-decode split-K + ATLAS_FLASH_ATTN_KGAMMA_SPLITK=1"),
    "attn_o_proj": dict(phase="attention", group="ATTN",
        status="FLOOR", flag="-", note="o_proj GEMM; short-K, not occupancy-starved. Banked (SSM-sink note)."),
    "attn_rope": dict(phase="attention", group="ATTN", status="FLOOR", flag="-", note="rope floor"),
    "attn_cache_write": dict(phase="attention", group="ATTN", status="FLOOR", flag="-", note="kv write floor"),
    "attn_rms_norm_residual": dict(phase="norm", group="ATTN", status="FLOOR", flag="-", note="norm floor"),

    # ---- Shared / head ----
    "lm_head": dict(phase="head", group="SHARED",
        status="OPTIMIZED", flag="ATLAS_LM_HEAD_T=1",
        note="transposed lm_head. Still BF16-unquantized (2.37GB); NVFP4 lm_head = open lever #66, pass@1-gated."),
    "embed": dict(phase="head", group="SHARED", status="FLOOR", flag="-", note="embedding gather floor"),
    "final_norm": dict(phase="norm", group="SHARED", status="FLOOR", flag="-", note="norm floor"),
    "argmax": dict(phase="head", group="SHARED", status="FLOOR", flag="-", note="sampling floor"),
    "rms_norm": dict(phase="norm", group="SHARED", status="FLOOR", flag="-", note="norm floor"),

    # ---- FFN generic labels (fire depending on m16/m32/fused routing) ----
    "ffn_gateup_fused_kgamma": dict(phase="ffn", group="SHARED",
        status="OPTIMIZED", flag="ATLAS_FFN_FUSED_GATEUP=1",
        note="fused gate+up+silu, one launch (supersedes gateup split-K)."),
    "ffn_down_kgamma": dict(phase="ffn", group="SHARED",
        status="OPTIMIZED", flag="ATLAS_FFN_DOWN_SPLITK=4",
        note="down_proj split-K=4 (80->320 CTAs; 91->163 GB/s)."),
    "ffn_gate_kgamma": dict(phase="ffn", group="SHARED", status="OPTIMIZED",
        flag="ATLAS_FFN_KGAMMA_M128=1", note="gate GEMM (unfused path)"),
    "ffn_up_kgamma": dict(phase="ffn", group="SHARED", status="OPTIMIZED",
        flag="ATLAS_FFN_KGAMMA_M128=1", note="up GEMM (unfused path)"),
    "ffn_silu_mul_kgamma": dict(phase="ffn", group="SHARED", status="FLOOR", flag="-", note="silu*mul floor"),
    "ffn_gate_up_dual_batch3": dict(phase="ffn", group="SHARED", status="OPTIMIZED", flag="-", note="dual gate/up batched"),
    "ffn_down_batch3": dict(phase="ffn", group="SHARED", status="OPTIMIZED", flag="ATLAS_FFN_DOWN_SPLITK=4", note="down batched"),
    "ffn_silu_mul": dict(phase="ffn", group="SHARED", status="FLOOR", flag="-", note="silu*mul floor"),
    "ffn_gate_up_dual_m1": dict(phase="ffn", group="SHARED", status="OPTIMIZED", flag="-", note="n=1 dual"),
    "ffn_down_m1": dict(phase="ffn", group="SHARED", status="OPTIMIZED", flag="-", note="n=1 down"),
    "ffn_down_silu_m1": dict(phase="ffn", group="SHARED", status="OPTIMIZED", flag="-", note="n=1 down+silu"),
    "ffn_silu_mul_m1": dict(phase="ffn", group="SHARED", status="FLOOR", flag="-", note="n=1 silu floor"),
}

# Present-but-disabled fast paths surfaced by the prefill dispatch line.
# key = the has/kernel boolean that proves the kernel is embedded;
# gate = the raw env flag that would enable it; scope = where it applies.
DISABLED_FASTPATHS = {
    "e2m1_kernel": dict(gate="ATLAS_E2M1_GEMM", scope="prefill-only (W4A4)",
        risk="activation quant -> changes md5; NOT wired to decode/verify FFN",
        lever="prefill TTFT only; quality-gate before ship"),
    "m128_v2_kernel": dict(gate="ATLAS_FFN_M128_V2", scope="prefill-only",
        risk="accumulation order differs -> re-prove md5",
        lever="8-warp shadow of w4a16_gemm_t_m128; A/B vs v1 on prefill GEMM"),
    "fp8_m128_kernel": dict(gate="ATLAS_FFN_PREDEQUANT_FP8", scope="prefill-only",
        risk="~17GB extra weights for 3xNxK FP8 buffers",
        lever="skips per-K dequant phase; memory-cost gated"),
}

PHASE_ORDER = ["ffn", "attention", "ssm", "norm", "head", "other"]


def parse_log(path):
    """Extract KPROF rows, module count, and dispatch fast-path booleans."""
    kprof = OrderedDict()          # label -> (calls, total_us, per_call_us)
    steps = None
    modules = None
    dispatch = {}                  # boolean_name -> True/False
    gates = {}                     # ATLAS_* -> True/False (raw env==1)
    kprof_re = re.compile(
        r"KPROF kernel=(\S+) calls=(\d+) total_us=(\d+) per_call_us=(\d+)")
    sum_re = re.compile(r"KPROF SUMMARY steps=(\d+) kernels=(\d+)")
    mod_re = re.compile(r"Selected kernel target: .*\((\d+) modules\)")

    with open(path, "r", errors="replace") as fh:
        for raw in fh:
            line = strip_ansi(raw)
            m = mod_re.search(line)
            if m:
                modules = int(m.group(1))
            m = sum_re.search(line)
            if m:
                steps = int(m.group(1))
            m = kprof_re.search(line)
            if m:
                # keep the LAST dump (most steps accumulated)
                kprof[m.group(1)] = (int(m.group(2)), int(m.group(3)), int(m.group(4)))
            if "atlas-prefill-ffn" in line or "forward_prefill dispatch" in line:
                for key in ("e2m1_kernel", "m128_v2_kernel", "fp8_m128_kernel",
                            "m128_kernel", "has_transposed", "has_e2m1",
                            "e2m1_fast_path", "fp8_fast_path", "v2_fast_path",
                            "fast_path"):
                    mm = re.search(rf"{key}[=\x1b m]*?(true|false)", line)
                    if mm:
                        dispatch[key] = mm.group(1) == "true"
                for g in re.findall(r"(ATLAS_[A-Z0-9_]+)=(true|false)", line):
                    gates[g[0]] = g[1] == "true"
    return dict(kprof=kprof, steps=steps, modules=modules,
                dispatch=dispatch, gates=gates)


def strip_ansi(s):
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def build_rows(parsed):
    """One row per KPROF kernel with ms/step, %step, classification."""
    kprof = parsed["kprof"]
    steps = parsed["steps"] or 0
    if not kprof or steps <= 0:
        return [], 0.0
    rows = []
    for label, (calls, total_us, per_call_us) in kprof.items():
        ms_step = (total_us / 1000.0) / steps
        kb = KB.get(label, dict(phase="other", group="OTHER",
                                status="UNKNOWN", flag="?", note="unclassified"))
        rows.append(dict(
            kernel=label,
            calls_step=round(calls / steps, 1),
            us_call=per_call_us,
            ms_step=ms_step,
            phase=kb["phase"], group=kb["group"],
            status=kb["status"], flag=kb["flag"], note=kb["note"],
        ))
    total_ms = sum(r["ms_step"] for r in rows)
    for r in rows:
        r["pct"] = (r["ms_step"] / total_ms * 100.0) if total_ms else 0.0
    rows.sort(key=lambda r: r["ms_step"], reverse=True)
    return rows, total_ms


def phase_shape(rows, total_ms):
    agg = OrderedDict((p, 0.0) for p in PHASE_ORDER)
    for r in rows:
        agg[r["phase"]] = agg.get(r["phase"], 0.0) + r["ms_step"]
    out = OrderedDict()
    for p, ms in agg.items():
        if ms > 0:
            out[p] = round(ms / total_ms * 100.0, 1) if total_ms else 0.0
    return out


def render_full(parsed, rows, total_ms):
    out = []
    out.append("=" * 100)
    out.append("AEON-Q36-27B DENSE CHAMPION — KERNEL-BY-KERNEL OPTIMIZATION AUDIT  (INTERNAL — DO NOT PUBLISH)")
    out.append("=" * 100)
    mods = parsed["modules"]
    steps = parsed["steps"]
    out.append(f"kernels embedded: {mods}   |   profiled verify steps: {steps}   |   "
               f"total measured GPU/step: {total_ms:.1f} ms   |   kprof kernels: {len(rows)}")
    out.append("")
    hdr = f"{'#':>2} {'kernel':<30} {'ms/step':>8} {'%':>6} {'/step':>6} {'us/call':>8}  {'status':<10} {'flag':<28} note"
    out.append(hdr)
    out.append("-" * len(hdr))
    for i, r in enumerate(rows, 1):
        out.append(f"{i:>2} {r['kernel']:<30} {r['ms_step']:>8.2f} {r['pct']:>5.1f}% "
                   f"{r['calls_step']:>6} {r['us_call']:>8}  {r['status']:<10} {r['flag']:<28} {r['note']}")
    out.append("")

    # ---- opportunity list ----
    out.append("TOP OPPORTUNITIES (actionable levers, ranked)")
    out.append("-" * 100)
    opps = []
    # disabled fast paths present in the build
    disp = parsed["dispatch"]
    gates = parsed["gates"]
    for key, meta in DISABLED_FASTPATHS.items():
        present = disp.get(key, False) or disp.get(key.replace("_kernel", ""), False)
        gate_on = gates.get(meta["gate"], False)
        if present and not gate_on:
            opps.append((f"[DISABLED] {meta['gate']}", meta["scope"],
                         f"{meta['lever']}  (risk: {meta['risk']})"))
    # kernels classified DISABLED with a hot footprint
    for r in rows:
        if r["status"] == "DISABLED":
            opps.append((f"[DISABLED] {r['flag']}", f"{r['ms_step']:.2f} ms/step ({r['pct']:.1f}%)",
                         r["note"]))
        if r["status"] == "FALLBACK":
            opps.append((f"[FALLBACK] {r['kernel']}", f"{r['ms_step']:.2f} ms/step ({r['pct']:.1f}%)",
                         "MISSING kernel -> slow generic. Build/embed the fast variant. " + r["note"]))
    if not opps:
        out.append("  (none — every hot kernel is OPTIMIZED or at FLOOR)")
    for lever, where, detail in opps:
        out.append(f"  {lever:<34} {where:<24} {detail}")
    out.append("")
    out.append("STATUS LEGEND: OPTIMIZED=on best available fast path | DISABLED=faster variant present but flag off |")
    out.append("               FALLBACK=needed kernel MISSING -> slow generic | FLOOR=at bandwidth/compute floor (banked)")
    return "\n".join(out)


def render_public(parsed, rows, total_ms):
    """Sanitized health snippet — NO proprietary flags/kernel-names/timings."""
    mods = parsed["modules"] or 0
    used = len(rows)
    # A hot kernel on FALLBACK (missing) is the only unhealthy state we surface.
    hot = [r for r in rows if r["pct"] >= 3.0]
    missing = sum(1 for r in hot if r["status"] == "FALLBACK")
    healthy = missing == 0
    # generic phase shape, top-5, rounded, GENERIC names only
    shape = phase_shape(rows, total_ms)
    top5 = OrderedDict(list(shape.items())[:5])
    snippet = OrderedDict()
    snippet["kernel_audit"] = OrderedDict([
        ("kernels_embedded", mods),
        ("kernels_used", used),
        ("kernels_missing_fallback", missing),
        ("time_by_phase_pct", top5),
        ("fast_paths_healthy", healthy),
    ])
    return json.dumps(snippet, indent=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True, help="path to a serve log (FULL_PROFILE for timing)")
    ap.add_argument("--mode", choices=["full", "public"], default="full",
                    help="full=internal audit (default); public=sanitized JSON health snippet")
    args = ap.parse_args()

    try:
        parsed = parse_log(args.log)
    except FileNotFoundError:
        sys.exit(f"error: log not found: {args.log}")

    rows, total_ms = build_rows(parsed)
    if not rows:
        if args.mode == "public":
            # still emit counts if we at least saw the module line
            print(json.dumps({"kernel_audit": {
                "kernels_embedded": parsed["modules"] or 0,
                "kernels_used": 0, "kernels_missing_fallback": 0,
                "time_by_phase_pct": {}, "fast_paths_healthy": None,
                "note": "no KPROF timing (serve without ATLAS_FULL_PROFILE=1 or <150 verify steps)"}},
                indent=2))
        else:
            sys.exit("error: no KPROF rows found. Serve with ATLAS_FULL_PROFILE=1 and "
                     "drive >150 verify steps (~1500 tokens) so the 150-step dump fires.")
        return

    if args.mode == "public":
        print(render_public(parsed, rows, total_ms))
    else:
        print(render_full(parsed, rows, total_ms))


if __name__ == "__main__":
    main()
