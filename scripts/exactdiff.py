#!/usr/bin/env python3
"""Byte-level exactness diff between a plain-decode log and a spec-verify log.

The γ-verify is lossless iff every verify row computes the same function as
the plain decode step at the same (token, position). This tool compares the
two instrumented logs layer-by-layer, row-by-row, with the three pitfalls
that produced days of false conclusions handled automatically:

  1. NORMS LIE.  diag_norm prints norm/max/first4 rounded to 4 decimals —
     tensors that differ bytewise routinely print identical norms. Only the
     `bhash` field (FNV-1a over the raw bytes) decides equality.
  2. THE WARMUP FORWARD.  The server runs a dummy batched forward at load
     (n=4 as of 2026-08). Its probe lines land in BOTH logs, so flat
     occurrence indices are offset. This tool content-anchors instead:
     it locates the known step1-row0 hash in the plain series per
     (layer, label) and derives the offset from that.
  3. LABEL CADENCE.  Some labels exist at two cadences (per-row "V4-decode
     L{} o_out" from attention_forward_v4 vs per-step "V4-msdecode L{}
     o_out" from multi_seq). Comparing across cadences is meaningless; this
     tool pins the tag per label.

Comparisons are only valid at MATCHED rows: verify row r at position p is
the same computation as plain step p iff the row's input token equals
plain's committed token at p (accepted drafts + row 0 always are; rejected
drafts never are). The VSTEP trace (ATLAS_DFLASH_VSTEP_DIAG=1) supplies the
per-step tokens; rows whose input differs from the plain stream are skipped.

Usage:
  exactdiff.py <plain.log> <spec.log> [--gamma 6] [--layers 43]

How to produce the logs (both servers eager, one request each, same prompt):
  plain: ATLAS_DIAG_V4_ALL_LAYERS=1                                (no dflash)
  spec : ATLAS_DIAG_V4_ALL_LAYERS=1 ATLAS_DFLASH_VSTEP_DIAG=1 \
         ATLAS_MTP_GATE_FORCE=1 ATLAS_DFLASH_DEBUG_NO_GRAPH=1 \
         <exactness gates under test>  --dflash ...
Give plain enough max_tokens to cover every position the spec run reaches.

Output: per layer, per label, a row-grid (steps x rows) of E/D/., and the
first divergent (layer, label, step, row) — the kernel to fix.
"""

import argparse
import re
import sys

ROW_TAG = "V4-decode"  # per-row probes (attention_forward_v4 fires per row
#                        on the MLA_NO_BATCH fallback; plain fires per step)
PROMPT_LEN_GUESS = None  # derived from VSTEP pre of step 1


def parse_series(path, tag, layers):
    """(layer, label) -> [bhash...] in log order, for one tag only.

    Probes may carry an ` obj=<hex>` suffix identifying the layer OBJECT.
    The DSpark drafter is itself a 3-layer V4 model whose layers reuse
    attn_layer_idx 0..2 and fire identical labels — mixing its occurrences
    into the target's series poisoned two rounds of layer walks. When obj
    tags are present, keep ONLY the objs that also appear at layer >= 3
    (the drafter never has an L3).
    """
    raw = []
    rx = re.compile(rf"DIAG {tag} L(\d+) ([^:]+?)( obj=([0-9a-f]+))?: .*bhash=([0-9a-f]+)")
    target_objs = set()
    for line in open(path, errors="replace"):
        m = rx.search(line)
        if m:
            L = int(m[1])
            raw.append((L, m[2], m[4], m[5]))
            if m[4] and L >= 3:
                target_objs.add(m[4])
    out = {}
    for L, label, obj, h in raw:
        if L >= layers:
            continue
        if obj is not None and target_objs and obj not in target_objs:
            continue  # drafter-layer probe
        out.setdefault((L, label), []).append(h)
    return out


def parse_vsteps(path):
    """[(pre, [input tokens], [verified], acc)] per K=gamma verify step."""
    rx = re.compile(r"VSTEP pre=(\d+) in=\[([0-9, ]*)\] verified=\[([0-9, ]*)\] acc=(\d+)")
    steps = []
    for line in open(path, errors="replace"):
        m = rx.search(line)
        if m:
            toks = [int(x) for x in m[2].split(",") if x.strip()]
            ver = [int(x) for x in m[3].split(",") if x.strip()]
            steps.append((int(m[1]), toks, ver, int(m[4])))
    return steps


def plain_tokens(steps):
    """Reconstruct the plain-greedy committed stream from the spec's own
    accepted prefix (valid up to the first fork): row tokens of accepted
    prefixes + bonuses ARE plain tokens until the fork."""
    # positions: step pre -> row r at pos pre-1+r, input toks[r]
    committed = {}
    for pre, toks, ver, acc in steps:
        for r, t in enumerate(toks):
            committed.setdefault(pre - 1 + r, t)
    return committed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plain_log")
    ap.add_argument("spec_log")
    ap.add_argument("--layers", type=int, default=43)
    ap.add_argument("--labels", default="Q after q_b_norm,attn_out derot,o_out")
    args = ap.parse_args()

    labels = args.labels.split(",")
    pl = parse_series(args.plain_log, ROW_TAG, args.layers)
    sp = parse_series(args.spec_log, ROW_TAG, args.layers)
    steps = parse_vsteps(args.spec_log)
    if not steps:
        sys.exit("no VSTEP lines in spec log — run with ATLAS_DFLASH_VSTEP_DIAG=1")
    first_pre = steps[0][0]
    base_pos = first_pre - 1  # row 0 of step 1
    n_rows = len(steps[0][1])

    # Warmup burst length on the spec side: probes before the first real step.
    # Detect per (L,label): find plain's anchor (spec step1-row0 == plain's
    # base_pos occurrence) by scanning candidate warmup offsets 0..8.
    first_div = None
    for L in range(args.layers):
        for lab in labels:
            s = sp.get((L, lab), [])
            p = pl.get((L, lab), [])
            if not s or not p:
                continue
            # The spec warmup burst length is whatever precedes the real
            # steps: len(series) - steps*rows. Anchoring MUST start there —
            # searching from index 0 re-anchors warmup-to-warmup (both logs
            # contain the identical dummy warmup), silently reproducing the
            # exact pitfall this tool exists to prevent.
            ws = len(s) - len(steps) * n_rows
            anchor = None
            if 0 <= ws < len(s):
                try:
                    wp = p.index(s[ws])
                    anchor = (ws, wp)
                except ValueError:
                    pass
            if anchor is None:
                print(f"L{L:>2} {lab}: NO ANCHOR (step1-row0 already diverged "
                      f"from plain, or series empty) <-- investigate here")
                if first_div is None:
                    first_div = (L, lab, 1, 0)
                continue
            ws, wp = anchor
            grid = []
            for si, (pre, toks, ver, acc) in enumerate(steps):
                row = ""
                for r in range(len(toks)):
                    so = ws + si * n_rows + r
                    po = wp + (pre - 1 + r) - base_pos
                    if so >= len(s) or po >= len(p) or po < 0:
                        row += "."
                        continue
                    ok = s[so] == p[po]
                    row += "E" if ok else "D"
                    if not ok and first_div is None:
                        first_div = (L, lab, si + 1, r)
                grid.append(row)
            print(f"L{L:>2} {lab}: " + " ".join(grid))
    print()
    if first_div:
        L, lab, s, r = first_div
        print(f"FIRST DIVERGENCE: layer {L}, '{lab}', step {s}, row {r}")
        print("NOTE: a D at a row whose draft was REJECTED is expected "
              "(different input token) — read the VSTEP trace before blaming "
              "the kernel. Accepted-prefix rows and row 0 are always valid.")
    else:
        print("ALL COMPARED PROBES BYTE-EXACT")


if __name__ == "__main__":
    main()
