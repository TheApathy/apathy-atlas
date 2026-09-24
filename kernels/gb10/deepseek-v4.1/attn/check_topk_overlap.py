#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Per-layer TOP-K OVERLAP of an Atlas run against a Python oracle capture (runK_32k: the first
capture where top-512 truly prunes -- ~16K compressed positions at L2, 32K at L20+).

    check_topk_overlap.py <oracle ref dir> <atlas tap dir> [layers=2,20,24]

Both dirs hold `L{ll}.topk.{occ:03}.bin` = [t, 512] int64, -1 padded (occurrences in pass order:
L2/L20 = the encoder chunks then decode steps; L24 = the replay then decode steps). Per layer and
occurrence: overlap = |ours & oracle| / |oracle valid| per row -> mean, min, rows < 0.99, exact rows.

If the Atlas dir also holds `L{ll}.index_score_rows.{occ:03}.bin` (ATLAS_DSV41_TAP_INDEX_SCORE=1:
fp32 score rows for every 64th row of the pass, [ceil(t/64), n_pad]), every oracle index we MISSED
on those rows is classified by OUR score: its rank and its margin below our 512th value --
  tie      margin == 0 (an exact boundary tie: either choice is the reference's own ambiguity)
  near     margin <= 2^-16 x |512th| (a few bf16/fp32 ulps of the head sum: numerics, not logic)
  real     anything else (a ranking difference: the thing to chase)
PASS criteria, PRE-REGISTERED 2026-09-23 before runK_32k existed (free-running Atlas vs the oracle,
so upstream drift is included; it compounds with depth and L24 also inherits L20's candidate pool):
  L02  mean >= 0.999, no row < 0.95
  L20  mean >= 0.99,  rows < 0.90 <= 1% of rows,  'real' <= 10% of classified misses
  L24  mean >= 0.98,  rows < 0.90 <= 2% of rows,  'real' <= 15% of classified misses (pruned pool)
L20/L24 REQUIRE the classification (index_score_rows taps): without it they are UNCLASSIFIED = FAIL.
Every 'real' miss is listed with its rank either way.
"""
import os
import sys

import numpy as np

TOPK = 512
SAMPLE = 64
# layer -> (min mean overlap, max share of rows below 0.90, max 'real' share of classified misses)
BANDS = {20: (0.99, 0.01, 0.10), 24: (0.98, 0.02, 0.15)}


def load_i64(path, width=TOPK):
    a = np.fromfile(path, dtype=np.int64)
    return a.reshape(-1, width)


def occurrences(d, layer, name):
    out, occ = [], 0
    while True:
        p = os.path.join(d, f"L{layer:02d}.{name}.{occ:03d}.bin")
        if not os.path.exists(p):
            return out
        out.append(p)
        occ += 1


def classify(score_row, ours_row, miss):
    valid = ours_row[ours_row >= 0]
    kth = score_row[valid].min() if len(valid) else np.inf
    res = []
    for m in miss:
        s = score_row[m]
        margin = float(kth - s)
        rank = int((score_row > s).sum())
        kind = "tie" if margin == 0 else ("near" if margin <= abs(kth) * 2.0 ** -16 else "real")
        res.append((kind, rank, margin))
    return res


def main():
    ref, ours = sys.argv[1], sys.argv[2]
    layers = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "2,20,24").split(",")]
    fail = 0
    for l in layers:
        ro, oo = occurrences(ref, l, "topk"), occurrences(ours, l, "topk")
        if not ro or not oo:
            print(f"L{l:02d}: oracle {len(ro)} / atlas {len(oo)} topk occurrences -- MISSING")
            fail += 1
            continue
        if len(ro) != len(oo):
            print(f"L{l:02d}: occurrence count differs (oracle {len(ro)}, atlas {len(oo)}); comparing the first {min(len(ro), len(oo))}")
        kinds = {"tie": 0, "near": 0, "real": 0}
        reals = []
        all_ov = []
        for occ, (rp, op) in enumerate(zip(ro, oo)):
            R, O = load_i64(rp), load_i64(op)
            if R.shape != O.shape:
                print(f"  L{l:02d} occ {occ}: shape oracle {R.shape} vs atlas {O.shape} -- SKIPPED")
                fail += 1
                continue
            ov = np.empty(len(R))
            for r in range(len(R)):
                rv, ovv = R[r][R[r] >= 0], O[r][O[r] >= 0]
                if len(rv) == 0:   # nothing visible yet (L2 row 0): agreement iff we chose nothing too
                    ov[r] = float(len(ovv) == 0)
                else:
                    ov[r] = len(np.intersect1d(rv, ovv, assume_unique=True)) / len(rv)
            all_ov.append(ov)
            line = (f"  L{l:02d} occ {occ:3d} t={len(R):5d} valid/row={int((R >= 0).sum(1).mean()):3d}: overlap mean {ov.mean():.5f} "
                    f"min {ov.min():.4f} | rows<0.99 {int((ov < 0.99).sum())} | exact {int((ov == 1).sum())}/{len(R)}")
            sp = os.path.join(ours, f"L{l:02d}.index_score_rows.{occ:03d}.bin")
            if os.path.exists(sp):
                srows = np.fromfile(sp, dtype=np.float32)
                n_s = (len(R) + SAMPLE - 1) // SAMPLE
                srows = srows.reshape(n_s, -1)
                c = {"tie": 0, "near": 0, "real": 0}
                for i in range(n_s):
                    r = i * SAMPLE
                    miss = np.setdiff1d(R[r][R[r] >= 0], O[r][O[r] >= 0])
                    for kind, rank, margin in classify(srows[i], O[r], miss):
                        c[kind] += 1
                        if kind == "real":
                            reals.append((occ, r, rank, margin))
                for k in c:
                    kinds[k] += c[k]
                line += f" | sampled-row misses: tie {c['tie']} near {c['near']} real {c['real']}"
            print(line)
        if all_ov:
            ov = np.concatenate(all_ov)
            print(f"L{l:02d} TOTAL {len(ov)} rows: overlap mean {ov.mean():.5f} min {ov.min():.4f} rows<0.99 {int((ov < 0.99).sum())} "
                  f"| misses tie {kinds['tie']} near {kinds['near']} real {kinds['real']}")
            for occ, r, rank, margin in reals[:20]:
                print(f"    real miss: occ {occ} row {r}: our rank {rank} (512 kept), margin {margin:.3e}")
            n_cls = sum(kinds.values())
            real_share = kinds["real"] / n_cls if n_cls else 0.0
            low = (ov < 0.90).mean()
            if l == 2:
                ok = ov.mean() >= 0.999 and ov.min() >= 0.95
                band = "mean >= 0.999, no row < 0.95"
            elif l in BANDS:
                mean_min, low_max, real_max = BANDS[l]
                classified = any(os.path.exists(os.path.join(ours, f"L{l:02d}.index_score_rows.{o:03d}.bin")) for o in range(len(oo)))
                ok = classified and ov.mean() >= mean_min and low <= low_max and real_share <= real_max
                band = (f"mean >= {mean_min}, rows<0.90 <= {low_max:.0%} (got {low:.2%}), real <= {real_max:.0%} of classified "
                        f"(got {real_share:.1%} of {n_cls})" + ("" if classified else " -- UNCLASSIFIED: no index_score_rows taps"))
            else:
                ok, band = True, "reported only"
            print(f"L{l:02d} PRE-REGISTERED ({band}): {'PASS' if ok else 'FAIL'}")
            fail += not ok
    print("TOPK OVERLAP", "PASS" if fail == 0 else f"FAIL ({fail})")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
