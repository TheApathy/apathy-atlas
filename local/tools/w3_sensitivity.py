#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""W3 sensitivity ranking for the Atlas mixed-precision FFN lane.

Ranks every layer's FFN tensors by W4->W3 requantization error (weight-space
proxy) and emits candidate ATLAS_FFN_W3_LAYERS sets (safest 8 / 16 / 24).

METHOD ------------------------------------------------------------------------
For each layer i and proj in {gate_proj, up_proj, down_proj}:
  1. Dequantize the shipped NVFP4 weight to f32 (the W4 values ARE the
     reference — the engine never sees anything more precise at decode).
  2. Requantize to the W3 format with the exact repack_w3.py simulator
     (per-group optimal e4m3 scale, nearest-level codes, runtime-exact
     dequant reconstruction).
  3. relMSE(proj) = ||W3 - W4||^2 / ||W4||^2.

Layer score = output-dim-weighted mean of the three projection relMSEs,
with down_proj OVERWEIGHTED 2x: its output feeds the residual stream
directly (gate/up errors are partially absorbed by the SiLU gating and the
down projection's contraction). This is a weight-space proxy for true
activation sensitivity; an activation-weighted refinement needs calibration
activations, which the CPU-only window doesn't have. The ABBA eval gate is
the ground-truth quality check downstream.

Rows are subsampled (--row-stride, default 4) — relMSE is a per-group
statistic over millions of groups, so 1/4 sampling changes layer scores by
<1e-4 while cutting runtime 4x.

OUTPUT: ranked table (safest first) + ready-to-paste env strings, optional
JSON (--out).
"""

import argparse
import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from repack_w3 import PROJS, SafetensorsReader, layer_prefix, repack_tensor  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/flocka/models/AEON-Q36-27B-Full")
    ap.add_argument("--row-stride", type=int, default=4)
    ap.add_argument("--factors", type=int, default=9)
    ap.add_argument("--out", default=None, help="JSON output path")
    ap.add_argument("--down-weight", type=float, default=2.0)
    args = ap.parse_args()

    reader = SafetensorsReader(args.model)
    layers = sorted(
        {int(k.split(".layers.")[1].split(".")[0])
         for k in reader.header if ".mlp.gate_proj.weight" in k})
    factors = np.linspace(0.75, 1.25, args.factors)

    per_layer = {}
    for i in layers:
        row = {}
        num = den = 0.0
        for proj in PROJS:
            r = repack_tensor(reader, layer_prefix(i, proj), factors,
                              row_stride=args.row_stride)
            row[proj] = {"rel_mse": r["rel_mse"], "max_abs_err": r["max_abs_err"]}
            w = (args.down_weight if proj == "down_proj" else 1.0) * r["n"] * r["k"]
            num += w * r["rel_mse"]
            den += w
        row["score"] = num / den
        per_layer[i] = row
        print(f"layer {i:2d}: score={row['score']:.5f} "
              + " ".join(f"{p}={row[p]['rel_mse']:.5f}" for p in PROJS), flush=True)

    ranked = sorted(per_layer, key=lambda i: per_layer[i]["score"])
    print("\n== RANKED (safest first) ==")
    print(" ".join(str(i) for i in ranked))
    for n in (8, 16, 24, 32):
        sel = sorted(ranked[:n])
        print(f'\nsafest-{n}: ATLAS_FFN_W3_LAYERS="{",".join(map(str, sel))}"')

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"per_layer": {str(k): v for k, v in per_layer.items()},
                       "ranked_safest_first": ranked}, f, indent=2)
        print(f"\nstats -> {args.out}")


if __name__ == "__main__":
    main()
