#!/usr/bin/env python3
"""Analyse an ATLAS_MOE_GATE_HIST=1 dump — the measurement adaptive top-K lives on.

Reads the JSONL written by `crates/spark-model/src/layers/moe/gate_hist.rs` and
answers, per routing kind and per layer:

  * mean gate-mass fraction at each WEIGHT rank (rank 0 = largest weight)
  * P(smallest slot < t) and the fraction of DROPPABLE slots below t, for
    t in 0.01 / 0.02 / 0.05 / 0.10
  * the byte and ms/step saving each threshold implies

"Droppable" excludes the arg-max slot, exactly as `moe_adaptive_topk_prune`
does — a token always reaches at least one routed expert.

Usage:
  python3 scripts/moe_gate_hist.py /tmp/gate_hist.jsonl [--by-layer]
  python3 scripts/moe_gate_hist.py /tmp/gate_hist.jsonl \\
      --layers 43 --expert-mb 13.37 --achieved-gbs 192 --step-ms 45.3

Defaults are the DeepSeek-V4-Flash-162B / GB10 numbers from
docs/DECODE-WATERFALL-2026-08-10.md.
"""
import argparse
import collections
import json
import sys

THRESHOLDS = (0.01, 0.02, 0.05, 0.10)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--by-layer", action="store_true")
    ap.add_argument("--kind", default="gate", choices=("gate", "hash", "all"),
                    help="hash-routed layers pick experts from a static table; "
                         "their weight distribution is a different population")
    ap.add_argument("--layers", type=int, default=43,
                    help="MoE layers per token (fires per token per kind)")
    ap.add_argument("--expert-mb", type=float, default=13.37,
                    help="bytes streamed per routed expert per layer, MB")
    ap.add_argument("--achieved-gbs", type=float, default=192.0,
                    help="measured MoE expert-GEMV bandwidth, GB/s")
    ap.add_argument("--step-ms", type=float, default=45.3)
    args = ap.parse_args()

    # rank -> [sum_mass, count]; plus per-threshold counters
    agg = collections.defaultdict(
        lambda: {"rank": collections.defaultdict(lambda: [0.0, 0]),
                 "fires": 0, "slots": 0, "droppable": 0,
                 "below_min": [0] * len(THRESHOLDS),
                 "below_slots": [0] * len(THRESHOLDS)}
    )

    with open(args.path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if args.kind != "all" and r.get("kind") != args.kind:
                continue
            sm = r["sorted_mass"]
            keys = ["ALL"] + ([f"L{r['layer']:02d}"] if args.by_layer else [])
            for key in keys:
                a = agg[key]
                a["fires"] += 1
                a["slots"] += len(sm)
                a["droppable"] += max(0, len(sm) - 1)
                for i, m in enumerate(sm):
                    a["rank"][i][0] += m
                    a["rank"][i][1] += 1
                for j, t in enumerate(THRESHOLDS):
                    if sm[-1] < t:
                        a["below_min"][j] += 1
                    a["below_slots"][j] += sum(1 for m in sm[1:] if m < t)

    if not agg:
        print(f"no records of kind={args.kind} in {args.path}", file=sys.stderr)
        return 2

    # Bytes per dropped slot per TOKEN, and what that is worth in ms and tok/s.
    mb_per_slot_token = args.expert_mb * args.layers
    ms_per_slot = mb_per_slot_token / 1e3 / args.achieved_gbs * 1e3
    base_tps = 1e3 / args.step_ms

    for key in sorted(agg, key=lambda k: (k != "ALL", k)):
        a = agg[key]
        ranks = sorted(a["rank"])
        print(f"\n=== {key}  kind={args.kind}  fires={a['fires']} ===")
        print("  mean gate-mass by weight-rank (rank0 = largest):")
        print("   " + "  ".join(
            f"r{i}={a['rank'][i][0] / a['rank'][i][1]:.4f}" for i in ranks))
        print(f"  per-slot cost: {mb_per_slot_token:.0f} MB/token = "
              f"{ms_per_slot:.2f} ms/step at {args.achieved_gbs:.0f} GB/s")
        print("  thr   P(min<thr)  droppable-slots<thr  mean-dropped/fire   "
              "GB/token   ms/step   tok/s")
        for j, t in enumerate(THRESHOLDS):
            p_min = a["below_min"][j] / a["fires"]
            frac = a["below_slots"][j] / max(1, a["droppable"])
            dropped = a["below_slots"][j] / a["fires"]
            gb = dropped * mb_per_slot_token / 1e3
            ms = dropped * ms_per_slot
            tps = 1e3 / max(1e-6, args.step_ms - ms)
            print(f"  {t:<5.2f} {p_min:10.3f}  {frac:19.3f}  {dropped:17.3f}   "
                  f"{gb:8.3f}   {ms:7.2f}   {tps:5.2f} (+{tps - base_tps:.2f})")
    print("\nThe tok/s column is a BYTE-MODEL projection, not a measurement: it "
          "assumes the dropped slot's bytes were the only cost and the step is "
          "otherwise unchanged. Confirm with decode_ab_probe.py, and read the "
          "quality columns in docs/ADAPTIVE-TOPK.md before believing any of it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
