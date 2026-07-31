#!/usr/bin/env python3
"""Check a decode_bench.py run against this branch's published reference hashes.

    python3 bench/laguna/check_repro.py --arm gproj bench/laguna/ab/gproj.json

Exits 0 only on a genuine pass. The point of this script is that "my numbers
look about right" is not a reproduction: two stacks can agree on tok/s to a
percent and still be emitting different tokens. Comparing completion hashes is
the only cheap check that actually constrains the computation.

Three per-row states, deliberately not two:

  MATCH          the hash equals the published one
  MISMATCH       it does not, on a row we publish as stable -- a real failure
  KNOWN-UNSTABLE the row is published as nondeterministic and the hash is one
                 of the variants we have seen (or a new one, reported as such)

A row that is legitimately unstable must not be reported the same way as a row
that broke, and it must not be quietly dropped either -- both of those turn a
known limitation into either a false alarm or a false pass.

The script also refuses to pass on a short comparison. A run missing half its
prompts would otherwise report zero mismatches and read exactly like a clean
one: a count of zero must never be readable as "nothing wrong".

Exit codes:  0 pass  1 mismatch on a stable row  2 incomplete run
             3 nothing to compare (inconclusive)
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", help="a --json-out file written by decode_bench.py")
    ap.add_argument("--arm", required=True,
                    help="which published configuration this run is: serial, nogproj or gproj. "
                         "Required, because the arms legitimately disagree with each other -- "
                         "guessing from the tag would silently check against the wrong one.")
    ap.add_argument("--reference", default=os.path.join(HERE, "reference_hashes.json"))
    args = ap.parse_args()

    ref = json.load(open(args.reference))
    if args.arm not in ref["arms"]:
        sys.exit(f"FATAL: unknown arm '{args.arm}'; published arms are: "
                 f"{', '.join(sorted(ref['arms']))}")
    expected = ref["arms"][args.arm]["prompts"]

    run = json.load(open(args.results))
    # This branch's decode_bench.py writes "results"; the qwen sibling writes
    # "rows". Accept
    # either rather than failing on a key name, but fail loudly if neither is
    # present -- an empty list here would sail through as zero mismatches.
    rows = run.get("rows", run.get("results"))
    if not rows:
        # Exit 3, not 1. "I could not check" and "I checked and it failed" are
        # different outcomes and a caller that treats them alike will eventually
        # retry the wrong one.
        print(f"FATAL: {args.results} contains no per-prompt rows -- nothing to check.\n"
              "INCONCLUSIVE, which is not a pass and not a mismatch.", file=sys.stderr)
        return 3
    got = {r["name"]: r["hash"] for r in rows}

    n_match = n_mismatch = n_unstable = n_new = 0
    missing = []
    print(f"=== checking {os.path.basename(args.results)} against arm '{args.arm}' ===")
    for name, spec in expected.items():
        if name not in got:
            missing.append(name)
            print(f"  {name:<12} MISSING from this run")
            continue
        h = got[name]
        if spec.get("stable", True):
            if h == spec["sha"]:
                n_match += 1
                print(f"  {name:<12} MATCH           {h}")
            else:
                n_mismatch += 1
                print(f"  {name:<12} MISMATCH        {h}  (published {spec['sha']})")
        else:
            if h in spec.get("observed", []):
                n_unstable += 1
                print(f"  {name:<12} KNOWN-UNSTABLE  {h}  (a variant we have seen)")
            else:
                n_new += 1
                print(f"  {name:<12} KNOWN-UNSTABLE  {h}  ** new variant, not seen before")

    print("  " + "-" * 56)
    total = len(expected)
    print(f"  {n_match}/{total} stable rows match, {n_mismatch} mismatched, "
          f"{n_unstable + n_new} unstable rows ({n_new} showing a new variant), "
          f"{len(missing)} missing")

    if missing:
        print("\nFAIL: the run did not cover every published prompt, so a clean "
              "result here would be a statement about the rows that ran, not "
              "about the suite.")
        return 2
    if n_mismatch:
        print("\nFAIL: at least one row we publish as deterministic produced "
              "different tokens. Check the serve asserts first (kernel target, "
              "gamma, KV dtype) -- a config difference is far more likely than "
              "a numerics one.")
        return 1

    print("\nPASS: every deterministic row reproduces this branch's published output.")
    if n_new:
        print("      (An unstable row showed a variant we had not recorded. That is "
              "expected behaviour for that row, not a failure -- see README section 5.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
