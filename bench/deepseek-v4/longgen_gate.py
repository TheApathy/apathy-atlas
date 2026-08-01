#!/usr/bin/env python3
"""Long-generation quality gate for reduced-precision decode configs.

The short coherence + GSM8K probes in `quality_probe.py` do NOT catch the
failure mode that `--lm-head-dtype fp8|nvfp4` warns about: low-margin argmax
flips in the final vocab projection COMPOUND over long structured generation,
so a config can look perfect on 128-token answers and still derail completely
at 800 tokens. Atlas measured exactly that on Qwen3.6-35B-A3B (bf16 10/10,
fp8 and nvfp4 both collapsed). The same warning applies to any change that
lowers precision on the decode path, not just the lm-head.

So this gate asks for LONG, STRUCTURED output and checks that structure holds
all the way to the end:

  * the requested scaffolding is actually present (all N sections/steps),
  * the tail is still on-task rather than degenerate,
  * no token-level degeneration (a short phrase repeated forever), which is
    what an argmax flip cascade looks like in practice,
  * generation ran long enough for the failure to have had a chance to appear.

  python3 bench/deepseek-v4/longgen_gate.py --port 8977

Exit 0 iff every task passes.

ABSOLUTE vs RELATIVE: some models fail tasks here at FULL precision (measured
2026-08-01 on DeepSeek-V4-Flash-162B REAP: the BF16 head passes only 1/4 —
it EOSes the arithmetic chain at 49 tokens and ignores the exact-header
instruction). Against such a model the absolute verdict gates the MODEL, not
the precision change. For that, record the full-precision behavior once and
gate the reduced-precision config on REGRESSIONS only:

  python3 longgen_gate.py --port 8977 --save-baseline longgen_bf16.json   # bf16 serve
  python3 longgen_gate.py --port 8977 --baseline longgen_bf16.json        # fp8 serve
"""
import argparse, json, re, sys, urllib.request

# Each task: name, prompt, min tokens we insist on seeing, and a structural
# check over the completed text.
TASKS = [
    (
        "numbered-20",
        "List exactly 20 distinct programming languages. Number them 1. through 20., "
        "and give one sentence about each. Do not stop early.",
        # Every marker 1..20 must appear at the start of a line.
        lambda t: all(re.search(rf"(?m)^\s*{i}[.)]", t) for i in range(1, 21)),
        "all 20 numbered items present",
    ),
    (
        "structured-essay",
        "Write a technical article about how a CPU cache hierarchy works. "
        "Use exactly these four markdown section headers, in this order: "
        "## Overview, ## L1 and L2, ## Cache Coherence, ## Conclusion. "
        "Write at least three paragraphs under each header.",
        lambda t: [m.group(1) for m in re.finditer(r"(?m)^##\s*(.+?)\s*$", t)][:4] == [
            "Overview", "L1 and L2", "Cache Coherence", "Conclusion"],
        "all four headers, in order",
    ),
    (
        "long-chain-arithmetic",
        "Start with the number 3. Apply these steps one at a time, showing each result "
        "on its own line as 'Step K: <value>'. Step 1: multiply by 2. Step 2: add 7. "
        "Step 3: multiply by 3. Step 4: subtract 5. Step 5: multiply by 4. "
        "Step 6: add 100. Step 7: divide by 2. Step 8: subtract 36. "
        "Then state 'Final: <value>'.",
        # 3*2=6, +7=13, *3=39, -5=34, *4=136, +100=236, /2=118, -36=82
        lambda t: re.search(r"Final:\s*\**\s*82\b", t) is not None,
        "correct final value 82 after an 8-step chain",
    ),
    (
        "code-block",
        "Write a complete, runnable Python module that implements a fixed-size LRU cache "
        "class with get/put, plus a __main__ block that exercises it. Put it in a single "
        "```python fenced code block. Include docstrings and type hints.",
        lambda t: (t.count("```") >= 2 and "class" in t and "def get" in t
                   and "def put" in t and "__main__" in t),
        "one closed fenced block with a complete class",
    ),
]

# A run that degenerates emits the same short phrase over and over. Detect it
# rather than trusting the structural check, which a repeated header can fool.
def degenerate(text: str) -> str | None:
    words = text.split()
    if len(words) < 60:
        return None
    for n in (4, 6, 10):
        grams = [" ".join(words[i:i + n]) for i in range(len(words) - n)]
        if not grams:
            continue
        top, count = max(((g, grams.count(g)) for g in set(grams)), key=lambda kv: kv[1])
        if count >= max(8, len(grams) // 8):
            return f"{n}-gram {top!r} repeats {count}x"
    return None


def call(host, port, model, prompt, max_tokens):
    body = json.dumps({"model": model,
                       "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0.0,
                       "stream": False}).encode()
    req = urllib.request.Request(f"http://{host}:{port}/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1200) as r:
        d = json.load(r)
    ch = d["choices"][0]
    return ch["message"]["content"], d.get("usage", {}).get("completion_tokens", 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8977)
    ap.add_argument("--model", default="/home/flocka/models/DeepSeek-V4-Flash-162B")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--min-tokens", type=int, default=200,
                    help="a task that stops this early never exercised the failure mode")
    ap.add_argument("--save-baseline", metavar="FILE",
                    help="record per-task pass/fail to FILE (run against the "
                         "full-precision serve) instead of gating")
    ap.add_argument("--baseline", metavar="FILE",
                    help="gate on REGRESSIONS vs a recorded full-precision "
                         "baseline instead of absolute task success")
    a = ap.parse_args()

    baseline = None
    if a.baseline:
        with open(a.baseline) as f:
            baseline = json.load(f)

    print(f"== long-generation gate (max_tokens={a.max_tokens}, greedy) ==")
    results = {}
    failures = 0
    for name, prompt, check, what in TASKS:
        try:
            out, ntok = call(a.host, a.port, a.model, prompt, a.max_tokens)
        except Exception as e:
            print(f"  [FAIL] {name}: request error: {e}")
            results[name] = {"passed": False, "tokens": 0, "why": f"request error: {e}"}
            failures += 1
            continue

        why = None
        if ntok < a.min_tokens:
            why = f"only {ntok} tokens generated (< {a.min_tokens}); gate not exercised"
        elif (d := degenerate(out)) is not None:
            why = f"degenerate output: {d}"
        elif not check(out):
            why = f"structure check failed: expected {what}"

        results[name] = {"passed": why is None, "tokens": ntok, "why": why}
        if why:
            failures += 1
            print(f"  [FAIL] {name} ({ntok} tok): {why}")
            print(f"         tail: ...{out[-160:].strip()!r}")
        else:
            print(f"  [PASS] {name} ({ntok} tok): {what}")

    if a.save_baseline:
        with open(a.save_baseline, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  baseline recorded to {a.save_baseline} "
              f"({len(TASKS) - failures}/{len(TASKS)} pass at this precision)")
        sys.exit(0)

    if baseline is not None:
        regressions = [n for n, r in results.items()
                       if not r["passed"] and baseline.get(n, {}).get("passed")]
        for n in regressions:
            print(f"  REGRESSION: {n} passed in baseline, fails here")
        print(f"\n  VERDICT: {'PASS' if not regressions else 'FAIL'} "
              f"({len(regressions)} regression(s) vs baseline; "
              f"{len(TASKS) - failures}/{len(TASKS)} absolute)")
        sys.exit(1 if regressions else 0)

    print(f"\n  VERDICT: {'PASS' if not failures else 'FAIL'} "
          f"({len(TASKS) - failures}/{len(TASKS)} long-generation tasks)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
