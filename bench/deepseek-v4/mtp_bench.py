#!/usr/bin/env python3
"""Atlas mirror of the ds4-on-spark llama-benchy mtp-bench suite.

Reproduces the reference measurement shape — 9 workloads, 512 tokens,
non-streaming WALL tok/s (includes prefill, like the published table) — so
the Atlas numbers are directly comparable to the Entrpi/ds4 DSpark table:

  workload        plain  DSpark  gain   tok/step  accept
  stepwise_math   20.1   34.5    1.71x  4.00      89 %
  ...             (suite mean: 20.1 -> 27.7, 1.38x)

Per-workload accept and tokens/step are read from the serve log's per-step
verify lines, so the serve must run with
`RUST_LOG=info,spark::scheduler::verify_k2_step=debug` for those columns
(they print as `-` otherwise; wall tok/s works regardless).

  # speculative columns (serve with --speculative + verify debug):
  python3 bench/deepseek-v4/mtp_bench.py --port 8977 --log serve.log \
      --out /tmp/mtp_bench_spec.json
  # plain columns (serve without --speculative):
  python3 bench/deepseek-v4/mtp_bench.py --port 8977 --out /tmp/mtp_bench_plain.json
  # merge into the comparison table:
  python3 bench/deepseek-v4/mtp_bench.py --table /tmp/mtp_bench_plain.json /tmp/mtp_bench_spec.json
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.request

REVIEW_SNIPPET = '''
def process_orders(orders, inventory):
    total = 0
    for o in orders:
        if o["sku"] in inventory:
            if inventory[o["sku"]] > 0:
                inventory[o["sku"]] -= o["qty"]
                total += o["price"] * o["qty"]
            else:
                print("out of stock: " + o["sku"])
        else:
            pass
    return total

class OrderManager:
    def __init__(self):
        self.orders = []
        self.lock = None
    def add(self, order):
        self.orders.append(order)
    def process_all(self, inventory):
        results = []
        for i in range(len(self.orders)):
            results.append(process_orders([self.orders[i]], inventory))
        return sum(results)
'''

SUMMARIZE_PASSAGE = (
    "The transition from vacuum tubes to transistors in the 1950s marked the "
    "beginning of the modern computing era. Early machines like ENIAC consumed "
    "enormous amounts of power and required constant maintenance as tubes burned "
    "out. The invention of the transistor at Bell Labs in 1947 offered a smaller, "
    "cooler, and far more reliable switching element. By the early 1960s, "
    "integrated circuits allowed multiple transistors on a single silicon die, "
    "and Gordon Moore observed in 1965 that the number of components per chip "
    "was doubling roughly every year, later revised to every two years. This "
    "observation, known as Moore's Law, became a self-fulfilling roadmap for the "
    "semiconductor industry. Successive process nodes shrank feature sizes from "
    "micrometers to nanometers, enabling microprocessors, personal computers, "
    "mobile phones, and eventually data centers filled with accelerators for "
    "machine learning. Along the way, architects introduced caches, pipelining, "
    "out-of-order execution, and multicore designs to convert transistor budgets "
    "into usable performance, while power density and memory bandwidth emerged "
    "as the dominant constraints of the current era."
)

# name -> prompt. Mirrors the reference suite's workload mix: structured math,
# factual QA, summarization, two code generations, translation, explanation,
# code review, and open-ended creative writing.
WORKLOADS = [
    ("stepwise_math",
     "Solve this step by step, showing every intermediate calculation on its own "
     "line. A water tank holds 2,400 liters. Pump A fills it at 45 L/min and pump "
     "B drains it at 18 L/min. Both start at 09:00 with the tank 25% full. At what "
     "time is the tank full? Then redo the problem with pump A slowing to 30 L/min "
     "after 20 minutes. Show all steps for both cases."),
    ("qa_factual",
     "Answer each question in one short sentence, numbered 1-15: 1. Capital of "
     "Australia? 2. Year the Berlin Wall fell? 3. Chemical symbol for gold? 4. "
     "Largest planet in the solar system? 5. Author of 1984? 6. Speed of light in "
     "km/s? 7. Smallest prime number? 8. Currency of Japan? 9. Longest river in "
     "Africa? 10. Inventor of the telephone? 11. Boiling point of water at sea "
     "level in Fahrenheit? 12. Number of bones in the adult human body? 13. "
     "Painter of the Mona Lisa? 14. Deepest ocean trench? 15. First element on "
     "the periodic table?"),
    ("summarize",
     "Summarize the following passage in three detailed paragraphs, then give a "
     "bulleted list of the five most important facts:\n\n" + SUMMARIZE_PASSAGE),
    ("code_cpp",
     "Write a complete C++17 implementation of a fixed-capacity, thread-safe "
     "MPMC ring buffer using std::mutex and std::condition_variable, with "
     "push/pop/try_pop, a small main() demonstrating producer and consumer "
     "threads, and comments explaining the synchronization."),
    ("translation",
     "Translate the following paragraph into French, then German, then Spanish, "
     "labeling each: 'The lighthouse stood at the edge of the cliff for two "
     "hundred years. Every evening the keeper climbed the spiral stairs to light "
     "the lamp, and every morning ships passed safely between the rocks. When "
     "the automated beacon arrived, the keeper stayed on as a caretaker, unable "
     "to imagine the tower without a human heartbeat inside it.'"),
    ("code_python",
     "Write a complete Python module implementing a least-recently-used cache "
     "class with get/put/resize, full type hints, docstrings, and a pytest test "
     "class covering eviction order, resize behavior, and edge cases."),
    ("explain_concept",
     "Explain how TCP congestion control works, covering slow start, congestion "
     "avoidance, fast retransmit, and fast recovery, with a worked example of "
     "how the congestion window evolves across packet losses."),
    ("long_code_review",
     "Review the following Python code line by line. For each issue, quote the "
     "line, explain the problem, and show the corrected code:\n```python\n"
     + REVIEW_SNIPPET + "\n```"),
    ("creative_short",
     "Write a short story (about 400 words) about a cartographer who discovers "
     "that the blank spaces on an old map are not unexplored — they are places "
     "that have been deliberately erased."),
]

ACCEPT_RE = re.compile(rb"K2 (ACCEPT|REJECT):")


def call(host, port, prompt, max_tokens):
    body = json.dumps({
        "model": "bench",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"http://{host}:{port}/v1/chat/completions",
        data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.load(r)
    wall = time.time() - t0
    ct = d.get("usage", {}).get("completion_tokens", 0)
    return ct, wall


def log_size(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def count_steps(path, start, end):
    """(accepts, rejects) among per-step verify lines in log[start:end)."""
    if not path:
        return 0, 0
    with open(path, "rb") as f:
        f.seek(start)
        chunk = f.read(max(0, end - start))
    acc = rej = 0
    for m in ACCEPT_RE.finditer(chunk):
        if m.group(1) == b"ACCEPT":
            acc += 1
        else:
            rej += 1
    return acc, rej


def run_suite(a):
    results = {}
    for name, prompt in WORKLOADS:
        toks, walls, accs, rejs = [], [], 0, 0
        for _ in range(a.runs):
            mark = log_size(a.log) if a.log else 0
            ct, wall = call(a.host, a.port, prompt, a.max_tokens)
            if a.log:
                time.sleep(0.3)  # let the serve flush its step lines
                acc, rej = count_steps(a.log, mark, log_size(a.log))
                accs += acc
                rejs += rej
            toks.append(ct)
            walls.append(wall)
        total_t, total_w = sum(toks), sum(walls)
        steps = accs + rejs
        results[name] = {
            "tok_s": total_t / total_w if total_w else 0.0,
            "tokens": total_t,
            "accept": accs / steps if steps else None,
            "tok_per_step": total_t / steps if steps else None,
        }
        r = results[name]
        acc_s = f"{r['accept'] * 100:.0f}%" if r["accept"] is not None else "-"
        tps_s = f"{r['tok_per_step']:.2f}" if r["tok_per_step"] else "-"
        print(f"  {name:<18} {r['tok_s']:6.1f} tok/s  accept={acc_s:<5} "
              f"tok/step={tps_s}  ({total_t} tok / {total_w:.1f}s)", flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"saved: {a.out}")


def print_table(plain_path, spec_path):
    plain = json.load(open(plain_path))
    spec = json.load(open(spec_path))
    print(f"{'workload':<18} {'plain t/s':>9} {'spec t/s':>9} {'gain':>6} "
          f"{'tok/step':>9} {'accept':>7}")
    p_sum = s_sum = n = 0
    for name, _ in WORKLOADS:
        p = plain.get(name, {}).get("tok_s")
        s = spec.get(name, {}).get("tok_s")
        if p is None or s is None:
            continue
        acc = spec[name].get("accept")
        tps = spec[name].get("tok_per_step")
        print(f"{name:<18} {p:9.1f} {s:9.1f} {s / p:5.2f}x "
              f"{tps if tps is None else f'{tps:9.2f}'} "
              f"{'-' if acc is None else f'{acc * 100:6.0f} %'}")
        p_sum += p
        s_sum += s
        n += 1
    if n:
        print(f"{'suite mean':<18} {p_sum / n:9.1f} {s_sum / n:9.1f} "
              f"{s_sum / p_sum:5.2f}x")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8977)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--log", help="serve log (verify_k2_step=debug) for accept/steps")
    ap.add_argument("--out", help="write per-workload JSON here")
    ap.add_argument("--table", nargs=2, metavar=("PLAIN_JSON", "SPEC_JSON"),
                    help="print the comparison table from two saved runs")
    a = ap.parse_args()
    if a.table:
        print_table(a.table[0], a.table[1])
        return
    run_suite(a)


if __name__ == "__main__":
    main()
