#!/usr/bin/env python3
"""Quantify the results gap of a single-node DeepSeek-V4-Flash Atlas serve.

Answers "are the results the same?" with a number, not a guess. Two parts:

  1. Coherence gate — the four prompts the Atlas EP=2 recipe was validated on
     (Paris, counting, haiku, math). Every one must produce clean, on-task text.
  2. Accuracy probe — a small fixed GSM8K-style arithmetic-reasoning set graded
     on exact final-integer match. This is a fast proxy for the ds4 reference's
     GSM8K 97.5% gate; a REAP-pruned checkpoint that has drifted will show here.

Greedy (temperature 0) so the run is deterministic and comparable across builds.

  python3 bench/deepseek-v4/quality_probe.py --port 8899 [--model deepseek-v4-flash]

Exit 0 iff coherence is 4/4 AND accuracy >= --min-acc (default 0.80).
"""
import argparse, json, re, sys, urllib.request

COHERENCE = [
    ("paris",    "What is the capital of France? Answer in one sentence.", lambda t: "paris" in t.lower()),
    ("counting", "Count from 1 to 10, separated by commas.",              lambda t: all(str(n) in t for n in range(1, 11))),
    ("haiku",    "Write a haiku about the ocean.",                          lambda t: len(t.split()) >= 6),
    ("math",     "What is 17 times 24? Give only the number.",             lambda t: "408" in t.replace(",", "")),
]

# Fixed arithmetic-reasoning items with known integer answers (GSM8K-style).
GSM = [
    ("Natalia sold clips to 48 friends in April, then half as many in May. How many clips did she sell altogether?", 72),
    ("Weng earns $12 an hour for babysitting. Yesterday she babysat for 50 minutes. How much did she earn?", 10),
    ("Betty needs $100 for a wallet. She has half of that. Her parents give her $15 and her grandparents twice as much as her parents. How much more does she need?", 5),
    ("A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts total?", 3),
    ("James writes a 3-page letter to 2 friends twice a week. How many pages does he write a year?", 624),
    ("There are 15 trees. Workers plant more so there are 21. How many did they plant?", 6),
    ("Shawn has 5 toys. For Christmas he got 2 from mom and 2 from dad. How many toys now?", 9),
    ("There were 9 computers. 5 more were installed each day from Monday to Thursday. How many now?", 29),
    ("Michael had 58 golf balls. He lost 23 on Tuesday and 2 more on Wednesday. How many at end of Wednesday?", 33),
    ("Olivia had $23. She bought 5 bagels for $3 each. How much money does she have left?", 8),
    ("A group ordered 2 dozen scarves at $2 each. What was the total cost in dollars?", 48),
    ("If a train travels 60 miles per hour for 3 hours, how many miles does it travel?", 180),
]

def call(host, port, model, prompt, max_tokens):
    body = json.dumps({"model": model,
                       "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0.0, "stream": False}).encode()
    req = urllib.request.Request(f"http://{host}:{port}/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    return d["choices"][0]["message"]["content"]

def last_int(text):
    nums = re.findall(r"-?\d[\d,]*", text.replace("$", ""))
    if not nums:
        return None
    try:
        return int(nums[-1].replace(",", ""))
    except ValueError:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--min-acc", type=float, default=0.80)
    a = ap.parse_args()

    print("== coherence gate ==")
    coh = 0
    for name, prompt, check in COHERENCE:
        try:
            out = call(a.host, a.port, a.model, prompt, 128)
            ok = check(out)
        except Exception as e:
            out, ok = f"ERROR: {e}", False
        coh += ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {out[:90].strip()!r}")

    print("== accuracy probe (GSM8K-style, greedy) ==")
    correct = 0
    for q, ans in GSM:
        try:
            out = call(a.host, a.port, a.model, q + " Show your reasoning, then end with 'Answer: <number>'.", 512)
            got = last_int(out)
            ok = got == ans
        except Exception as e:
            got, ok = f"ERR({e})", False
        correct += bool(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] want={ans} got={got}")
    acc = correct / len(GSM)

    print("\n== summary ==")
    print(f"  coherence : {coh}/{len(COHERENCE)}")
    print(f"  accuracy  : {correct}/{len(GSM)} = {acc:.2%}   (ds4 reference GSM8K gate ~97.5%)")
    passed = coh == len(COHERENCE) and acc >= a.min_acc
    print(f"  VERDICT   : {'PASS' if passed else 'FAIL'} (need coherence {len(COHERENCE)}/{len(COHERENCE)} and acc >= {a.min_acc:.0%})")
    sys.exit(0 if passed else 1)

if __name__ == "__main__":
    main()
