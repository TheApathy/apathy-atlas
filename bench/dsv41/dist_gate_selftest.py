#!/usr/bin/env python3
"""Self-test of dist_gate.test on synthetic draws: same distribution must PASS (p > 0.01) in the large
majority of repeats; shifted distributions (realistic draft-vs-model gaps) must FAIL (p < 0.01); and a
high-entropy distribution shows how much power the rare-category pooling leaves."""
import random, sys
sys.path.insert(0, __import__('os').path.dirname(__file__))
from dist_gate import test2 as test

def zipf(n, s=1.1):
    w = [1 / (k + 1) ** s for k in range(n)]
    z = sum(w)
    return [x / z for x in w]

def shift(p, tv):
    # move `tv` of mass from the top token to the tail uniformly (total variation = tv)
    q = list(p)
    take = min(tv, q[0] - 1e-6)
    q[0] -= take
    for i in range(1, len(q)):
        q[i] += take / (len(q) - 1)
    return q

def draw(p, n, rng):
    return [(rng.choices(range(len(p)), p)[0],) for _ in range(n)]

def rate(p, q, n=500, reps=20, seed=0):
    rng = random.Random(seed)
    fails = sum(test(draw(p, n, rng), draw(q, n, rng), lambda x: x)[0] < 0.01 for _ in range(reps))
    return fails / reps

p = zipf(60)
same = rate(p, p)
print(f"same distribution: fail rate {same:.2f} (expect ~0.01)")
for tv in (0.05, 0.10, 0.20, 0.30):
    print(f"shift TV={tv:.2f}: fail rate {rate(p, shift(p, tv)):.2f}")
flat = [1 / 400] * 400
print(f"high-entropy (400 equal tokens) same: fail rate {rate(flat, flat):.2f}; shifted: {rate(flat, [(2 if i < 200 else 0) / 400 for i in range(400)]):.2f}")
ok = same <= 0.05 and rate(p, shift(p, 0.20)) >= 0.9 and rate(flat, [(2 if i < 200 else 0) / 400 for i in range(400)]) >= 0.9
print("SELFTEST", "PASS" if ok else "FAIL")
