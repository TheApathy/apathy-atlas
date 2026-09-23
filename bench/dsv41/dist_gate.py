#!/usr/bin/env python3
"""End-to-end sampled-spec distribution gate (lead ruling): same prompt, N seeds, T=1.0, first 2
generated tokens, plain serve vs DSpark serve. Two-sample chi-square (categories with expected < 5
pooled) on token 1 (prefill-sampled: a sanity arm), token 2 (the first speculated position) and the
joint pair. PASS: p > 0.01 for DSpark vs plain; the accept-all control vs plain must give p < 0.01."""
import glob, json, os, sys
from collections import Counter
from scipy.stats import chi2_contingency

def load(d, prefix):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file('/home/flocka/models/DeepSeek-V4.1-Flash-Next-DGX-Spark-512K/tokenizer.json')
    out = []
    for f in sorted(glob.glob(os.path.join(d, prefix + '_*.response.json'))):
        r = json.load(open(f))
        if 'choices' not in r:
            continue
        ids = tok.encode(r['choices'][0]['text'], add_special_tokens=False).ids
        out.append(tuple(ids[:2]) if len(ids) >= 2 else tuple(ids))
    return out

def perm_tv(a, b, key, n_perm=500, seed=0):
    """Permutation test on the empirical total-variation distance: no pooling, so it keeps power
    where chi-square pooling folds a high-entropy distribution into one bucket."""
    import random
    xa, xb = [key(x) for x in a], [key(x) for x in b]
    def tv(u, v):
        cu, cv = Counter(u), Counter(v)
        return 0.5 * sum(abs(cu[c] / len(u) - cv[c] / len(v)) for c in set(cu) | set(cv))
    obs = tv(xa, xb)
    pool, rng, hits = xa + xb, random.Random(seed), 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        hits += tv(pool[:len(xa)], pool[len(xa):]) >= obs
    return (hits + 1) / (n_perm + 1)

def test2(a, b, key):
    # Both statistics, Bonferroni: the arm passes if min(p) > 0.005 (reported as 2*min p).
    p1, k1 = test(a, b, key)
    p2 = perm_tv(a, b, key)
    return min(1.0, 2 * min(p1, p2)), k1

def test(a, b, key):
    ca, cb = Counter(key(x) for x in a), Counter(key(x) for x in b)
    cats = sorted(set(ca) | set(cb), key=lambda c: -(ca[c] + cb[c]))
    na, nb = sum(ca.values()), sum(cb.values())
    rows, other = [], [0, 0]
    for c in cats:
        exp = (ca[c] + cb[c]) * min(na, nb) / (na + nb)
        if exp < 5:
            other[0] += ca[c]; other[1] += cb[c]
        else:
            rows.append([ca[c], cb[c]])
    if other[0] + other[1] > 0:
        rows.append(other)
    if len(rows) < 2:
        return 1.0, len(rows)
    _, p, _, _ = chi2_contingency(list(zip(*rows)))
    return p, len(rows)

def main(base):
 for prompt in ('nouns', 'prose'):
   plain, spec, ctrl = (load(os.path.join(base, d), prompt) for d in ('dist_plain', 'dist_dspark', 'dist_control'))
   print(f"== prompt {prompt}: n plain {len(plain)} dspark {len(spec)} control {len(ctrl)}")
   ok = True
   for name, key in (('token1', lambda x: x[0] if x else None), ('token2', lambda x: x[1] if len(x) > 1 else None), ('pair', lambda x: x)):
       p_s, k_s = test2(plain, spec, key)
       p_c, k_c = test2(plain, ctrl, key)
       print(f"{name}: dspark vs plain p={p_s:.4f} ({k_s} cats) | control vs plain p={p_c:.4g} ({k_c} cats)")
       if name != 'token1':
           ok &= p_s > 0.01
   print("control FAILS (token2 or pair p < 0.01):", test2(plain, ctrl, lambda x: x[1] if len(x) > 1 else None)[0] < 0.01 or test2(plain, ctrl, lambda x: x)[0] < 0.01)
   print("GATE", "PASS" if ok else "FAIL")


if __name__ == '__main__':
    main(sys.argv[1])
