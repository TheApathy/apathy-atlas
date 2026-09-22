#!/usr/bin/env python3
"""compare.py REF_ARM CAND_ARM: per-layer last-token hidden-state error (from the
ATLAS_NEMO_DUMP capture of each request) and greedy 64-token continuation agreement."""
import json, os, sys, struct, math
B = "/home/flocka/atlas/flashnext-prefill-work/bench/results"
ref, cand = sys.argv[1], sys.argv[2]
def load(path):
    b = open(path, "rb").read(); n = len(b)//4
    return struct.unpack("<%df" % n, b)
def toks(path):
    try:
        r = json.load(open(path)); return r["choices"][0]["text"]
    except Exception as e:
        return None
print(f"ref={ref} cand={cand}")
print("per-layer relative L2 error of the LAST prompt token's post-layer hidden (48 layers), per request:")
for req in ["warmup", "1", "2", "3", "4", "5"]:
    d1, d2 = f"{B}/{ref}/dump-{req}", f"{B}/{cand}/dump-{req}"
    if not (os.path.isdir(d1) and os.path.isdir(d2)):
        print(f"  req {req}: missing dumps"); continue
    errs = []; cos = []
    for L in range(48):
        a = load(f"{d1}/atlas_L{L}.bin"); b = load(f"{d2}/atlas_L{L}.bin")
        num = math.sqrt(sum((x-y)**2 for x, y in zip(a, b))); den = math.sqrt(sum(x*x for x in a)) or 1.0
        dot = sum(x*y for x, y in zip(a, b)); nb = math.sqrt(sum(y*y for y in b)) or 1.0
        errs.append(num/den); cos.append(dot/(den*nb))
    print(f"  req {req}: L0 {errs[0]:.2e}  L11 {errs[11]:.2e}  L23 {errs[23]:.2e}  L35 {errs[35]:.2e}  L47 {errs[47]:.2e}  max {max(errs):.2e}  cos(L47) {cos[47]:.6f}")
import re
def logits(arm):
    out=[]
    for line in open(f"{B}/{arm}/server.log", errors="replace"):
        m = re.search(r"top-10 logits = \[(.*?)\]", line)
        if m:
            pairs = re.findall(r"\((\d+), ([-\d.]+)\)", m.group(1))
            out.append([(int(a), float(b)) for a, b in pairs])
    return out
la, lb = logits(ref), logits(cand)
print(f"top-10 logits per request (ref has {len(la)} records, cand {len(lb)}): argmax match / top-5 overlap / max |dlogit| over shared top-10 / ref margin")
for i in range(min(len(la), len(lb))):
    A = dict(la[i]); Bd = dict(lb[i])
    am = la[i][0][0] == lb[i][0][0]
    top5 = len(set(t for t,_ in la[i][:5]) & set(t for t,_ in lb[i][:5]))
    shared = set(A) & set(Bd)
    dmax = max((abs(A[t]-Bd[t]) for t in shared), default=float("nan"))
    margin = la[i][0][1] - la[i][1][1]
    print(f"  rec {i}: argmax {'MATCH' if am else 'DIFF'} ({la[i][0][0]} vs {lb[i][0][0]}) top5 {top5}/5 max|dlogit| {dmax:.3f} ref margin {margin:.3f}")
print("first-token agreement:")
same = 0; tot = 0
for req in ["warmup", "1", "2", "3", "4", "5"]:
    a = toks(f"{B}/{ref}/response-{req}.json"); b = toks(f"{B}/{cand}/response-{req}.json")
    tot += 1; same += (a == b); print(f"  req {req}: {a!r} vs {b!r} {'MATCH' if a==b else 'DIFF'}")
print(f"  {same}/{tot} match")
print("greedy 64-token continuation agreement (prefix match length / chars):")
for req in ["1", "2", "3", "4", "5"]:
    a = toks(f"{B}/{ref}/response-gen-{req}.json"); b = toks(f"{B}/{cand}/response-gen-{req}.json")
    if a is None or b is None: print(f"  req {req}: missing"); continue
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]: n += 1
    print(f"  req {req}: common prefix {n}/{max(len(a),len(b))} chars | ref={a[:60]!r} | cand={b[:60]!r}")
