#!/usr/bin/env python3
"""score.py <window-dir> <reference-arm>

Per arm and prompt: median decode tok/s, completion tokens, finish reasons, distinct output shas.
Identity gate: every trial of every arm must produce text byte-identical to the reference arm's trial
of the same prompt. The perturbed-prompt control (trial-1-ctlperturb) must DIFFER from the reference
short output, or the gate is declared unable to fail. DFlash arms: phase-receipt decomposition.
"""
import collections, glob, json, os, re, statistics, sys

win, ref = sys.argv[1], sys.argv[2]
arms = sorted(d for d in os.listdir(win) if os.path.isdir(os.path.join(win, d)))


def text(path):
    try:
        return json.load(open(path))["choices"][0]["text"]
    except Exception:
        return None


def trials(arm):
    out = collections.defaultdict(list)
    for f in sorted(glob.glob(os.path.join(win, arm, "trial-*.response.json"))):
        m = re.match(r"trial-(\d+)-(\w+)\.response\.json", os.path.basename(f))
        out[m.group(2)].append((int(m.group(1)), f))
    return out


def timing(arm):
    rows = {}
    p = os.path.join(win, arm, "timing.txt")
    if os.path.exists(p):
        for line in open(p):
            m = re.match(r"(\S+) (\w+): .*completion=(\d+).*decode_tok_s=([\d.]+).*finish=(\S+) sha=(\w+)", line)
            if m:
                rows[m.group(1)] = (int(m.group(3)), float(m.group(4)), m.group(5), m.group(6))
    return rows


def phases(arm):
    p = os.path.join(win, arm, "server.log")
    if not os.path.exists(p):
        return None
    acc = collections.Counter()
    sums = collections.defaultdict(list)
    prop = []
    for line in open(p, errors="replace"):
        if "atlas.glm53.phase_timing.v1" not in line or "receipt" not in line:
            continue
        j = line[line.index("receipt") :]
        j = j[j.index("{") :]
        j = re.sub(r"\x1b\[[0-9;]*m", "", j).strip()
        try:
            r = json.loads(j)
        except Exception:
            continue
        ph = {name: v["elapsed_ns"] / 1e6 for name, _, v in r["phase_rows"]}
        if r["kind"] == "proposal":
            prop.append(ph.get("proposal", 0))
        elif r["kind"] == "verify":
            acc[r["accepted"]] += 1
            for k in ("verify_total", "wide_stage", "partial_replay", "full_commit_body", "policy_pick"):
                if k in ph:
                    sums[k].append(ph[k])
    if not acc:
        return None
    n = sum(acc.values())
    mean_acc = sum(k * v for k, v in acc.items()) / n
    top = max(acc)
    per_pos = " ".join(f"p{i}={sum(v for k, v in acc.items() if k >= i) / n:.3f}" for i in range(1, top + 1))
    s = (f"steps={n} mean_accepted={mean_acc:.2f} hist={dict(sorted(acc.items()))} "
         f"reach[{per_pos}] proposal={statistics.mean(prop):.1f}ms")
    for k, v in sums.items():
        s += f" {k}={statistics.mean(v):.1f}ms(n={len(v)})"
    return s


reftr = trials(ref)
fail = False
print(f"reference arm: {ref}")
for arm in arms:
    tr, tm = trials(arm), timing(arm)
    print(f"\n== {arm}")
    for prompt, lst in sorted(tr.items()):
        speeds = [tm[f"trial-{i}-{prompt}"][1] for i, _ in lst if f"trial-{i}-{prompt}" in tm]
        cts = sorted({tm[f"trial-{i}-{prompt}"][0] for i, _ in lst if f"trial-{i}-{prompt}" in tm})
        fins = sorted({tm[f"trial-{i}-{prompt}"][2] for i, _ in lst if f"trial-{i}-{prompt}" in tm})
        shas = sorted({tm[f"trial-{i}-{prompt}"][3] for i, _ in lst if f"trial-{i}-{prompt}" in tm})
        ident = []
        if prompt != "ctlperturb" and prompt in reftr:
            rtext = text(reftr[prompt][0][1])
            for i, f in lst:
                t = text(f)
                if t is None or rtext is None:
                    ident.append("MISSING"); fail = True
                elif t == rtext:
                    ident.append("same")
                else:
                    n = next((k for k, (a, b) in enumerate(zip(t, rtext)) if a != b), min(len(t), len(rtext)))
                    ident.append(f"DIFF@{n}"); fail = True
        med = statistics.median(speeds) if speeds else 0
        print(f"  {prompt:10s} tok/s median={med:.3f} {['%.3f' % x for x in speeds]} completion={cts} finish={fins} shas={len(shas)} vs_ref={ident}")
        if cts and cts != [160]:
            print(f"  *** DO-NOT-SCORE {arm}/{prompt}: completion {cts} != 160"); fail = True
    ph = phases(arm)
    if ph:
        print(f"  phases: {ph}")
    if "ctlperturb" in tr:
        c = text(tr["ctlperturb"][0][1]); r = text(reftr["short"][0][1])
        verdict = "DIFFERS (gate can fail: OK)" if c != r else "IDENTICAL (gate cannot fail!)"
        print(f"  CONTROL perturbed short prompt vs {ref}/short: {verdict}")
        if c == r:
            fail = True
print("\nGATE:", "FAIL" if fail else "PASS (all arms byte-identical to reference on every trial)")
