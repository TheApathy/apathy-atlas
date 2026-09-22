#!/usr/bin/env python3
"""Compare every tap a Rust run dumped against an oracle run, via compare.py (so every row
carries compare.py's own automatic negative control). Prints a table sorted by (layer, occ,
forward order) and the FIRST_DIVERGENT_LAYER: the lowest layer with any FAIL on `h`."""
import json, os, re, subprocess, sys
ref_dir, cand_dir = sys.argv[1], sys.argv[2]
extra = sys.argv[3:]
HERE = "/home/flocka/atlas/DSV41_PORT/oracle/compare.py"
man = json.load(open(os.path.join(ref_dir, "manifest.json")))
ref = {os.path.splitext(e["file"])[0]: e for e in man["tensors"]}
ORDER = ["engram_out", "attn_x", "attn_out", "moe_in", "moe_routed", "moe_shared", "h", "pre_mix", "logits_last"]
rows = []
for f in sorted(os.listdir(cand_dir)):
    m = re.match(r"L(\d+)\.(\w+)\.(\d+)\.bin$", f)
    if not m:
        continue
    name = f[:-4]
    if name not in ref:
        rows.append((int(m[1]), int(m[3]), 99, name, "NO-REF", "", ""))
        continue
    cand_bytes = os.path.getsize(os.path.join(cand_dir, f))
    n = 1
    for s in ref[name]["shape"]:
        n *= s
    cdt = {2: "bfloat16", 4: "float32"}.get(cand_bytes // max(n, 1), ref[name]["dtype"])
    r = subprocess.run([sys.executable, HERE, "--ref-dir", ref_dir, "--name", name,
                        "--cand", os.path.join(cand_dir, f), "--cand-dtype", cdt, "--json", *extra],
                       capture_output=True, text=True)
    try:
        j = json.loads(r.stdout)
    except Exception:
        rows.append((int(m[1]), int(m[3]), 98, name, f"ERR rc={r.returncode}", r.stderr[-200:], ""))
        continue
    verdict = j.get("verdict") or {0: "PASS", 1: "FAIL", 2: "UNRUNNABLE"}.get(r.returncode, str(r.returncode))
    tap = m[2]
    rows.append((int(m[1]), int(m[3]), ORDER.index(tap) if tap in ORDER else 50, name, verdict,
                 f"rel_l2={j.get('rel_l2', j.get('mismatch', '?')):.3e}" if isinstance(j.get('rel_l2'), float) else f"{ {k: v for k, v in j.items() if 'mismatch' in k} }",
                 f"max_abs={j.get('max_abs', '')} ctrl_stat={(j.get('control') or {}).get('stat', '')} cos={j.get('cosine', '')}"))
rows.sort()
first = None
for L, occ, _, name, v, a, b in rows:
    print(f"{name:32s} {v:10s} {a} {b}")
    if first is None and v != "PASS" and name.split(".")[1] == "h":
        first = L
print(f"FIRST_DIVERGENT_LAYER (on h) = {first if first is not None else 'none among dumped layers'}")
