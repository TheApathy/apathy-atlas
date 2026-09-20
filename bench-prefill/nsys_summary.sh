#!/usr/bin/env bash
# usage: nsys_summary.sh <profile.nsys-rep> -> kernel time table for the whole capture (server load + warmup + 1 trial)
R=${1:?}
/usr/local/bin/nsys stats --report cuda_gpu_kern_sum --format csv -o ${R%.nsys-rep}-kern $R >/dev/null 2>&1
/usr/local/bin/nsys stats --report cuda_api_sum --format csv -o ${R%.nsys-rep}-api $R >/dev/null 2>&1
python3 - ${R%.nsys-rep}-kern_cuda_gpu_kern_sum.csv <<'PY'
import csv,sys
rows=list(csv.DictReader(open(sys.argv[1])))
tot=sum(float(r['Total Time (ns)']) for r in rows)/1e6
print(f"total kernel time {tot:.1f} ms")
print(f"{'ms':>9} {'%':>5} {'calls':>6} {'avg us':>8}  name")
for r in rows[:40]:
    ms=float(r['Total Time (ns)'])/1e6
    print(f"{ms:9.1f} {100*ms/tot:5.1f} {int(r['Instances']):6d} {float(r['Avg (ns)'])/1e3:8.1f}  {r['Name'][:90]}")
PY
