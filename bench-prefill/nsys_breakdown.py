#!/usr/bin/env python3
"""nsys_breakdown.py <run-dir> [prefills]: kernel-time table grouped by component (ms per prefill)."""
import csv, subprocess, sys, os, re, collections
run=sys.argv[1]; prefills=int(sys.argv[2]) if len(sys.argv)>2 else 2
rep=os.path.join(run,'profile.nsys-rep'); csvp=os.path.join(run,'stats_cuda_gpu_kern_sum.csv')
if not os.path.exists(csvp):
    subprocess.run(['/usr/local/bin/nsys','stats','--report','cuda_gpu_kern_sum','--format','csv','-o',os.path.join(run,'stats'),rep],capture_output=True)
rows=list(csv.DictReader(open(csvp)))
groups=[('MoE experts (staged gate_up/down + gather/activate/scatter/combine/pack/build)',r'staged_|pack_routes|combine_shared|build_chunks'),
 ('MoE router',r'router_logits|topk_sigmoid|prompt_bf16_to_f32'),
 ('Dense EXL3 trellis GEMMs (row_x, k32_n128, exl3_gemm)',r'exl3_gemm_kernel|kda_qkv_n256|k32_n128|kda_qkv_hadamard'),
 ('Reconstruct kernels',r'reconstruct_had'),
 ('cuBLASLt GEMMs (nvjet/cutlass: reconstructed dense, KDA low-rank, router/hc mixing)',r'nvjet|cutlass|gemm_|sm90|sm100|sm120|xmma|Kernel'),
 ('f16<->bf16 casts',r'exl3_f16_to_bf16|exl3_bf16_to_f16'),
 ('KDA (recurrence, conv, split, norm, gate)',r'kda_'),
 ('DSA (dense causal, absorb, transpose, index, norms)',r'dsa_'),
 ('mHC glue (hc_pre/post/mix/mean/expand)',r'hc_pre|hc_post|hc_mix|hc_mean|hc_expand'),
 ('norms / swiglu / misc',r'rms_norm|swiglu|dflash2|memset|copy|Memcpy|Memset'),]
tot=sum(float(r['Total Time (ns)']) for r in rows)/prefills/1e6
agg=collections.OrderedDict((g,0.0) for g,_ in groups); agg['other']=0.0
per=[]
for r in rows:
    t=float(r['Total Time (ns)'])/prefills/1e6; name=r['Name']
    for g,pat in groups:
        if re.search(pat,name): agg[g]+=t; break
    else: agg['other']+=t
    per.append((t,int(r['Instances'])//prefills,float(r['Avg (ns)'])/1e3,name[:90]))
print(f"kernel time per prefill: {tot:.0f} ms ({sum(int(r['Instances']) for r in rows)//prefills} launches)")
print("| component | ms/prefill | share |\n|---|---|---|")
for g,t in agg.items(): print(f"| {g} | {t:.0f} | {100*t/tot:.1f}% |")
print("\ntop kernels (ms/prefill, launches/prefill, avg us):")
for t,n,a,name in sorted(per,reverse=True)[:32]: print(f"{t:8.1f} {n:6d} {a:9.1f}  {name}")
