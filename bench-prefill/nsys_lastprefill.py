import csv,sys,collections
rows=list(csv.DictReader(open(sys.argv[1])))
rows.sort(key=lambda r: float(r['Start (ns)']))
# segment by idle gaps > 100 ms
segs=[[rows[0]]]
for a,b in zip(rows,rows[1:]):
    if float(b['Start (ns)'])-(float(a['Start (ns)'])+float(a['Duration (ns)']))>100e6: segs.append([])
    segs[-1].append(b)
print('segments:',[(len(s), round((float(s[-1]['Start (ns)'])+float(s[-1]['Duration (ns)'])-float(s[0]['Start (ns)']))/1e6,1)) for s in segs][-6:])
seg=segs[-1]
span=(float(seg[-1]['Start (ns)'])+float(seg[-1]['Duration (ns)'])-float(seg[0]['Start (ns)']))/1e6
agg=collections.defaultdict(lambda:[0,0.0])
for r in seg:
    n=r['Name']; n=n if not n.startswith('[CUDA') else n
    agg[n][0]+=1; agg[n][1]+=float(r['Duration (ns)'])/1e6
tot=sum(v[1] for v in agg.values())
print(f'last segment: {len(seg)} launches, wall span {span:.1f} ms, kernel+copy busy {tot:.1f} ms, idle {span-tot:.1f} ms')
print(f"{'ms':>8} {'%':>5} {'n':>5} {'avg us':>8}  name")
for n,(c,ms) in sorted(agg.items(), key=lambda x:-x[1][1])[:32]:
    print(f"{ms:8.1f} {100*ms/span:5.1f} {c:5d} {1000*ms/c:8.1f}  {n[:95]}")
