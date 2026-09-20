#!/usr/bin/env python3
"""Teacher-forced capture: one max_tokens=1 request per truncation, moving the
ATLAS_NEMO_DUMP full-logits file into place after each. Env: BENCH OUT PORT."""
import json, os, shutil, time, urllib.request
B=os.environ["BENCH"]; OUT=os.environ["OUT"]; PORT=os.environ["PORT"]
pos=json.load(open(f"{B}/tf_positions.json"))
names=json.load(open(f"{B}/tf_corpus.json"))
log=open(f"{OUT}/tf.log","a")
t_all=time.time()
for name in names:
    req=json.load(open(f"{B}/tf-req-{name}.json")); ids=req["prompt_token_ids"]
    os.makedirs(f"{OUT}/{name}", exist_ok=True)
    for p in pos:
        if p>=len(ids): continue
        dst=f"{OUT}/{name}/logits_{p:04d}.bin"
        if os.path.exists(dst): continue
        body=dict(req); body["prompt_token_ids"]=ids[:p]; body["max_tokens"]=1
        data=json.dumps(body).encode()
        t0=time.time()
        r=urllib.request.urlopen(urllib.request.Request(f"http://localhost:{PORT}/v1/completions", data=data, headers={"Content-Type":"application/json"}), timeout=600)
        resp=json.loads(r.read())
        dt=time.time()-t0
        shutil.move(f"{OUT}/dump/atlas_logits.bin", dst)
        shutil.rmtree(f"{OUT}/dump", ignore_errors=True)
        log.write(f"{name} {p} {resp['usage']['prompt_tokens']} {dt:.3f} {resp['choices'][0]['text']!r}\n"); log.flush()
log.write(f"TOTAL {time.time()-t_all:.1f}s\n")
print("capture done")
