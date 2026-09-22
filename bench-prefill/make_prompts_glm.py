#!/usr/bin/env python3
"""Four real texts -> exactly N token ids each (GLM tokenizer), for the teacher-forced gate."""
import json, os, sys
from tokenizers import Tokenizer
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
tok = Tokenizer.from_file('/home/flocka/models/GLM-5.3-Flash-exl3-2.05bpw/tokenizer.json')
W = '/home/flocka/atlas/glm53-prefill-work'
specs = [
    ('code_rust', f'{W}/crates/spark-model/src/model/glm53/dsa_attention.rs'),
    ('code_cuda', f'{W}/kernels/gb10/glm5.3-flash/iq3/glm53_hyper.cu'),
    ('prose_readme', f'{W}/README.md'),
    ('notes_md', f'{W}/DEBUGGING_METHODOLOGY.md'),
]
out = []
for name, path in specs:
    text = open(path, encoding='utf-8', errors='replace').read()
    ids = tok.encode(text, add_special_tokens=False).ids
    assert len(ids) >= N, (name, len(ids))
    ids = [154841, 154842] + ids[: N - 2]   # same two leading specials as the P57 request
    out.append({'name': name, 'source': path, 'n_source_tokens': len(tok.encode(text).ids), 'ids': ids})
    print(f'{name:14s} {len(ids)} tokens  tail={tok.decode(ids[-8:])!r}')
json.dump(out, open(f'prompts_{N}.json', 'w'))
