#!/usr/bin/env python3
"""Independent GLM-5.3-Flash EXL3 reference: teacher-forced logits from ExLlamaV3.

Usage: exl3_reference.py <prompts.json> <out_dir> [--cap-gb 40]

<prompts.json> is a list of {"name", "ids"}. For each prompt this writes
<out_dir>/ref-<i>-r<rows>.f32 (raw little-endian float32, [rows, 154880]) and
<out_dir>/meta.json.

Streams the model one module at a time (load, uncached forward over every prompt,
unload), the same loop as ExLlamaV3's own eval/model_diff.py plus an unload. Peak
device use is one decoder layer plus activations, not the 80 GB checkpoint: on
GB10 an over-allocation takes the host down rather than raising, so the torch
allocator is also capped and a breach raises instead.

Nothing here is Atlas code: KDA runs through fla's chunk_kda, MLA/DSA through
ExLlamaV3's Triton kernels, EXL3 GEMMs through its own extension.
"""
import argparse
import hashlib
import json
import os
import sys
import time

EXL3_SRC = "/var/tmp/exllamav3-glm53-r28"
EXL3_EXT = "/var/tmp/exllamav3-arm-oracle-ext-r1/exllamav3_ext"
MODEL = "/home/flocka/models/GLM-5.3-Flash-exl3-2.05bpw"
VOCAB = 154_880

sys.path.insert(0, EXL3_EXT)
sys.path.insert(0, EXL3_SRC)

import torch  # noqa: E402
from exllamav3 import Config, Model  # noqa: E402


def mem_available_gb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) / 1024 / 1024
    raise RuntimeError("MemAvailable missing from /proc/meminfo")


def write_logits(out_dir, i, prompt, logits):
    logits = logits.reshape(-1, logits.shape[-1])[:, :VOCAB].float().cpu().contiguous()
    if logits.shape != (len(prompt["ids"]), VOCAB):
        sys.exit(f"refusing: prompt {i} logits shape {tuple(logits.shape)}")
    if not torch.isfinite(logits).all():
        sys.exit(f"refusing: prompt {i} logits are not finite")
    path = os.path.join(out_dir, f"ref-{i}-r{len(prompt['ids'])}.f32")
    logits.numpy().tofile(path)
    return os.path.basename(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompts")
    ap.add_argument("out_dir")
    ap.add_argument("--cap-gb", type=float, default=40.0)
    ap.add_argument("--causality-probe", action="store_true",
                    help="per module, max |state| difference over the rows prompts 0 and 1 share")
    args = ap.parse_args()

    prompts = json.load(open(args.prompts))
    if not prompts or any(len(p["ids"]) < 2 for p in prompts):
        sys.exit("refusing: need at least one prompt, each with >= 2 tokens")
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda:0")
    total = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, args.cap_gb * 1024**3 / total), device)

    config = Config.from_directory(MODEL)
    config.override_dynamic_seq_len(max(len(p["ids"]) for p in prompts))
    model = Model.from_config(config)

    states = [torch.tensor([p["ids"]], dtype=torch.long) for p in prompts]
    files = []
    params = [{} for _ in prompts]
    shared = 0
    if args.causality_probe:
        a, b = prompts[0]["ids"], prompts[1]["ids"]
        while shared < min(len(a), len(b)) and a[shared] == b[shared]:
            shared += 1
        if shared == 0:
            sys.exit("refusing: --causality-probe needs prompts 0 and 1 to share a prefix")
    low_water = mem_available_gb()
    started = time.time()
    last = len(model.modules) - 1
    for idx, module in enumerate(model.modules):
        config.stc.begin_deferred_load()
        module.load(torch.device("cpu") if module.caps.get("prefer_cpu") else device)
        config.stc.end_deferred_load()
        for b in range(len(states)):
            params[b]["dev_cache"] = None
            x = module.prepare_for_device(states[b], params[b])
            states[b] = module.forward(x, params[b])
            if idx == last:
                # 16K rows of logits are 10 GB: write each prompt's out and free
                # it before the next, so at most one is resident.
                files.append(write_logits(args.out_dir, b, prompts[b], states[b]))
                states[b] = None
        module.unload()
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        now = mem_available_gb()
        low_water = min(low_water, now)
        probe = ""
        if shared and idx != last:
            # Rows before the first differing token must be identical in a causal model.
            sa, sb = states[0], states[1]
            seq = len(prompts[0]["ids"])
            axis = next(d for d in range(sa.dim()) if sa.shape[d] == seq)
            ra, rb = sa.narrow(axis, 0, shared).float(), sb.narrow(axis, 0, shared).float()
            probe = f" shared_rows={shared} max|d|={(ra - rb).abs().max().item():.3e}"
        print(
            f"[{time.time() - started:7.1f}s] module {idx:2d}/{last} {module.key} "
            f"peak_alloc={torch.cuda.max_memory_allocated(device) / 1024**3:.2f}GB "
            f"mem_avail={now:.1f}GB mem_avail_low={low_water:.1f}GB{probe}",
            flush=True,
        )

    meta = {
        "reference": "exllamav3 streaming uncached forward (flash_attn_nc)",
        "exllamav3_src": EXL3_SRC,
        "exllamav3_ext": EXL3_EXT,
        "model": MODEL,
        "torch": torch.__version__,
        "prompts_sha256": hashlib.sha256(open(args.prompts, "rb").read()).hexdigest(),
        "names": [p["name"] for p in prompts],
        "files": files,
        "peak_alloc_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
        "mem_available_low_water_gb": low_water,
        "seconds": time.time() - started,
    }
    json.dump(meta, open(os.path.join(args.out_dir, "meta.json"), "w"), indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
