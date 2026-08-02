# SPDX-License-Identifier: AGPL-3.0-only
"""Offline DSpark drafter acceptance probe.

Replays an ATLAS_DSPARK_DUMP capture (hc-mean target hiddens at layers
40/41/42 + the greedy token stream) through the OFFICIAL DeepSeek DSpark
reference implementation (model.py, verbatim from the 0731 checkpoint's
inference/ dir; kernel.py is a pure-torch shim). Measures how many of the
drafter's block_size=5 proposals match what the Atlas server actually
generated — the acceptance rate that decides whether the in-engine port
(docs/dspark_port.md) is worth its verify cost, and specifically whether the
drafter (trained against non-REAP 0731 hiddens) survives our REAP-162B
target's hidden drift.

Usage: python3 probe.py [dump.bin]
"""

import json
import os
import struct
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as M  # noqa: E402

DRAFTER_DIR = "/home/flocka/models/DeepSeek-V4-Flash-0731-drafter"
TARGET_DIR = "/home/flocka/models/DeepSeek-V4-Flash-162B"
DUMP = sys.argv[1] if len(sys.argv) > 1 else "/home/flocka/deepseek-flash/dspark_dump.bin"

E2M1_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
)


def dequant_fp8_block(w, s, block=128):
    """[out,in] e4m3 + [ceil(out/128), ceil(in/128)] e8m0 -> bf16."""
    out, inn = w.shape
    sf = s.float().repeat_interleave(block, 0)[:out].repeat_interleave(block, 1)[:, :inn]
    return (w.float() * sf).to(torch.bfloat16)


def dequant_mxfp4(w_packed, s, group=32):
    """[out, in//2] packed e2m1 pairs + [out, in//32] e8m0 -> bf16 [out, in].
    Low nibble = even element (matches torch float4_e2m1fn_x2 packing and the
    Atlas kernels' `byte & 0xF` / `byte >> 4` convention)."""
    b = w_packed.view(torch.uint8)
    lo = (b & 0xF).long()
    hi = (b >> 4).long()
    out, half = b.shape
    vals = torch.empty(out, half * 2, dtype=torch.float32, device=b.device)
    vals[:, 0::2] = E2M1_LUT.to(b.device)[lo]
    vals[:, 1::2] = E2M1_LUT.to(b.device)[hi]
    sf = s.float().repeat_interleave(group, 1)[:, : half * 2]
    return (vals * sf).to(torch.bfloat16)


def read_dump(path):
    recs = []
    with open(path, "rb") as f:
        while True:
            hdr = f.read(28)
            if len(hdr) < 28:
                break
            magic, kind, start, n, h, nl, token = struct.unpack("<7I", hdr)
            assert magic == 0x4453504B
            data = f.read(nl * n * h * 2)
            t = torch.frombuffer(bytearray(data), dtype=torch.bfloat16).view(nl, n, h)
            recs.append(dict(kind=kind, start=start, n=n, token=token, h=t))
    return recs


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(dev)
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(0)

    cfg = json.load(open(f"{DRAFTER_DIR}/config-0731.json"))
    args = M.ModelArgs(
        max_batch_size=1,
        max_seq_len=4096,
        temperature=0.0,  # greedy — matches the dump run
        dtype="bf16",
        scale_fmt=None,
        expert_dtype=None,  # everything dequantized to BF16 at load
        scale_dtype="fp32",
        vocab_size=cfg["vocab_size"],
        dim=cfg["hidden_size"],
        moe_inter_dim=cfg["moe_intermediate_size"],
        n_layers=cfg["num_hidden_layers"],
        n_hash_layers=cfg.get("num_hash_layers", 0),
        n_mtp_layers=3,
        n_heads=cfg["num_attention_heads"],
        n_routed_experts=cfg["n_routed_experts"],
        n_shared_experts=cfg["n_shared_experts"],
        n_activated_experts=cfg["num_experts_per_tok"],
        score_func=cfg["scoring_func"],
        route_scale=cfg["routed_scaling_factor"],
        swiglu_limit=cfg["swiglu_limit"],
        q_lora_rank=cfg["q_lora_rank"],
        head_dim=cfg["head_dim"],
        rope_head_dim=cfg["qk_rope_head_dim"],
        norm_eps=cfg["rms_norm_eps"],
        o_groups=cfg["o_groups"],
        o_lora_rank=cfg["o_lora_rank"],
        window_size=cfg["sliding_window"],
        compress_ratios=tuple(cfg["compress_ratios"]),
        rope_theta=cfg["rope_theta"],
        rope_factor=cfg["rope_scaling"]["factor"],
        beta_fast=cfg["rope_scaling"]["beta_fast"],
        beta_slow=cfg["rope_scaling"]["beta_slow"],
        original_seq_len=cfg["rope_scaling"]["original_max_position_embeddings"],
        hc_mult=cfg["hc_mult"],
        hc_sinkhorn_iters=cfg["hc_sinkhorn_iters"],
        hc_eps=cfg["hc_eps"],
        dspark_block_size=cfg["dspark_block_size"],
        dspark_noise_token_id=cfg["dspark_noise_token_id"],
        dspark_target_layer_ids=tuple(cfg["dspark_target_layer_ids"]),
        dspark_markov_rank=cfg["dspark_markov_rank"],
    )
    M.world_size, M.rank = 1, 0
    M.default_dtype = torch.bfloat16
    M.scale_fmt = cfg["quantization_config"]["scale_fmt"]  # KV round-trip fidelity
    M.scale_dtype = torch.float32
    M.Attention.__init__.__globals__  # noqa: B018 — sanity that model is loaded

    print(f"building 3 DSpark stages on {dev} ...")
    blocks = torch.nn.ModuleList(
        [M.DSparkBlock(args.n_layers + i, args) for i in range(args.n_mtp_layers)]
    )
    embed = M.ParallelEmbedding(args.vocab_size, args.dim)
    head = M.ParallelHead(args.vocab_size, args.dim, args.norm_eps, args.hc_eps)
    for b in blocks:
        b.embed, b.head = embed, head

    # ── drafter weights ──
    params = dict(blocks.named_parameters())
    params.update(dict(blocks.named_buffers()))
    tensors = {}
    for s in (46, 47, 48):
        p = f"{DRAFTER_DIR}/model-000{s}-of-00048.safetensors"
        with safe_open(p, framework="pt", device="cpu") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
    loaded = 0
    for name, t in tensors.items():
        if name.endswith(".scale"):
            continue
        # mtp.{i}.rest -> ModuleList path {i}.rest
        assert name.startswith("mtp.")
        pname = name[len("mtp."):]
        if pname.endswith(".weight") is False and ".weight" not in pname:
            pass
        key = pname[: -len(".weight")] if pname.endswith(".weight") else pname
        target_key = pname if pname in params else None
        if target_key is None:
            # parameters registered without ".weight" (hc_* tensors, attn_sink)
            target_key = key if key in params else None
        if target_key is None:
            print(f"  SKIP (no module param): {name} {tuple(t.shape)} {t.dtype}")
            continue
        dst = params[target_key]
        scale = tensors.get(name.replace(".weight", ".scale"))
        if t.dtype == torch.float8_e4m3fn:
            src = dequant_fp8_block(t, scale)
        elif t.dtype in (torch.int8, torch.uint8) or "float4" in str(t.dtype):
            src = dequant_mxfp4(t, scale)
        else:
            src = t
        assert src.shape == dst.shape, f"{name}: {tuple(src.shape)} vs {tuple(dst.shape)}"
        with torch.no_grad():
            dst.copy_(src.to(dst.dtype))
        loaded += 1
    print(f"drafter: {loaded} tensors loaded")

    # ── shared embed + lm_head from the 162B TARGET ──
    idx = json.load(open(f"{TARGET_DIR}/model.safetensors.index.json"))["weight_map"]
    for tname, dst in (("embed.weight", embed.weight), ("head.weight", head.weight)):
        shard = idx[tname]
        with safe_open(f"{TARGET_DIR}/{shard}", framework="pt", device="cpu") as f:
            t = f.get_tensor(tname)
            scale_name = tname.replace(".weight", ".scale")
            if t.dtype == torch.float8_e4m3fn:
                sh = idx[scale_name]
                with safe_open(f"{TARGET_DIR}/{sh}", framework="pt", device="cpu") as f2:
                    t = dequant_fp8_block(t, f2.get_tensor(scale_name))
            with torch.no_grad():
                dst.copy_(t.to(dst.dtype))
        print(f"target {tname}: {tuple(t.shape)} loaded")

    # ── replay ──
    # The dump may hold several sequences back-to-back (one per request);
    # each starts with a prefill record at start == 0.
    recs = read_dump(DUMP)
    seqs = []
    for r in recs:
        if r["kind"] == 0 and r["start"] == 0:
            seqs.append([])
        if seqs:
            seqs[-1].append(r)
    print(f"dump: {len(recs)} records, {len(seqs)} sequences")
    assert seqs, "no prefill records — rebuild the server with the prefill_b hook"

    def fuse(rec_slice):  # [nl, n, h] -> [1, n, nl*h]
        return rec_slice.permute(1, 0, 2).reshape(1, rec_slice.shape[1], -1).to(dev)

    bs = args.dspark_block_size
    n_props = 0
    chain_hist = [0] * (bs + 1)
    pos_match = [0] * bs
    pos_total = [0] * bs
    conf_kept_chain = []
    with torch.inference_mode():
        for si, seq in enumerate(seqs):
            pre = [r for r in seq if r["kind"] == 0]
            dec = [r for r in seq if r["kind"] == 1]
            if not dec:
                continue
            # Seed each stage's ring from the prompt (forward_spec start_pos=0).
            main_hidden = torch.cat(
                [fuse(r["h"]) for r in sorted(pre, key=lambda r: r["start"])], dim=1
            )
            ids0 = torch.tensor([[dec[0]["token"]]], device=dev)
            h, main_x = blocks[0].forward_embed(main_hidden, ids0[:, 0])
            for b in blocks:
                h = b(h, 0, ids0, main_x)
            print(f"seq {si}: ring seeded from {main_hidden.shape[1]} prompt rows, "
                  f"{len(dec)} decode positions")

            tok_at = {r["start"]: r["token"] for r in dec}
            for r in dec:
                # Reference alignment (model.py __main__): forward_spec is
                # called with the token GENERATED from position p (= the next
                # record's input token) as the committed block row, and
                # main_hidden@p. Record p's own token is already folded into
                # its hidden — embedding it again is off-distribution.
                p = r["start"]
                nxt = tok_at.get(p + 1)
                if nxt is None:
                    continue
                mh = fuse(r["h"])
                ids = torch.tensor([[nxt]], device=dev)
                h, main_x = blocks[0].forward_embed(mh, ids[:, 0])
                for b in blocks:
                    h = b(h, p, ids, main_x)
                out_ids, logits, conf = blocks[-1].forward_head(h, ids[:, 0])
                drafts = out_ids[0, 1:].tolist()
                confs = torch.sigmoid(conf[0].float()).tolist()
                actual = [tok_at.get(p + 2 + j) for j in range(bs)]
                if actual[0] is None:
                    continue
                n_props += 1
                chain = 0
                for j in range(bs):
                    if actual[j] is None:
                        break
                    pos_total[j] += 1
                    if drafts[j] == actual[j]:
                        pos_match[j] += 1
                        if chain == j:
                            chain = j + 1
                chain_hist[chain] += 1
                kept = 0
                for j in range(bs):
                    if confs[j] < 0.9:
                        break
                    kept += 1
                conf_kept_chain.append((kept, chain))

    print(f"\n== DSpark offline acceptance ({n_props} propose points) ==")
    for j in range(bs):
        if pos_total[j]:
            print(f"  draft[{j}] match: {pos_match[j]}/{pos_total[j]} = {pos_match[j]/pos_total[j]*100:.1f}%")
    mean_chain = sum(i * c for i, c in enumerate(chain_hist)) / max(n_props, 1)
    print(f"  chain hist (0..{bs} accepted): {chain_hist}")
    print(f"  mean accepted chain = {mean_chain:.2f}  -> tok/step = {mean_chain + 1:.2f}")
    if conf_kept_chain:
        # How well the confidence head predicts the chain (kept vs chain)
        over = sum(1 for k, c in conf_kept_chain if k > c)
        under = sum(1 for k, c in conf_kept_chain if k < c)
        kept_mean = sum(k for k, _ in conf_kept_chain) / len(conf_kept_chain)
        print(f"  confidence@0.9: mean kept {kept_mean:.2f} (over-keep {over}, under-keep {under})")
        eff = sum(min(k, c) for k, c in conf_kept_chain) / len(conf_kept_chain)
        print(f"  confidence-gated mean accepted = {eff:.2f} -> tok/step = {eff + 1:.2f}")


if __name__ == "__main__":
    main()
