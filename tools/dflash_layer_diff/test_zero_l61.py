#!/usr/bin/env python3
"""Test: zero out L61 (the most-drifted capture layer) and see if drafter
predictions improve.

Reads Atlas's /tmp/atlas_target_hidden.bin, zeros out the L61 slice
(slot 4 of 5), runs drafter forward, compares drafts to target's
verified[].
"""
import os, sys, math, struct
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

DRAFTER_PATH = "/home/flocka/models/z-lab-Qwen3.6-27B-DFlash/model.safetensors"
TARGET_PATH  = "/home/flocka/models/AEON-Q36-27B-Full/model.safetensors"
TARGET_HIDDEN_BIN = "/tmp/atlas_target_hidden.bin"

NUM_LAYERS = 5
HIDDEN = 5120
NUM_Q_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
RMS_EPS = 1e-6
ROPE_THETA = 10_000_000.0
ROTARY_DIM = HEAD_DIM
LAYER_TYPES = ['sliding', 'sliding', 'sliding', 'sliding', 'full']
SLIDING_WINDOW = 2048
MASK_TOKEN_ID = 248070
GAMMA = 16


def bf16_bytes_to_f32(b):
    u = np.frombuffer(b, dtype=np.uint16)
    return ((u.astype(np.uint32)) << 16).view(np.float32)


def rms_norm(x, w, eps):
    xf = x.float()
    return (xf * (xf.pow(2).mean(-1, keepdim=True) + eps).rsqrt() * w.float()).to(x.dtype)


def inv_freq(dim, theta):
    return 1.0 / (theta ** (torch.arange(0, dim // 2, dtype=torch.float32) * 2.0 / dim))


def rope(qk, pos, ifreq):
    n, h, d = qk.shape
    half = d // 2
    angles = pos.float().unsqueeze(1) * ifreq.unsqueeze(0)
    cos = angles.cos().unsqueeze(1).expand(n, h, half)
    sin = angles.sin().unsqueeze(1).expand(n, h, half)
    x0 = qk[..., :half].float()
    x1 = qk[..., half:].float()
    out = torch.empty_like(qk, dtype=torch.float32)
    out[..., :half] = x0 * cos - x1 * sin
    out[..., half:] = x1 * cos + x0 * sin
    return out.to(qk.dtype)


def load_weights():
    w = {}
    with safe_open(DRAFTER_PATH, framework='pt') as st:
        for k in st.keys():
            w[k] = st.get_tensor(k)
    with safe_open(TARGET_PATH, framework='pt') as st:
        for k in st.keys():
            if 'embed_tokens' in k and 'layers' not in k:
                w['_embed'] = st.get_tensor(k)
            if 'lm_head' in k and 'weight' in k:
                w['_lm_head'] = st.get_tensor(k)
    return w


def run_drafter(target_hidden_stack, last_token, position, eff_ctx, w, label):
    fc_proj = F.linear(target_hidden_stack.float(), w['fc.weight'].float()).to(torch.bfloat16)
    fc_proj = rms_norm(fc_proj, w['hidden_norm.weight'], RMS_EPS)

    ids = torch.tensor([last_token] + [MASK_TOKEN_ID] * (GAMMA - 1), dtype=torch.long)
    noise = w['_embed'][ids].to(torch.bfloat16)

    n_attn = eff_ctx + GAMMA
    stream = torch.zeros(n_attn, HIDDEN, dtype=torch.bfloat16)
    stream[eff_ctx:] = noise

    pos_off = position - eff_ctx
    pos = torch.cat([
        torch.arange(pos_off, pos_off + eff_ctx, dtype=torch.long),
        torch.arange(pos_off + eff_ctx, pos_off + eff_ctx + GAMMA, dtype=torch.long),
    ])
    ifreq = inv_freq(ROTARY_DIM, ROPE_THETA)
    Q_DIM = NUM_Q_HEADS * HEAD_DIM
    KV_DIM = NUM_KV_HEADS * HEAD_DIM
    INV_D = 1.0 / math.sqrt(HEAD_DIM)

    for layer_idx in range(NUM_LAYERS):
        p = f'layers.{layer_idx}'
        in_norm = rms_norm(stream, w[f'{p}.input_layernorm.weight'], RMS_EPS)
        q = F.linear(in_norm.float(), w[f'{p}.self_attn.q_proj.weight'].float()).to(torch.bfloat16)
        k = F.linear(in_norm.float(), w[f'{p}.self_attn.k_proj.weight'].float()).to(torch.bfloat16)
        v = F.linear(in_norm.float(), w[f'{p}.self_attn.v_proj.weight'].float()).to(torch.bfloat16)
        if eff_ctx > 0:
            k_ctx = F.linear(fc_proj.float(), w[f'{p}.self_attn.k_proj.weight'].float()).to(torch.bfloat16)
            v_ctx = F.linear(fc_proj.float(), w[f'{p}.self_attn.v_proj.weight'].float()).to(torch.bfloat16)
            k[:eff_ctx] = k_ctx
            v[:eff_ctx] = v_ctx
            q[:eff_ctx] = 0

        q_h = q.view(n_attn, NUM_Q_HEADS, HEAD_DIM)
        k_h = k.view(n_attn, NUM_KV_HEADS, HEAD_DIM)
        q_n = rms_norm(q_h, w[f'{p}.self_attn.q_norm.weight'], RMS_EPS)
        k_n = rms_norm(k_h, w[f'{p}.self_attn.k_norm.weight'], RMS_EPS)
        q_r = rope(q_n, pos, ifreq)
        k_r = rope(k_n, pos, ifreq)
        n_rep = NUM_Q_HEADS // NUM_KV_HEADS
        v_h = v.view(n_attn, NUM_KV_HEADS, HEAD_DIM)
        k_full = k_r.repeat_interleave(n_rep, dim=1)
        v_full = v_h.repeat_interleave(n_rep, dim=1)
        qp = q_r.permute(1, 0, 2).float()
        kp = k_full.permute(1, 0, 2).float()
        vp = v_full.permute(1, 0, 2).float()
        scores = torch.matmul(qp, kp.transpose(-1, -2)) * INV_D
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, vp).permute(1, 0, 2).contiguous().reshape(n_attn, Q_DIM).to(torch.bfloat16)

        stream_acc = F.linear(out.float(), w[f'{p}.self_attn.o_proj.weight'].float()).to(torch.bfloat16)
        stream = (stream.float() + stream_acc.float()).to(torch.bfloat16)
        post = rms_norm(stream, w[f'{p}.post_attention_layernorm.weight'], RMS_EPS)
        gate = F.linear(post.float(), w[f'{p}.mlp.gate_proj.weight'].float())
        up = F.linear(post.float(), w[f'{p}.mlp.up_proj.weight'].float())
        gated = (F.silu(gate) * up).to(torch.bfloat16)
        mlp = F.linear(gated.float(), w[f'{p}.mlp.down_proj.weight'].float()).to(torch.bfloat16)
        stream = (stream.float() + mlp.float()).to(torch.bfloat16)

    normed = rms_norm(stream[eff_ctx:], w['norm.weight'], RMS_EPS)
    logits = F.linear(normed.float(), w['_lm_head'][:248320].float())
    drafts = logits.argmax(-1).tolist()
    print(f"  [{label}] drafts: {drafts}")
    return drafts


def main():
    import json
    info = json.load(open('/tmp/atlas_tokens.json'))
    tokens = info['all_tokens']
    last_token = info['last_token']
    position = info['position']

    raw = open(TARGET_HIDDEN_BIN, 'rb').read()
    n_floats = len(raw) // 2
    n_ctx = n_floats // (5 * HIDDEN)
    print(f"n_ctx={n_ctx} last_token={last_token} position={position}")

    arr = bf16_bytes_to_f32(raw).reshape(n_ctx, 5, HIDDEN)

    print(f"\nLoading drafter + target embed/lm_head...")
    w = load_weights()

    print(f"\n=== Run 1: original L1+L16+L31+L46+L61 ===")
    th_orig = torch.from_numpy(arr.reshape(n_ctx, 5 * HIDDEN).copy()).to(torch.bfloat16)
    d1 = run_drafter(th_orig, last_token, position, n_ctx, w, "original")

    print(f"\n=== Run 2: zero out L61 (most-drifted) ===")
    arr2 = arr.copy()
    arr2[:, 4, :] = 0  # zero L61 slot
    th_zero61 = torch.from_numpy(arr2.reshape(n_ctx, 5 * HIDDEN)).to(torch.bfloat16)
    d2 = run_drafter(th_zero61, last_token, position, n_ctx, w, "zero L61")

    print(f"\n=== Run 3: zero out L46 + L61 (top-2 drifted) ===")
    arr3 = arr.copy()
    arr3[:, 3, :] = 0  # zero L46
    arr3[:, 4, :] = 0  # zero L61
    th_zero46_61 = torch.from_numpy(arr3.reshape(n_ctx, 5 * HIDDEN)).to(torch.bfloat16)
    d3 = run_drafter(th_zero46_61, last_token, position, n_ctx, w, "zero L46+L61")

    print(f"\nDiffs from original:")
    print(f"  zero L61:     {sum(1 for a,b in zip(d1,d2) if a != b)} / {GAMMA} drafts changed")
    print(f"  zero L46+L61: {sum(1 for a,b in zip(d1,d3) if a != b)} / {GAMMA} drafts changed")


if __name__ == '__main__':
    main()
