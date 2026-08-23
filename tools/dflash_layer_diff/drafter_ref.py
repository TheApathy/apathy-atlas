#!/usr/bin/env python3
"""Pure-PyTorch reference implementation of the Qwen3.6-DFlash drafter.

Loads drafter weights from safetensors and runs the SAME forward chain
vLLM does. Dumps every layer's intermediates so we can compare element-wise
to Atlas's /tmp/atlas_layer*.bin dumps.

Usage:
    # First: run Atlas with ATLAS_DFLASH_DEBUG_DUMP_FULL=1 and
    # ATLAS_DFLASH_DEBUG_DUMP_ALL_LAYERS=1 to populate /tmp/atlas_*.bin
    # Then:
    python3 drafter_ref.py
    python3 compare_layers.py

Reads:
    /tmp/atlas_target_hidden.bin     -- target's captured hidden states
                                        (eff_ctx slots × 5 layers × 5120 BF16)
    $MODELS/z-lab-Qwen3.6-27B-DFlash/model.safetensors
                                     -- drafter weights
    AEON-Q36-27B-Full embed/lm_head -- shared with target (Atlas uses these)

Writes:
    /tmp/ref_layer{i}_{tag}.bin     -- ref intermediates matching Atlas tags
    /tmp/ref_final_logits.bin
    /tmp/ref_final_drafts.bin
"""
from __future__ import annotations
import os, sys, math, struct
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

# Checkpoint directory; override with the MODELS env var.
MODELS_DIR = os.environ.get("MODELS", os.path.expanduser("~/models"))

DRAFTER_PATH = os.path.join(MODELS_DIR, "z-lab-Qwen3.6-27B-DFlash/model.safetensors")
TARGET_PATH  = os.path.join(MODELS_DIR, "AEON-Q36-27B-Full/model.safetensors")
TARGET_HIDDEN_BIN = "/tmp/atlas_target_hidden.bin"
DUMP_DIR = Path("/tmp")

# Drafter config (from config.json)
NUM_LAYERS = 5
HIDDEN = 5120
INTER = 17408
NUM_Q_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
RMS_EPS = 1e-6
ROPE_THETA = 10_000_000.0
ROTARY_DIM = HEAD_DIM
SLIDING_WINDOW = 2048
LAYER_TYPES = ['sliding', 'sliding', 'sliding', 'sliding', 'full']
TARGET_LAYER_IDS = [1, 16, 31, 46, 61]
MASK_TOKEN_ID = 248070
GAMMA = 16


def bf16_bytes_to_fp32(b: bytes) -> np.ndarray:
    u = np.frombuffer(b, dtype=np.uint16)
    f = ((u.astype(np.uint32)) << 16).view(np.float32)
    return f


def fp32_to_bf16_bytes(arr: np.ndarray) -> bytes:
    a = np.ascontiguousarray(arr, dtype=np.float32)
    u32 = a.view(np.uint32)
    u16 = (u32 >> 16).astype(np.uint16)
    return u16.tobytes()


def load_drafter():
    """Load all drafter weights into a dict of torch tensors (BF16)."""
    weights = {}
    with safe_open(DRAFTER_PATH, framework='pt') as st:
        for k in st.keys():
            weights[k] = st.get_tensor(k)
    print(f"Loaded {len(weights)} drafter weights")
    return weights


def load_target_embed():
    """Load shared embed_tokens (used for both noise embedding and mask token)."""
    with safe_open(TARGET_PATH, framework='pt') as st:
        for k in st.keys():
            if 'embed_tokens' in k and 'layers' not in k:
                t = st.get_tensor(k)
                print(f"Loaded target embed: {k} shape={list(t.shape)}")
                return t
    raise RuntimeError("embed_tokens not found")


def load_target_lm_head():
    with safe_open(TARGET_PATH, framework='pt') as st:
        for k in st.keys():
            if 'lm_head' in k and 'weight' in k:
                t = st.get_tensor(k)
                print(f"Loaded target lm_head: {k} shape={list(t.shape)}")
                return t
    raise RuntimeError("lm_head not found")


def rms_norm_vanilla(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """HF-style vanilla RMSNorm: out = x * w / sqrt(mean(x^2) + eps)."""
    x_f32 = x.float()
    rms = (x_f32.pow(2).mean(-1, keepdim=True) + eps).rsqrt()
    return (x_f32 * rms * weight.float()).to(x.dtype)


def vanilla_inv_freq(dim: int, theta: float) -> torch.Tensor:
    n_pairs = dim // 2
    return 1.0 / (theta ** (torch.arange(0, n_pairs, dtype=torch.float32) * 2.0 / dim))


def apply_rope_neox(qk: torch.Tensor, positions: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
    """NeoX-style RoPE on shape [N, num_heads, head_dim].

    NeoX pairing: rotate (d_i, d_{i+half_rot}) per head, where i < half_rot.
    """
    n, h, d = qk.shape
    half = d // 2
    # positions: [N]; inv_freq: [half]
    angles = positions.float().unsqueeze(1) * inv_freq.unsqueeze(0)  # [N, half]
    cos = angles.cos().unsqueeze(1).expand(n, h, half)
    sin = angles.sin().unsqueeze(1).expand(n, h, half)
    x0 = qk[..., :half].float()
    x1 = qk[..., half:].float()
    y0 = x0 * cos - x1 * sin
    y1 = x1 * cos + x0 * sin
    out = torch.empty_like(qk, dtype=torch.float32)
    out[..., :half] = y0
    out[..., half:] = y1
    return out.to(qk.dtype)


def dump_bin(path: Path, t: torch.Tensor):
    arr = t.detach().to(torch.bfloat16).cpu().contiguous().view(torch.uint16).numpy()
    path.write_bytes(arr.tobytes())
    print(f"  wrote {path} shape={list(t.shape)} n_elems={t.numel()}")


def main():
    # Read target_hidden_stack from Atlas dump
    if not os.path.exists(TARGET_HIDDEN_BIN):
        print(f"ERROR: {TARGET_HIDDEN_BIN} not found. Run Atlas with "
              f"ATLAS_DFLASH_DEBUG_DUMP_FULL=1 first.")
        sys.exit(1)
    raw = open(TARGET_HIDDEN_BIN, 'rb').read()
    n_floats = len(raw) // 2
    assert n_floats % (5 * HIDDEN) == 0, f"file size {len(raw)} not divisible by 5*hidden*2={5*HIDDEN*2}"
    n_ctx = n_floats // (5 * HIDDEN)
    print(f"target_hidden_stack: n_ctx={n_ctx} slots × 5 layers × {HIDDEN} BF16")
    t_arr = bf16_bytes_to_fp32(raw).reshape(n_ctx, 5 * HIDDEN)
    target_hidden_stack = torch.from_numpy(t_arr).to(torch.bfloat16)  # [n_ctx, 5*5120]

    # Load drafter weights
    w = load_drafter()
    embed = load_target_embed()       # [vocab, 5120] BF16
    lm_head = load_target_lm_head()   # [vocab, 5120] BF16

    # Inputs
    print("\n=== Step 0: fc + hidden_norm ===")
    fc_weight = w['fc.weight']        # [5120, 25600]
    hidden_norm = w['hidden_norm.weight']  # [5120]
    fc_proj = F.linear(target_hidden_stack.float(), fc_weight.float()).to(torch.bfloat16)
    fc_proj = rms_norm_vanilla(fc_proj, hidden_norm, RMS_EPS)
    print(f"fc_proj shape={list(fc_proj.shape)} mean={fc_proj.float().mean():.4f} std={fc_proj.float().std():.4f}")

    # === Step 1: build noise inputs ===
    # vLLM-aligned: query = [bonus=last_token, mask × γ] → noise_count = γ+1.
    # Read inputs from /tmp/atlas_dump_meta.json (written by forward_block.rs
    # alongside the target_hidden binary) so reference + Atlas use identical
    # last_token / position / eff_ctx.
    import json as _json
    meta_path = "/tmp/atlas_dump_meta.json"
    if os.path.exists(meta_path):
        meta = _json.load(open(meta_path))
        LAST_TOKEN = int(meta["last_token"])
        POSITION = int(meta["position"])
        meta_eff_ctx = int(meta["eff_ctx"])
        if meta_eff_ctx != n_ctx:
            print(f"WARN: meta eff_ctx={meta_eff_ctx} != binary n_ctx={n_ctx} — using binary")
        print(f"[meta] last_token={LAST_TOKEN} position={POSITION} eff_ctx={n_ctx}")
    else:
        LAST_TOKEN = 248068
        POSITION = n_ctx
        print(f"WARN: no {meta_path}; falling back to LAST_TOKEN={LAST_TOKEN} POSITION={POSITION}")

    NOISE_COUNT = GAMMA + 1  # 1 bonus + γ MASK
    print(f"\n=== Step 1: noise embeddings [bonus={LAST_TOKEN}, mask × {GAMMA}] noise_count={NOISE_COUNT} ===")
    ids = torch.tensor([LAST_TOKEN] + [MASK_TOKEN_ID] * GAMMA, dtype=torch.long)
    noise_embeds = embed[ids].to(torch.bfloat16)  # [γ+1, 5120]
    print(f"noise_embeds shape={list(noise_embeds.shape)} mean={noise_embeds.float().mean():.5f} std={noise_embeds.float().std():.5f}")

    # ctx slots zeroed, noise rows = noise_embeds
    eff_ctx = n_ctx
    n_attn = eff_ctx + NOISE_COUNT
    stream = torch.zeros(n_attn, HIDDEN, dtype=torch.bfloat16)
    stream[eff_ctx:] = noise_embeds

    # Position IDs (Atlas layout):
    #   ctx_pos_i  = POSITION - eff_ctx + i   for i in [0, eff_ctx)
    #   noise_pos_i = POSITION + i            for i in [0, NOISE_COUNT)
    ctx_start = max(0, POSITION - eff_ctx)
    pos = torch.cat([
        torch.arange(ctx_start, ctx_start + eff_ctx, dtype=torch.long),
        torch.arange(POSITION, POSITION + NOISE_COUNT, dtype=torch.long),
    ])
    print(f"[pos] ctx=[{ctx_start}..{ctx_start+eff_ctx}) noise=[{POSITION}..{POSITION+NOISE_COUNT}) n_attn={n_attn}")

    inv_freq = vanilla_inv_freq(ROTARY_DIM, ROPE_THETA)

    print("\n=== Step 3: drafter layers ===")
    Q_DIM = NUM_Q_HEADS * HEAD_DIM
    KV_DIM = NUM_KV_HEADS * HEAD_DIM
    INV_SQRT_D = 1.0 / math.sqrt(HEAD_DIM)

    for layer_idx in range(NUM_LAYERS):
        p = f'layers.{layer_idx}'
        input_norm_w = w[f'{p}.input_layernorm.weight']
        post_norm_w  = w[f'{p}.post_attention_layernorm.weight']
        q_proj_w = w[f'{p}.self_attn.q_proj.weight']  # [Q_DIM, hidden]
        k_proj_w = w[f'{p}.self_attn.k_proj.weight']  # [KV_DIM, hidden]
        v_proj_w = w[f'{p}.self_attn.v_proj.weight']  # [KV_DIM, hidden]
        o_proj_w = w[f'{p}.self_attn.o_proj.weight']  # [hidden, Q_DIM]
        q_norm_w = w[f'{p}.self_attn.q_norm.weight']  # [HEAD_DIM]
        k_norm_w = w[f'{p}.self_attn.k_norm.weight']  # [HEAD_DIM]
        gate_w = w[f'{p}.mlp.gate_proj.weight']       # [INTER, hidden]
        up_w   = w[f'{p}.mlp.up_proj.weight']
        down_w = w[f'{p}.mlp.down_proj.weight']       # [hidden, INTER]

        # 3a. input layer norm
        normed = rms_norm_vanilla(stream, input_norm_w, RMS_EPS)

        # 3b. q/k/v proj (NOISE rows only — vLLM does this)
        q = F.linear(normed.float(), q_proj_w.float()).to(torch.bfloat16)  # [n_attn, Q_DIM]
        k = F.linear(normed.float(), k_proj_w.float()).to(torch.bfloat16)
        v = F.linear(normed.float(), v_proj_w.float()).to(torch.bfloat16)

        # 3b'. ctx K/V override from fc_proj
        if eff_ctx > 0:
            k_ctx = F.linear(fc_proj.float(), k_proj_w.float()).to(torch.bfloat16)
            v_ctx = F.linear(fc_proj.float(), v_proj_w.float()).to(torch.bfloat16)
            k[:eff_ctx] = k_ctx
            v[:eff_ctx] = v_ctx
            q[:eff_ctx] = 0  # zero ctx Q

        # 3c. per-head q/k norm
        q_h = q.view(n_attn, NUM_Q_HEADS, HEAD_DIM)
        k_h = k.view(n_attn, NUM_KV_HEADS, HEAD_DIM)
        q_normed = rms_norm_vanilla(q_h, q_norm_w, RMS_EPS)
        k_normed = rms_norm_vanilla(k_h, k_norm_w, RMS_EPS)
        dump_bin(DUMP_DIR / f"ref_layer{layer_idx}_q_post_norm.bin", q_normed.reshape(n_attn, Q_DIM))
        dump_bin(DUMP_DIR / f"ref_layer{layer_idx}_k_post_norm.bin", k_normed.reshape(n_attn, KV_DIM))
        dump_bin(DUMP_DIR / f"ref_layer{layer_idx}_v_buf.bin", v.reshape(n_attn, KV_DIM))

        # 3d. RoPE (NeoX-style)
        q_rope = apply_rope_neox(q_normed, pos, inv_freq)
        k_rope = apply_rope_neox(k_normed, pos, inv_freq)
        dump_bin(DUMP_DIR / f"ref_layer{layer_idx}_q_post_rope.bin", q_rope.reshape(n_attn, Q_DIM))
        dump_bin(DUMP_DIR / f"ref_layer{layer_idx}_k_post_rope.bin", k_rope.reshape(n_attn, KV_DIM))

        # 3e. attention (GQA: repeat K/V to num_q_heads)
        n_rep = NUM_Q_HEADS // NUM_KV_HEADS  # 4
        # k_rope/q_rope are 3D [n_attn, num_heads, head_dim]; v is 2D [n_attn, KV_DIM]
        v_h = v.view(n_attn, NUM_KV_HEADS, HEAD_DIM)
        k_full = k_rope.repeat_interleave(n_rep, dim=1)
        v_full = v_h.repeat_interleave(n_rep, dim=1)
        q_perm = q_rope.permute(1, 0, 2).float()       # [num_q, n_attn, head_dim]
        k_perm = k_full.permute(1, 0, 2).float()
        v_perm = v_full.permute(1, 0, 2).float()

        # Atlas: causal=false for all layers. Per-layer SWA for first 4 layers.
        scores = torch.matmul(q_perm, k_perm.transpose(-1, -2)) * INV_SQRT_D  # [num_q, n_attn, n_attn]

        if LAYER_TYPES[layer_idx] == 'sliding':
            # SWA mask only applies when causal=true in Atlas's kernel, which it isn't.
            # So no mask in non-causal mode → full attention. Skip SWA mask.
            pass

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v_perm)  # [num_q, n_attn, head_dim]
        out_perm = out.permute(1, 0, 2).contiguous()  # [n_attn, num_q, head_dim]
        attn_out = out_perm.reshape(n_attn, Q_DIM).to(torch.bfloat16)
        dump_bin(DUMP_DIR / f"ref_layer{layer_idx}_attn_out.bin", attn_out)

        # 3f. o_proj
        stream_acc = F.linear(attn_out.float(), o_proj_w.float()).to(torch.bfloat16)
        dump_bin(DUMP_DIR / f"ref_layer{layer_idx}_stream_acc_post_o_proj.bin", stream_acc)

        # 3g. residual
        stream = (stream.float() + stream_acc.float()).to(torch.bfloat16)

        # 3h. post-attention layer norm
        normed = rms_norm_vanilla(stream, post_norm_w, RMS_EPS)

        # 3i-j. MLP
        gate = F.linear(normed.float(), gate_w.float())
        up = F.linear(normed.float(), up_w.float())
        gated = (F.silu(gate) * up).to(torch.bfloat16)
        mlp_out = F.linear(gated.float(), down_w.float()).to(torch.bfloat16)

        # 3k. residual
        stream = (stream.float() + mlp_out.float()).to(torch.bfloat16)
        dump_bin(DUMP_DIR / f"ref_layer{layer_idx}_stream_buf_post_mlp.bin", stream)

    # === Final norm + LM head (noise rows only) ===
    print("\n=== Final ===")
    final_norm_w = w['norm.weight']
    stream_noise = stream[eff_ctx:]
    normed = rms_norm_vanilla(stream_noise, final_norm_w, RMS_EPS)
    dump_bin(DUMP_DIR / "ref_final_norm_buf.bin", normed)

    # lm_head over shared target weights
    LM_VOCAB = min(lm_head.shape[0], 248320)
    logits = F.linear(normed.float(), lm_head[:LM_VOCAB].float()).to(torch.bfloat16)
    dump_bin(DUMP_DIR / "ref_final_logits.bin", logits)

    drafts = logits.float().argmax(-1).to(torch.uint32)
    print(f"ref drafts: {drafts.tolist()}")
    (DUMP_DIR / "ref_final_drafts.bin").write_bytes(drafts.numpy().astype(np.uint32).tobytes())


if __name__ == "__main__":
    main()
