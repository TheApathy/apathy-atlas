#!/usr/bin/env python3
"""DFlash drafter retrainer v2 — matches inference regime (γ=16 sequence).

Key fix vs v1: at inference the drafter sees
  - eff_ctx ≤ ctx_window prior target hiddens (each = 5 captures concat'd)
  - 1 BONUS row using last_token's actual embedding (not MASK)
  - γ MASK rows that it predicts into

For training we mirror exactly that:
  Given a stream of (hidden_5x5120, token) pairs from our dump, for each
  position P:
    target_hidden = hiddens[P-ctx_window .. P]  (ctx_window past positions)
    noise_embed  = [embed(tokens[P]), MASK_embed × γ]
    predict tokens[P+1 .. P+γ+1]  (γ targets)
    loss = CE over γ positions, optional decay weighting

This sees the same input distribution as inference. The bonus row +
mask block lets the drafter learn block-coherent γ-token prediction.
"""
import argparse
import os
import struct
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset, DataLoader

HIDDEN_MAGIC = 0xA71A5DEE
TOKEN_MAGIC = 0xA71B5DEE


def parse_dump_stream(path, hidden_dim=5120, n_capture=5):
    """Walk dump in emission order → flat list (hidden_5x5120 bf16, token_id long).

    Each verify step emits k_verify × n_capture HIDDEN records (k_verify=3 for K=3,
    17 for DFlash γ=16) followed by N_accepted TOKEN records. The hiddens for
    accepted-token position i correspond to the i-th block in the step.
    """
    data = open(path, "rb").read()
    nbytes = len(data)
    bytes_per_hidden = 16 + hidden_dim * 2
    stream_h, stream_t = [], []
    # Per-step buffers: token_idx -> ordered list of hidden tensors (one per
    # capture layer, in slot order). Layout (per flush_hidden_dump in impl_b3.rs):
    #   for t in 0..k {  for (slot, layer) in capture_layers { write(t, layer, h) } }
    #   then N_accepted TOKEN records.
    step_hiddens_by_tok: dict[int, list] = {}
    last_was_token = False
    off, nh, nt = 0, 0, 0
    n_accepted_in_step = 0
    while off + 16 <= nbytes:
        magic = struct.unpack_from("<I", data, off)[0]
        if magic == HIDDEN_MAGIC:
            if off + bytes_per_hidden > nbytes: break
            _layer_idx, token_idx, _hd = struct.unpack_from("<III", data, off + 4)
            # Step boundary: previous record was a TOKEN — flush state, start new step.
            if last_was_token:
                step_hiddens_by_tok = {}
                n_accepted_in_step = 0
                last_was_token = False
            payload = bytes(data[off + 16 : off + bytes_per_hidden])
            t = torch.frombuffer(payload, dtype=torch.bfloat16).clone()
            step_hiddens_by_tok.setdefault(token_idx, []).append(t)
            nh += 1
            off += bytes_per_hidden
        elif magic == TOKEN_MAGIC:
            tok = struct.unpack_from("<I", data, off + 4)[0]
            # The N-th TOKEN of this step pairs with hiddens at token_idx = N.
            if n_accepted_in_step in step_hiddens_by_tok:
                hs = step_hiddens_by_tok[n_accepted_in_step]
                if len(hs) == n_capture:
                    stream_h.append(torch.stack(hs))
                    stream_t.append(tok)
            n_accepted_in_step += 1
            last_was_token = True
            nt += 1
            off += 16
        else:
            break
    print(f"[parse] hidden={nh:,} token={nt:,} stream_pairs={len(stream_h):,}", file=sys.stderr)
    H = torch.stack(stream_h)           # [N, n_cap, H] bf16
    T = torch.tensor(stream_t, dtype=torch.long)
    return H, T


class SequenceDataset(Dataset):
    """Yield (ctx_hiddens [ctx_len, n_cap*H], bonus_token_id, future_tokens [γ]).

    Each item picks a position P with ≥γ tokens after it. ctx is the
    sliding window of up to ctx_window prior positions; bonus = token at P;
    future = tokens at P+1..P+γ.

    For positions P < ctx_window, ctx is just the available prefix
    (variable length); we'll right-pad in collate.
    """
    def __init__(self, H, T, gamma: int, ctx_window: int, n_cap: int):
        self.H = H              # [N, n_cap, hidden]
        self.T = T              # [N]
        self.gamma = gamma
        self.ctx_window = ctx_window
        self.n_cap = n_cap
        self.hidden_dim = H.shape[-1]
        # Valid positions: need γ future tokens
        N = len(T)
        self.valid_P = [P for P in range(N) if P + 1 + gamma <= N]

    def __len__(self): return len(self.valid_P)

    def __getitem__(self, i):
        P = self.valid_P[i]
        ctx_start = max(0, P - self.ctx_window)
        ctx = self.H[ctx_start : P]                # [ctx_len, n_cap, H]
        ctx_flat = ctx.reshape(-1, self.n_cap * self.hidden_dim)  # [ctx_len, n_cap*H]
        bonus_tok = self.T[P].item()
        future = self.T[P + 1 : P + 1 + self.gamma]  # [γ]
        return ctx_flat, bonus_tok, future


def collate(batch, ctx_window, n_cap_h):
    """Right-pad ctx to ctx_window, return mask of real positions."""
    B = len(batch)
    ctx = torch.zeros(B, ctx_window, n_cap_h, dtype=torch.bfloat16)
    ctx_mask = torch.zeros(B, ctx_window, dtype=torch.bool)
    bonus = torch.zeros(B, dtype=torch.long)
    futures = torch.zeros(B, batch[0][2].shape[0], dtype=torch.long)
    for i, (c, b, f) in enumerate(batch):
        L = c.shape[0]
        ctx[i, ctx_window - L : ctx_window] = c.to(torch.bfloat16)
        ctx_mask[i, ctx_window - L : ctx_window] = True
        bonus[i] = b
        futures[i] = f
    return ctx, ctx_mask, bonus, futures


def load_drafter_and_target(drafter_dir, target_path, device):
    from transformers import AutoConfig, AutoModel
    cfg = AutoConfig.from_pretrained(drafter_dir, trust_remote_code=True)
    cfg._attn_implementation = "sdpa"
    model = AutoModel.from_pretrained(
        drafter_dir, config=cfg, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to(device)
    embed, lm_head = None, None
    with safe_open(target_path, framework="pt") as f:
        for k in f.keys():
            if "embed_tokens.weight" in k:
                embed = f.get_tensor(k).to(device, dtype=torch.bfloat16)
            elif k == "lm_head.weight":
                lm_head = f.get_tensor(k).to(device, dtype=torch.bfloat16)
    return model, embed, lm_head, cfg


def freeze_all_but_fc(model):
    trainable, frozen = 0, 0
    for name, p in model.named_parameters():
        if name in ("fc.weight", "hidden_norm.weight"):
            p.requires_grad = True; trainable += p.numel()
        else:
            p.requires_grad = False; frozen += p.numel()
    return trainable, frozen


def forward_drafter_sequence(model, ctx, ctx_mask, bonus_ids, gamma, mask_id,
                              embed_table, lm_head_weight, device):
    """γ-token prediction matching inference regime.
    ctx:        [B, ctx_window, n_cap*H] bf16
    ctx_mask:   [B, ctx_window] bool (True = real position)
    bonus_ids:  [B] long (last_token at P)
    Returns logits [B, γ, vocab].
    """
    B, ctx_window, ncH = ctx.shape
    H = embed_table.shape[1]
    noise_count = gamma + 1
    # Noise: [bonus_emb, MASK_emb × γ]
    bonus_emb = F.embedding(bonus_ids, embed_table).unsqueeze(1)  # [B, 1, H]
    mask_emb = F.embedding(
        torch.full((B,), mask_id, device=device, dtype=torch.long), embed_table
    )  # [B, H]
    mask_block = mask_emb.unsqueeze(1).expand(B, gamma, H)        # [B, γ, H]
    noise = torch.cat([bonus_emb, mask_block], dim=1).to(torch.bfloat16)  # [B, γ+1, H]

    # Position ids cover ALL attention positions = ctx_window + noise_count
    # Layout: ctx positions are [0..ctx_window), noise positions [ctx_window..ctx_window+noise_count).
    total_len = ctx_window + noise_count
    position_ids = (
        torch.arange(total_len, device=device, dtype=torch.long)
        .unsqueeze(0).expand(B, total_len)
    )

    # Attention mask: noise positions can attend to all ctx + earlier noise (causal).
    # ctx positions don't act as queries here (q comes from noise only inside layer).
    # But our HF custom layer concatenates ctx K/V with noise K/V — q only from noise.
    # So we don't strictly need a mask in the conventional sense, but for masked-out
    # ctx slots (padding) we should suppress them.
    # Build [B, 1, q_len, ctx_len + q_len] additive mask, -inf for invalid ctx slots.
    q_len = noise_count
    kv_len = ctx_window + noise_count
    am = torch.zeros(B, 1, q_len, kv_len, device=device, dtype=torch.bfloat16)
    # Mask out padded ctx positions
    am[:, 0, :, :ctx_window] = ctx_mask.unsqueeze(1).expand(B, q_len, ctx_window).logical_not() \
        .to(torch.bfloat16) * (-1e4)
    # Causal mask on noise→noise: position i can only attend to noise[0..i]
    noise_causal = torch.triu(
        torch.full((q_len, q_len), -1e4, device=device, dtype=torch.bfloat16), diagonal=1
    )
    am[:, 0, :, ctx_window:] = noise_causal.unsqueeze(0).expand(B, q_len, q_len)

    h = model(
        position_ids=position_ids,
        attention_mask=am,
        noise_embedding=noise,
        target_hidden=ctx,
    )  # [B, γ+1, H]

    # Predictions come from MASK rows [1..γ+1]; row 0 is the bonus.
    h_preds = h[:, 1:, :]                                # [B, γ, H]
    logits = F.linear(h_preds, lm_head_weight, None)     # [B, γ, vocab]
    return logits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dump", default="/tmp/test_dump2.bin")
    p.add_argument("--drafter-dir", default="/path/to/models/z-lab-Qwen3.6-27B-DFlash")
    p.add_argument("--target-safetensors", default="/path/to/models/AEON-Q36-27B-XS/model.safetensors")
    p.add_argument("--output-dir", default="/path/to/models/z-lab-Qwen3.6-27B-DFlash-aeon-v2")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--mask-token-id", type=int, default=248070)
    p.add_argument("--hidden-dim", type=int, default=5120)
    p.add_argument("--n-capture", type=int, default=5)
    p.add_argument("--gamma", type=int, default=16)
    p.add_argument("--ctx-window", type=int, default=64,
                   help="Smaller than production 512 to fit batches in memory")
    p.add_argument("--loss-decay-gamma", type=float, default=7.0,
                   help="Exponential decay across γ positions (paper Eq.4)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--probe-only", action="store_true")
    args = p.parse_args()

    H, T = parse_dump_stream(args.dump, args.hidden_dim, args.n_capture)
    if len(H) < args.gamma + 1:
        sys.exit(f"need ≥ {args.gamma+1} stream pairs, got {len(H)}")

    model, embed_table, lm_head_weight, cfg = load_drafter_and_target(
        args.drafter_dir, args.target_safetensors, args.device
    )
    trainable, frozen = freeze_all_but_fc(model)
    print(f"[freeze] trainable={trainable:,} frozen={frozen:,} pct={trainable/(trainable+frozen)*100:.2f}%",
          file=sys.stderr)

    ds = SequenceDataset(H, T, args.gamma, args.ctx_window, args.n_capture)
    print(f"[data] sequence samples: {len(ds):,}", file=sys.stderr)
    n_cap_h = args.n_capture * args.hidden_dim
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0,
                    collate_fn=lambda b: collate(b, args.ctx_window, n_cap_h))

    # Loss decay weights (paper Eq.4 style): w_i = exp(-i / γ_decay)
    if args.loss_decay_gamma and args.loss_decay_gamma > 0:
        decay = torch.exp(-torch.arange(args.gamma, dtype=torch.float32) / args.loss_decay_gamma)
        decay = decay / decay.mean()  # normalize so total weight ~= γ
    else:
        decay = torch.ones(args.gamma, dtype=torch.float32)
    decay = decay.to(args.device)
    print(f"[loss] decay weights (first/last): {decay[0].item():.3f}..{decay[-1].item():.3f}",
          file=sys.stderr)

    print("[probe] one forward...", file=sys.stderr)
    model.eval()
    with torch.no_grad():
        ctx, ctx_mask, bonus, fut = next(iter(dl))
        ctx = ctx.to(args.device); ctx_mask = ctx_mask.to(args.device)
        bonus = bonus.to(args.device); fut = fut.to(args.device)
        try:
            logits = forward_drafter_sequence(
                model, ctx, ctx_mask, bonus, args.gamma, args.mask_token_id,
                embed_table, lm_head_weight, args.device,
            )
            per_pos_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                            fut.reshape(-1), reduction="none").reshape(logits.shape[:2])
            weighted = (per_pos_loss * decay.unsqueeze(0)).mean()
            top1 = logits.argmax(-1)
            acc_per_pos = (top1 == fut).float().mean(0)  # [γ]
            print(f"[probe] loss={weighted.item():.4f} acc_pos[0]={acc_per_pos[0].item()*100:.1f}% "
                  f"acc_pos[γ-1]={acc_per_pos[-1].item()*100:.1f}%", file=sys.stderr)
        except Exception as e:
            import traceback; traceback.print_exc(); sys.exit(1)

    if args.probe_only:
        print("[probe-only] OK", file=sys.stderr); return

    model.train()
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0,
    )

    print(f"[train] {args.epochs} ep × {len(dl)} batches × bs={args.batch_size} γ={args.gamma} ctx={args.ctx_window}",
          file=sys.stderr)
    t0 = time.time()
    for epoch in range(args.epochs):
        ep_loss = 0.0; ep_acc_first = 0.0; ep_acc_last = 0.0; n = 0
        for ctx, ctx_mask, bonus, fut in dl:
            ctx = ctx.to(args.device); ctx_mask = ctx_mask.to(args.device)
            bonus = bonus.to(args.device); fut = fut.to(args.device)
            logits = forward_drafter_sequence(
                model, ctx, ctx_mask, bonus, args.gamma, args.mask_token_id,
                embed_table, lm_head_weight, args.device,
            )
            per_pos = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(), fut.reshape(-1),
                reduction="none"
            ).reshape(logits.shape[:2])  # [B, γ]
            loss = (per_pos * decay.unsqueeze(0)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            top1 = logits.argmax(-1)
            B = fut.shape[0]
            ep_loss += loss.item() * B
            ep_acc_first += (top1[:, 0] == fut[:, 0]).float().sum().item()
            ep_acc_last += (top1[:, -1] == fut[:, -1]).float().sum().item()
            n += B
        print(f"[train] ep {epoch+1}/{args.epochs} loss={ep_loss/n:.4f} "
              f"acc_first={ep_acc_first/n*100:.1f}% acc_last={ep_acc_last/n*100:.1f}% "
              f"elapsed={time.time()-t0:.1f}s", file=sys.stderr)

    out = Path(args.output_dir); out.mkdir(exist_ok=True)
    print(f"[save] -> {out}", file=sys.stderr)
    for fname in os.listdir(args.drafter_dir):
        src = Path(args.drafter_dir) / fname
        if src.is_file() and fname != "model.safetensors":
            os.system(f"cp -f '{src}' '{out / fname}'")
    original = load_file(Path(args.drafter_dir) / "model.safetensors")
    original["fc.weight"] = model.fc.weight.detach().cpu()
    original["hidden_norm.weight"] = model.hidden_norm.weight.detach().cpu()
    save_file(original, out / "model.safetensors")
    print(f"[save] wrote {out / 'model.safetensors'}", file=sys.stderr)


if __name__ == "__main__":
    main()
