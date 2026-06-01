#!/usr/bin/env python3
"""Minimal DFlash drafter retrainer (offline, single-token prediction).

Architecture (per z-lab dflash.py):
  noise_embedding      [B, q_len, H]
  target_hidden        [B, ctx_len, n_cap * H]
         |
    fc + hidden_norm   → [B, ctx_len, H]
         |
  drafter transformer (5 layers, custom DFlash attention with target_hidden ctx)
         |
  final norm           → [B, q_len, H]
         |
  TARGET's lm_head     → [B, q_len, vocab]

Only `fc.weight` + `hidden_norm.weight` are unfrozen. The drafter loads
via HF AutoModel (DFlashDraftModel registered in dflash.py). Target's
embed_tokens + lm_head are loaded as DETACHED tensors from the target
safetensors (no need to instantiate the 27B model).
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


def parse_dump(path, hidden_dim=5120, n_capture=5, max_pairs=None):
    data = open(path, "rb").read()
    n_bytes = len(data)
    bytes_per_hidden = 16 + hidden_dim * 2
    h_buf, t_buf = [], []
    pending = []
    off, nh, nt = 0, 0, 0
    while off + 16 <= n_bytes:
        magic = struct.unpack_from("<I", data, off)[0]
        if magic == HIDDEN_MAGIC:
            if off + bytes_per_hidden > n_bytes: break
            payload = bytes(data[off+16:off+bytes_per_hidden])
            pending.append(torch.frombuffer(payload, dtype=torch.bfloat16).clone())
            nh += 1
            off += bytes_per_hidden
        elif magic == TOKEN_MAGIC:
            tok = struct.unpack_from("<I", data, off+4)[0]
            if len(pending) >= n_capture:
                h_buf.append(torch.stack(pending[-n_capture:]))
                t_buf.append(tok)
            nt += 1
            off += 16
            if max_pairs is not None and len(h_buf) >= max_pairs: break
        else:
            break
    H = torch.stack(h_buf)  # [N, n_cap, H] bf16
    T = torch.tensor(t_buf, dtype=torch.long)
    print(f"[parse] hidden={nh:,} token={nt:,} paired={len(H):,}", file=sys.stderr)
    return H, T


def load_drafter_and_target_pieces(drafter_dir, target_path, device):
    from transformers import AutoConfig, AutoModel
    cfg = AutoConfig.from_pretrained(drafter_dir, trust_remote_code=True)
    cfg._attn_implementation = "sdpa"
    print(f"[load] drafter config: hidden={cfg.hidden_size} layers={cfg.num_hidden_layers}", file=sys.stderr)
    model = AutoModel.from_pretrained(
        drafter_dir, config=cfg, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to(device)
    print(f"[load] drafter loaded — fc:{model.fc.weight.shape} hidden_norm:{model.hidden_norm.weight.shape}",
          file=sys.stderr)
    # Pull target pieces (embed + lm_head) WITHOUT loading the 27B model
    embed = None
    lm_head = None
    with safe_open(target_path, framework="pt") as f:
        for k in f.keys():
            if "embed_tokens.weight" in k:
                embed = f.get_tensor(k).to(device, dtype=torch.bfloat16)
            elif k == "lm_head.weight":
                lm_head = f.get_tensor(k).to(device, dtype=torch.bfloat16)
    assert embed is not None and lm_head is not None, "target embed/lm_head not found"
    print(f"[load] target embed:{embed.shape} lm_head:{lm_head.shape}", file=sys.stderr)
    return model, embed, lm_head, cfg


def freeze_all_but_fc(model):
    trainable, frozen = 0, 0
    for name, p in model.named_parameters():
        if name in ("fc.weight", "hidden_norm.weight"):
            p.requires_grad = True
            trainable += p.numel()
        else:
            p.requires_grad = False
            frozen += p.numel()
    return trainable, frozen


def forward_drafter(model, target_hidden_concat, embed_table, lm_head_weight, mask_token_id, device):
    """Single-token forward.
    target_hidden_concat: [B, n_cap*H] bf16
    Returns logits [B, vocab].
    """
    B = target_hidden_concat.shape[0]
    H = embed_table.shape[1]
    # ctx_len = 1, q_len = 1
    target_hidden = target_hidden_concat.unsqueeze(1)        # [B, 1, n_cap*H]
    mask_ids = torch.full((B, 1), mask_token_id, device=device, dtype=torch.long)
    noise = F.embedding(mask_ids, embed_table)                # [B, 1, H] bf16
    position_ids = torch.zeros(B, 1, device=device, dtype=torch.long)
    # Drafter's forward returns the final-norm output
    h = model(
        position_ids=position_ids,
        attention_mask=None,
        noise_embedding=noise,
        target_hidden=target_hidden,
    )  # [B, 1, H]
    logits = F.linear(h, lm_head_weight, None).squeeze(1)     # [B, vocab]
    return logits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dump", default="/tmp/test_dump2.bin")
    p.add_argument("--drafter-dir", default="/path/to/models/z-lab-Qwen3.6-27B-DFlash")
    p.add_argument("--target-safetensors", default="/path/to/models/AEON-Q36-27B-XS/model.safetensors")
    p.add_argument("--output-dir", default="/path/to/models/z-lab-Qwen3.6-27B-DFlash-aeon-tuned")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--mask-token-id", type=int, default=248070)
    p.add_argument("--hidden-dim", type=int, default=5120)
    p.add_argument("--n-capture", type=int, default=5)
    p.add_argument("--max-pairs", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--probe-only", action="store_true")
    args = p.parse_args()

    H, T = parse_dump(args.dump, args.hidden_dim, args.n_capture, args.max_pairs)
    if len(H) == 0:
        sys.exit("no paired samples")

    model, embed_table, lm_head_weight, cfg = load_drafter_and_target_pieces(
        args.drafter_dir, args.target_safetensors, args.device
    )

    trainable, frozen = freeze_all_but_fc(model)
    print(f"[freeze] trainable={trainable:,} frozen={frozen:,} pct={trainable/(trainable+frozen)*100:.2f}%",
          file=sys.stderr)

    n_cap, hdim = args.n_capture, args.hidden_dim

    print("[probe] forward...", file=sys.stderr)
    model.eval()
    with torch.no_grad():
        h0 = H[:2].to(args.device, dtype=torch.bfloat16).reshape(2, n_cap * hdim)
        t0 = T[:2].to(args.device)
        try:
            logits = forward_drafter(model, h0, embed_table, lm_head_weight,
                                     args.mask_token_id, args.device)
            loss = F.cross_entropy(logits.float(), t0)
            top1 = logits.argmax(-1).tolist()
            print(f"[probe] loss={loss.item():.4f}  argmax_top1={top1}  target={t0.tolist()}",
                  file=sys.stderr)
        except Exception as e:
            import traceback; traceback.print_exc()
            sys.exit(1)

    if args.probe_only:
        print("[probe-only] OK", file=sys.stderr); return

    model.train()
    ds = list(zip(H, T))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0,
                    collate_fn=lambda b: (torch.stack([x[0] for x in b]),
                                          torch.stack([x[1] for x in b])))
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0,
    )

    print(f"[train] epochs={args.epochs} batches/ep={len(dl)} bs={args.batch_size}", file=sys.stderr)
    t0 = time.time()
    for epoch in range(args.epochs):
        ep_loss, ep_acc = 0.0, 0
        n_seen = 0
        for h_batch, t_batch in dl:
            h_batch = h_batch.to(args.device, dtype=torch.bfloat16).reshape(-1, n_cap * hdim)
            t_batch = t_batch.to(args.device)
            logits = forward_drafter(model, h_batch, embed_table, lm_head_weight,
                                     args.mask_token_id, args.device)
            loss = F.cross_entropy(logits.float(), t_batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            ep_loss += loss.item() * len(t_batch)
            ep_acc += (logits.argmax(-1) == t_batch).sum().item()
            n_seen += len(t_batch)
        print(f"[train] ep {epoch+1}/{args.epochs} loss={ep_loss/n_seen:.4f} acc={ep_acc/n_seen*100:.1f}% elapsed={time.time()-t0:.1f}s",
              file=sys.stderr)

    out = Path(args.output_dir)
    out.mkdir(exist_ok=True)
    print(f"[save] -> {out}", file=sys.stderr)
    for fname in os.listdir(args.drafter_dir):
        src = Path(args.drafter_dir) / fname
        if src.is_file() and fname != "model.safetensors":
            os.system(f"cp -n '{src}' '{out / fname}'")
    original = load_file(Path(args.drafter_dir) / "model.safetensors")
    original["fc.weight"] = model.fc.weight.detach().cpu()
    original["hidden_norm.weight"] = model.hidden_norm.weight.detach().cpu()
    save_file(original, out / "model.safetensors")
    print(f"[save] wrote {out / 'model.safetensors'}", file=sys.stderr)


if __name__ == "__main__":
    main()
