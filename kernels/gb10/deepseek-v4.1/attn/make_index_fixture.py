#!/usr/bin/env python3
"""Indexer fixtures for sparse_index_gate.cu, straight from the oracle taps (no GPU, no weights).

Inputs are the indexer's own taps (idx_q, ik, wts, cand_in) and positions; targets are the
engine's idx_score, topk and (layer 20) cand_out. One directory per (run, layer, chunk).
Layers above 20 run in the decoder replay, whose first position is win_lo, not the chunk S.
"""
import os
import sys

import numpy as np

from oracle_io import Run

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "idx_fixtures")
CASES = [(2, 0), (2, 1), (14, 1), (20, 0), (20, 1), (24, 0)]


def topk_occ(R, L, occ):
    n_ch = sum(1 for i in range(4) if R.has(L, "kv_new", i))
    n_tk = sum(1 for i in range(2 * n_ch) if R.has(L, "topk", i))
    return 2 * occ if n_tk == 2 * n_ch else occ


def main(runs):
    for rn in runs:
        R = Run(rn)
        for L, occ in CASES:
            if not R.has(L, "idx_q", occ):
                continue
            d = os.path.join(OUT, f"{rn}_L{L:02d}_{occ}")
            os.makedirs(d, exist_ok=True)
            q = R.load(L, "idx_q", occ, raw=True)
            ik = R.load(L, "ik", occ, raw=True)
            wts = R.load(L, "wts", occ).astype(np.float32)
            score = R.load(L, "idx_score", occ).astype(np.float32)
            topk = R.load(L, "topk", topk_occ(R, L, occ)).astype(np.int64)
            ratio = int(R.load(L, "ratio", occ))
            n_c = int(R.load(L, "n_c", occ))
            pos0 = int(R.load(L, "win_lo", occ)) if L > 20 else R.meta(L, "kv_new", occ)["S"]
            cl = R.load(L, "compress_lens", occ)
            assert np.array_equal(cl, (pos0 + np.arange(len(cl)) + 1) // ratio), (rn, L, occ)
            cin = R.load(L, "cand_in", occ)
            cand_ld = cin.shape[1] if cin.size else 0
            is_src = R.has(L, "cand_out", occ)
            T, n_pad = score.shape
            q.tofile(f"{d}/q.bin"); ik.tofile(f"{d}/ik.bin"); wts.tofile(f"{d}/wts.bin")
            score.tofile(f"{d}/ref_score.bin"); topk.tofile(f"{d}/ref_topk.bin")
            if cand_ld:
                cin.astype(np.uint8).tofile(f"{d}/cand_in.bin")
            if is_src:
                R.load(L, "cand_out", occ).astype(np.uint8).tofile(f"{d}/ref_cand.bin")
            with open(f"{d}/dims.txt", "w") as f:
                f.write(f"{T} {ik.shape[0]} {n_pad} {pos0} {ratio} {cand_ld} {int(is_src)} {n_c}\n")
            print(d, T, ik.shape[0], n_pad, pos0, ratio, cand_ld, int(is_src), n_c)


if __name__ == "__main__":
    main(sys.argv[1:] or ["runD_L20_kernel", "runE_torch"])


# ---------------------------------------------------------------------------------------------
# Synthetic-POOL cases. At 1024 tokens every visible block survives layer 20's top-2048, so
# the candidate mask equals the visibility mask: its top-k-of-blocks, its force-keep and
# the pruning it does at 24..36 are all UNEXERCISED by any capture (they need > 16384
# compressed rows). These cases run the reference's OWN _select_candidates (verbatim below)
# with topk_blocks = SMALL_BLOCKS (96 of 128 blocks at n_c=1024) on the engine's real idx_score, so the kernel's block selection and
# the score kernel's mask plumbing are tested on real values where they actually bite.
SMALL_BLOCKS = 96


def select_candidates_ref(logits, compress_lens, topk_blocks, block_size):
    """engine/model.py Model._select_candidates, verbatim."""
    import torch
    import torch.nn.functional as F
    width = logits.size(-1)
    scores = F.pad(logits, (0, -width % block_size), value=float("-inf"))
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)
    last = ((compress_lens - 1) // block_size)[:, None]
    scores = scores.masked_fill(torch.arange(num_blocks)[None, :] == last, float("inf"))
    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, top.indices, top.values > float("-inf"))
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


def topk_ref(score, compress_lens, n_c):
    """engine/model.py Model._indexer's selection + _pad_topk, verbatim."""
    import torch
    k_ = min(512, n_c)
    idx = score.topk(k_, dim=-1, sorted=False).indices.sort(dim=-1).values
    idx = torch.where(idx < compress_lens[:, None], idx, torch.full_like(idx, -1))
    return torch.nn.functional.pad(idx, (0, 512 - idx.size(1)), value=-1)


def small_pool_cases(runs):
    import torch
    for rn in runs:
        for L, occ in ((20, 1), (24, 0)):
            d = os.path.join(OUT, f"{rn}_L{L:02d}_{occ}")
            T, n_keys, n_pad, pos0, ratio, cand_ld, is_src, n_c = map(int, open(f"{d}/dims.txt").read().split())
            score = torch.from_numpy(np.fromfile(f"{d}/ref_score.bin", np.float32).reshape(T, n_pad))
            cl = (pos0 + torch.arange(T) + 1) // ratio
            pool = select_candidates_ref(score, cl, SMALL_BLOCKS, 8)
            if is_src:
                pool.numpy().astype(np.uint8).tofile(f"{d}/ref_cand{SMALL_BLOCKS}.bin")
            # a layer consuming this pool: mask = (its own cand_in, if any) & pool
            mask = pool.clone()
            if cand_ld:
                mask &= torch.from_numpy(np.fromfile(f"{d}/cand_in.bin", np.uint8).reshape(T, cand_ld) != 0)
            masked = score.masked_fill(~mask, float("-inf"))
            finite = torch.isfinite(masked).sum(1)
            mask.numpy().astype(np.uint8).tofile(f"{d}/pool_mask{SMALL_BLOCKS}.bin")
            topk_ref(masked, cl, n_c).numpy().astype(np.int64).tofile(f"{d}/ref_topk_pool{SMALL_BLOCKS}.bin")
            print(d, f"pool{SMALL_BLOCKS}: kept {pool.float().mean():.3f}, finite/row min {int(finite.min())} "
                  f"(k={min(512, n_c)}; rows short of k: {int((finite < min(512, n_c)).sum())})")


if __name__ == "__main__":
    small_pool_cases(sys.argv[1:] or ["runD_L20_kernel", "runE_torch"])
