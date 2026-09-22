"""SPEC 8e.1: rebuild the indexer's inputs (idx_q, wts) from qr/attn_x taps + checkpoint
weights, and measure how often upstream ulps change the top-k (fp64 dot on top). CPU only."""
import numpy as np
import torch

import check_spec_sections as C
from oracle_io import Run, f32_to_bf16_bits, round_bf16

torch.set_num_threads(16)


def selection_rows_differ(R, L, occ, tko, q, wts):
    ik = torch.from_numpy(R.load(L, "ik", occ)).double()
    d = torch.from_numpy(round_bf16(torch.einsum("thd,nd->thn", q.double(), ik).float().numpy()))
    s = (d.relu() * wts[:, :, None].float()).sum(1).numpy()
    cl = R.load(L, "compress_lens", occ)
    s[np.arange(s.shape[1])[None, :] >= cl[:, None]] = -np.inf
    if R.meta(L, "cand_in", occ)["shape"][0]:
        s[~R.load(L, "cand_in", occ)] = -np.inf
    idx = np.sort(np.argpartition(-s, 511, axis=1)[:, :512], axis=1)
    tk = R.load(L, "topk", tko)
    return sum(set(a) != set(b[b >= 0]) for a, b in zip(idx, tk))


for rn in ["runD_L20_kernel", "runE_torch"]:
    R = Run(rn)
    for L, occ, S in [(20, 1, 512), (24, 0, 896)]:
        tko = 2 if (rn.startswith("runD") and L == 20) else occ
        qr = torch.from_numpy(R.load(L, "qr", occ)).double()
        x = torch.from_numpy(R.load(L, "attn_x", occ)).double()
        w = C.weight(L, "attn.indexer.wq_b.weight").float()
        sc = C.weight(L, "attn.indexer.wq_b.scale").float()
        wd = (w * sc.repeat_interleave(32, 0).repeat_interleave(32, 1)).double()
        q = torch.from_numpy(round_bf16((qr @ wd.T).float().numpy())).view(-1, 32, 128)
        q = torch.from_numpy(round_bf16(C.rope_tail(q, C.TABLES["freqs_c"][S:S + q.shape[0]]).numpy()))
        wp = C.weight(L, "attn.indexer.weights_proj.weight").double()
        wts = torch.from_numpy(round_bf16((x @ wp.T).float().numpy())) * (128 ** -0.5 * 32 ** -0.5)
        tq, tw = R.load(L, "idx_q", occ), R.load(L, "wts", occ)
        qe = (f32_to_bf16_bits(q.numpy()) == R.load(L, "idx_q", occ, raw=True)).mean()
        print(f"{rn} L{L}: idx_q bit-exact {qe:.4f}, wts exact {(wts.numpy() == tw).mean():.4f} | "
              f"top-k rows differ: taps {selection_rows_differ(R, L, occ, tko, torch.from_numpy(tq), torch.from_numpy(tw))}"
              f", rebuilt q+wts {selection_rows_differ(R, L, occ, tko, q, wts)}")
