# SPDX-License-Identifier: AGPL-3.0-only
"""Pure-torch stand-ins for the tilelang kernels the official DeepSeek-V4
inference/model.py imports. Semantics transcribed from inference/kernel.py
(sparse_attn_kernel / hc_split_sinkhorn_kernel); the quantized GEMMs are
unused because the probe dequantizes every weight to BF16 at load, which
routes model.linear() onto plain F.linear. Only act_quant is exercised (the
FP8 KV-cache round-trip inside DSparkAttention) and is implemented
faithfully: per-group e4m3 quantize + dequantize in place.
"""

import torch

FP8_MAX = 448.0  # torch.finfo(torch.float8_e4m3fn).max


def act_quant(x: torch.Tensor, group: int, scale_fmt=None, scale_dtype=None, inplace: bool = False):
    """Per-`group` (last dim) FP8-E4M3 quantization. With `inplace=True`,
    round-trips x through FP8 in place (the KV-cache write simulation) and
    returns None; otherwise returns (x_fp8_as_float, scales)."""
    orig_shape = x.shape
    # x may be a non-contiguous slice (the KV latent view); reshape copies as
    # needed for compute, and the in-place branch writes back via copy_.
    g = x.reshape(-1, group)
    amax = g.abs().amax(dim=-1, keepdim=True).float().clamp_min(1e-12)
    scale = amax / FP8_MAX
    if scale_fmt == "ue8m0":
        scale = torch.exp2(torch.ceil(torch.log2(scale)))
    q = (g.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    deq = (q.float() * scale).to(x.dtype).view(orig_shape)
    if inplace:
        x.copy_(deq)
        return None
    return deq, scale.view(*orig_shape[:-1], -1)


def fp4_act_quant(*args, **kwargs):
    raise NotImplementedError("probe loads BF16 weights; fp4 path must not be hit")


def fp8_gemm(*args, **kwargs):
    raise NotImplementedError("probe loads BF16 weights; fp8 path must not be hit")


def fp4_gemm(*args, **kwargs):
    raise NotImplementedError("probe loads BF16 weights; fp4 path must not be hit")


def sparse_attn(q: torch.Tensor, kv: torch.Tensor, attn_sink: torch.Tensor,
                topk_idxs: torch.Tensor, softmax_scale: float) -> torch.Tensor:
    """Gathered multi-head attention with a per-head sink logit.
    q [b, m, h, d]; kv [b, n, d] (MQA: shared across heads); topk_idxs
    [b, m, topk] int32, -1 = hole. The sink joins the softmax denominator
    only (contributes no value). Mirrors sparse_attn_kernel exactly."""
    b, m, h, d = q.shape
    idx = topk_idxs.long()
    hole = idx < 0
    gathered = kv.gather(1, idx.clamp_min(0).view(b, -1, 1).expand(-1, -1, d))
    gathered = gathered.view(b, m, -1, d)                      # [b, m, topk, d]
    scores = torch.einsum("bmhd,bmtd->bmht", q.float(), gathered.float()) * softmax_scale
    scores = scores.masked_fill(hole.unsqueeze(2), float("-inf"))
    smax = scores.amax(dim=-1, keepdim=True)                   # [b, m, h, 1]
    ex = torch.exp(scores - smax)
    denom = ex.sum(dim=-1) + torch.exp(attn_sink.float().view(1, 1, h) - smax.squeeze(-1))
    o = torch.einsum("bmht,bmtd->bmhd", ex, gathered.float()) / denom.unsqueeze(-1)
    return o.to(q.dtype)


def hc_split_sinkhorn(mixes: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor,
                      hc_mult: int = 4, sinkhorn_iters: int = 20, eps: float = 1e-6):
    """mixes [b, s, (2+hc)*hc] -> (pre [b,s,hc], post [b,s,hc], comb [b,s,hc,hc]).
    Transcribed from hc_split_sinkhorn_kernel."""
    hc = hc_mult
    m = mixes.float()
    pre = torch.sigmoid(m[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2.0 * torch.sigmoid(m[..., hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])
    comb = (m[..., 2 * hc:] * hc_scale[2] + hc_base[2 * hc:]).view(*m.shape[:-1], hc, hc)
    comb = comb.softmax(dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    dt = mixes.dtype
    return pre.to(dt), post.to(dt), comb.to(dt)
