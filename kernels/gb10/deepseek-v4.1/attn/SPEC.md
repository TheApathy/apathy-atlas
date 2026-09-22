# DeepSeek-V4.1 cross-layer sparse index attention — algorithm, derived from the Python engine

Source of truth (read 2026-09-22, not re-derived):
- `/home/flocka/atlas/dsv41-prefill-work/engine/model.py` — `attention`, `_compressed`,
  `_indexer`, `_pad_topk`, `_select_candidates`, `Shared`, `Caches`
- `/home/flocka/atlas/dsv41-prefill-work/tools/prefill_attn.py` — one-pass streaming gather (Triton)
- `/home/flocka/atlas/dsv41-prefill-work/tools/indexer_kernel.py` — fused indexer score tile
- `/home/flocka/atlas/dsv41-prefill-work/tools/decode_attn.py` — decode-path flash softmax
- `/home/flocka/atlas/dsv41-prefill-work/tools/v41_ref.py` — `Args`

## 0. Constants (v41_ref.Args)

    n_heads 64, head_dim 512, rope_head_dim 64, o_groups 8, window_size 128
    index_topk 512, index_n_heads 32, index_head_dim 128
    kv_source_layers      = (2, 8, 14, 20)
    index_source_layers   = (2, 8, 14, 20, 24, 28, 32, 36)
    candidate_source_layer = 20, candidate_topk_blocks 2048, candidate_block_size 8
    compress_ratios = [0,0] + [2]*18 + [1]*20 + [0,0,0]   # 40 layers + 3 DSpark blocks
    rope_theta 10000 (no YaRN), compress_rope_theta 160000 (YaRN, original_seq_len 65536)

## 1. What "cross-layer" actually means — THREE independent inheritance chains

The brief calls this "cross-layer sparse index attention". Concretely, three different things
are computed on a few layers and *reused unchanged* by the layers after them, carried in a
per-forward `Shared` struct:

1. **Compressed KV cache + index keys** — computed ONLY on `kv_source_layers` = {2, 8, 14, 20}
   (`w.is_kv_source`). Layer L reads `sh.ckv`/`sh.ik`/`sh.ratio` left by the most recent
   kv-source layer <= L. Layers 3..7 reuse layer 2's, 9..13 reuse 8's, 21..39 reuse 20's.
   `assert sh.ratio == w.ratio` — the ratio changes 2 -> 1 exactly at layer 20, which is why 20
   is a kv-source layer.
2. **Top-k selection** — recomputed ONLY on `index_source_layers` = {2,8,14,20,24,28,32,36}
   (8 layers). Every other layer attends with the `sh.topk` its predecessor left. So 40 layers
   run 8 indexers, not 40.
3. **Candidate block mask** — produced ONLY by layer 20's indexer, consumed by the indexers at
   24, 28, 32, 36 to prune their score matrices. Layers 2, 8, 14 run *before* 20 and use no
   candidate mask (`use_cand = (L != 20) and 0 <= 20 < L and sh.candidates is not None`).

Layers 0 and 1 have ratio 0: **window only, no compressed rows, no indexer**.

## 2. Per-layer attention (engine/model.py `attention`)

    pos  = S .. S+T-1
    freqs = freqs_c if w.ratio != 0 else freqs_w        # PER-LAYER ROPE TABLE — see §6
    qr = rmsnorm(x @ wq_a, q_norm)                      # [T, 1280] q_lora
    q  = (qr @ wq_b).view(T, 64, 512);  rope_tail(q, freqs[pos])   # rope on LAST 64 dims
    kv = rmsnorm(x @ wkv, kv_norm);     rope_tail(kv, freqs[pos])  # [T, 512], MQA: one kv head
    ring[pos % RING] = kv                               # window ring write, RING=4096
    wpos[t, i] = pos[t] - (127 - i), or -1 if < 0        # [T, 128] absolute positions
    if w.ratio: (ckv, cidx) = compressed(...)           # §3, cidx = [T, 512] abs rows or -1
    o = sparse_attention(q, ring, wpos, win_lo, ckv, cidx, attn_sink, head_dim**-0.5)
    o = rope_tail(o, freqs[pos], inverse=True)          # INVERSE rope on the OUTPUT
    o = o.reshape(T, 8, 64); o = wo_a_proj(o, wo_a); out = o.flatten(1) @ wo_b

Notes that bite:
- MQA: `num_key_value_heads = 1`. All 64 query heads share one 512-dim KV row per position.
  K and V are **the same tensor** — `_attend_block` does `dot(q, k^T)` then `dot(p, k)`.
  There is no separate V. This halves the gather traffic and is the reason the one-pass
  streaming gather is so favourable here.
- The output gets an **inverse RoPE** before the o-projection. Easy to miss; silently wrong model.
- `attn_sink` is a learned fp32 per-head scalar added to the softmax *denominator only*:
  `denom = sum_n exp(s_n - m) + exp(sink_h - m)`. It contributes no value to the numerator.
- `win_lo` is 0 everywhere except decoder replay; keep the parameter.

## 3. `_compressed` — building ckv / ik (kv-source layers only)

For `r = 2` layers (2, 8, 14):

    kvl = x_f32 @ comp_wkv ; sc = x_f32 @ comp_wgate      # both [T, 512], fp32
    prepend Caches.pending[L] (the unpaired position carried from the last chunk), if any
    cut = n_tok - n_tok % r ; leftover (if any) becomes the new pending
    group into (n/r, r, 512):  latent = sum_r kvl * softmax(sc, dim=r)   # gated mean over r
    latent = rmsnorm(latent.to(bf16), comp_norm)
    j0 = first // r ; jpos = (j0 + arange(nj)) * r        # absolute token pos of each comp row
    if L in indexers:  k = rmsnorm(latent @ iw.wk, iw.k_norm); rope_tail(k, freqs_c[jpos])
                       ik[L][j0 : j0+nj] = k              # [*, 128] index keys, PRE-RoPE latent
    ckv[L][j0 : j0+nj] = rope_tail(latent, freqs_c[jpos]) # [*, 512]

For `r = 1` (layer 20): `latent = rmsnorm(x @ comp_wkv, comp_norm)`, `j0 = S`, no gating,
no pending. Same ik/ckv writes.

**Order matters**: the index key is built from the latent *before* its RoPE; `lat` is then
RoPE'd in place. A port that RoPEs first and then derives `ik` is a different model.

`compress_lens[t] = (pos[t] + 1) // r` — how many compressed rows query t may see.
`n_c = (S + T) // r`.

## 4. `_indexer` — the top-k (8 layers)

    q_i  = (qr @ iw.wq_b).view(T, 32, 128); rope_tail(q_i, freqs_c[pos])   # rope on last 64 of 128
    wts  = (x @ iw.weights_proj).float() * (128**-0.5 * 32**-0.5)          # [T, 32]
    n_pad = ceil(n_c / 512) * 512   (min 512)
    score[t, n] = sum_h relu( bf16( q_i[t,h,:] . ik[n,:] ) ) * wts[t, h]   # fp32 head sum
    score[t, n] = -inf where n >= compress_lens[t]
    if use_cand:  score = -inf where not candidates[t, n]
    if L == 20:   candidates = select_candidates(score, compress_lens, 2048, 8)
    k_   = min(512, n_c)
    idx  = sort( topk(score, k_, sorted=False).indices )                   # ASCENDING positions
    idx  = where(idx < compress_lens[t], idx, -1)
    return pad_to_512(idx, fill=-1)

- The dot is accumulated in fp32 but **rounded to bf16 before the relu** (the torch einsum
  returned bf16). `indexer_kernel.py` reproduces this explicitly: `s.to(bf16).to(f32)`.
  Skipping that rounding changes which rows get selected near ties.
- `topk` is unsorted then sorted ascending — so the attention gather walks compressed rows in
  increasing position order. Ties in `topk` are resolution-dependent; this is the one place a
  bit-exact port may legitimately diverge, and it is *discrete* (a different row selected),
  so it must be checked as a set-overlap, not an L2.
- The output is ALWAYS padded to exactly 512 columns with -1. Reason (verbatim from the
  reference): a varying N makes cuBLAS pick a different kernel and the ulp differences flip
  router decisions downstream. Keep the fixed 512 width in the Rust port too.

`select_candidates`: pad score to a multiple of 8, `amax` over each block of 8, force-keep the
block containing `compress_lens-1` (score set to +inf), take top-2048 blocks, keep only those
with score > -inf, `repeat_interleave(8)` back to the column axis.

## 5. The kernel — one-pass streaming gather (port of `tools/prefill_attn.py`)

One program = one token x HB heads (HB=32 -> 2 programs/token for 64 heads). Online (flash)
softmax in registers, in log2 domain with `scale * log2(e)` folded in.

    m = -inf; l = 0; acc = 0                              # [HB], [HB], [HB, 512] fp32
    for kb in 0..128 step BK:                             # WINDOW FIRST
        p = wpos[t, kb:kb+BK]; valid = (p >= 0) & (p >= win_lo)
        k = ring[p % RING]                                # gathered IN-KERNEL, never materialised
        attend(k, valid)
    for kb in 0..512 step BK:                             # THEN COMPRESSED
        j = cidx[t, kb:kb+BK]; valid = j >= 0
        k = ckv[j]                                        # gathered IN-KERNEL
        attend(k, valid)
    denom = l + exp2(sink[h] * log2e - m)
    o[t, h, :] = acc / denom

`attend(k, valid)`:
    s = dot(q, k^T) * scale_log2 ; s = where(valid, s, -inf)
    m_new = max(m, rowmax(s)) ; m_safe = (m_new == -inf) ? 0 : m_new
    alpha = exp2(m - m_safe) ; p = exp2(s - m_safe)
    l = l*alpha + rowsum(p) ; acc = acc*alpha + dot(p_as_bf16, k)

**Block order (window, then compressed) is part of the numerics** — online softmax rescaling is
not associative in floating point. A port that walks compressed-then-window will not reproduce
the reference bit-for-bit even when everything else is right.

The `m_safe` clamp exists so an all-masked row (every key invalid) gives `acc=0, l=0,
denom=exp2(sink)` -> `o=0` rather than NaN. Negative control candidate: feed an all-`-1`
`wpos`/`cidx` row and require finite zeros out.

Numerics gap to know about: this kernel rounds P to bf16 before the PV product; the *reference*
`_softmax_attn` keeps P in fp32, and `decode_attn.py` splits P into bf16 hi+lo (PV_SPLIT=1) to
keep ~16 mantissa bits. The Python engine accepted the bf16-P rounding for prefill only.
So "which reference" must be named on every comparison. I will oracle against the fp32
`_softmax_attn` path and report the kernel's error against *both*.

## 6. Two RoPE tables, selected per layer

    freqs_c: theta 160000, YaRN with original_seq_len 65536   -> layers with ratio != 0 (2..39)
    freqs_w: theta 10000,  NO YaRN (original_seq_len 0)        -> layers 0, 1 and the DSpark blocks

`freqs_c` is also always used for the compressed rows and the indexer query, on every layer.
Getting this backwards produces a plausible-looking but wrong model on 38 of 40 layers.

## 7. Memory/traffic (why the one-pass gather is the right call on 273 GB/s)

Per token per layer the gathered set is 128 window + 512 compressed = 640 rows x 512 dims x bf16
= 640 KB. Materialising it as the reference's slow path does costs ~3 GB of fp32 intermediates
per layer at T=2048 (~50 ms). Gathering inside the kernel makes the traffic 640 KB of *reads*
with no write-back, and MQA means those rows are shared by all 64 heads, so arithmetic intensity
is 64 FLOP/byte-ish rather than 1. This is the single highest-value item in the lane.

## 8. What I need from `dsv41-engine` (seam)

See the message to the lead; signature request lives there, not here, until it is agreed.

## 8b. Decode path (engine/fastdecode.py) — static buckets, CUDA-graph capture

Decode runs a fixed T = T_VERIFY = T_DRAFT + 1 query rows (a verify block; the DSpark draft runs
T_DRAFT). Everything is preallocated so the whole step captures into a CUDA graph: no host
round-trip, no allocation, no shape that depends on the sequence length.

    context_bucket(required, max_seq)
        = min(max_seq, max(CONTEXT_BUCKET_MIN, next_pow2(required)))
        CONTEXT_BUCKET_MIN = max(512, env DSV41_CONTEXT_BUCKET_MIN default 32768)
        -- smallest power-of-two capacity covering the request; the backing caches stay max_seq
           long, this only avoids scoring future-zero index rows the old path masked anyway.

    indexer_score_rows(cache_rows, context_cap, ratio)
        = min(cache_rows, context_cap // ratio + 1)        # rows visible in this bucket
    indexer_topk_width(index_topk, score_rows)
        = min(index_topk, score_rows)                      # then pad back to index_topk with -1

The `+1` in score_rows is the pending/unpaired compressed row and is the exact shape of the
GLM `usable_pools` off-by-one the lead warned about. A short 512-position bucket over a ratio-2
cache gives 257 score rows, so `topk(512)` would be invalid — hence the width clamp, and hence
the fixed-width -1 padding afterwards. Port both or the port breaks only on short requests.

Startup invariant worth keeping (fastdecode asserts it): the candidate mask produced at layer 20
is only wide enough for downstream indexers whose ratio is >= layer 20's. It raises on
`[L for L in indexers if L > 20 and ratio[L] < ratio[20]]`. For this checkpoint ratio[20]=1 and
every later indexer is also 1, so the set is empty — but the check must survive the port or a
future config silently slices a too-short mask.

Decode kernel differs from prefill in one numeric: `decode_attn.py` splits the probability matrix
into a bf16 high part and a bf16 remainder (PV_SPLIT=1) and accumulates both, keeping ~16 mantissa
bits of P instead of 8. It does this because with ~640 dense keys a single bf16 rounding of P costs
about as much accuracy as the final bf16 rounding of o. The prefill kernel does NOT do this.
Three different P precisions therefore exist in the reference (fp32 `_softmax_attn`, split-bf16
decode, single-bf16 prefill); name which one any comparison is against.

## 8c. Gate results (kernels/gb10/deepseek-v4.1/attn/sparse_attn.cu, commit aab4dad99)

Synthetic fixture, T=128, S=4096, RING=4096 (the window wraps the ring). Reproduced identically
across two runs.

    kernel vs fp32-P reference       rel_l2 4.750e-07   worst_abs 1.164e-08
    kernel vs bf16-P reference       rel_l2 1.419e-03   worst_abs 1.460e-05
    all-masked row (token 0)         finite, max|o| = 0.000e+00   (no NaN)
    CONTROL gather (sequential rows) rel_l2 1.197e+00   -> 2,520,753x separation
    CONTROL order  (reordered)       rel_l2 6.086e-07 vs the correct order  -> DOES NOT SEPARATE
                                     and does not separate at ANY skew in 0.031..32 (see sec 5)

Independently confirmed by `DSV41_PORT/oracle/compare.py` (PASS against the fp32 reference;
its own `--negative-control` watched rejecting on this data).

Reading of these numbers: the kernel sits essentially ON the fp32 reference, and the entire
1.419e-03 against the bf16-P reference is `prefill_attn`'s P-rounding, not error here. The
kernel is MORE accurate than the Python prefill path. Do not "fix" it toward the bf16-P number.

The consequence, which is real: more accurate means DIVERGING from the Python engine, and ulp
differences flip MoE router decisions. End-to-end validation is therefore on OUTPUT QUALITY, not
bit-identity, and a per-layer bisect is valid only up to the FIRST routing flip — the same
lesson as GLM's MoE expert flip.

WHAT THIS GATE DOES NOT COVER: block order (above); the orchestration of sections 1-4 (which
layer sources the cache, which RoPE table, the inverse RoPE on the output) — that needs the
captured oracle; and any real weight, since the fixture is synthetic.

## 9. Open items / things NOT yet verified

- No captured oracle yet. Nothing in sections 1-4 has been checked against a real tensor.
- The layer-20 capture (T>=512, S>0, plus a candidate-mask consumer) is queued with
  dsv41-oracle. The first capture (layers 0,1,2,14 at T=64) cannot cover this kernel: at T=64
  there is no full window and no compressed rows, which are the two things it exists to do.
- Decode path (`engine/fastdecode.py` + `tools/decode_attn.py`) has its own static-bucket
  scheduling (`indexer_score_rows`, `indexer_topk_width`) not yet transcribed here.
- Atlas Rust has NO GLM DSA code in this tree (grep for "dsa" over crates/ and kernels/ is
  empty) — the GLM lessons are lessons, not reusable code, in this worktree.
- `crates/spark-model/src/weight_loader/deepseek_v4/indexer.rs` admits indexer weights only
  where `compress_ratio == 4`. V4.1's ratios are 0/2/1, so it admits ZERO tensors for this
  checkpoint and returns Ok(0) silently. That is a V4-Flash-0731 assumption, not a V4.1 one.
