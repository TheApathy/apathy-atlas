# NVFP4 MLA KV cache — measured on paper, NO-GO

Question asked: the reference stack runs `kv_cache_dtype: nvfp4_ds_mla` with a
584-byte record (and an aspirational `experimental_true_nvfp4_record_bytes:
432`). We run FP8 KV. Every verify row reads the KV cache, so a smaller record
should compound with speculation width — one of the two multipliers in the
measured 2× gap (`docs/SPEC-3X-PLAN.md`: they commit ~4.5 tok at ~121 ms, we
commit 3.46–3.73 at ~199 ms).

**Answer: the lever does not exist at our context lengths.** NVFP4 KV saves
**0.17 ms of a 199 ms verify step** at 4096 tokens — 18× below the 3 ms/step
go threshold set for this investigation. Break-even is ~91K context. No kernel
was built. Two by-products of the measurement are reported at the end: our
record is *already smaller* than the reference's, and the `nvfp4` KV knob is a
live correctness trap on V4-Flash (now guarded).

---

## 1. What our record actually is

DeepSeek-V4-Flash stores one compressed MLA latent per token per layer.
`num_kv_heads = 1`, `head_dim = kv_lora_rank + qk_rope_head_dim = 512 + 64 =
576` (`run_paged_decode.rs:52`, `deepseek_v4_mtp.rs:148`).

| | bytes/token/layer | where |
|---|---:|---|
| FP8 record, **as read at decode** | **576** | K pool only — `mla_paged_decode_fp8_kvalias` deletes the V load |
| FP8 record, as *allocated* | 1152 | `paged_impl.rs:25-28` allocates K and V pools separately |
| reference `nvfp4_ds_mla`, physical | 584 | "padded_fp8_compatibility" |
| reference, aspirational true NVFP4 | 432 | `experimental_true_nvfp4_record_bytes` |
| our NVFP4, if built | 324 | 576/2 data + 576/16 FP8 group scales (`kv_cache.rs:236-245`, `NVFP4_GROUP_SIZE=16`) |

The KV-alias is already default-on and bit-exact by construction: V4 MLA writes
`v_cache[i] == k_cache[i]` on every dim (`mla_absorbed.cu:288-320`) and the FP8
scales are calibrated off those byte-identical buffers, so the host selects the
alias entry point after an exact `k_scale == v_scale` compare
(`run_paged_decode.rs`; `ATLAS_MLA_NO_V_FUSE=1` forces the original).

**We already read 576 B/token/layer where the reference reads 584.** On the
record-size axis we are 1.4% ahead, not behind. The 2× gap is not here — and
`SPEC-3X-PLAN.md` already convicts the real cause: draft acceptance (ours
55–62%, theirs ~95%).

## 2. Why context length does not rescue it — the read is not L·43·576

The premise "every verify row reads the KV cache, so this compounds" is true
about the *rows* (grid is `[num_q_heads, num_seqs, 1]`; each of the m verify
rows re-walks the cache) but wrong about the *bytes per row*. Two architectural
facts cap the per-row stream far below the naive `L × 43 × 576`:

1. **Sliding window 128.** `config.json: sliding_window = 128`, passed
   hard-coded at `run_paged_decode.rs:267`. The raw paged read is `min(L,128)`
   positions — **context-independent**. Naive at 4096 would be 101 MB/step;
   the raw window is 3.1 MB/step no matter how long the context gets.
2. **CSA compressed pool at ratio 4/128.** `compress_ratios` over 43 layers is
   21 layers at ratio 4, 20 at ratio 128, 2 at 0. Compressed blocks are
   `seq_len / ratio` at `COMP_BLOCK_DIM = 512` FP8 bytes. This is the only term
   that grows with L, and 20 of the 41 CSA layers grow 32× slower than the
   other 21.

Per-layer read at context L:

```
ratio 4   (21 layers):  128·576 + (L/4)·512
ratio 128 (20 layers):  128·576 + (L/128)·512
ratio 0   ( 2 layers):  128·576
```

## 3. The arithmetic

Total KV bytes read per step, one row:

| ctx | KV/step (m=1) | @229 GB/s | m=6 | m=9 |
|---:|---:|---:|---:|---:|
| 512 | 4.59 MB | 0.020 ms | 0.120 ms | 0.180 ms |
| 2048 | 8.84 MB | 0.039 ms | 0.232 ms | 0.347 ms |
| **4096** | **14.51 MB** | **0.063 ms** | **0.380 ms** | **0.570 ms** |
| 16384 | 48.52 MB | 0.212 ms | 1.271 ms | 1.907 ms |
| 65536 | 184.57 MB | 0.806 ms | 4.836 ms | 7.254 ms |
| 262144 | 728.78 MB | 3.182 ms | 19.10 ms | 28.64 ms |

229 GB/s is the measured real kernel ceiling on this GB10
(`gb10-bw-ceiling-and-expert-structure`), so these are *lower bounds on time*
— i.e. upper bounds on the achievable saving.

At 4096 tokens the entire KV read is **14.51 MB/step**, against the plain
decode weight stream of **6.7 GB/token** (`DECODE-WATERFALL-2026-08-10.md` §1).
KV is **0.217%** of the byte budget.

NVFP4 (576→324, 512→288) removes **43.75%** of those bytes:

| | saving | share of a 199 ms verify step |
|---|---:|---:|
| m=1 | 0.028 ms | 0.014% |
| **m=6** | **0.166 ms** | **0.084%** |
| m=9 (the actual γ=8 / 199 ms config) | 0.249 ms | 0.125% |

**Verdict: 0.17 ms/step at m=6, against a 3 ms/step go threshold. 18× short.**

**Break-even** (where NVFP4 KV would save 3 ms/step at m=6):

- **~91K context** at the 229 GB/s ceiling.
- **~39K context** even at a pessimistic 100 GB/s achieved.

## 4. The measured cross-check — the kernel is latency-bound, not byte-bound

The paper number above is generous, because it assumes the paged kernel is
bandwidth-limited. It is not.

`DECODE-WATERFALL-2026-08-10.md` §3 measures `V4 paged_attn` at **1.13
ms/token, 26.3 µs/call × 43** on a 254-token prose run. At that context the
formula above gives ~4.6 MB of KV — so the kernel achieves **~4 GB/s**, about
1.7% of the 229 GB/s ceiling. The cost is launch and occupancy, not DRAM. The
same doc's verify decomposition puts the entire `B_attn` bucket
(rope + cache write + paged read, *combined*) at **4.0 ms of a 113 ms step**
with a stated floor of ~4 ms — i.e. already at its floor, and the paged read is
only a slice of it.

Halving the bytes under a launch-latency floor buys approximately nothing. The
paper 0.17 ms is the optimistic end.

## 5. Where the verify time actually is

For the record, from the same waterfall — the buckets that matter are two
orders of magnitude above KV:

| bucket | ms/step | floor |
|---|---:|---:|
| MoE expert union (`exp_splitk_m_t`) | 54.1 | ~35 |
| MLA `C_oproj` | 11.5 | ~7.5 |
| MLA `A_proj` | 10.8 | ~4.5 |
| HC/norms/glue | ~16 | ~8 |
| MoE gate | 6.2 | known wash |
| **`B_attn` incl. the entire KV read** | **4.0** | **~4** |

And per `SPEC-3X-PLAN.md`, engine speed is the smaller half of the 2× gap at
all: kernel work on decode is bounded at roughly +6 tok/s, **draft acceptance
is worth +19**.

## 6. By-products worth acting on (neither is NVFP4)

### 6a. The `nvfp4` KV knob is a correctness trap on V4-Flash — now guarded

`--kv-cache-dtype nvfp4` parses (`kv_cache.rs:156`) and dispatches to
`ops::mla_paged_decode_nvfp4` (`run_paged_decode.rs:50`). But
`mla_paged_decode_nvfp4` (`mla_paged_decode.cu:78`) predates the V4 hybrid
attention bring-up. Its parameter list has **no** `sliding_window`, **no**
`attn_sink`, and **no** `comp_pool` / `comp_block_count` / `comp_ratio` —
compare `mla_paged_decode_fp8`, which takes all five. On V4-Flash it therefore
computes **full causal attention over the raw pool and ignores the CSA
compressed pool entirely**, on a model that is `sliding_window=128` with
compression on 41 of 43 layers. Its header additionally documents
`inv_sqrt_d // 1/sqrt(576)` — the softmax scale the FP8 path records as a
measured regression (correct: `1/sqrt(512)`).

That is a different attention operator, not a precision tier. This commit makes
the arm `bail!` with a pointer to this doc when `mla.compressor.is_some() ||
sliding_window.is_some()`. Non-V4 models and the generic NVFP4 paged-decode
path are untouched. Anyone reviving NVFP4 KV must first port the five missing
parameters into the kernel — and read §3 before spending the time.

### 6b. Dead V pool — a capacity lever, not a speed one

The KV-alias deletes the V *load*, but `paged_impl.rs:25-28` still *allocates*
and the write path still *fills* a full V pool that is byte-identical to K and
never read on the V4 decode path:

| ctx | raw K+V pool | dead (V half) |
|---:|---:|---:|
| 4096 | 0.203 GB | 0.101 GB |
| 131072 | 6.493 GB | 3.246 GB |

Aliasing the *allocation* (V pool = K pool for symmetric V4 MLA layers) frees
half the raw KV footprint bit-exactly, and also removes the redundant V write
from the cache-assemble kernel. That is an HBM-capacity and write-bandwidth
win, independent of everything above, and it is far cheaper than an NVFP4
kernel. It is **not** claimed to move the verify step — it removes a write, not
a read. Filed here as the one genuine finding this investigation turned up;
building it was out of scope for this task.

## 7. Numerics tier and quality protocol — for the record, unused

Had this been built, it would have been a **Tier-2 precision change** (a real
change to attention *inputs*, not a bit-exact rewrite): NVFP4 e2m1 with per-16
FP8 group scales on the K latent and rope tail, replacing FP8 E4M3 with a
per-tensor calibrated scale. It would violate the exact-GEMV law
(`DECODE-WATERFALL` §6: *partial exactness is worse than none*) on the verify
side unless the drafter's captures were quantized identically, so the acceptance
histogram — not just the quality gate — would have been the binding measurement.

Protocol that would have gated it:

1. `tool-eval-bench` ≥ 90/100, the standing gate.
2. **Verbatim recall**, the check that killed NVFP4 lm_head
   (`DECODE-WATERFALL` §4 item 4b): ask for the first stanza of *Stopping by
   Woods on a Snowy Evening* at temp 0 and diff against the FP8-KV output. NVFP4
   lm_head passed prose/code/repeat and still fabricated the stanza; a
   short smoke does not catch this class.
3. Accept histogram at fixed γ against the FP8-KV build — a KV precision change
   that costs acceptance loses even if it saves time.

**Calibration**: NVFP4 would need none. Our FP8 KV uses *online* calibration
(the 256-token absmax freeze, which is also what suppresses CUDA graphs on short
benches — see `atlas-fp8-calibration-suppresses-graphs`). NVFP4's scales are
per-16-element groups computed at write time from the group's own absmax, so
there is no global statistic to accumulate, no freeze, and no
calibration-vs-graph interaction. That is the one genuine advantage of the
format here — and it is an advantage over a problem that is already solved.

## 8. Reproducing the arithmetic

Every input is an in-tree constant or a cited measurement; there is no GPU run
to reproduce.

```
# record layout
crates/spark-runtime/src/kv_cache.rs:236-245        # NVFP4 = elems/2 + elems/16
crates/spark-model/.../decode/run_paged_decode.rs:52   # 512 + 64 = 576
kernels/.../mla_paged_decode_fp8.cu:25              # COMP_BLOCK_DIM 512
kernels/.../mla_paged_decode_fp8.cu (kvalias)       # V load deleted, bit-exact

# model geometry
python3 -c "import json;c=json.load(open('\$MODEL/config.json'));\
print(c['num_hidden_layers'],c['sliding_window'],c['compress_ratios'])"
# -> 43, 128, [0,0,4,128,...,4,0]   (21×ratio4, 20×ratio128, 2×ratio0)

# the read model
ratio4   (21): 128*576 + (L/4)*512
ratio128 (20): 128*576 + (L/128)*512
ratio0   ( 2): 128*576
# L=4096 -> 14,508,032 B/step/row
```

Measured anchors: `DECODE-WATERFALL-2026-08-10.md` §1 (6.7 GB/token, 229 GB/s
ceiling), §3 (paged_attn 1.13 ms/token), §6 (`B_attn` 4.0 ms of 113 ms);
`SPEC-3X-PLAN.md` (199 ms step at γ=8, 2.0× gap is acceptance).
