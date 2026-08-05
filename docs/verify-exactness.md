# γ-Verify Exactness: Method, Tooling, and Findings

Speculative decoding at temperature 0 is supposed to be **lossless**: the
committed token stream must be byte-identical to plain greedy decode. On
DeepSeek-V4-Flash/GB10 it was not, which silently capped drafter acceptance
at ~1.1 (vs 3.69 offline) and made every DSpark throughput number a lie.
This document records the method that localized the divergences, the tools
that automate it, and the fixes banked so far. Task #45 is the ledger.

## The measurement stack

1. **Plain-greedy oracle.** Plain decode is bit-deterministic. Record its
   output hashes once per prompt; every spec config is judged against them.
   `scripts/bisect-probe.py <log>` runs the oracle prompt 3× and reports
   hash matches + steady-state acceptance.
2. **One-toggle serve legs.** `scripts/bisect-verify.sh <name> [ENV=V ...]`
   starts a forced-spec server (`ATLAS_MTP_GATE_FORCE=1` so the throughput
   gate cannot silently serialize the run) with exactly the envs you pass.
3. **Byte-level layer probes.** `ATLAS_DIAG_V4_ALL_LAYERS=1` (eager only —
   pair with `ATLAS_DFLASH_DEBUG_NO_GRAPH=1` on spec) makes every layer
   print `DIAG <tag> L<n> <label>: ... bhash=<fnv1a-of-bytes>`.
4. **Per-step accept trace.** `ATLAS_DFLASH_VSTEP_DIAG=1` prints one line
   per verify step: `VSTEP pre=<len> in=[tokens] verified=[argmaxes]
   acc=<n> bonus=<tok>` — the ground truth for which rows are comparable
   and where the committed stream forks.
5. **The comparator.** `scripts/exactdiff.py plain.log spec.log` aligns the
   two logs and prints an E/D grid per (layer, label, step, row) plus the
   first divergence. It automates the three pitfalls below.

## The three pitfalls (each cost us a wrong conclusion)

* **Norms lie.** `diag_norm`'s norm/max/first4 are rounded; tensors that
  differ bytewise print identically. Two full investigation rounds chased
  phantom culprits (the MoE, then the O-projection) before the `bhash`
  field existed. Only byte hashes decide.
* **The warmup forward.** The server runs a dummy batched forward (n=4) at
  load. Its probes land in every log, so flat occurrence indexing compares
  warmup-vs-warmup (trivially equal — this forged an "all 43 layers exact"
  result) or offsets everything by one. Content-anchor: locate the known
  step1-row0 hash in the plain series and derive the offset.
* **Label cadence.** `o_out` exists per-row (`V4-decode`, from
  `attention_forward_v4`) and per-step (`V4-msdecode`, from `multi_seq`).
  Mixing cadences produced a false conviction of the O-projection. Pin the
  tag; never fall back across tags.

Also: a `D` at a row whose draft was **rejected** is expected — its input
token differs from the plain stream. Read the VSTEP trace first.

## Exactness gates (all banked, env-controlled)

| Env | What it does | Verdict |
|---|---|---|
| `ATLAS_MLA_NO_BATCH=1` | per-row MLA fallback (plain kernels per row) | byte-exact; the batched Q-chain/O-proj kernels are NOT (reduction order) |
| `ATLAS_LMHEAD_EXACT=1` | zeroes the fp8_gemm lm_head tiers that cast activations to FP8-E4M3 in-kernel | real accuracy bug; keep on |
| `ATLAS_MOE_GATE_EXACT=1` | per-row router GEMV identical to m=1 | no-op here (router wasn't flipping); keeps the door shut |
| `ATLAS_OPROJ_EXACT=1` | per-row Step-6 O-proj staged through plain's buffers | superseded by MLA_NO_BATCH; kept for kernel bisection |
| `ATLAS_V4_FORCE_NO_COMP=1` | compressed-KV arm off entirely | A/B lever for the CSA-layer divergence |

## State of the hunt (2026-08-05)

With `MLA_NO_BATCH=1 + LMHEAD_EXACT=1`: layer front (embed→hc→norm), MLA,
O-projection, and the m-row MoE are **byte-exact vs plain at every
matched-token row, including rows > 0** — embeddings byte-match the
safetensors table. The committed stream tracks plain for 4 verify steps.

**First divergence: layer 2 — the first CSA (compressed-arm) layer — for
row 0 of step 1.** Layers 0-1 (`compress_ratios` 0, no compressor) are
exact; every compressor layer differs. The compressed-arm state machinery
during verify (pool count device word, ring/prev_win, the γ-speculate
advance, or the kernel's visible-count derivation) is the remaining
suspect. The confirmation A/B is both servers under
`ATLAS_V4_FORCE_NO_COMP=1`: if spec goes fully exact against the no-comp
plain oracle, the bug is confined to the compressed arm.

## Reproduction recipe (fork localization, ~15 min)

```sh
# 1. plain oracle probes (once per prompt; give it MORE max_tokens than spec)
ATLAS_DIAG_V4_ALL_LAYERS=1 <plain serve...>          # request max_tokens=20
# 2. spec under test
scripts/bisect-verify.sh mytest ATLAS_MLA_NO_BATCH=1 ATLAS_LMHEAD_EXACT=1 \
    ATLAS_DFLASH_DEBUG_NO_GRAPH=1 ATLAS_DIAG_V4_ALL_LAYERS=1 \
    ATLAS_DFLASH_VSTEP_DIAG=1                        # request max_tokens=16
# 3. compare
python3 scripts/exactdiff.py serve-plain.log serve-bisect-mytest.log
```

Oracle hashes on record (paged-attention prompt, 128 tok): plain-default
`0d20ac629078b9f9`, plain-`HC_SPLIT=0` `c09d417714e424fb`.
