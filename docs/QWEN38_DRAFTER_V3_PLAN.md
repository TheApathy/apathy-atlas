# Drafter v3 — plan and rationale

Filed 2026-08-23. v3 combines three things we learned separately: the thinking-
span fix, a serve-weighted prompt mix including security, and the corpus
provenance discipline.

## Why v3 exists

Decode is at 63.9 tok/s. The cycle is 80.5% verify, and verify now runs at ~1.5x
its bandwidth floor — the kernel work is close to exhausted. The remaining lever
is **acceptance**, not kernels:

- measured per-token accept p ~= 0.87
- p ~= 0.92 reaches 70 tok/s at the current verify cost

So v3 is a data problem, not an engine problem.

## What was wrong with v2

v2 measured p=0.9011 in training but ~0.80 at serve. Two defects explain the gap:

1. **63% of its corpus was off-policy passthrough** — completions not generated
   by the target model. A drafter learns to predict *what this target emits*;
   text from anywhere else teaches the wrong distribution.
2. **Zero thinking-span text**, while thinking is 80-95% of the tokens the
   drafter actually drafts at serve time. It had never seen the distribution it
   spends most of its time predicting.

## v3 corpus recipe

Generated on a rented RTX PRO 6000 (vLLM 0.27.1) against the byte-verified
target `unsloth/Qwen3.8-27B-NVFP4` @ `7d6f8d4d…`, weights confirmed identical to
the local serving copy by config hash and safetensors size before spending.

- **On-policy by construction.** Every completion comes from the target itself.
- **Thinking forced on every row** via `chat_template_kwargs.enable_thinking`.
- **Serve-weighted prompt mix**, `build/v4_pool_weighted.jsonl`, 2000 rows:
  900 code / 600 security / 500 general. The pool is weighted toward the
  workload we actually serve rather than sampled uniformly.
- `max_tokens` 8192, temperature 0.

### The thinking-tag trap

The Qwen3.8 chat template prefills a literal `<think>\n` as the last thing in
the **prompt** (`chat_template.jinja:181`), so generation begins already inside
the thinking block and the opening tag never appears in the response. Separately,
vLLM only populates `reasoning_content` when launched **with** a reasoning
parser; by default the thinking flows inline into `content`.

Both together mean the naive `if reasoning_content:` reconstruction writes rows
as untagged reasoning prose, the `'<think>' in content` audit reports 0%, and
the corpus silently repeats the exact defect it exists to fix. `regen_qwen38.py`
now handles both cases. Verified on the live run: **100% of rows carry a
`<think>` span.**

The template also defaults to `reasoning_effort='xhigh'`
(`chat_template.jinja:59`), which is why 3328 tokens never closed a thinking
block. At 8192 roughly half of rows close; the rest are truncated mid-thought.
Truncated rows still teach the thinking-token distribution — the point of the
corpus — they just do not teach the `</think>` -> answer transition. Both are
kept, and the closed fraction is recorded per run.

## Security content

Security is a first-class slice (30% of the pool), not a garnish, because it is
a real part of the serving workload and its token distribution differs from
general code. The DEFCON corpora on the exited Vast instances
(`defcon-distill`, `defcon-archive`) are candidate *additional* material — see
the Vast storage notes in the working area for how to retrieve them and
what that costs. They must be run through the same on-policy pipeline: raw
DEFCON text is **not** drafter training data, for exactly the reason v2 failed.
Use it as a prompt pool; the completions must come from the target.

## Training

`block_size: 32` (v2 was 16). `block_size` sets the usable draft width via
`trained_drafts = block_size - 1`, so 32 lifts the gamma ceiling from 15 to 31.
Note the trap: `--block-size` is **silently ignored** when `--draft-config-path`
is given (`train_dflash.py:217` sits in the `else` branch), so the seed config
itself must carry `block_size: 32` — hence `drafter-seed-block32/`.

Whether a wider gamma pays is an open question, not an assumption: verify cost
is linear in draft width at 1.890 ms/node, so a wider draft only wins if
acceptance holds up across the added positions. Per-position hazard is currently
flat at 0.87 out to position 12, which is the encouraging sign.

## Acceptance criteria

v3 ships only if it beats v2 **at serve time**, not in training loss:

1. measured serve-time accept p > 0.87 on the MinHeap probe
2. tok/s > 63.9 median on the same probe, interleaved with v2, not sequential
3. no AEON regression

A training-loss improvement is not evidence — v2's 0.9011 training figure is
precisely how we got a 0.80 serve-time drafter.
