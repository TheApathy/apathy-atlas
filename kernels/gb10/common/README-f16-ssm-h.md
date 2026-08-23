# The seven `*_f16.cu` files here are an INCOMPLETE and UNWIRED port

Copied from `upstream-latest` on 2026-08-15 while investigating
`--ssm-h-dtype f16`. **Nothing references them.** They do not affect the build.
Do not assume the feature exists because the files do.

Files: `gdn_f16_state.cuh`, `gated_delta_rule_wy_f16.cu`,
`gated_delta_rule_wy3_f16.cu`, `gated_delta_rule_wy4_f16.cu`,
`gated_delta_rule_wy2_resident_f16.cu`, `gated_delta_rule_wy3_resident_f16.cu`,
`ssm_h_dtype.cu`.

## Two reasons to read the analysis before touching them

**1. These are the wrong kernels.** They are the *stage-2 MTP-verify* twins plus
the dtype converters. The two kernels that actually produced upstream's decode
win are `gated_delta_rule_decode_f16_norm` and
`gated_delta_rule_decode_f16_strided_norm_half`, which live in
`upstream-latest/kernels/gb10/qwen3.6-27b/nvfp4/gated_delta_rule.cu` (:2353 and
:2134). Neither those nor their four FP32 fused-norm parents exist anywhere in
this tree.

**2. f16 h-state is structurally incompatible with the DFlash/DDTree champion
arm.** That arm's GDN h-state readers and writers are `gated_delta_rule_tree`,
`gated_delta_rule_tree_wy`, and the six-kernel `wy17` family — all fork-only,
all FP32, and none has an f16 twin here or upstream. **An FP32 kernel reading an
FP16 pool does not fault. It emits fluent garbage.** Any future port must be
hard-gated off whenever DFlash/DDTree is active.

## And it is probably not worth it

The GDN h-state is 144 MiB per full pass against 24.57 GB of weights per step.
Halving it saves ≤0.6% of traffic at K=1 and ≤1.5% at K=4 — arithmetically
incapable of the ~15% prose swing it was reached for. That swing is more likely
a K mismatch: the arm it was compared against ran `--num-drafts 3` while ours
ran `1`.

Full analysis, including the CLI/model plumbing diff and the md5-determinism
consequences: **`~/atlas/qwen38/analysis/F16-PORT-PLAN.md`**.

Branch `flocka/ssm-h-f16-port` exists as a bookmark and never diverged from
`flocka/abliterated-gb10-fixes`.
