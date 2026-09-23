# fn-spec sweep s1 — pre-registered 2026-09-23 before any measurement
Binary: allmodels-bench/bin/spark-allmodels (sha 46454931..., built from 93fecd9d5 = perf/fn-spec base).
Env: recipe env (env-fn.txt). Arms C K2 C K5H C K3 C, 4 prompts x (1 warmup + 5 reps), greedy.

Expectations (from memory F27/F40/F41/README):
- C: 40-43 tok/s on code/chat. Self-determinism: F41 saw 3 distinct outputs in 5 reps on this
  model; the cublasLt coin-flip fix (67e3cf5ce) may or may not have cured it. Expect >=1 prompt
  with >1 distinct sha across C1..C4 (cross-process) = 50/50.
- K2: 34-38 tok/s (0.85-0.9x), output == C (README: same SHA).
- K5H: code 48-52 tok/s (1.15-1.25x), chat lower (~1.0-1.1x); acceptance ~2.6/4 on code.
  Output == C expected on the no-CONV flag set (F41), not guaranteed.
- K3: 32-38 tok/s (F27: 0.78x).
- think_code: spec arms with THINK_SPEC=1 speculate inside <think>; expect same sha as C.
A spec arm "wins" only if every measured trial's sha equals the controls' sha for that prompt
AND its median decode tok/s beats the mean of its two neighbouring controls.

## s1 outcome (aborted after 2 arms)
- C1 (recipe env): 33.5-34.2 tok/s on all 4 prompts, NOT 41. code_py rep3 differed from the other
  5 trials (diverges at char 892 of 1254): baseline nondeterminism at T=0 reproduces (F41).
- K2: server refused to build: ATLAS_QWEN4_PREFILL_MOE_COMPACT "requires canonical C1
  single-GPU original-layout Flash-Next" once the MTP sidecar is attached. Spec cannot run with
  the recipe env at all.

# s2 — pre-registered before measurement
Env: EMPTY (allmodels fndef profile, which measured 41.1 decode) for C/K arms; R = recipe env plain.
Order C1 K5H C2 K2 C3 R C4.
- C: 40-42 tok/s. Determinism: expect >=1 prompt with 2 distinct shas across 20 control trials
  (60%): the empty env still has the base-decode nondeterminism unless it lives in a prefill lever.
- R: 33-35 tok/s = ~0.82x of its neighbours (reproduces s1 C1). If R ~= C, the s1 gap was
  cross-window drift, not the env.
- K5H: code 1.15-1.25x, chat ~1.0-1.1x, think_code ~1.0-1.2x; output identical to C on every
  prompt where C itself is single-valued (70%).
- K2: 0.85-0.92x, output identical.

## s2 outcome (C1 K5H C2 K2 C3 R C4, empty env, base binary 4645)
- C: 41.3 tok/s. Nondeterminism: C1 chat warmup differed from its 5 reps (char 539); C2 code_rs
  rep5 differed. Not a verify issue: it's base decode, ~1 in 12 trials.
- K5H code_py 50.2 (1.21x) code_rs 46.1 (1.12x) chat 34.8 (0.84x) think 35.6 (0.86x); per-draft
  acceptance 0.83/0.73/0.66/0.59.
- K2 code 38.2 (0.93x); acceptance 0.855.
- chat: BOTH spec arms emitted "</think>" as the first content token. ROOT CAUSE: the Qwen4 MTP
  prompt replay (prefill_last_k, default K=64 for qwen4) runs the MTP LM head into
  buffers.logits() AFTER the target computed its prefill logits and BEFORE the scheduler samples
  the first token from that same buffer. So the first output token was the MTP head's argmax, not
  the target's. Fixed: replay skips the head (qwen4_mtp.rs, emit_draft=false).
- think_code: K2 and K5H both diverge from C at the same char 239 (in <think>) -> systematic, not
  verify numerics (which would differ between K=2 and K=5).

# s3 — pre-registered (binary lastk2 = fix + ATLAS_SPEC_SKIP_SSM_RESTORE_CONTROL)
Order C1 K2 C2 K5H C3 K2X C4 K2nt C5.
- Prefill first token for chat is 43 in EVERY arm (the fix). 90%.
- K2/K5H chat output == C chat output (whenever C is single-valued). 70%.
- K2X (skip-restore control): output DIFFERS from C on >=3 of 4 prompts, early. Must fail; if it
  does not, the branch I gated is not the one Qwen4 commits through and the gate is void.
- K2nt (no THINK_SPEC): think_code == C. If K2 (THINK_SPEC=1) still differs, the defect is in the
  think-spec accept path. 60%.
- Speeds unchanged vs s2 (fix only touches prefill replay): K2 ~0.93x, K5H code 1.1-1.2x.
