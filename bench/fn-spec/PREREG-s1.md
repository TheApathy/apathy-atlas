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

## s3 partial outcome (lastk2)
- chat prefill first token in spec arms: 59383 (was 248069, plain 43). The head-skip fix removed
  the MTP LM head write but the MTP LAYER itself also uses buffers.logits() as scratch, so the
  target's logits were still clobbered. lastk3/4: park the target logits in the capture buffer's
  tail across the replay and restore them.
- K2X (control v1, skipped only the live copy) produced BYTE-IDENTICAL output to K2 on all four
  prompts while its warn line proved the branch ran: the control could not fail, because
  pre_verify_copy re-seeds live state from the checkpoint before every verify. Control v2 skips
  the checkpoint write too.

# s4 — pre-registered (binary lastk4)
Order C1 K2 C2 K5H C3 K2X C4. THINK_SPEC env dropped (native MTP enables it by default per
think_spec_accept.rs).
- chat prefill first token == 43 in K2/K5H. 85%.
- K2 and K5H chat output == C. 65%. code prompts == C (as before). think_code: unknown (s3 K2nt
  decides whether the think divergence is policy-path or prefill-related).
- K2X: output differs from K2 on >=3 of 4 prompts. 85%. If it does not, the Qwen4 K2 commit does
  not go through this function at all and I must find the real one before claiming rollback.

## s4 partial: chat prefill first token is 43 in K2 (fix works); K2 chat output == plain (f171);
think_code still differs (93d3, char 239) in K2.

## Profiles (lastk2, code_py, nsys, one request each; attribution only)
- plain: 25.3 ms/step span, 23.1 busy (91%); 1 graph launch + 34 eager launches per token.
- K2: 51.9 ms/step span, 42.7 busy (82%): SSM projections (w4a16_gemv_sw) run per row
  (170 calls vs 84 plain, 13.3 ms vs 6.0), MoE batch2 13.2 ms vs 8.0 plain.
- K5H: 79.5 ms/step span, 67.8 busy (85%), 3.95 tok/step on code: exact_m8 projections 16.2 ms,
  MoE batch3 26.6 ms (~65% of its expert-traffic floor), 4x MTP LM head ~5 ms, GDN sequence 5.6 ms.
  2458 cuLaunchKernel + 16 stream syncs per step: Qwen4 verify is EAGER by design
  (verify_d.rs gate `!is_qwen4_exp()`), ~12 ms/step idle.
Lever 1 (built, binary vg1): ATLAS_QWEN4_VERIFY_GRAPH=1 = segmented verify graph (layer 0 + PLE
eager, layers 1.. + head captured), same split as the plain-decode suffix graph.

# s5o — oracle sweep (vg1, 1 rep): O2 O5 O5G with ATLAS_DFLASH_SERIAL_COMMIT=1 +
SKIP_REPROPOSE=1, which logs DFLASH_SERIAL_ORACLE batch_argmax vs serial(plain decode)_argmax per
verify row and commits the serial choice.
- think_code: batch_argmax != serial_argmax at the divergence row (verify numerics, not policy). 60%.
- total mismatching rows across 4 prompts: 1-10 for O5, 0-5 for O2.
- oracle-mode outputs == plain controls on all 4 prompts (serial commit). 75%.
- O5G mismatches == O5 mismatches (graph does not change arithmetic). 85%.
# s5 — timing (vg1, 5 reps): C K5H C K5HG C K2G C
- K5HG faster than K5H by 8-15% on code; outputs identical to K5H. 70%.
- K2G faster than s4 K2 by 5-15%. 60%.

## s5o O2 (K2 + serial oracle): 1512/1512 rows batch_argmax == serial_argmax, and outputs with
serial commit: code/chat == plain, think_code == fd692dc8 (K5H's think output), NOT plain's 166af6bb.
So the think divergence is NOT verify numerics: the spec server's own one-token decode picks
"times" where the plain server picks "weighted". The spec server decodes eagerly; the plain
server uses the PLE segmented graph.
# s6 — pre-registered: E1 = plain server with ATLAS_QWEN4_PLE_SEGMENTED_GRAPHS=0 (eager decode).
- think_code in E1 == fd692dc8/"times" (eager and graph decode differ numerically; the plain
  GRAPH path is the odd one out, e.g. a capture-time scalar reused across lengths). 55%.
- code/chat in E1 == plain census. 80%.

## s5o/s5 outcome (vg1)
- Oracle: O2 1512/1512, O5 2615/2615, O5G 2580/2580 verify rows batch_argmax == plain one-token
  argmax. Verify is argmax-exact per row at K=2 and K=5, graphed or not.
- think_code in oracle mode (serial commit + skip-repropose) still != plain and even varied
  between reps in O5 -> the thinking defect is in the spec server's thinking path, not verify.
- K5HG (segmented verify graph): outputs identical to K5H on all prompts; code_py 50.35 -> 52.25
  (+3.8%), code_rs 46.20 -> 48.26 (+4.5%). Pre-registered 8-15%: MISS (launch overhead was mostly
  overlapped; idle is syncs, not launches).

# s7 (REPLACED before running; vg2 sweep cancelled) — now binary vg4 = graph + think opt-out +
MoE dedup (ATLAS_QWEN4_MOE_DEDUP) + sync-free MTP draft chain (ATLAS_QWEN4_MTP_CHAIN).
s7c (1 rep): DC = dedup with ATLAS_QWEN4_MOE_DEDUP_CHECK (runs both kernels every call, compares
bytes); DCX = same + CHECK_CORRUPT (flips one byte: the checker MUST report every call).
- DC: 0 mismatching calls. 80% (per-row expression, lane map and shuffle order are identical and
  the target builds --fmad=false). DCX: every call reported. 97%.
s7 timing C G C GV96 C GV32 C GD C GC C ALL C NT C, 5 reps:
- GV96 vs G: +4-7% code (LM head 357 MB -> 138 MB per draft); GV32: +6-9% if acceptance holds.
- GD vs G: +5-12% (MoE 26.6 ms/step in the profile, union traffic ~0.56x of per-slot).
- GC vs G: +2-4% (8 of 16 syncs per step gone).
- ALL: code_py >= 60 tok/s (1.45x). 50%.
- Outputs == plain census on code/chat for every arm. 90%.
- NT think_code == plain 166af6bb. 50%.

## Think divergence ROOT CAUSE (found by reading, 2026-09-23 ~14:30Z)
Plain greedy = host `Iterator::max_by` over the row = ties go to the HIGHEST token id. The device
`argmax_bf16` breaks ties to the LOWEST id. Every spec path that trusts the raw verify argmax
(the think-spec walk's fast path) therefore disagrees with plain decode on an exact BF16 tie.
" times" = 2942, " weighted" = 35604: device picks "times", plain picks "weighted" - the observed
divergence. The content walk was already exact because its fast path re-derives the argmax on
the host with last-wins. Fix (vg5+): `argmax_bf16_last_wins` for the Qwen4 verify rows;
`ATLAS_QWEN4_VERIFY_ARGMAX_FIRST_WINS=1` restores the old kernel as the control.

# s8 — pre-registered (vg6 = vg5 + ATLAS_LOGITS_FNV row hashing), C ALL C ALLF C, 5 reps
ALL = K5 hybrid + graph + dedup + chain + mtp-vocab 96000.
- ALL think_code == plain 166af6bb on all 5 reps. 85%.
- ALLF (control) think_code != plain ("times" at char 239) on all reps. 90%.
- FNV: plain-vs-plain rows identical up to the first flip index in any flipped trial; spec
  content rows (fast-path hashes) bit-identical to plain rows at the same output index for most
  indices — if NOT bit-identical, verify is argmax-exact but not logits-exact (the DSpark bar). 50%.

## s7 partial (vg4): G (K5H+graph) code_py 52.17, GV96 (+mtp-vocab 96000) 53.97 (+3.4%),
code_rs 47.67 -> 51.41 (+7.8%). Acceptance per draft position G 0.823/0.741/0.661/0.586.
Box heavily contended; s7 aborted before GV32 while waiting on the lock. Output flips at known
near-tie words appeared in BOTH plain (C2 chat, C3 code_rs/think) and spec arms, at higher rate
than this morning.
# s8 revised — C1 V96 C2 ALL C3 ALLF C4 on vg6 (all arms log ATLAS_LOGITS_FNV).
V96 = K5H + graph + mtp-vocab 96000 (tie fix on); ALL = V96 + dedup + chain; ALLF = ALL with the
first-wins argmax control. Predictions as before plus: ALL vs V96 +5-12%.

# s9 — pre-registered (vg7 = vg6 + GDN lazy on the Qwen4 exact rows route), C ALL C ALLZ C GATE C
- ALLZ vs ALL: +2-4% on code (GDN sequence 5.6 ms/step, ~half is intermediate snapshot writes),
  outputs identical. 60%. If ALLZ output differs, the lazyfinal/nosnap replay is not bit-exact with
  the sequence kernel on this model -> keep it off.
- GATE (throughput arbiter with a serial arm): chat >= 0.97x (switches to plain), code within 3% of
  ALLZ after warm-up. 50% (per-request probing may cost more than it saves at 256-320 tokens).

# s10 — merged replacement for s8+s9 (box contention), binary vg7, all arms log LOGITS_FNV
C1 ALL C2 ALLZ C3 GATE C4 ALLF. Predictions carried over from s8/s9:
- ALL think_code == plain 166af6bb (tie fix); ALLF think_code == "times" divergence (control). 85%/90%.
- ALL code_py >= 60 tok/s (1.45x). 50%.  ALLZ vs ALL +2-4%, identical outputs. 60%.
- GATE chat >= 0.97x. 50%.

## s10 outcome (vg7)
- ALL: lost to the FN load hang (layer 42, log silent 13.5 min; reported to infra-audit).
- ALLZ (= ALL + GDN lazy): code_py 51.90 (1.32x vs C2/C3, 1.36x vs C1/C2), code_rs 47.82
  (1.22-1.26x), chat 35.93 (0.92-0.95x), think 39.71 (1.01-1.05x). Outputs: code_py/chat/think 5/5
  == plain; code_rs 4/5 + one 9cd942 (the same flip plain C4 shows). Pre-registered 1.45x: MISS.
  Tie fix: think == plain 166af6bb 5/5 (pre-registered 85%: HIT).
- LOGITS_FNV: ALLZ verify rows bit-identical to plain decode rows at every output index on clean
  trials (319/319, 255/255, 122/122); divergences only after a base-race hit (same as plain).
- GATE: 1.25x / 1.16x / 0.87x / 0.90x, worse than ALLZ everywhere (pre-registered chat >= 0.97x: MISS).
- ALLF (first-wins control): pending on the lock at report time.
