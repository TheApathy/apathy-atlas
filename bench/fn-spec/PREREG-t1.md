# t1: exact lane-parallel _t decode MoE on the FN recipe env (pre-registered 2026-09-24)

Binary spark-tl1 (perf/fn-spec-r2 + moe_expert_*_shared_t_lanes_o4t16u1, dispatched in
dispatch_unified_t_decode behind ATLAS_MOE_T_LANES=1). Plain greedy decode, 5 prompts x 3 reps
plus a ~2k-token prefill probe. Arms: R0 recipe, RL recipe+lanes, E0 empty env, RU recipe with
ATLAS_UNIFIED_MOE_LAYOUT=0 (untransposed decode kernels, everything else recipe).

Expected:
1. RL per-step logits FNV == RU on every clean trial (same prefill path, decode MoE now bit-identical
   to the untransposed kernel). If RU fails to load, compare RL to E0 instead and say so.
2. Control: R0 != RL on most trials (the old _t kernel is inexact: 7/14080 gate_up and 140/28160 down
   bytes differ in the harness), so the FNV comparison can fail.
3. Speed: RL plain decode >= 1.20x R0 (R0 ~32.6 tok/s; kernel harness 150 vs 257 us/layer).
4. Prefill probe: RL prompt_tokens/ttft within 3% of R0 (the lanes switch touches decode dispatch only).
5. Low-water >= 12 GB in R0/RL (no memory change).
