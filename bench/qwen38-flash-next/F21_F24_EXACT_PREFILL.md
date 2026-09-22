# F21-F29: exact Flash-Next prefill projection scheduling

2026-09-16, `perf/qwen38-flash-next`. These paths are default-off experiments
for the canonical single-GB10, C1, BF16-KV, initial 2,048-token window. They do
not establish the separate 2,000 tok/s recipe claim, DFlash2, vision, long-QSA,
or fleet-wide qualification.

## Result summary

Effective input rate is `prompt_tokens / server TTFT`, not isolated GPU kernel
throughput. Each median below uses five independently retained M2013 responses
with 2,013 prompt tokens, 27 completion tokens, cache0, `stop`, and the exact
answer and stable semantic hashes.

| Path | Median input tok/s | Change from prior | Exact M2013 |
| --- | ---: | ---: | --- |
| F16 load-fixed, PLE off | 126.3219 | baseline | 5/5 |
| F17 PLE QD256 | 130.3763 | +3.21% vs F16 | 5/5 |
| F21 exact QKV16 | 160.3795 | +23.01% vs F17 | 5/5 |
| F22 exact SSM RT2 | 173.0076 | +7.87% vs F21 | 5/5 |
| F23 exact O16 candidate | 175.0525 | +1.18% vs F22 | 5/5 |
| F24 same-ELF O16 off | 172.2578 | controlled baseline | 5/5 |
| F24 same-ELF O16 on | 174.4729 | +1.29% vs F24 off | 5/5 |
| F25 same-ELF MoE K32 off | 172.8522 | controlled baseline | 5/5 |
| F25 same-ELF MoE K32 on | 176.6274 | +2.18% vs F25 off | 5/5 |
| F29 final K32-only release candidate | 177.1964 | +2.51% vs F25 off | 5/5 |

F21 and its same-ELF selector-off control produced byte-identical complete
1,303-token MinHeap responses. F22 and F23 remain byte-identical to that
response. Matched 400-token decode was 41.39 tok/s for the F21 control, 41.23
for F21, and 41.08 for F22. Later sustained-load samples fell together with
host thermal state; these prefill-only selectors do not enter decode.
F25 K32-on produced the same established 400-token coding content SHA256
`806c9e4f8ed152e481102332bafa568b8d5e6ca4b5d0fcde39da4d83aa4095c3`
at 41.04 tok/s under sustained load. F29 reproduced that content at 41.46
tok/s with 801.1 ms TTFT. Its five M2013 runs were 176.0943, 178.2536,
176.3752, 177.5596, and 177.1964 input tok/s. The live success-only receipt
was `MOE_PREFILL_COMPACT_K32_ENGAGED` for 2,013 rows.

Three bounded follow-ups were exact but rejected on same-binary performance:

| Rejected path | Control median | Candidate median | Change |
| --- | ---: | ---: | ---: |
| F26 compact K64 | 177.2792 | 143.7146 | -18.94% |
| F27 compact M128/K32 | 177.7517 | 161.4429 | -9.18% |
| F28 compact grid 2,048 | 176.5845 | 174.4398 | -1.21% |
| F30 exact attention tile 17 | 177.6555 | 176.9345 | -0.41% |
| F31 exact attention QKV RT2 | 177.0292 | 175.7875 | -0.70% |

K64, M128, and the runtime grid selector are absent from F29. The 1,024-grid
trial was skipped under the declared stop rule after 2,048 regressed. F30 used
the existing exact M17 QG, dual-KV, and O kernels to reduce the scalar tail from
13 to 7 rows; the route remained exact but regressed and was also removed.
F31 reused each activation slab across two adjacent attention-projection output
chains while preserving the exact K8 tree. Its extra M17 accumulator pressure
also regressed, so the private module and selector were removed.

## Exact scheduling changes

F21 packs 16 already-prepared attention inputs and calls the existing exact
gated-Q plus dual-KV projection kernels. QSA updates, KV writes, causal
attention, O projection, and HC injection originally remained token ordered.
The scalar tail is unchanged.

F22 permits `ATLAS_W4A16_GEMV_RT2=1` in F12's exact SSM projection tiles. RT2
reuses each activation slab for two adjacent output rows while retaining the
ordinary K1 lane ownership, operand order, reduction tree, and BF16 output
rounding. Every selected RT2 tier is preflighted before model effects. A
success-only runtime line records `SSM_PREFILL_EXACT_RT2_ENGAGED`.

F23 retains each token-ordered raw attention result, then applies the existing
exact M16 O-projection tier and injects the projected rows in order. Raw rows
are 6,144 BF16 elements and use the checked `ssm_qkvz` arena; projected rows
are 2,560 elements in `norm_output`. F24 separates this behind
`ATLAS_QWEN4_PREFILL_ATTN_O16=1`, which requires the exact QKV16 route, so the
small F23 gain can receive a same-binary A/B decision.

F25 adds `ATLAS_QWEN4_PREFILL_MOE_COMPACT_K32=1`, which requires the compact
MoE selector. It retains M64/N64 and the exact ordered K16 MMA operations, but
loads two adjacent K16 slabs into shared memory before the barrier. This halves
the barrier rounds without changing accumulation or BF16 store order.

## Artifacts

- F21 ELF: `/var/tmp/atlas-flashnext-f21-native.8csmx2/release/spark`, SHA256
  `8a094a8124b6bd509e624659b64bde891e47a6b2e0f1edaeef0b8f47ed3fdb46`.
- F22 ELF: `/var/tmp/atlas-flashnext-f22-native.1FpPmh/release/spark`, SHA256
  `5c92c7efdbe4c7ea44ef34a9b48d49a91c1296cb3f21e24edf7ab19a3c52c98e`.
- F22 five-run JSONL:
  `/var/tmp/atlas-flashnext-f15-ab.VUBZCA/f22-rt2.m2013.jsonl`, SHA256
  `cca0a7dd143193373bc7ba4d1c52e7f77174f3cb09dccf912e1fa101444484e7`.
- F23 ELF: `/var/tmp/atlas-flashnext-f23-native.qqwEi8/release/spark`, SHA256
  `1a7b349cbdca4b9269d031537d60e3f9ea89c55a45e32d4fcf3782d37b13e7eb`.
- F23 five-run JSONL:
  `/var/tmp/atlas-flashnext-f15-ab.VUBZCA/f23-qkvo16-rt2.m2013.jsonl`, SHA256
  `4d64b5cdaa5541d77c6e5999b72ab32ed41445d48ca243fd9f38f91679870df7`.
- F24 ELF: `/var/tmp/atlas-flashnext-f24-native.qVR4yQ/release/spark`, SHA256
  `ed675b95eb2c626063485a061f6c5845d941f3afd62038941785cb66ac10d29a`.
- F24 O16-off/on JSONL SHA256:
  `446a6d0b6c7568e9b5188c5cc269c59d98ee8c7dde9fdc46ed965fc73fcfb803` /
  `4e9a8d227e12c8a91fa71b23f6f19b9d86faf7e898941ccf5dda8e07ec70d84e`.
- F25 ELF: `/var/tmp/atlas-flashnext-f25-native.9QVeaW/release/spark`, SHA256
  `292c78d95c7257c5bf82a2b1d57293fb2780f6e0b8bcddb7469a1b281a9a5b3c`.
- F25 K32-off/on JSONL SHA256:
  `c851ae8738baeaf98c6f940863a109942d0ec80d37423720738637405438e2c4` /
  `750a4dc8d16da9cc281118aff8b11eedca2bb56ac9c84fa193f27003a2be6ec0`.
- F29 final K32-only ELF:
  `/var/tmp/atlas-flashnext-f29-native.vgIXbf/release/spark`, SHA256
  `3653243922a884c7faf5563540239ddae1803819ff5e83f7bc397154dccd4178`,
  build ID `e7c9e677a83f341e90bef9ac3d1437b5b7cf96ec`.
- F29 five-run JSONL:
  `/var/tmp/atlas-flashnext-f15-ab.VUBZCA/f29-final-k32.m2013.jsonl`, SHA256
  `a30260bdd66a48d345e96fdb060c8347a32a509e35798b8cc7c1e073629f301d`.
- F29 coding JSON:
  `/var/tmp/atlas-flashnext-f15-ab.VUBZCA/f29-final-k32.coding.json`, SHA256
  `7694a5663d692c9692ad3f18cce579cc140d047d696d7a4229d8cef284b781e0`.
- F30 same-ELF tile-16/tile-17 JSONL SHA256:
  `d8c24f524f4d5e06309def897c99e12a0b3372502cec45b0165bcabe7a55250a` /
  `16dd8ae82940c532e21ea0a596e5f66c17a2d481ad7e943dbd351c91abe59758`.
- F31 same-ELF QKV-RT2-off/on JSONL SHA256:
  `7a51dbacab7421418f2e8e6cba134e99adc734dd477fa4c2708543d4854664a8` /
  `fffe941fb36954532b3f89f138645744bcd4e1f9e245954f1431260cabf1b6b2`.

F21 profiling measured about 326 ms per attention layer and 230 ms per SSM
layer. F22 lowered the 36 SSM layers to about 200-212 ms while attention stayed
317-331 ms. Compact routed MoE remained about 65-74 ms per layer and is now a
major independent floor. F25 K32 profiling lowered compact MoE from a 69.812
ms per-layer median (K16 reference) to 67.298 ms, a 3.60% kernel-phase gain.
The loader also reports that retaining transposed gate/up experts would require
42.2 GB in addition to the resident 78.19 GB checkpoint, while only about 13
GB remains after construction. A permanent larger gain therefore needs a
replace-instead-of-duplicate weight layout, a decode-compatible transposed
kernel, or a different offload/residency design; another environment toggle
cannot make that layout fit on one 119.7 GB GB10.

## Gates

The final tree passed the complete `spark-model` library suite (639 tests), the
focused compact contract suite (6/6), the compact model suite (5/5), and
`cargo fmt --check`. Its native GB10 build contains 161 modules. These selectors
remain default-off. Default promotion still requires long-QSA, reset and
interleaving qualification plus separate vision and DFlash2 qualification.
