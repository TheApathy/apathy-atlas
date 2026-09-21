# F13: original-layout SSM tensor-core projections

2026-09-05, `perf/qwen38-flash-next`, HEAD
`11e76f29a68ed5e22be8fd14cc72e318bdcc6111` plus preserved dirty source.
This extends F12 experimentally; no default promotion, fleet-wide completion,
2,000 tok/s, DFlash2, or vision qualification is implied.

## Implemented contract

`ATLAS_QWEN4_PREFILL_SSM_GEMM` accepts absent/`0` (off), `out` (output only),
or `all` (QKVZ and output). `1`, `off`, typos and non-UTF8 values reject.
It requires the F8 MoE batch, F12 HC exact and SSM exact selectors. The F12
canonical eager C1/BF16/F32-state/2,048-token geometry and arena admission remain.
Active strict SSM CHECK conflicts reject before effects; it is never changed
to a tolerance checker. Missing original-layout GEMM handles also reject.

The two projection sites use existing original-layout weights and
`ops::w4a16_gemm`, with original exact tiles in the off branch. No new CUDA
production code, weight copy, transpose, GPU allocation or FP8 dispatch.
Singleton continuation uses unchanged decode. The receipt labels `off`, `out`
and `all` separately and does not call tensor-core projections exact.

QKVZ is N16,384/K2,560; output is N2,560/K6,144. Both use BF16 operands, FP32
MMA accumulation, and BF16 output. Per-tensor scale2 is included in weight
dequantization **before** its BF16 rounding boundary. This differs numerically
from the F12 exact GEMV family and needs separate model-quality evaluation.

## Source and CPU evidence

Frozen build-input fingerprint:
`f589a434108efd39bf69495a82c2f405617c181bdd85d9b80120f8a3b469a037`.
Current-tree offline/locked skip-native Cargo tests:618 library +72 focused
=690 pass, exit0. CPU artifacts:
`/var/tmp/atlas-f13-receipt-cpu.9eMxnHnZ/integrated-summary.json`.
Standalone selector/source tests24/24 and receipt source tests2/2 also pass.

Synthetic numerical harness: `f13_projection_gate/`, six new files. Independent
host E2M1/E4M3 decoding includes non-power-of-two scale2 before BF16-RNE; cuBLAS
GEMMEx consumes BF16 operands and produces FP32 reference on the same nondefault
stream. Reduced-precision reductions and atomics are disabled and checked.

Fixed regression criterion: at most one BF16 ULP from the rounded reference,
or absolute error at most1e-4 against FP32 near cancellation. This is **not**
a universal error bound or full-model accuracy threshold. The harness reports
floor-only admissions and worst coordinates, rejects nonfinite/corrupt outputs,
checks poisoned padded rows/redzones and immutable inputs, and repeats after
reset. An explicit wrong post-dot-scale reference must fail its canary.

Corpus: both shapes, M1/2/24/31/32/33/63/64/65/256/2048, two resets =44 cases.
CPU decode/comparator/parser tests pass; final CPU executable SHA256
`ff1c69c5c915cbee37d0c6579bd9ad0dff7f6aad8f41d7a4a2f813910e5939a2`.

## Qualification state

Native server build completed, exit0 in2m24s,160 selected kernels, in
`/var/tmp/atlas-flashnext-f13-native.erkxYz4N`. ELF SHA256:
`96dea1a4e74fd86b4a63cc681e68ec3acb6d7f96a3ae6eec40b3d5869d708aea`.
Source fingerprint unchanged; generated target table matches F12.

Synthetic GPU passed first output/M24 and then all44 cases, exit0. Evidence:
`/var/tmp/atlas-f13-projection-gpu.rlIdYWqc/summary.json` and complete `full.jsonl`.
Raw JSONL SHA256 `22c380f203f462964500315c50d30b30af50e457408494df270079b54c09f2c4`;
bench ELF `137083c6e811468324b9046e3c390fcce68d2090d6f8bc879041fd365b75999b`.
QKVZ matched rounded-reference BF16 in all cases; output had120 floor-only
admissions across repeats, with max32 ULP near zero. The fixed numerical gate
passed, but this is not byte-exact GEMV or model-state parity. Relative L2 error
against FP32 was at most0.001662 in this synthetic corpus, including ordinary
BF16 output rounding. Raw diagnostic FNV integers remain unmodified.

Output-only actual-model run completed ten requests. Eight short exact-output
checks and the256-token comparison pass. The2,048-token response changes wording
from F12 ("detailing the specific counts ..." instead of "detailing the number
..."). It is coherent, not demonstrated semantic failure, but does not pass the
predeclared output-equivalence gate. No further candidate timing or `all` model
run was performed. The candidate stays default-off and unpromoted.

All ten requests had cache0 and complete layer receipts. Same-input HC960 and
compact480 checks pass;20 path receipts and zero logged errors. Strict SSM CHECK
was0 by the conflict contract. These checks do **not** compare F13 SSM state to
F12; changing output-projection rounding can also change later-layer state.
All timings from this checked run are ineligible for benchmark credit.

Model evidence: `/var/tmp/atlas-flashnext-f13-out.07WZAm9S/summary.json`, SHA256
`ac8e7489c4f419b897fdb3424b37837af4df42565a0427cded0ddce1843bfe49`.
Complete server log SHA256
`63e044653b874b1279ec5efc94f9a509282741a07cb34b797ff874295de62540`.
Bound PID2642804/start8685449 received SIGINT after response completion and
exited0; GPU and port8898 are empty. Frozen source remains unchanged. No lanes
are retained. Resume only with a new root reservation.

Next numerical gate: localize output/state deltas on identical model inputs,
then use a predefined representative quality criterion before timing `out`
or `all`. Do not weaken the exact checker or call a wording change a speed win.
