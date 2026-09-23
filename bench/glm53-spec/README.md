# GLM-5.3 DFlash2 speculative decode: verify-cost fixes and their gates

Result (window w3, 2026-09-23, EXL3 2.05 bpw, greedy, 160 tokens, 3 prompts x 2 trials,
fresh server per arm, target arm first and last):

| arm | short | long | prose | output |
|---|---|---|---|---|
| target-only (T, T2) | 14.09 / 14.51 | 12.23 / 12.55 | 14.13 / 14.54 | reference |
| DFlash2 gamma 3, no fixes (w1 S0) | 7.86 | 9.79 | 8.16 | identical |
| DFlash2 gamma 3, all fixes (S4) | 16.26 | 16.72 | 16.70 | identical |
| DFlash2 gamma 2, all fixes (S4g2) | 16.56 | 16.23 | 17.44 | identical |

All numbers run under `ATLAS_GLM53_UNVALIDATED_BRINGUP=1` (GLM kernel admission is closed).

## The fixes (all default off)

- `ATLAS_GLM53_EXL3_ROWBATCH=1`: the exact-verify EXL3 projection ran the pinned M=1 GEMM once
  per row inside one cooperative launch (weights re-read and re-decoded per row). It now runs
  every row in one pass of the same inner kernel. Bit-identical per row by construction.
- `ATLAS_GLM53_VERIFY_DSA_PRECOMPUTE=1`: DSA row-local inputs (q/kv/indexer projections, norms,
  key absorption) computed once for all verify rows before the causal per-row loop.
- `ATLAS_GLM53_VERIFY_DSA_BATCH_OUTPUT=1`: value absorption and output projection deferred out
  of the per-row loop and run once over all rows.
- `ATLAS_GLM53_PREFIX_COMMIT=1` (decode agent B's design, plus a fix): a partial acceptance commits
  per-row KDA/DSA snapshots instead of replaying the accepted prefix through the target. The
  original re-staged every KDA layer's conv shift register from the verify scratch, which all
  34 layers share, i.e. from the LAST layer's inputs. Each layer's conv inputs are now saved.

## Gates

- `harness/rowbatch_equiv.cu`: rowbatch vs rowexact vs M=1, 4 tile shapes x rows 1..8, memcmp;
  negative control perturbs one input row and requires only that row to change.
- `score.py <run> <ref-arm>`: every trial's text must equal the reference arm's; a perturbed
  prompt must differ (the gate can fail); phase receipts give the per-step decomposition and
  per-position acceptance.
- `statehash_compare.py`: `ATLAS_GLM53_STATE_HASH=<file>` hashes the committed conv and
  recurrent state of KDA layers 0/17/33 after every commit; spec must equal the target walk at
  every shared position. Control: `ATLAS_GLM53_PREFIX_COMMIT_CONTROL_LEGACY_RESTAGE=1` re-enables
  the original bug and must fail (it does, from the first prefix commit, on layers 0 and 17 but
  not 33 -- the last layer's own inputs).
