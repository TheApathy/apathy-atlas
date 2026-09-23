# GLM-5.3 mixed layer-major prefill P45 plan

## Evidence

- P44 natural-image timing preserves exact output on all three fixtures.
- At 1,220/1,221 prompt tokens, `prefill_chunk` takes 53.45/53.95 seconds,
  or 84.5% of TTFT. At 429 tokens it takes 17.06 seconds, or 93.0%.
- `requested_wide_prefill(true)` deliberately excludes layer-major mode and
  sends mixed text/image inputs through at most eight rows per target pass.

## Bounded change

1. Add `ATLAS_GLM53_LAYER_MAJOR_VISION_PREFILL=1` as a separate, default-off
   admission gate. Existing text, serial mixed, and 2/4/8 mixed recipes retain
   their current behavior.
2. Reuse the existing whole-prompt target/drafter/capacity and prepared-owner
   admission. No new unchecked model allocation is allowed.
3. For each admitted layer-major chunk, gather normal token embeddings into the
   existing destination, then overwrite only contiguous image-pad spans from the
   prepared BF16 owner. Validate range, extent, disjointness, global pad ordinal,
   and copy failure before publishing state.
4. Generalize the existing DFlash2 layer-major capture body with a fill closure;
   retain its locks, capture receipt, commit, synchronization, observation,
   abort, poison, and owner lifecycle unchanged.
5. Keep the existing 1..=8 mixed path unchanged. Never credit performance until
   same-image exact parity and sampled-GPU qualification pass.

## Gates

1. RED: isolated environment selection and greater-than-eight image overwrite.
2. PASS: focused selector/input/capture/wiring tests, then full `spark-model` lib.
3. Format, SPDX, file-size, and source-diff audit.
4. Native build only from a frozen source snapshot.
5. Device gate: exact mixed-input parity against P44 on small adversarial layouts.
6. Serving gate: exact three-image answers, then same-request TTFT comparison.

## Stop conditions

- Any target/drafter cursor disagreement, capture receipt drift, source alias,
  partial-copy ownership ambiguity, output mismatch, runtime fault, foreign GPU
  process, or worse TTFT stops promotion.
