# Engram lane: fixtures, oracle and the I/O floor

Everything here runs on CPU/NVMe only. No GPU lock is needed.

| file | what it does |
|---|---|
| `export_map.py` | Exports `token_map_i32.bin` (129,280 x i32) from the checkpoint tokenizer. Asserts the compressed vocab is exactly 99,092 — the reference's own load-bearing check, because every hash multiplier derives from it. |
| `ref_hashes.py` | Runs the checkpoint's `inference/engram.py` `NgramHashState` on a prompt and dumps `ref_hashes.bin` `[T, 2, 24]` int64 plus `ref_ids.bin`. |
| `cmp_rows.py` | Dequantizes the reference rows for those hashes and compares a candidate `[T, 2, 24, 256]` f32 dump **bitwise**. Both sides are exact in f32 (3-bit mantissa x power-of-two scale), so anything but bit-equality is a real bug, not rounding. |
| `iofloor.rs` | The I/O floor probe (`rustc -O -o iofloor iofloor.rs`). Args: `<tokens> <iters> <split|cached> <thread list> [seed]`. Pass a FRESH seed each run — replaying a seed replays the row ids and measures the page cache instead of the device. |

## Measured I/O floor (cold, this box)

| point | threads | per step | cap |
|---|---|---|---|
| prefill, 2048-token chunk (98,304 rows) | 256 | 104.3 ms | 19,637 tok/s |
| decode, 1 token (48 rows) | 32 | 1.21 ms | 827 tok/s |
| decode, 1 token | 1 | 20.3 ms | 49 tok/s |

Cross-checked against `/proc/diskstats`: 3.94 GB/s at 917k IOPS, **4684 device bytes
per requested row** — one 4 KiB page per 256 B row, which is what a genuinely cold
random gather costs and what a page-cache artifact could not produce.

So engram I/O is ~4% of a ~30 ms decode step and 16x under a ~1190 tok/s prefill
engine. It is **not** the bottleneck. The thread count is the whole game.

## Validation status

- n-gram hashing: **98,304 / 98,304 row ids exact** vs the Python `NgramHashState`
  on a real 2048-token prompt.
- gather + dequantize: **25,165,824 / 25,165,824 f32 values bit-exact**.
- Negative controls, each watched to FAIL: perturbed multiplier (hash), swapped
  prime order (layout), non-sticky look-back, perturbed scale byte (dequant), and
  wrong row ids (gather).
- **UNVALIDATED: the dead-head mask.** On a text-only prompt `engram_dead_heads` is
  identically all-False, so `masked_fill` is a no-op and any comparison passes by
  construction. Validating it needs a capture containing a real image span.

## Row reuse (why `gather_dedup` exists)

Measured on the real 2048-token prompt above:

| lookups | unique rows | saving |
|---|---|---|
| 98,304 (both layers) | 75,483 | **23.2%** |
| 2-gram columns | 20,847 / 32,768 | 36.4% |
| 3-gram columns | 26,191 / 32,768 | 20.1% |
| 4-gram columns | 28,443 / 32,768 | 13.2% |

A duplicate costs a full 4 KiB page fault, so the dedup removes ~23% of the I/O for
the price of a hash map. The gradient — shorter n-grams repeat more — is what
natural text should produce, and is a soft check that the hash is not degenerate.

## Validated against the oracle capture (runA, the real engine)

`validate_oracle.sh` compares against `DSV41_PORT/oracle/ref/runA`, a 37-token
capture of the production engine. This is a stronger check than running the
reference myself: it is the engine's own taps.

| tap | result |
|---|---|
| `L01.engram_hashes` | **0 / 888 mismatches** (exact integer compare) |
| `L14.engram_hashes` | **0 / 888 mismatches** |
| `L01.engram_rows` | **rel_l2 0.0, max_abs 0.0** over 227,328 f32 |
| `L14.engram_rows` | **rel_l2 0.0, max_abs 0.0** over 227,328 f32 |

compare.py's own auto negative control was rejected in every float case, so the
tolerance is not what produced the PASS.

### Negative controls — each fails with the count the structure predicts

Failing is not enough; the failure must land where the n-gram geometry says it must.

| control | predicted | observed |
|---|---|---|
| flip one multiplier (`mult[0][2]`) | enters `rolling` at i=2, so the 3- and 4-gram groups = 16 cols x 37 = **592**, layer 0 only | 592, L14 correctly untouched |
| swap two primes (layer 0, 2-gram, heads 0/1) | exactly those 2 cols x 37 = **74**, layer 0 only | 74, L14 correctly untouched |
| perturb `token_map` at one id | id 260 occurs at positions 5/23/29/35; a change at p reaches p+0..3 but the 2-gram looks back only 1 and the 3-gram 2, giving 24+24+16+8 per isolated occurrence and 48 for the one at the tail = **264**, BOTH layers | 264 on both |

The third control is the team lead's: its failure would otherwise be mistaken for
a tokenizer-version difference rather than a bug.

### Dead-head path: still UNVALIDATED

`engram_dead` in runA is 888 entries, **0 True**, on both layers — confirmed
empirically, as predicted. `masked_fill` is a no-op here, so the dead-head path is
NOT exercised and a PASS on `engram_rows` does not cover it. Needs an
image-bearing capture.
