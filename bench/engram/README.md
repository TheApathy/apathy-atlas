# Engram lane: fixtures, oracle and the I/O floor

Everything here runs on CPU/NVMe only. No GPU lock is needed.

| file | what it does |
|---|---|
| `export_map.py` | Exports `token_map_i32.bin` (129,280 x i32) from the checkpoint tokenizer. Asserts the compressed vocab is exactly 99,092 — the reference's own load-bearing check, because every hash multiplier derives from it. |
| `ref_hashes.py` | Runs the checkpoint's `inference/engram.py` `NgramHashState` on a prompt and dumps `ref_hashes.bin` `[T, 2, 24]` int64 plus `ref_ids.bin`. |
| `cmp_rows.py` | Dequantizes the reference rows for those hashes and compares a candidate `[T, 2, 24, 256]` f32 dump **bitwise**. Both sides are exact in f32 (3-bit mantissa x power-of-two scale), so anything but bit-equality is a real bug, not rounding. |
| `iofloor.rs` | The I/O floor probe (`rustc -O -o iofloor iofloor.rs`). Args: `<tokens> <iters> <split|cached> <thread list> [seed]`. Pass a FRESH seed each run — replaying a seed replays the row ids and measures the page cache instead of the device. |
| `dump_chunk_ids.py` | Pulls one occurrence's LOCAL chunk token ids out of an oracle manifest (`engram_dead_heads` is called fresh per forward chunk, no cross-chunk carry — see `dead_heads.rs`'s module doc). Writes `ids_NNN.bin` per occurrence. |
| `oracle_dead_heads.rs` | Splice with `dead_heads.rs` (zero external deps, compiles standalone) and run against `ids_NNN.bin` files to validate `engram_dead_heads` against a real capture's `engram_dead` taps. |
| `oracle_apply_mask.rs` | Splice with `dead_heads.rs`; applies `apply_dead_mask` to a captured `engram_rows_premask` and checks it reproduces `engram_rows` bit-exactly. |

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
- **Dead-head mask: NOW VALIDATED, 2026-09-22.** `runE_image` (1031 tokens, a real
  IMAGE_SENTINEL/IMAGE_PAD span at positions 8-14) makes `engram_dead_heads`
  non-degenerate: occurrence 0 has 216/12288 True, matching the brief exactly.
  `crates/spark-model/src/layers/deepseek_v41_engram/dead_heads.rs` (in-crate,
  `cargo test -p spark-model --lib deepseek_v41_engram::dead_heads`) unit-tests the
  pure function; `oracle_dead_heads.rs` + `oracle_apply_mask.rs` (below) validate the
  SAME code against the real engine's taps, bit-exact, with negative controls
  watched failing (an off-by-one shift: 24/12288 mismatches at the boundary row; a
  skipped mask: rel_l2 0.15).
- **Chunk boundary: NOW VALIDATED, 2026-09-22.** `hash.rs`'s
  `chunk_boundary_matches_the_oracle_exactly` replays `runD_L20_kernel` (1024 real
  tokens, two 512-token chunks) through `EngramHashState` and matches the oracle's
  `engram_hashes` at the S=512 occurrence exactly, both engram layers. Negative
  control `dropping_the_carry_breaks_the_match`: re-hashing chunk 1 without the
  cross-chunk cache moves exactly 48/12288 row ids — 24+16+8, the count the n-gram
  geometry predicts for how far a 4-gram's look-back reaches past the boundary, not
  merely "some". Needs `token_map_i32.bin` (regenerate with `export_map.py`;
  `*.bin` is gitignored so it is not checked in).

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

### Dead-head path: validated on `runE_image`

`engram_dead` in runA is 888 entries, 0 True, on both layers — confirmed empirically,
as predicted, and it is why runA alone could never validate this path.
`runE_image` fixes that: a synthetic-token image span (`IMAGE_PAD_ID`/
`IMAGE_SENTINEL_ID` present, vision encoder did not run — valid for engram and the
router's vl-bias branch, not the vision encoder itself) gives occurrence 0 a
non-degenerate 216/12288 True.

| tap | result |
|---|---|
| `L01.engram_dead.000` / `L14.engram_dead.000` | **0/12288 mismatches**, both PASS |
| `L01.engram_dead.001/.002`, `L14.` (post-image, all-False) | **0 mismatches** too — low-information per compare.py, kept as a sanity check that the mask correctly turns back off once the span ends |
| `L01.engram_rows.000` = `apply_dead_mask(premask.000)` | **rel_l2 0.0** (3,145,728 f32) |
| `L14.engram_rows.000` = `apply_dead_mask(premask.000)` | **rel_l2 0.0** |

Negative controls, each watched FAIL:

| control | predicted | observed |
|---|---|---|
| shift the mask forward one position (`p > offset` instead of `p >= offset`) | only the boundary row where the span's own look-back reaches position 0 of the chunk moves = 1 row x 24 cols = **24** | 24 |
| skip `apply_dead_mask` entirely (compare premask directly) | premask != rows wherever the mask is True | rel_l2 0.151 (L01), 0.155 (L14) |

Both engram layers' masks are IDENTICAL (same sha256) at every occurrence, as
expected — `engram_dead_heads` is a pure function of token ids, not of which
engram layer is asking.
