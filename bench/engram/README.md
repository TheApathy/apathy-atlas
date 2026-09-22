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
