# DeepSeek V4 Flash K2 — one-Spark measured checkpoint

Date: 2026-08-24

## Reproducibility

- Atlas branch: `apathy-deepseek`
- Implementation commit: `6803bf4d7fa05189482de790bf38db38ab9d89a2`
- Checkpoint: `wrldsuksgo2mars/DeepSeek-V4-Flash-0731-EXL3-K2-calibrated-v1`
- Checkpoint revision: `68eaca43e99bfbfd697a5559c7796b983deb38f8`
- Upstream recipe: `tpurtell/ds4-mia-exl3-k2-1spark@f20b97dfd7666c00c316f29542e2e53f33cabb19`
- Hardware: one NVIDIA DGX Spark / GB10
- Checkpoint validation: 10/10 shards opened, 142,973 tensors enumerated

## Decode results

All rows are concurrency one and use varied prompts. Rates are visible decode
tokens per second; they are not synthetic kernel throughput.

| Profile | Code | Repetition | Quotation | Prose |
| --- | ---: | ---: | ---: | ---: |
| Plain K2, BF16 attention | 22.98 | 22.97 | 22.94 | 22.90 |
| DSpark launcher, warm, legacy gamma 5 (four drafts) | 26.91 | 28.40 | 24.73 | 25.87 |

The warm DSpark configuration is a speed profile, not a byte-identical K2
quality baseline: it enables Atlas's lossy NVFP4 attention-residency path.
Code, quotation, and prose output hashes differed from the plain profile;
repetition matched. Quality must be evaluated before promoting that profile.

The embedded five-token DSpark proposer loads and runs, but this measurement
used the legacy raw `GAMMA=5` launcher spelling, which asks Atlas for four
drafts because gamma counts the target bonus row. It is retained as historical
evidence, not as the checkpoint-native five-draft result. The online scheduler
measured that four-draft MTP arm at 14.6 tok/s versus 24.8 tok/s for serial
decode and switched back to serial.
The current result therefore proves the DFlash integration works, while also
showing that speculation does not yet move the performance frontier.

## Memory and context

- Plain tuned boot: 300,000 resident KV tokens; the API exposes the native
  1,048,576-token YaRN ceiling through paged-KV overcommit.
- Untuned boots varied between 117K and 134K resident tokens as free unified
  memory changed.
- The 8K speculative boot reported capacity for 403,104 resident tokens after
  target load, BF16 mirror release, and zero-copy drafter reuse.
- The embedded drafter now aliases 9,313 target-store tensors, avoiding a
  duplicate 5,455,341,084-byte allocation.
- Speculative 1M is not claimed. DSpark capture is currently an absolute BF16
  buffer (`3 * max_seq_len * hidden`), which is 24,576,000,000 bytes at 1M.
  It needs a windowed capture ring before the 1M speculative profile is viable.

## Validation

- K2 CPU/GPU dequant gate: bit-exact; retained K3 decode, prefill, fused,
  deterministic, and m-row gates pass.
- `cargo build --release -p spark-server`: pass, 176 GB10 kernels compiled.
- `filtered_view_aliases_only_selected_tensors`: pass.
- `from_safetensors_str_matches_disk_mapping`: pass.
- Full `spark-runtime` unit run had one unrelated existing failure:
  `buffers::tests::test_buffer_arena_alloc` expected 29 allocations and saw 31.
