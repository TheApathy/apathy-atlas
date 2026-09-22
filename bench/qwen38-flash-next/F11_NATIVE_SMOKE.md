# F11: native F8 build and same-binary short smoke

2026-09-05. **Pass: native build and8/8 short candidate outputs identical to
8/8 baseline outputs. Not full numerical parity, speed, coding or vision
qualification. F8 remains default-off.** Both test servers were drained.

## Frozen native build

Worktree `perf/qwen38-flash-next`, HEAD
`11e76f29a68ed5e22be8fd14cc72e318bdcc6111` plus preserved dirty source.
No production source or CUDA changes were made during F11.

```sh
cd /home/flocka/atlas/qwen38-flash-next
env -u ATLAS_SKIP_BUILD -u SKIP_ATLAS_BUILD \
  CUDA_HOME=/usr/local/cuda-13.0 CUDARC_CUDA_VERSION=13000 \
  ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=qwen3.8-flash-next \
  ATLAS_TARGET_QUANT=nvfp4 \
  CARGO_TARGET_DIR=/var/tmp/atlas-flashnext-f11-native.dTvuWPdb \
  RUSTFLAGS='-C target-cpu=native' \
  cargo build --locked --offline --release -p spark-server --bin spark
```

Build exited0 in2m25s,159 selected model kernels. The native CUDA build supplies
cuBLASLt linkage; the F8 CPU skip-build linker workaround was not used.

- ELF: `/var/tmp/atlas-flashnext-f11-native.dTvuWPdb/release/spark`.
- ELF SHA256: `ead677cb479c529fcbc4dd2c880bd32b487bdeb90ca95ae107e2139799eb1eb1`.
- Build ID: `6d82449d8f973f4b65c3da632a3cd1920d8f10e5`.
- Generated target table SHA256:
  `9edc645c17a5f00016493b57c9635a149dcc71a322f3921b6f08c79f417386c0`.
- Source fingerprint before build, after build and after both runs:
  `7dcddd13afc1baff99673702f117375b0bdfee1907aa5b44a5c93a92e4777ca0`.

Fingerprint input: sorted unique tracked+untracked nonignored files under
Cargo.toml/Cargo.lock/rust-toolchain.toml/.cargo/crates/kernels, with registered
kernel directory aliases expanded to their regular-file contents; each line is
the path-qualified sha256sum, then the complete listing is hashed. An initial
attempt to hash directory symlinks as files failed and was discarded, not used
as a valid source fingerprint. Bench/docs are outside these build inputs.

## Exact A/B recipe

No existing ATLAS/SPARK/NCCL/CUDA tuning variables were inherited. These four
were explicit; only the final selector changed from0 to1 for B:

```sh
ATLAS_PLE_CACHE_MB=512 \
ATLAS_QWEN4_PLE_SEGMENTED_GRAPHS=0 \
ATLAS_SSM_H_FP16=0 \
ATLAS_QWEN4_PREFILL_MOE_BATCH=0 \
/var/tmp/atlas-flashnext-f11-native.dTvuWPdb/release/spark serve \
  --model-from-path /home/flocka/models/Qwen3.8-Flash-Next-NVFP4-Offload \
  --model-name qwen3.8-flash-next --kernel-target qwen3.8-flash-next \
  --bind 127.0.0.1 --port 8898 \
  --max-seq-len 4096 --max-prefill-tokens 2048 \
  --max-num-seqs 1 --max-batch-size 1 --ssm-cache-slots 8 \
  --kv-cache-dtype bf16 --enable-prefix-caching false --qwen4-qsa --no-tui
```

Config SHA256 `e765305daba0951974308f4d32c075b52a6a45974730d273f2216718a994d624`;
index `c654034a19be39baf2348dc02c818b555d3e0f2dc036346f58fe3623c1bc311d`;
PLE manifest `01911bc91039510642c8ebec4aa4f777bd533b317a0f448063707072aa3b5520`.
Tokenizer `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`;
tokenizer_config `b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27`;
chat template `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`.
Full tensor payloads were not independently hashed during this smoke.

Both use no speculation, BF16 KV, BF16 residual and F32 SSM. Optional63.3GiB
full/42.2GiB gate-up transpose copies cannot fit, so the loader explicitly keeps
the original-layout uncoalesced MoE prefill route. Its log's predicted slowdown
is not a measured performance result. Original-layout F8 admission succeeded.

## Short result

Each request was repeated twice in each arm. Temperature0/thinking off, max32
outputs except max64 for the small code request; all finished naturally at stop.

| Request | Input tokens | Output tokens | Exact output, both repeats/arms |
| --- | ---: | ---: | --- |
| C3 canary instruction | 24 | 6 | ATLAS_CANARY_OK |
| 17 + 25 | 28 | 3 | 42 |
| Named-box retrieval | 41 | 2 | BLUE |
| Python add(a,b) function | 30 | 18 | Same complete fenced function returning a + b |

All16 responses: HTTP200/curl0, identical A/B prompt and completion counts,
cache0, expected content and stop. PID/start/executable/port ownership were
checked around every request. Actual argv equality and tuning-name inventory
were checked per arm. Requests and saved output files were also compared on
disk after drain. Saved output files have a normalized terminal LF; raw JSON
retains the exact output string and is authoritative for content comparison.

B emitted16 ordered receipts:12 attention and36 SSM layers for each of8
requests, with matching M, `ffn=grouped core=serial_token_ordered`. Receipts mean
enqueued paths, not bitwise state equality. No census or timing A/B was run.

Evidence:

- A: `/var/tmp/atlas-flashnext-f11a-smoke.TDLsXLtX/`, summary SHA256
  `dfe86624ce95001d7fa6a0ede797342bde8a06771314fb81e51d1f37ed47248c`.
  Producer `cf3022f8a6dc95d7d081a52ce3096d7428aedd3bb1299ce2dbc8884da80d7e21`.
- B: `/var/tmp/atlas-flashnext-f11b-smoke.Ht5NWaAU/`, summary SHA256
  `d0f814f870cd1d7f58663f6d7bf0b54f16d0fad2ca7727be4a89bd7df7c80439`.
  Producer `01538190c43f173a43f2a63f47d4b4db1a3d4ceacd435ab07291403366b66773`.
  Complete request-engagement log:
  `5be0b429e654b361a525258d8fc99cf1d5b290edfd68fd94afca73e04ceac85b`.

B's captured startup output was truncated during middle weight loading; this is
flagged in its summary and is not complete startup-log proof. All request,
response and engagement records are complete. Native source/binary are unchanged.

## Remaining gates and resources

Short text equality is insufficient: router logits/IDs/weights, routed/shared
FFN outputs, complete final KV/SSM/HC state, singleton/chunk/reset boundaries,
interleaved requests and next-token continuation remain to be captured and
compared. Existing PLE capture covers only PLE state, not this full F8 contract.
Only then run varied/coding and256/2048 uncached same-binary ABBA with separate
GPU prefill/TTFT/decode measurements. Long QSA and multimodal remain separate.

A PID2455431/start7997266 and B PID2463915/start8033865 each received a
binding-checked SIGINT and exited0. Final GPU process/listener inventory was
empty, with109GiB available host memory. No GPU/server/native-build reservation
is carried forward; fresh root claims are required for every next lane.
