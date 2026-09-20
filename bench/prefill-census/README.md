# Branch-independent uncached prefill census

CPU-only HTTP client for a **separately reserved and started** Atlas server.
This preserves the existing V4 measurement protocol used for the September 5
DeepSeek Vision census. It does not link Atlas, CUDA, Torch or model code,
start/stop services, change recipes, download models, or select a winner.

## Build and test

From this directory, with Rust and the locked dependencies already cached:

```sh
CARGO_TARGET_DIR=/absolute/task-specific/build-directory \
  RUSTFLAGS=-Dwarnings cargo test --locked --offline
CARGO_TARGET_DIR=/absolute/task-specific/build-directory \
  RUSTFLAGS=-Dwarnings cargo build --release --locked --offline
cargo fmt -- --check
```

This is an independent Cargo workspace. Its lockfile and exact serde_json
version replace the old temporary build script's hardcoded cached rlib path.
No changes to the parent Atlas workspace are needed. Source files are split
below 250 lines; protocol function bodies and the 13 existing tests are retained.

## Measurement contract

- Explicit server model, port, input-ID wire format and producer provenance.
- Local `127.0.0.1` only; curl configuration, proxies and redirects disabled.
- Admission verifies model/config and executable hashes, PID/start time,
  actual argv, safe tuning environment, and ownership of the listening port.
  PID/executable/argv/environment/port are rechecked around each request.
- Exact 256/2048/8192 token bins, or an explicit ascending subset. One warm
  request and five measured requests per bin, temperature zero, max32 outputs.
- Every response requires exact input usage, explicit zero cached tokens,
  consistent counts, finished nonempty output and finite positive timing.
- Retain raw requests/responses, transport errors, token IDs, output text and
  SHA256s. Output directory must be new. A complete summary is written only
  after every selected bin succeeds; failures retain partial evidence.

The prefill metric is **input tokens / server TTFT**, not isolated GPU prefill.
Client wall time includes the complete request and is separate. The server's
reported decode metric is retained but is not a representative decode benchmark.
Missing cache telemetry is an error, not zero. Unmeasured bins stay explicit.

## Invocation

After fresh coordination and server startup, replace every illustrative path:

```sh
/absolute/task-specific/build-directory/release/atlas-prefill-census \
  --port 8977 --model EXACT_SERVED_NAME \
  --ids-mode prompt-array \
  --prompt-format deepseek-chat \
  --bins 256,2048 \
  --output-dir /absolute/new-nonexistent-evidence-directory \
  --provenance /absolute/frozen-producer-provenance.json \
  --timeout-seconds 600
```

`prompt-array` sends integer IDs in `prompt`; `prompt_token_ids` uses the
older dedicated field plus an empty `prompt`. Confirm the actual branch API.
No speculative-decode setting is inferred from the model name or environment.

Available prompt formats preserve V4 bytes:

| Format | Intended use |
| --- | --- |
| `qwen-chatml` | Inspected dense Qwen / Flash-Next instruction completion paths |
| `deepseek-chat` | Inspected DeepSeek native thinking-off template |
| `plain` | Historical plain completion corpus; legacy default, not universal chat |

Always choose the format explicitly in a fleet recipe. Other model templates
and translation need separately tested adapters; registration alone does not
make these prompts valid. The complete separately tokenized suffix is retained
in every exact-length input. This is a deterministic factual-summary corpus,
not a coding, vision, retrieval or varied-workload quality evaluation.

## Producer provenance

Example schema only, **not a launch recipe**:

```json
{
  "schema": "atlas-prefill-census-provenance-v1",
  "pid": 12345,
  "port": 8977,
  "model_name": "EXACT_SERVED_NAME",
  "binary_path": "/absolute/frozen-spark",
  "binary_sha256": "REPLACE_WITH_64_LOWERCASE_HEX_CHARACTERS",
  "config_path": "/absolute/checkpoint/config.json",
  "config_sha256": "REPLACE_WITH_64_LOWERCASE_HEX_CHARACTERS",
  "context_limit": 12288,
  "argv": ["/absolute/frozen-spark", "serve", "EXACT_REMAINING_ARGUMENTS"],
  "env": {"CUDA_VISIBLE_DEVICES": "0"}
}
```

`argv` must exactly equal `/proc/PID/cmdline`. `env` includes every effective
`ATLAS_*`, `SPARK_*`, `NCCL_*`, and any present `CUDA_VISIBLE_DEVICES`,
`CUDARC_CUDA_VERSION`, `CUDA_DEVICE_ORDER`, `CUDA_MODULE_LOADING`, and no others.
Secret-like keys/arguments are rejected without recording their values. Do not
put credentials or a full generic environment in this file. Context must cover
the largest selected bin plus32 output tokens. Prefix caching must be off.

## Qualification is a separate gate

The canary is printed and recorded verbatim with `semantic_status: unreviewed`.
Like V4, a wrong but nonempty canary does **not** automatically stop the client.
Every successful summary is labelled
`MEASURED_SELECTED_BINS_SEMANTICS_UNREVIEWED`; it is not a quality pass.
An operator must check the canary, answer quality and baseline/candidate
output/continuation parity before attributing or promoting any speed result.

This client does not hash all checkpoint payloads or tokenizer/template files,
detect every competing GPU process, or profile CUDA intervals. Keep a separate
frozen branch/source/kernel/checkpoint/tokenizer/template/recipe manifest,
exclusive resource reservation and pre/postflight evidence. Run it only in an
owned evidence directory. Never reuse provenance from an exited server.

For optimization: first verify baseline determinism and candidate correctness;
then use same-binary ABBA comparisons and independent decode/vision checks.
Use existing Atlas profiling/microbenchmarks for disjoint on-GPU attribution.
Do not combine instrumented and uninstrumented timings, nested profile buckets,
different checkpoints, cache hits or different prompt formats.

## Lineage

Original frozen client:
`/var/tmp/atlas-prefill-census-v4.lnBBu3/census`, SHA256
`9646ca7538c74f0489a992d47c044f1e57f370bfcdf1baf830b7fc82d1138fea`.
Those original sources, binaries and measurement artifacts are unchanged.
This durable packaging is CPU-qualified only; its newly built binary requires
fresh server admission before any new measurement. See `QUALIFICATION.md`.
