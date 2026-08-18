# Reproducing the Apathy Atlas TUI benchmark work

This experimental branch adds an in-process benchmark plugin system and a
Benchmarks section to the `spark serve` terminal UI. It also replaces mutable
process-global model/run state with explicit owners so benchmark runs and
future in-process model changes do not inherit stale state.

It is infrastructure work, not a new model-throughput record. See
[`BRANCHES.md`](BRANCHES.md) for the measured model branches.

## Build and test

The plugin crate intentionally has no CUDA, model, or server dependency:

```bash
cargo test -p atlas-plugin
cargo test -p spark-server tui
```

For a full server build, follow the upstream hardware/model build instructions.
Start `spark serve`, open the terminal UI, and select **Benchmarks** in the
sidebar. The suite contains concurrency, warm and cold TTFT, agentic, and BFCL
entries. Runs target the attached server by default and can be pointed at a
different OpenAI-compatible endpoint in the parameter view.

## TUI controls

- `j`/`k` or arrow keys: move through benchmarks and rows;
- `Enter`/`l`: open parameters or the active run;
- `s`: start from the parameter view;
- `d`: restore a benchmark's declared defaults;
- `c`: cancel the active run;
- `h`/`Esc`: return to the suite list;
- `v`: reopen the latest run from the suite list.

The agentic entry executes model-authored shell and therefore requires a second
explicit confirmation. BFCL provisions its pinned Python environment beneath
`~/.atlas/artifacts`; other entries are native Rust HTTP clients. Persisted run
frames and baselines live beneath `~/.atlas/runs`.

## What changed on this branch

- a typed plugin/benchmark API, registry, executor, parameter schema and result
  frames in `crates/atlas-plugin`;
- concurrency, TTFT, agentic and BFCL benchmark implementations;
- Suite and History views in the server TUI, including cancellation and run
  persistence;
- end-to-end tests against a mock OpenAI-compatible SSE endpoint;
- a broad run/model/backend state-ownership refactor that removes mutable
  process statics and resets run mailboxes explicitly.

## Maturity and limits

- Experimental: it has not been promoted into an Apathy model release branch.
- The model-specific CUDA build and live benchmark still require the relevant
  branch's hardware and checkpoint prerequisites.
- Benchmark numbers are only comparable when endpoint, model, prompt shape,
  concurrency, sampling, clocks and output fingerprint are held fixed.
