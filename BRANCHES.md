# Apathy Atlas branch map

This repository keeps model-specific work on separate branches so a measured
tree is not rewritten merely to make its history look tidy. Use this page to
choose a branch before building or comparing results.

| Branch | Status | Scope | Start here |
|---|---|---|---|
| [`main`](https://github.com/TheApathy/apathy-atlas/tree/main) | Upstream mirror | Current Avarok Atlas tree; no Apathy performance claims | Upstream `README.md` |
| [`base`](https://github.com/TheApathy/apathy-atlas/tree/base) | Stable upstream anchor | Same upstream commit as `main`, retained for clean three-dot diffs | Upstream `README.md` |
| [`laguna`](https://github.com/TheApathy/apathy-atlas/tree/laguna) | Maintained, measured | Laguna-S-2.1 on GB10, including the unmerged MonumentalSystems enablement lineage and Apathy DFlash work | `bench/laguna/README.md` |
| [`qwen`](https://github.com/TheApathy/apathy-atlas/tree/qwen) | Maintained, measured | Qwen3.6-27B dense DFlash, drafter comparisons, language evals and deterministic reproduction gates | `bench/qwen/README.md` |
| [`ds4-flash`](https://github.com/TheApathy/apathy-atlas/tree/ds4-flash) | Research, measured | Single-GB10 DeepSeek-V4-Flash/REAP bring-up, prefill, decode, MTP and quality evidence | `bench/deepseek-v4/README.md` |
| [`feat/tui-benchmarks`](https://github.com/TheApathy/apathy-atlas/tree/feat/tui-benchmarks) | Experimental | In-process benchmark plugin, TUI benchmark views and run-scoped state refactor | `docs/APATHY_TUI_BENCHMARKS.md` |
| [`feat/qwen38-repro-public`](https://github.com/TheApathy/apathy-atlas/tree/feat/qwen38-repro-public) | Review branch, incomplete release | Public-safe Qwen3.8 coding evidence and endpoint verifier; source and drafter publication are still pending | `docs/reproduction/qwen38.md` |

## What the status labels mean

- **Upstream mirror**: intentionally contains no Apathy-specific product work.
- **Maintained, measured**: has a branch-local harness and checked-in evidence;
  claims still apply only to the documented hardware, model and prompt mix.
- **Research, measured**: contains real measurements, but model or method
  caveats prevent treating it as a general production recommendation.
- **Experimental**: useful implementation work whose public reproduction or
  release qualification is not complete.
- **Review branch, incomplete release**: documentation and sanitized evidence
  are reviewable, but the branch is not yet an immutable one-command release.

## Comparing branches correctly

The model branches forked at different points. Prefer a three-dot diff against
the upstream anchor:

```bash
git fetch origin
git diff --stat origin/base...origin/laguna
git diff --stat origin/base...origin/qwen
git diff --stat origin/base...origin/ds4-flash
```

For exact lineage boundaries and model-specific caveats, read
[`REPRODUCING.md`](REPRODUCING.md) on the branch you intend to run.

## Publication policy

Public documentation and evidence may include commands, hashes, aggregate
metrics and sanitized result records. It must not include credentials, private
prompts, raw generated responses, hostnames, personal home paths, model
weights, process dumps or mutable claims presented as released results.
