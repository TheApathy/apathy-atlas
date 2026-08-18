# Apathy Atlas branch map

Model-specific work stays on separate branches so measured trees keep their
real history. Choose a branch before building or comparing results.

| Branch | Status | Scope | Start here |
|---|---|---|---|
| [`main`](https://github.com/TheApathy/apathy-atlas/tree/main) / [`base`](https://github.com/TheApathy/apathy-atlas/tree/base) | Upstream mirrors | Avarok Atlas; `base` is the stable diff anchor | Upstream `README.md` |
| [`laguna`](https://github.com/TheApathy/apathy-atlas/tree/laguna) | Maintained, measured | Laguna-S-2.1 on GB10, MonumentalSystems enablement plus Apathy DFlash work | `bench/laguna/README.md` |
| [`qwen`](https://github.com/TheApathy/apathy-atlas/tree/qwen) | Maintained, measured | Qwen3.6-27B dense DFlash, drafter and language evaluations | `bench/qwen/README.md` |
| [`ds4-flash`](https://github.com/TheApathy/apathy-atlas/tree/ds4-flash) | Research, measured | Single-GB10 DeepSeek-V4-Flash/REAP | `bench/deepseek-v4/README.md` |
| [`feat/tui-benchmarks`](https://github.com/TheApathy/apathy-atlas/tree/feat/tui-benchmarks) | Experimental | In-process benchmark plugin, TUI views and run-scoped state | `docs/APATHY_TUI_BENCHMARKS.md` |
| [`feat/qwen38-repro-public`](https://github.com/TheApathy/apathy-atlas/tree/feat/qwen38-repro-public) | Review branch | Public-safe Qwen3.8 coding evidence; source/drafter publication pending | `docs/reproduction/qwen38.md` |

Use three-dot diffs because branches forked from different upstream points:

```bash
git diff --stat origin/base...origin/laguna
git diff --stat origin/base...origin/qwen
git diff --stat origin/base...origin/ds4-flash
```

Public evidence may contain commands, hashes and sanitized metrics. It must not
contain credentials, private prompts or responses, personal home paths,
hostnames, model weights, process dumps, or mutable claims presented as
released results.
