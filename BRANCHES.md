# Apathy Atlas branch map

| Branch | Status | Scope | Start here |
|---|---|---|---|
| [`main`](https://github.com/TheApathy/apathy-atlas/tree/main) / [`base`](https://github.com/TheApathy/apathy-atlas/tree/base) | Upstream mirrors | Avarok Atlas; `base` is the stable diff anchor | Upstream `README.md` |
| [`laguna`](https://github.com/TheApathy/apathy-atlas/tree/laguna) | Maintained, measured | Laguna-S-2.1 on GB10 | `bench/laguna/README.md` |
| [`qwen`](https://github.com/TheApathy/apathy-atlas/tree/qwen) | Maintained, measured | Qwen3.6-27B dense DFlash | `bench/qwen/README.md` |
| [`ds4-flash`](https://github.com/TheApathy/apathy-atlas/tree/ds4-flash) | Research, measured | Single-GB10 DeepSeek-V4-Flash/REAP | `bench/deepseek-v4/README.md` |
| [`feat/tui-benchmarks`](https://github.com/TheApathy/apathy-atlas/tree/feat/tui-benchmarks) | Experimental | Benchmark plugin, TUI views and run-scoped state | `docs/APATHY_TUI_BENCHMARKS.md` |
| [`feat/qwen38-repro-public`](https://github.com/TheApathy/apathy-atlas/tree/feat/qwen38-repro-public) | Review branch | Public-safe Qwen3.8 coding evidence; release inputs pending | `docs/reproduction/qwen38.md` |

Use three-dot diffs against `origin/base`; the model branches were measured on
different historical Atlas trees. Public evidence must exclude credentials,
private prompts/responses, personal paths, hostnames, process dumps and model
weights.
