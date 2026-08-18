# Reproducing the measurements in this fork

This is a fork of [Atlas](https://github.com/Avarok-Cybersecurity/atlas). Apathy
Atlas keeps upstream mirrors, measured model branches, and experimental
infrastructure separate. The GitHub default branch is `laguna`; use
[`BRANCHES.md`](BRANCHES.md) rather than assuming the default branch is the
upstream mirror.

Upstream is Avarok-Cybersecurity/atlas. The `laguna` branch additionally builds
on 36 commits from [MonumentalSystems/atlas](https://github.com/MonumentalSystems/atlas)
that were never merged upstream — see [Where every commit came from](#where-every-commit-came-from),
which gives you a one-line diff for each contributor's share.

| Branch | Model | Harness | What it measures |
|---|---|---|---|
| `base` / `main` | — | — | Upstream Atlas, unmodified. Same commit on both; `base` exists as a stable anchor to diff against. |
| `laguna` | poolside/Laguna-S-2.1-NVFP4 | `bench/laguna/` | Single-stream DFlash speculative decode, prefill, capacity, and a 69-case tool-calling eval. |
| `qwen` | Qwen3.6-27B (dense) + DFlash drafter | `bench/qwen/` | The champion single-stream speculative-decode configuration. |
| `ds4-flash` | DeepSeek-V4-Flash/REAP | `bench/deepseek-v4/` | Single-GB10 bring-up, prefill, decode, MTP and quality evidence, with explicit model-quality caveats. |
| `feat/tui-benchmarks` | Atlas infrastructure | in-process plugin/TUI | Experimental benchmark UI and run-scoped state refactor; not a model-performance release. |
| `feat/qwen38-repro-public` | Qwen3.8-27B | `repro/qwen38-coding/` | Public-safe endpoint verifier and sanitized deterministic coding evidence; exact source/drafter publication pending. |

Only the branches listed above are maintained as Apathy Atlas surfaces. Other
remote branches may be upstream branches, historical work or short-lived review
branches; they are not implied release channels.

## Where every commit came from

The two branches do not have the same lineage, and `laguna` has **two**
upstreams rather than one. GitHub's "forked from" header can only ever name a
single parent, so it cannot state this correctly — these three tags do:

| Tag | Commit | What it marks |
|---|---|---|
| `upstream/avarok-laguna` | `f8ff5f78` | The Avarok commit `laguna` forked from |
| `upstream/monu-laguna-2.1` | `93ff1113` | Tip of MonumentalSystems' Laguna-S-2.1 enablement work |
| `upstream/avarok-qwen` | `ddc7080f` | The Avarok commit `qwen` forked from |

`laguna` is three contiguous stages with no interleaving:

```bash
# 36 commits, all by Richard Safier, from MonumentalSystems' unmerged
# feat/laguna-s-2.1 branch -- 124 files, +7,938 / -686
git diff upstream/avarok-laguna..upstream/monu-laguna-2.1

# 44 commits -- ours -- 185 files, +32,899 / -625
git diff upstream/monu-laguna-2.1..laguna
```

Those 36 commits are **load-bearing**: they include `spark-model: add
Laguna-S-2.1 inference support`, so the model does not load without them, and
they were never merged into Avarok. If you want the Laguna-S-2.1 support by
itself, that segment is where it is. This branch carries it so you do not have
to chase an unmerged upstream branch to build.

`qwen` is two stages and involves MonumentalSystems not at all:

```bash
# 294 commits -- ours -- 589 files, +118,579 / -3,251
git diff upstream/avarok-qwen..qwen
```

### If you diff against `base`, use three dots

`base` tracks upstream's *current* tip, which is newer than either fork point.
A two-dot diff against it subtracts that newer tip and reports upstream's own
later commits as though this fork had deleted them:

| | two-dot `base..X` — misleading | three-dot `base...X` — correct |
|---|---|---|
| `laguna` | 557 files, −29,129 lines | **267 files, +40,814 / −1,288** |
| `qwen` | 1,989 files, −283,621 lines, 1,066 "deletions" | **589 files, +118,579 / −3,251** |

The three-dot form resolves to each branch's own fork point, so it is equivalent
to the tag commands above and shows what changed here and nothing else.
`base...laguna` deletes zero files.

Neither branch has been rebased onto `base` to make that diff tidier. These are
the trees the published numbers were actually measured on, and rewriting them
into something neater would produce a history that has never been run.

Both branches target GB10-class hardware (Grace Blackwell, `sm_121f`, unified
LPDDR5x). Nothing in either harness is hardcoded to a particular machine: paths,
ports, and checkpoint locations all come from the environment, and the scripts
refuse to guess rather than silently pick a default that is wrong for you.

```bash
git checkout laguna && cat bench/laguna/README.md   # or
git checkout qwen   && cat bench/qwen/README.md
```

## The two harnesses are siblings, not variants

They are deliberately the same shape — one `env.sh` that owns every path and
setting, one self-verifying launcher, one deterministic decode benchmark — so
that if you can read one you can read the other.

They are **not** re-parameterizations of each other, and settings do not
transfer. The two stacks disagree on γ, on KV dtype, on which log line carries
the accept counter, and on which gates are safe to enable. Each README has a
"things that will cost you a day" section; read the one for the branch you are
actually on.

## What the asserts are for

Every launcher here refuses to report a healthy serve unless its invariants
hold, and each assert exists because its absence once produced a clean-looking
but wrong measurement. The recurring failure mode in this kind of work is not a
crash — it is a serve that comes up healthy while running a configuration nobody
asked for, and then produces plausible numbers about the wrong thing.

Three of those checks are worth calling out because they generalize beyond this
repo:

- **A gate the binary cannot read is a silent no-op.** Setting an `ATLAS_*`
  environment variable proves nothing about whether the binary honors it. Both
  launchers verify every gate they export actually exists as a string in the
  binary before launching.
- **A serve that silently falls back to non-speculative decode is fast enough to
  look plausible.** Every benchmark row is classified as speculative-or-serial
  *before* it is aggregated.
- **A count of zero must never be readable as "nothing wrong".** A row that
  cannot be graded is reported as `UNGRADED`, distinct from a row that was
  graded and found clean.

## Measurements are content-mix dependent

Neither branch has a single headline throughput number, because on both stacks
throughput varies by more than 4× across prompt types — highly predictable
content (counting, repetitive structure) accepts most drafted tokens, and novel
prose accepts very few. Any single number is a statement about a prompt mix.
Both harnesses therefore report per-content-type rows and say what the mix was.
