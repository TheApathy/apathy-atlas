# Reproducing the measurements in this fork

This is a fork of Atlas. The default branch tracks upstream unchanged; the work
lives on per-model branches, and each one carries its own self-contained
reproduction harness under `bench/<model>/`.

| Branch | Model | Harness | What it measures |
|---|---|---|---|
| `base` / `main` | — | — | Upstream Atlas, unmodified. |
| `laguna` | poolside/Laguna-S-2.1-NVFP4 | `bench/laguna/` | Single-stream DFlash speculative decode, prefill, capacity, and a 69-case tool-calling eval. Shares history with `base`, so `base...laguna` is a meaningful diff. |
| `qwen` | Qwen3.6-27B (dense) + DFlash drafter | `bench/qwen/` | The champion single-stream speculative-decode configuration. A standalone snapshot — see the warning below. |

### `qwen` is a snapshot, not a diff

`laguna` branches from `base`. **`qwen` does not.** It descends from a different
upstream lineage, and its tree is older than `base` by 1066 files. Comparing the
two renders as though this fork deleted 399 crates and 375 kernels, which is an
artifact of the two histories, not a change anyone made.

So do not read `base...qwen`. Check `qwen` out and build it: it is the tree the
published Qwen numbers were actually measured on, which is exactly why it has
not been rewritten into something tidier that has never been run.

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
