# Qwen3.8-27B production container (GB10)

Status: current as of 2026-08-23.

This page covers packaging and reproduction. The measurement record behind the
recipe — what each knob is worth, and which levers measured null — lives in
[`QWEN38_PERFORMANCE_RECIPE.md`](QWEN38_PERFORMANCE_RECIPE.md); this page does
not restate it. For the general Atlas deployment modes see
[`DEPLOYMENT.md`](DEPLOYMENT.md), and for the GB10 hardware envelope see
[`HARDWARE.md`](HARDWARE.md).

Sources live outside the repo, alongside the weights they package:
`qwen38/container/production-v2/`.

---

## 1. What the image is

One image, one model, one measured recipe, and an end-to-end reproduction that
either clears a throughput floor or fails.

| | |
|---|---|
| Base | `nvidia/cuda:13.0.0-runtime-ubuntu24.04` (digest-pinned) |
| Engine | pinned `spark`, kernel target `qwen3.8-27b`, sha256 `1ce470d6fa2ff301…` |
| Drafter | baked, 3.96 GiB BF16, 69 tensors, `block_size` 16 |
| Target | **mounted**, not baked — `unsloth/Qwen3.8-27B-NVFP4` @ `7d6f8d4d…` |
| Probe | `weschera_minheap_repro.py`, baked |
| Tag | `atlas-qwen38:production-v2` |

The 21.8 GB target stays out of the image on purpose. Its revision, its two
shard sizes, and the sha256 of every metadata file are recorded in
`/opt/atlas/provenance.json` and checked at every container start.

### Why the drafter is baked, and renamed

The drafter is 3.96 GiB — small enough to ship, and pinning it is the entire
point: it is private, it is the thing the throughput number depends on, and a
mounted drafter is a drafter that can silently change.

The copy is at `qwen38/drafter-production-v2`, taken byte-for-byte from
`qwen38/drafter-qwen38-v2-epoch4-step24852`. The source name encodes a training
checkpoint (`epoch4-step24852`), which is a *training* identity, not a release
identity. Pinning production to a training path means the next retrain either
cannot prune its checkpoints or breaks the image build. The stable name breaks
that coupling; `provenance.json` records the source path and the sha256 of every
file, so nothing about the lineage is lost.

### Why the binary is pinned, not built

The tree is now committed, so a from-source build has become *meaningful*. It
has not yet become *justified*, and the binary stays pinned until it is.

The binary that produced the reference number was built at 19:20 from a working
tree that was uncommitted at the time. That tree has since been committed as
`bedfb478` on `qwen38/gb10-perf-and-packaging` (verified: the commit exists, it
is `HEAD`, and the working tree is clean). What has **not** been established is
that this binary is what `bedfb478` produces. Nothing links them but sequence —
the build predates the commit, and no one has rebuilt and compared.

Two signals that look like evidence against correspondence are not, and are
recorded here so they do not get re-litigated:

- 909 files under `crates/` and `kernels/` carry mtimes later than the binary.
  That is a bulk touch, not editing: `crates/atlas-embed/src/token.rs` is among
  them and its last content-changing commit is the original open-source release,
  with a clean `git status` proving the working copy still matches it.
- A `git diff e8b00332 bedfb478` does not hash to the working-tree diff recorded
  before the commit. Expected: the commit range renders the previously-untracked
  files as new-file diffs, and `git diff HEAD` never showed them at all.

So correspondence is **unproven in both directions**, and only one thing settles
it: rebuild in-image, extract the binary, run the MinHeap probe at 400 tokens ×5,
and require the median within drift of 63.96 / 63.77. If it lands low, that is a
finding about the committed tree, not a reason to relax the gate.

Meanwhile `target/release/spark` is rebuilt continuously with *different kernel
targets* — at one point during this work it was `03a1dbb9…`, a different binary
from the pinned `1ce470d6…`. Pinning by path would have shipped whichever build
happened to be on disk.

So the pin is by **content hash**, enforced three times: `build-local.sh`
refuses to stage a mismatched binary, the Dockerfile fails the build on it, and
`entrypoint.sh` refuses to start on it. All three additionally check that the
string `qwen3.8-27b` is present in the binary, because kernels are compiled per
target and a binary built for the wrong one fails late and confusingly.

What a build stage now needs: `FROM nvidia/cuda:13.0.0-devel-ubuntu24.04`, copy
`Cargo.toml Cargo.lock rust-toolchain.toml crates/ kernels/ jinja-templates/`,
`ENV RUSTUP_TOOLCHAIN=stable` (the `rust-toolchain.toml` pin is 1.85 and a
transitive dep needs ≥1.88), the three `ATLAS_TARGET_*` variables, then
`cargo build --release -p spark-server`. It does **not** need `COPY vendor/` —
that line was stale in all nine `docker/gb10/*` Dockerfiles and has been removed;
the vendored C++ xgrammar tree was deliberately deleted when the grammar stack
went pure-Rust.

---

## 2. Build

```bash
qwen38/container/production-v2/build-local.sh
# or
cd qwen38/container/production-v2 && make build
```

The script hashes the binary (39 MB) and the drafter (4.0 GiB, ~20 s) before
staging anything, then hardlinks the drafter into the build context when the
staging directory shares a filesystem with it. Override `SOURCE_BINARY`,
`DRAFTER_DIR`, or `IMAGE_TAG` by environment.

---

## 3. Run

```bash
make serve                    # or the docker run below
```

```bash
docker run --rm --gpus all --ipc=host \
  -v /home/flocka/atlas/qwen38/optimized-qwen-unsloth-official:/model:ro \
  -v atlas-qwen38-weight-cache:/var/cache/atlas-weight-cache \
  -p 127.0.0.1:8896:8896 \
  atlas-qwen38:production-v2 serve
```

Modes: `serve` (default), `repro`, `verify`, or any other argv, which is
`exec`'d verbatim.

Mount the weight cache volume. Without it the 13 GB of post-transform artifacts
are rebuilt on every start and each restart costs minutes instead of ~17 s.

The listener binds `0.0.0.0` *inside the container namespace*; the publish flag
is what decides exposure. `-p 127.0.0.1:8896:8896` keeps it host-local. If you
publish it wider, pass `--require-auth` and a token.

Every knob in the recipe is an environment variable and every one is
overridable — `/opt/atlas/serve-recipe.env` sets them with `: "${VAR:=value}"`,
so anything the caller exports wins.

---

## 4. Reproduce the benchmark

```bash
make repro
```

Starts the server on the baked recipe, waits for health, runs the MinHeap probe
at `--max-tokens 400 --repetitions 5`, and asserts the median clears 60 tok/s.
It re-enters `entrypoint.sh serve` rather than carrying a second copy of the
launch flags, so the reproduced configuration is identical by construction.

Reference, single stream, greedy, thinking off:

| Arm | Median tok/s |
|---|---:|
| `ATLAS_DFLASH_DRAFT_SPLITK=8` | 63.96 / 63.77 |
| split-K unset | 62.86 / 62.92 |
| Historical, same probe | 51.26 |

The 60 tok/s floor sits ~4.5% below the slowest measured arm: enough to absorb
box drift, not enough to hide a regression. The actual median, the per-run
rates, and the determinism verdict are printed on every run, pass or fail — a
run that clears the floor by a hair is still worth reading. Both the floor and
the probe shape are overridable (`REPRO_FLOOR_TOK_S`, `REPRO_MAX_TOKENS`,
`REPRO_REPETITIONS`).

`repro` starts its own server, so nothing else may hold the GPU.

---

## 5. Guards

Three things about this box cost real time when they are rediscovered. Two are
enforced, one is documented because it cannot be.

**`SEQS` is capped at 8.** `SEQS=16` caused a global OOM and a hard host reboot.
On unified memory a GPU over-allocation takes the host down with it — there is
no separate VRAM to exhaust first. The entrypoint refuses anything outside
`1..8` and there is no override. Above 1 it also warns that you are on the
accuracy profile and must not report tok/s.

**`GPU_MEM_UTIL` is capped at 0.55.** It is not a KV-pool knob. Measured: 0.55
gives 16.6 GB allocatable, 0.68 gives 5.7 GB — raising it makes things *worse*.
The entrypoint refuses a raise unless `ATLAS_ALLOW_GPU_MEM_UTIL_RAISE=1` is set,
which exists so that a deliberate, re-measured change is possible and an
absent-minded one is not.

**The kernel build cache does not track `MODEL.toml`.** Nothing in the container
can detect this, because it happens on the host before the binary exists. After
editing `kernels/gb10/qwen3.8-27b/MODEL.toml`, run
`touch crates/atlas-kernels/build.rs` or the rebuild silently keeps the old
behaviour defaults.

One more, benign: the weight cache is keyed on a fingerprint that includes the
transform-affecting environment variables. Changing `ATLAS_FFN_*`,
`ATLAS_SSM_*_TC`, or `ATLAS_LM_HEAD_TC` invalidates it. That costs one slow
start. It never produces wrong output.

---

## 6. Provenance verification

`verify` (and the start of every `serve`) runs
`/opt/atlas/verify_provenance.py` in **fast** mode: it hashes every small
metadata file in full and checks the exact byte size of the two multi-GB shards.
Sub-second, and it catches the failure that actually happens — the wrong
checkpoint mounted.

Fast mode does **not** prove the weight bytes are intact. For that, set
`ATLAS_VERIFY_WEIGHTS=1` to hash the 22.5 GB target shard and the 4.0 GiB
drafter. That takes minutes and is not a sensible default for a service start.

`verify_provenance.py` is driven entirely by the record, so it is
provider-agnostic: a record carrying only a `target` block verifies a bare
checkpoint anywhere, and `--drafter-dir` is required only when the record has a
`drafter` block. The Vast.ai provisioning path reuses it against a
target-only slice of this record — a read-only dependency on one JSON block, not
a merge of the two serving paths.

The two target shard hashes in `provenance.json` were computed locally *and*
cross-checked against the HuggingFace download etags recorded under
`.cache/huggingface/download/*.metadata` for revision `7d6f8d4d`. Two
independent records agree.

---

## 7. Verification status

What has actually been executed, and what has not. An unevidenced claim in a
doc is worse than an absent one, so this section is the authority — if a
statement elsewhere on this page is not backed here, treat it as untested.

| Claim | Status | Evidence |
|---|---|---|
| Image builds; binary digest matches the pin inside the image | **verified** | `build-local.sh` exit 0; `docker run --entrypoint sha256sum` returns `1ce470d6…` |
| Provenance check passes against the real target and drafter | **verified** | `docker run … verify` exit 0 |
| Provenance check *rejects* a wrong mount and a truncated drafter | **verified** | wrong dir → 4 sha mismatches, exit 1; truncated drafter → size mismatch, exit 1 |
| `SEQS` guard refuses out-of-range values | **verified** | `-e SEQS=9` → exit 1 in-container |
| `GPU_MEM_UTIL` guard refuses a raise without the override | **verified** | `0.68` → exit 1; with `ATLAS_ALLOW_GPU_MEM_UTIL_RAISE=1` → warns and proceeds |
| Constructed serve argv matches the measured 63.96 tok/s launch | **verified** | argv and the 62-var `ATLAS_*` set diffed against the recorded run |
| Container resolves `libcuda.so.1` from the host driver | **verified** | `--gpus all` + `ldd`: `libcuda.so.1 => /usr/lib/aarch64-linux-gnu/libcuda.so.1`; in-container `nvidia-smi` reports `NVIDIA GB10, driver 580.126.09` |
| **`make repro` reaches the 60 tok/s floor** | **NOT VERIFIED** | never executed — see below |
| **Weight cache survives a container restart and makes the second start fast** | **NOT VERIFIED** | never executed — see below |
| Pinned binary is what `bedfb478` builds | **NOT VERIFIED** | build predates the commit; only a rebuild + re-measure settles it |

The two unverified rows both need exclusive use of the GPU, and the box has
been continuously occupied by a long-running corpus generation since this image
was built. Launching a second 22 GB model server alongside it on unified memory
is the exact condition that previously caused a global OOM and a hard host
reboot, so neither was attempted.

Specifically:

- The 63.96 / 63.77 and 62.86 / 62.92 reference numbers in section 4 **are**
  measured — they come from the recorded probe runs. What has not been shown is
  that *this image* reproduces them. The floor assertion logic itself was tested
  offline against those recorded probe outputs (63.96 passes a 60 floor; 62.86
  correctly fails a 65 floor), so the arithmetic and the pass/fail wiring are
  sound; the end-to-end path through server startup and health-wait is not.
- The "~17 s warm versus minutes cold" figures come from
  [`weight-cache.md`](weight-cache.md), where they were measured **directly on
  the host**. Whether that survives a container restart on a named Docker volume
  is an extrapolation, not a measurement.

Both are one GPU-free window away from being settled.

---

## 8. Files

```
qwen38/container/production-v2/
  Dockerfile                    runtime image, no build stage (see §1)
  build-local.sh                digest-checks inputs, stages, builds
  entrypoint.sh                 dispatch, guards, serve launch
  repro.sh                      E2E MinHeap reproduction + floor assertion
  serve-recipe.env              the measured knob set, single source of truth
  verify_provenance.py          fast/full provenance verification
  provenance.json               binary, target, drafter, reference numbers
  Makefile                      build / serve / repro / verify / shell
  README.md
qwen38/drafter-production-v2/   the pinned drafter copy
```
