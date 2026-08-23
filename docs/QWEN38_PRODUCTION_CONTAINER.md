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

### What the pin does and does not prove

**The gate is the measurement, not the digest.** That is a change of position,
and it was forced by evidence rather than chosen.

The original plan was: rebuild from the recorded commit, compare hashes, and let
digest equality prove that the shipped binary is what the source says it is. That
plan does not work here, for a reason worth stating plainly because it is easy to
believe the opposite:

> **This workspace is cache-stable, not reproducible.**

Two builds of the same commit that share `target/` are byte-identical — that was
measured twice, by two operators, 33 minutes apart. It is tempting to read that as
determinism. It is not. A build of the *same commit* from a *clean* `target/`
produces a **different hash**. The two binaries are the same size to the byte and
their only differing strings are a date and a time:

| build | embedded |
|---|---|
| shared `target/` | `Jun  3 2026` / `15:09:53` |
| clean `target/` | `Aug 23 2026` / `22:20:52` |

That is a C-style `__DATE__`/`__TIME__` pair from one of the C-compiling
dependencies (`xgrammar-rs`, `libmimalloc-sys`, `onig_sys`, `esaxx-rs` all have
objects cached since June 3). Every build that reuses `target/` inherits June 3
and therefore agrees with every other such build. Agreement between two builds
sharing a stale artifact is evidence that they shared an artifact, nothing more.

A corollary that applies to everyone working in this tree: our `target/` has been
serving five-month-old C objects to every build any of us has run. Nothing is
wrong with the result, but "it builds clean from our tree" has been a weaker
statement than it sounds, for all of us, all along. A clean build is the only one
that exercises the C dependencies — which is also how the `nvcc` requirement
below surfaced.

#### Correspondence: disproven

Hash is unusable, but **size is not** — the timestamp difference costs exactly
zero bytes, while real source differences cost real bytes. On that instrument:

| binary | size | delta vs pin |
|---|---:|---:|
| production pin | 39,068,536 | — |
| build of `bedfb478` | 39,138,920 | **+70,384** |
| build of tip `320e0060` | 39,138,832 | +70,296 |

Same source across timestamps differs by 0 bytes; different source (`bedfb478`
vs the tip) differs by 88. A 70 KB gap is three orders of magnitude past that
noise floor. **The pinned binary is not what `bedfb478` builds.**

The mechanism is deliberately left unattributed. The leading explanation — and it
is labelled a hypothesis because it has not been established — is that the pin was
built at 19:20:45 from a working tree that kept changing until it was committed
around 20:50, with several agents editing `crates/` in between. That intermediate
tree state no longer exists anywhere, so no rebuild can recover which edits landed
when. The decision this feeds does not depend on the answer.

#### What replaces it

Lineage is expressed as **commit + measured throughput**, and the floor assertion
in `make repro` is the gate. Not an adjunct to a hash check — the gate.

The digest pin stays, with its scope narrowed to what it can actually do: it stops
the wrong file being staged into an image. `build-local.sh`, the Dockerfile, and
`entrypoint.sh` all still refuse a mismatched binary, and all three still check
that `qwen3.8-27b` is present, because kernels are compiled per target and the
wrong one fails late and confusingly. What the digest no longer claims is that the
artifact came from a particular commit.

Chasing byte-for-byte reproducibility (`SOURCE_DATE_EPOCH`, `-D__DATE__`
overrides on the C dependencies) was considered and **rejected**: it is real work
with a maintenance tail across toolchain updates, and it buys a property that has
no consumer now that measurement is the gate.

#### If a from-source build stage is added later

It will produce a different hash on every build, by construction — a clean
`target/` stamps a fresh timestamp every time. Pin the commit and require the
floor; do not try to pin the resulting digest.

The stage needs: `FROM nvidia/cuda:13.0.0-devel-ubuntu24.04`, a copy of
`Cargo.toml Cargo.lock rust-toolchain.toml crates/ kernels/ jinja-templates/`,
`ENV RUSTUP_TOOLCHAIN=stable` (the toml pins 1.85; a transitive dep needs ≥1.88),
the three `ATLAS_TARGET_*` variables, and `cargo build --release -p spark-server`.
Assert on the build's own `compiled 151 kernels for target 0 (gb10, qwen3.8-27b,
nvfp4)` line — it fails at build time, where grepping the binary afterwards fails
later.

Two things it must **not** omit:

- `ENV PATH="/usr/local/cuda/bin:${PATH}"`. `cudarc` 0.19.2's `build.rs:115`
  shells out to a bare `nvcc --version` and panics if it is unresolvable — it does
  **not** consult `CUDA_HOME`. The devel images set this themselves, so a build
  that relies on inheritance works until a base-image bump quietly removes it.
  This is invisible from our tree because the cached `cudarc` build-script output
  predates any of us.
- `COPY vendor/`. There is no `vendor/` directory — the vendored C++ xgrammar tree
  was deliberately deleted when the grammar stack went pure-Rust. That stale line
  was present in all nine `docker/gb10/*` Dockerfiles and has been removed.

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
  -v <internal, not published>/optimized-qwen-unsloth-official:/model:ro \
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

**This is the gate.** The binary digest proves only that the intended file was
staged; it does not prove the artifact came from any particular source (section
1). What links this image to a claim about its behaviour is the measurement
below, and nothing else. A change that cannot clear the floor is a regression
regardless of what any hash says.


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
| Pinned binary is what `bedfb478` builds | **DISPROVEN** | built `bedfb478`: 39,138,920 bytes vs the pin's 39,068,536, a 70,384-byte gap against a 0-byte timestamp noise floor |
| Build is byte-reproducible | **DISPROVEN** | same commit, clean `target/` → different hash; differs only in an embedded `__DATE__`/`__TIME__` pair |
| Build is stable when `target/` is shared | **verified** | two operators, 33 min apart, same `target/` → identical sha256 `8568f485…` |
| A clean build needs `nvcc` on `PATH` | **verified** | clean-`target/` build panicked in `cudarc` 0.19.2 `build.rs:115` on a bare `nvcc --version`; passed once `/usr/local/cuda/bin` was on `PATH` |

The two remaining unverified rows both need exclusive use of the GPU, and the
box has been continuously occupied by a long-running corpus generation since this
image was built. Launching a second 22 GB model server alongside it on unified memory
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

Both are one GPU-free window away from being settled — and since the digest no
longer carries lineage (see section 1), the first of them is now the **primary**
gate on this image rather than a secondary check.

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
