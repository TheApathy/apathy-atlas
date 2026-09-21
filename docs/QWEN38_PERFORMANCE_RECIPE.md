# Qwen3.8-27B on GB10 — production recipe and measurement record

Status: current as of 2026-08-27. Every number here is measured on this box;
estimates are labelled as such. Where a lever was tested and found null, it is
recorded as null — the negative results are the most expensive part of this
document and the most useful.

---

## 1. Headline numbers

Canonical probe: `qwen38/benchmark/weschera_minheap_repro.py` — the "Weschera
MinHeap" prompt, single stream, greedy, thinking off, median of 5.

| Configuration | 400-token probe | 1500-token probe |
|---|---:|---:|
| Campaign start (2026-08) | ~48 | — |
| Historical reference | 51.26 | 41.22 |
| Shipped v2/BF16-KV container | 63.96 / 63.77 | 45.94 |
| Same, split-K off | 62.86 / 62.92 | — |
| Quarantined v3/NVFP4-KV profile (broken prefill-attention ABI) | 72.2518 | — |
| Clean HEAD + matched attention ABI | — | **55.3824** |
| Current corrected tree, new prefill routes disabled | — | **58.0120** |

Two arms are quoted for the v2 row because they were measured interleaved in
one session; the spread is run-to-run drift. The 2026-08-25 v3 samples were
72.3670, 72.4972, 72.2518, 72.2105, and 72.0784 tok/s with common SHA-256
`f51d8358ea2a5c63353ca00a29208ae2cccd3039b070043cad514cc4af9761c4`,
but they are invalid as a model-performance result. The binary embedded an
11-argument `inferspark_prefill{,_64}` PTX while Rust supplied 13 arguments;
the host's zero `query_start` became the kernel's zero `num_q_heads`, making
all 16 chunk-0 attention layers no-op.

A one-variable correction at clean HEAD produced five deterministic
1,500-token Weschera samples of 55.5466, 55.2232, 55.4252, 55.3824, and
55.2369 tok/s (median 55.3824, stable hash `8c459b89...`). The current matched-
ABI tree with all four new prefill routes disabled measured about 58.012 tok/s
and hash `9550504f...`. Those corrected trajectories replace `f51d8358...` as
the only admissible baseline family. NVFP4 KV remains a separate quality
qualification from BF16 KV.

AEON v4 text suite, best full board (greedy + all serving fixes, 2026-08-23):

| Category | Score | Notes |
|---|---:|---|
| Coding | 27/36 | easy 2/2, medium 3/3, hard 5/5, expert 7/8, frontier 9/12, god_mode 1/6 |
| Math | 27/34 | easy/medium 100%, hard 4/5 |
| Reasoning | 23/35 | easy/medium/hard 100% |
| Instruction | 19/34 | easy/medium 100%, hard 4/5 |
| Prose | 17/35 | |
| **Total** | **113/174 (65%)** | reference attested run: 147/174 (84.5%) on an RTX PRO 6000 |

The reference submission (`unsloth/Qwen3.8-27B-NVFP4`, stock vLLM) ran on an
RTX PRO 6000 Blackwell — ~1792 GB/s against GB10's ~273 GB/s. That 6.5x
bandwidth difference explains its 107-150 tok/s decode; the quality gap is a
separate question and is not fully explained by hardware.

---

## 2. The serve recipe

`qwen38/benchmark/serve-qwen38-quality.sh` carries the quality profile;
`arms/atlas-fork.sh` is the benchmark launcher. Speed profile:

```
MODEL_DIR=qwen38/optimized-qwen-unsloth-official     # see §3 for provenance
DRAFT_OVERRIDE=qwen38/drafter-qwen38-v2-epoch4-step24852
GAMMA=15  MTP_VOCAB=96000  ATLAS_DDTREE_MAX_NODES=16
KV_CACHE_DTYPE=bf16  KV_HIGH_PRECISION_LAYERS=0  SEQS=1  MAXLEN=8192
ATLAS_FFN_TC=1  ATLAS_SSM_PROJ_TC=1  ATLAS_LM_HEAD_TC=1  ATLAS_ACCEPT_FAST_ARGMAX=1
ATLAS_PREFILL_PROJ_FAST=0  ATLAS_PREFILL_FFN_FAST=0  ATLAS_DFLASH_FREE_SLOTS=0
ATLAS_SSM_GDN_SEQ_PERSISTENT=1  ATLAS_ATTN_QKV_FUSED=1  ATLAS_SSM_GDN_LAZY=1
ATLAS_DFLASH_DRAFT_SPLITK=8
ATLAS_WEIGHT_CACHE=1
CONFIDENCE_EARLY_STOP=off  SIMHASH_WATCHDOG=off  LOOP_WATCHDOG=off  THINK_LOOP_WATCHDOG=on
TOOL_CALL_PARSER=hermes
```

Quality profile differs in: `MAX_THINKING_BUDGET=57344`,
`ATLAS_MIN_THINKING_TOKENS=2048`, `MAXLEN=65536`,
`KV_HIGH_PRECISION_LAYERS=8`, greedy (no sampling override).

### Experimental 1M-token capacity

The checkpoint is native to 262,144 positions. Atlas now refuses to extend it
silently: a larger `--max-seq-len` also requires an explicitly selected
`--rope-theta-override`, applied before model construction. The startup shape
is:

```bash
spark serve "$MODEL_DIR" \
  --max-seq-len 1048576 \
  --rope-theta-override VALIDATED_CHECKPOINT_SPECIFIC_THETA
```

Retain the rest of the measured v3 serve profile. Do not copy a theta from
another model and call the result 1M support. The override unlocks allocation
and position handling only; 1M positional quality still requires
needle/retrieval and perplexity qualification against the native 262K range.

The DFlash target-hidden accumulator is circular and retains only its local
`ctx_window`. With five 8192-wide BF16 captures and a 4096-token draft window,
the allocation is 320 MiB at both 262K and 1M, instead of scaling to 80 GiB at
1M. Target KV remains the dominant capacity cost and must be measured on the
actual dtype/hardware before serving. This path has CPU/static coverage but no
1M GPU allocation, decode-speed, retrieval, or quality result yet.

### Prefill target and measurement

The single-Spark target is 2,000 effective prompt tok/s, measured as actual
server-reported prompt tokens divided by server TTFT. Because TTFT includes
sequence setup and the first-token boundary, this is stricter and more useful
than quoting an isolated kernel rate. The corresponding maximum TTFT is:

| Actual prompt tokens | TTFT budget at 2,000 tok/s |
|---:|---:|
| 2,048 | 1,024 ms |
| 8,192 | 4,096 ms |
| 32,768 | 16,384 ms |

Run `bench/qwen38-gb10/prefill_decode_repro.py`; it uses unique prompt prefixes
to prevent prefix-cache contamination, fails closed unless the server reports
zero cached prompt tokens, generates at least 32 continuation tokens at every
length, and requires each continuation to match a same-source/binary/kernel/
target/tokenizer no-spec reference. The report also binds supplied SHA-256
identities for those artifacts, the drafter, and effective launch environment,
then enforces the 85 tok/s decode stretch gate with stable output hashes. Its
`--budget-only` mode audits the wall-time targets without a GPU.
For M128/M256 qualification, use its explicit
`--comparison-kind same-mode --runtime-mode no-spec` workflow, then repeat with
`dflash-v3`. The same-mode reference binds the runtime family and, for V3, the
drafter index. Exactly one allowlisted kernel flag may change from `0` to `1`;
the command line and normalized full environment with that flag removed must
remain identical. The candidate must also beat control median and p90 TTFT at
2K and 8K. The ordinary no-spec-to-V3 reference remains a separate comparison
kind and cannot satisfy this implementation A/B.

The launcher defaults to the measured 8K profile. For the full sweep, start it
with `MAX_SEQ_LEN=65536 MAX_PREFILL_TOKENS=8192`; this keeps the documented 8K
chunk size while admitting the 32K row plus continuation headroom. Above the checkpoint-native 262K limit,
the launcher requires `ROPE_THETA_OVERRIDE` and forwards it explicitly rather
than silently extending positions.

The promoted launcher keeps the numerically conservative
`ATLAS_PREFILL_{FFN,PROJ}_FAST=0` routes but now enables their byte-exact
cp.async shadows with `ATLAS_PREFILL_{FFN,PROJ}_PIPE=1`. Historical single-Spark
evidence measured 604-token TTFT near 2.286 seconds after the FFN pipe change,
about 264 effective tok/s; that older short-prompt row is a diagnostic baseline,
not a result for the current source. The published 3,820 tok/s at 8K belongs to
vLLM TP=4 on four RTX PRO 2000 GPUs and is not a one-Spark comparison. A clean
current GB10 run is required before promoting any 2,000 tok/s result.

The default-off `ATLAS_PREFILL_FFN_FLASHINFER=1` production candidate now
replaces only the dense FFN for exact 2,079- and 8,192-row chunks. It retains
Atlas's activation quantization arithmetic and nibble bytes while writing the
activation block scales directly in CUTLASS 128x4 order. Original packed
checkpoint weights are reused; independent gate/up/down physical scale views
and device alpha scalars are admitted at load. Short final chunks retain the
ordinary route. The native library path is explicit and absolute through
`ATLAS_FLASHINFER_SM121_LIB`; every selected tactic is frozen to a measured
zero-workspace row before the model is retained. The path itself is not trusted:
the loader hashes a stable source read against the qualified
`a007a82566ca3d3115c8cc0e73e2bbfc0bd1c76b6342313fba7ee5483e49c020`
identity, copies those bytes into an anonymous memfd, applies and verifies
WRITE/GROW/SHRINK/SEAL, rehashes the sealed copy, and only then loads symbols
through its retained descriptor. Startup preflight also charges the maximum
retained activation arena exactly: 80,216,068 bytes per layer, or 4.78125 GiB
across 64 layers, in addition to 1.00 GiB of scale/alpha caches.

The latest 2026-08-27 strict same-binary, same-mode C=1 result changed exactly
that one enable bit while holding the library path and all other environment,
command-line, binary, kernel, model, tokenizer, context and cache identities
constant. The binary was
`6f06354edb3651486346ee58fa3647d6c940480f79c3f983fd700ea513a784fc`.
Five zero-cache repetitions per bin produced:

| Actual prompt | Control tok/s | FlashInfer FFN tok/s | Median TTFT speedup |
|---:|---:|---:|---:|
| 2,079 | 980.769 | 1,314.085 | 1.33985x |
| 8,223 | 954.987 | 1,050.050 | 1.09954x |
| 32,800 | 703.682 | 752.866 | 1.06989x |

Every prefill continuation and the required 400-token decode probe matched the
control output hashes. Independent review found no remaining P0/P1. Exact
M2079 tactics 4/4/4 and M8192 tactics 2/2/4 each emitted a distinct receipt.
The relative 2K/8K promotion gate passed; the absolute 2,000 tok/s target did
not. This is therefore a qualified FFN improvement, not a claim that FFN alone
closes prefill or changes the corrected 58.012 tok/s speculative-decode control. The
sealed reference/report SHA-256 values are
`cdc49b033c88754f4949f7a798f932eff9ed2860eb0fef42e860a3c085fc9256`
and `fa89ec07b6176123cb744707a1da96034b5723700088a8d91bbdcfa49c73ec2a`.

The next exact projection screen used the same pinned native-library identity
against immutable `optimized-qwen` tensors. All six real-checkpoint rows passed
BF16 byte parity, finite-output, redzone and immutable-input gates. Attention
QGKV tactic 4 measured 3.5425 -> 0.9029 ms at M2079 and 13.0573 -> 8.0888 ms
at M8192. Attention O measured 1.3994 -> 0.4320 ms with tactic 0 at M2079 and
4.7317 -> 1.4968 ms with tactic 2 at M8192. SSM O tactic 2 measured
1.3885 -> 0.4384 ms and 4.7387 -> 1.5408 ms. These are isolated
production-weight results, not production-route or TTFT claims. The passing
harness and library SHA-256 values are
`07b7b7465a28531398e8e9d258a1ee409aa38f6dace6b4a1c0bd623db2bb04a3`
and `a007a82566ca3d3115c8cc0e73e2bbfc0bd1c76b6342313fba7ee5483e49c020`.

SSM QKVZ cannot reuse one immutable checkpoint projection: the active loader
concatenates BF16 QKV/Z and requantizes the merged tensor. Its required
device-resident activation chain has now passed its raw GB10 gate at M2079 and
M8192. Across seven finite adversarial fixtures, packed E2M1 bytes, CUTLASS
128x4 scale bytes, `scale2`, and the combined alpha bits matched the host
reference exactly; padding, redzones, inputs and repeat determinism passed.
NaN and both infinities produced the required nonzero status, zeroed invalid
group, and fresh-context trap. This removes the D2H/synchronization boundary,
but remains default-unrouted until the complete QKVZ GEMM and production
state/output gates pass.

`ATLAS_PREFILL_ATTN_BR128=1` is the default-off NVFP4 HD256 attention
candidate for chunks of at least 2,048 tokens. It combines two BR64 query tiles
under one CTA, so each BC32 K/V page is fetched and E2M1-dequantized once for
128 query rows instead of once per 64 rows. The QK, online-softmax, and PV MMA
order remains the parent order within each half. The lower half stops at its
own parent causal-block ceiling while the upper half continues; running extra
masked operations would not be byte-equivalent. A single K/V shared tile is
aliased sequentially, reducing the required dynamic scratchpad to 95,808 bytes.
Direct SM121 compilation reports 126 registers/thread, a 64-byte stack, zero
spills, and 1,024 static shared bytes, so the total 96,832-byte block footprint
fits below the CC12.x 99 KiB limit. The raw gate is
`nvfp4_paged_attn_br128_microgate`: it requires parent-byte parity, 4-KiB
canaries, immutable Q/cache/table bytes, shuffled pages, tail/offset/window
coverage, and a win at both 2K and 8K before promotion. Eligible missing symbols
fail before launch; a qualified process must emit exactly one
`ENGAGED ATLAS_PREFILL_ATTN_BR128: nvfp4-hd256-br128` marker. The modeled
dequantized-cache traffic reduction is a prioritization signal, not a measured
TTFT result. The marker proves selection for the eligible long-prefill class,
not that deliberately ineligible sub-2K setup chunks used BR128; the strict
qualification sweep is C=1 and times only its 2K/8K/32K target rows. Both
`PREFILL_ATTN_BR128=1` and `ATLAS_PREFILL_ATTN_BR128=1` are accepted by
`serve.sh`; specifying both with different values, an empty value, or anything
other than exact `0`/`1` fails before process startup.

`ATLAS_PREFILL_QKNORM_ROPE=1` is a separate default-off dense-Qwen3.8 C=1
candidate. One 256-thread CTA per token replaces Q/G deinterleave plus Q norm,
four K RMSNorm rows, and the following RoPE launch while preserving the Q and K
parents' different reduction orders and the normalized-BF16 materialization
before rotation. The chunk-0 cache-skip path passes the same position pointer
for T/H/W so it remains byte-equivalent to its scalar-RoPE parent; continuation
chunks pass the actual interleaved T/H/W streams. Exact model identity, TP=1,
Q24/KV4/HD256/rotary64, per-head norms, and the expected parent RoPE symbol are
all required before selection. An eligible incomplete bundle fails before any
legacy Q/K/RoPE stage. A fresh 2K/8K/32K qualification log must contain exactly
one of each path-specific marker:

```text
ENGAGED ATLAS_PREFILL_QKNORM_ROPE: cache-skip-scalar
ENGAGED ATLAS_PREFILL_QKNORM_ROPE: paged-mrope-thw
```

The candidate removes 32 launches across the 16 attention layers and an
idealized 224 MiB/896 MiB of rotated-region traffic at 2K/8K. Those figures are
only a prioritization ceiling; promotion still requires byte/canary parity and
alternating GPU timing wins at both lengths. Direct CUDA-13 SM121 compilation
with `--fmad=false` reports max threads 256, 64 registers/thread, a 32-byte
stack, zero spills, LOCAL0, and 1,024 static shared bytes; the launch adds
24,576 dynamic shared bytes and pins a two-CTA compiler bound. The guarded raw
gate is `qwen38_prefill_qknorm_rope_microgate`. `PREFILL_QKNORM_ROPE=1` and the
full `ATLAS_PREFILL_QKNORM_ROPE=1` spelling are equivalent in `serve.sh`;
empty, conflicting, or non-binary values fail before startup.

The launcher also sets `ATLAS_DFLASH_PREFILL_PIPE=1`. V3 does not execute a
second full prompt prefill: after target prefill it ingests the captured prompt
context. The gate routes V3's large-M FC and per-layer context K/V projections
through the same byte-exact pipeline. FC uses the single-output kernel; each
layer's equal-shaped K/V pair uses the dual-output form so it reads the projected
context once and writes the same two independent BF16 cache buffers in one
launch. Small-M decode projections stay on the measured gamma-specialized
kernels. At V3's full 4,096-position window this avoids one 40 MiB input read
and one launch per layer, or 240 MiB plus six launches over its six layers.
This reduces the first speculative decode cycle, not server TTFT:
Atlas emits the first prefill token before invoking the drafter. TTFT therefore
isolates the target model, while the required 32-token continuation catches
corrupt target KV/recurrent state and the 400-token decode sweep includes and
amortizes V3 context ingestion.

The measured launcher removes all profiling/probe variables before process
startup. This matters because `ATLAS_PROFILE` and `ATLAS_PROFILE_FIRST` are
presence-based: exporting either as `0` still enables device synchronizations.
The production SSM prefill paths also no longer contain their historical
>4K crash-localization drains: an 8K chunk previously forced two barriers in
each of 48 SSM layers, while the phase-separated diagnostic path contained six
per layer. Explicitly gated profiling remains available through a direct server
launch, but it is excluded from throughput evidence.

The launcher also sets `ATLAS_SSM_PREFILL_PACK=1`. In the two-phase recurrent
path, phase 1 previously submitted two whole-chunk device copies plus one
strided Z copy per token. An 8,192-token chunk therefore issued 8,194 copies per
recurrent layer, or 393,312 across 48 layers. The direct route makes conv/L2 QKV
and FP32 gate/beta write into their full-sequence destinations immediately,
removing 130 MiB of copy payload per layer at 8K (6.09 GiB across 48 layers).
Projected Z remains strided, but an ABI-separate exact conv-prefill shadow has
the first 4,096 of its existing 8,192 channel threads copy the matching Z value
inside the same token loop. This removes every extra copy submission rather
than replacing them with another launch. Conv arithmetic/state, QKV, FP32
gate/beta, Z, and recurrence layout remain unchanged. Its effect is not included
in any published rate until the strict Spark sweep passes.

`ATLAS_SSM_PREFILL_CONV_L2=1` is a stronger default-off recurrent-layer
candidate. The ordinary route rounds conv+SiLU to BF16, writes Q/K, then has a
separate 128-thread-per-head L2 kernel reread packed BF16 pairs, reproduce its
two-warp reduction, and rewrite normalized BF16. The fused shadow keeps the
same BF16 boundary and reduction tree in shared memory while advancing the
causal-conv state sequentially. At 8,192 tokens it removes one launch and 128
MiB of Q/K global traffic per recurrent layer, or 6 GiB across 48 recurrent
layers. This is not the existing decode fusion, which normalizes FP32 SiLU with
a different reduction tree. The candidate remains outside the promoted
profile until canonical output hashes and the strict multi-length TTFT sweep
pass on the Spark.

`ATLAS_GDN_PREFILL_GATECACHE=1` is a default-off C=1 WY32 recurrence
candidate. The current 32-token WY correction recomputes every gate prefix and
inter-token gate product in each of 128 V-column threads: 5,456 multiplies per
thread, or 698,368 per CTA. The shadow computes each coefficient once in the
same left-to-right FP32 order and stores it in the unused diagonal/upper
triangle of the existing `smem_kd`; the lower triangle still holds the original
K-dot products. Each of those 496 K-dots now writes the same four warp-shuffle
partials into a 7,936-byte shared slab. A 992-byte packed pair map is built once
while H is staged, so each thread adds only its 3--4 assigned pairs in the
parent's exact `w0+w1+w2+w3` order instead of scanning all 496. Each lane reads
only the K dimension it just stored before the warp shuffle, allowing the first
input-visibility barrier to fold into partial publication. This replaces 1,488
pair-local full-CTA barriers with two chunk-wide barriers while retaining the
same one-CTA-per-SM shared-memory tier. Rolling left-to-right scans preserve
every stored product bit while reducing coefficient formation to 496
multiplies per CTA, a saving of 697,872 per 32-token chunk. For an 8K row that
is 178,655,232 fewer coefficient multiplies per recurrent layer, or
8,575,451,136 across all 48 recurrent layers. The exact host allocation grows
from 86,528 to 95,232 dynamic-shared bytes, below the CC12.x 99 KiB block limit;
direct SM121 compilation uses 127
registers, a 256-byte stack frame,
and zero spills, matching the current parent compile's register/stack tier.
H/K/Q, WY
correction ordering, state updates, output accumulation, and BF16 conversion
remain unchanged in source, but this candidate still requires the same live
hash and TTFT qualification before promotion.

The raw gate lives in the existing `atlas-spark-bench --bench ssm` target. It
uploads identical deterministic Qwen3.8 inputs and initial H to independent
parent/candidate buffers, requires exact final FP32-state and BF16-output bytes,
checks 4 KiB canaries around every mutated allocation, prints deterministic
fingerprints, then reports separate kernel-only rows. Run it only under the GPU
benchmark reservation, first at 32/256 tokens and then 2K/8K/32K:

```bash
ATLAS_TARGET_MODEL=qwen3.8-27b ATLAS_BENCH_SEQ=8192 \
  cargo bench -p atlas-spark-bench --bench ssm -- gdn_wy32_prefill
```

Do not substitute the first CUDA translation of the pinned SGLang c143
algorithm. Its corrected production-layout v2 passed compact-v1 byte equality,
determinism, canaries, finite checks and the declared numerical screen, but was
slower than the current Atlas kernel: 8.3364 versus 7.5262 ms at M2079 and
32.5435 versus 29.9384 ms at M8192. The pinned SGLang Triton reference measured
2.1197 and 8.4502 ms, respectively, so the useful evidence is the remaining
algorithmic ceiling, not the rejected port. Receipt SHA-256 is
`78c2502a8184216154739f1993fe771745e815394a75f1f272190cb33bbd638d`.

`ATLAS_PREFILL_FFN_FUSED_EPILOGUE=1` is a default-off next-stage candidate.
On the byte-exact pipe route it preserves the up GEMM accumulator and BF16
round trip, then applies the same SiLU expression directly into `gate_out`.
This removes the standalone SiLU launch plus the BF16 `up_out` write and read:
544 MiB of activation traffic per layer at 8,192 tokens, or 34 GiB across all
64 dense FFNs. The SM121 kernel and Rust ABI compile without a GPU, but the
flag must remain outside the promoted launcher until canonical Weschera output
hashes and the multi-length TTFT sweep pass on the Spark.

`ATLAS_PREFILL_FFN_DUAL_FUSED=1` is a stronger default-off candidate. It
accumulates gate and up independently with the same original-layout pipe math,
rounds each FP32 result through BF16 before applying SiLU, and writes only the
fused activation. At 8,192 tokens this removes 1,088 MiB of intermediate
activation traffic per layer (68 GiB across 64 FFNs), saves one 80 MiB input
read per layer, and removes two launches. SM121 compilation reports 186
registers in its historical dedicated form. The generalized dual-output form
now reports 194 registers, 23,360 bytes shared memory, and zero spills; both
resource counts remain in the same occupancy tier. The higher register pressure
makes this a measured candidate rather than a promoted default; it
must beat both the baseline and up-only fusion while preserving the canonical
Weschera hash.

The measured launcher explicitly requests the parent pipe. On any eligible
shape, `ATLAS_PREFILL_FFN_PIPE=1` now errors before the gate projection if its
symbol is absent. The two fusion flags use a pure disabled/ineligible/complete/
missing selector and likewise reject an incomplete parent/candidate pair;
dual fusion remains authoritative when both are set. Distinct one-time
`ENGAGED` lines identify ordinary pipe, up-only fusion, or dual fusion, so a
throughput row without the expected route proof is invalid.

The same fail-closed rule applies to `ATLAS_PREFILL_PROJ_PIPE=1`: after the
transposed-fast, dense, and alternate-quantization routes have retained their
precedence, every eligible original-layout attention Q/K/V/O or SSM QKVZ/out
projection requires the shared pipe symbol before launching. Distinct one-time
`ENGAGED ATLAS_PREFILL_PROJ_PIPE` lines identify the selected projection family;
missing proof invalidates attribution even when output bytes happen to match the
baseline.

The v5 qualification harness makes that rule executable. Each provenance
manifest declares every expected engagement-message fragment and exact positive
count, the canonical complete `ATLAS_*` environment, a privacy-preserving hash
of the full process environment, and a unique run nonce. `--route-log` must name
a fresh regular file directly owned by both server stdout and stderr, never a
`tee` pipe. The harness resolves the exact 127.0.0.1 listener through `/proc`
and rechecks socket, PID/start time, executable, command line, environments,
log inode, and nonce around model discovery and every request. Same-mode A/B
permits exactly one known kernel flag changing `0` to `1`, requires every other
environment entry and the command line to remain identical, and requires strict
median plus p90 TTFT wins at 2K and 8K. The reference/result retains hashes and
counts rather than raw logs or full potentially secret-bearing environments.
The no-spec and V3 wrappers source one target profile. A target-vs-DFlash row
must have identical `ATLAS_*` maps, identical normalized command lines after
removing only the canonical DFlash tuple, and identical normalized full
environments after removing only nonce, `DRAFT`, and `RUNTIME_MODE`; every
other target, CUDA, context, memory, or thread difference is a hard failure.

`ATLAS_PREFILL_KV_DUAL=1` is another default-off use of the generalized kernel.
Original-layout NVFP4 attention layers project K and V together into their
existing independent BF16 outputs in both chunk-0 cache-skip and later paged
paths. At 8,192 tokens the static ceiling is one 80 MiB normalized-input read
and one launch removed per attention layer: about 1.25 GiB plus 16 launches
over the model. Other
quantization, transposed layouts, M<=32, and incompatible shapes keep their
existing paths; an explicitly eligible request fails closed if the kernel
symbol is stale. A large aligned explicit request also rejects transposed-fast
or incompatible/non-NVFP4 K/V before Q rather than partially running or silently
measuring separate projections. A fresh candidate log must contain exactly one
`ENGAGED ATLAS_PREFILL_KV_DUAL: attention_kv_cache_skip` and exactly one
`ENGAGED ATLAS_PREFILL_KV_DUAL: attention_kv_paged`; the markers are emitted
only after their respective launch succeeds. These replace the ordinary
`attention_k`/`attention_v` projection-pipe markers for dual-eligible chunks.
Freeze exact tokenizer/chunker row sizes before launch and retain an ordinary
K/V marker only if a real terminal chunk has M<=32; v5 rejects both missing
declared markers and undeclared observed ENGAGED lines. This candidate also
requires the canonical output hash and strict 2K/8K/32K TTFT sweep before
promotion. The first live gate is
`w4a16_attention_kv_dual_microgate`: it compares both production parent GEMMs
against the one-launch dual route at M=33/63/64/65/127/128/129/2048/8192 plus
cancellation-sensitive rows, with distinct payload sentinels, 4-KiB redzones,
and immutable A/weight/scale checks. Only after every byte/canary case passes
may `ATLAS_PREFILL_KV_DUAL_MICROGATE_TIMING=1` run alternating parent/dual
timing at 2K and 8K; either non-win stops promotion before the server sweep.

`ATLAS_PREFILL_ATTN_GATE_FUSED=1` is an independent default-off chunk-0
candidate for the exact dense Qwen3.8 C=1 Q24/KV4/HD256/stride12288 route. It
uses an ABI-separate BR64 shadow whose only semantic change is the final store:
attention is explicitly rounded to BF16 and widened before the following
parent expression `1/(1+expf(-g))`, multiply, and BF16 conversion. The parent
BR64 body remained unchanged by source parameterization, but the first live
raw gate exposed an older host/device ABI defect from `3d260b4f`: Rust passed
the query-range pair `(0,seq_len)` while the HD256 CUDA symbols did not accept
it, so the parent saw zero Q heads and wrote nothing. HD256 now uses the same
range-aware ABI already established by HD128, and ordered source tests pin both
parent and fused launchers. Runtime driver attributes on GB10 report identical
resources for the corrected parent and candidate: max 512 threads, 126
registers, 90,112 shared bytes, and zero local bytes.

Across 16 attention layers, eliminating the intermediate attention write plus
gate read has a static ceiling of 768 MiB at 2K or 3.0 GiB at 8K and removes 16
launches. An explicit eligible stale bundle fails before Q/K/V projection or
KV-cache mutation. Successful launch emits exactly one process-global marker:

```text
ENGAGED ATLAS_PREFILL_ATTN_GATE_FUSED: cache-skip-br64
```

The candidate manifest adds that marker without removing any promoted marker
and declares exactly
`{"ATLAS_PREFILL_ATTN_GATE_FUSED":{"control":"0","candidate":"1"}}` as its
same-mode environment delta. The ordered 2K/8K/32K sweep emits the marker on
the first 2K request; 2K and 8K each exercise fused chunk 0, while 32K uses the
fusion only for its first chunk and retains paged parents for continuation.
Never label the entire 32K route fused.

First run `qwen38_prefill_attn_gate_microgate` with
`ATLAS_PREFILL_ATTN_GATE_MICROGATE_FULL=1`. It compares parent attention plus
parent gate with the fused symbol at M=63/64/65/127/128/129/2048/8192, global
and production-window masks, ordinary/cancellation/extreme gates, distinct
unwritten sentinels, 4-KiB redzones, and immutable Q/K/V/gate inputs. Only after
every BF16 byte and canary passes may
`ATLAS_PREFILL_ATTN_GATE_MICROGATE_TIMING=1` require alternating median and p90
wins at 2K and 8K. The raw gate replays parity from distinct output fills in
opposite launch order, uses three balanced warmups plus 21 timing pairs,
measures each pair with same-stream CUDA events over eight consecutive launches
and normalizes to per-layer time,
requires a positive paired-median saving after a `3*MAD` noise bound, and
requires modeled 16-layer savings of at least 0.5 ms at 2K and 2.0 ms at 8K.
It also requires the exact embedded `(sm_121,qwen3.8-27b,nvfp4)` bundle and a
runtime 48-SM GB10 at compute capability 12.1, recording the target/device,
case count, and matrix checksum. A default smoke run is explicitly labeled
`NOT QUALIFIED`; only full parity plus timing emits the promotion-ready raw
verdict. Two consecutive runs of binary SHA-256
`d42a9a827b9653a9f9c6dd661381506685cef79a88c8c7ee3161644be84f7508`
passed all 10 cases. Their normalized 2K parent/fused medians were
1.892/1.764 and 1.897/1.780 ms; 8K medians were 24.899/24.088 and
24.553/23.909 ms. This qualifies the raw kernel segment only, not end-to-end
TTFT or the 2,000 prompt tok/s target.
Then use fresh `PREFILL_ATTN_GATE_FUSED=0` and `=1` DFlash
V3 processes under the v5 same-mode harness. Any raw mismatch, timing non-win,
route-proof failure, output-hash drift, decode regression, or 2K/8K relative
TTFT non-win rejects promotion.

VeloGB10 v0.4.0's real long-prefill GEMM is an MXFP4/W4A4 OMMA path. Its
standalone NVFP4 GEMM prototype is explicitly abandoned because it converts
E4M3 scales as integers and is numerically wrong, so it is not a safe Atlas
port. Atlas already has a native block-scaled W4A4 kernel behind
`ATLAS_E2M1_GEMM=1`, but quantizing activations changes the numerical contract
relative to the promoted W4A16 pipe. The full route now prepares gate/up's
shared BF16 input once: at 8K it avoids a second 80 MiB absmax read, an 80 MiB
quantizer read, a 22.5 MiB packed/scales write, one D2H readback, one stream
synchronization, and one quantization launch per layer. Across 64 FFNs that is
about 11.4 GiB of activation traffic plus 64 host synchronization/readbacks
removed from the prior W4A4 implementation. `ATLAS_E2M1_GEMM_DOWN_ONLY=1`
keeps gate/up on W4A16 and tests only the historically favorable down shape.
Qualification order is control, exact fused candidates, down-only W4A4, then
full W4A4; the approximate candidates require output-quality checks and a new
reference rather than the byte-identical Weschera hash gate.

Eligible base W4A4 requests are now fail-closed. Full W4A4 is authoritative
when both scope flags are set; it never downgrades to a ready down-only route.
Before the gate projection, selection requires the actual row-major or K-major
M128/M256 runtime, scale source, optional fused-SwiGLU quantizer, and—in
down-only mode—the transformed W4A16 M128 gate/up path. Each selected M/shape
emits a keyed `ENGAGED ATLAS_E2M1_PREFILL` line containing actual scope,
gate/up route, W4A4 kernel, scale mode, down-input mode, activation, and
dimensions. Requested/present booleans are diagnostics only and cannot satisfy
the v5 route-evidence gate.

`darkdatter/gb10-repo@2db1ecda` independently reports 2,173 and 2,215 prompt
tok/s around 9.7K tokens on two GB10 systems with the pinned SGLang stack. That
is useful feasibility evidence, not an Atlas result or an isolated kernel A/B.
The pinned SGLang ModelOpt route reads `.input_scale` during model loading and
passes its reciprocal to device-side FP4 quantization; it does not perform a
request-time global absmax or host readback. Atlas's default-off
`ATLAS_E2M1_STATIC_SCALE=1` now mirrors that scale contract with the existing
quantizer and native OMMA kernel: each layer validates gate/up/down scales once
at construction, uses `max(gate,up)` for the shared merged input, then removes
both remaining runtime absmax scans and stream synchronizations from full
W4A4—or the single remaining scan/sync from down-only W4A4. It requires an
active W4A4 flag and valid ModelOpt scale metadata. Because fixed calibration
changes activation bytes relative to dynamic absmax, qualify it after the
dynamic W4A4 rows and require model-quality rather than byte-equality alone.

The original Atlas native W4A4 kernel is not an adequate gate/up endpoint: its
production-shape audit measured the checkpoint-row-major M64 implementation
about 2.2x slower than W4A16 because each output-column CTA fetches a disjoint
weight span. `ATLAS_E2M1_KMAJOR=1` selects an ABI-separate M128 shadow over the
already-retained `[K/2,N]` packed and `[K/16,N]` scale transforms. Eight warps
own 16 rows each, so one coalesced staged weight tile serves 128 prompt rows.
Its K-step, block-scaled OMMA, FP32 accumulation, global scale and BF16
epilogue order match the M64 parent; only global weight addressing changes.
Both its cp.async staging and compute-form B tiles are double-buffered, so the
next transform cannot overwrite the current tile while the eight compute
warps still consume it. The K loop retains one final full-CTA
visibility/lifetime barrier instead of a pre-transform and post-transform
pair.
It requires one of the W4A4 arms and fails closed on a stale symbol, missing
transforms, or incompatible projection geometry. Live qualification must show
both W4A4 output equivalence and a production-shape timing win before enabling
the later fused SwiGLU-to-down-quant boundary.

`ATLAS_E2M1_KMAJOR_M256=1` is a further default-off long-prefill shadow. One
512-thread CTA keeps sixteen one-fragment warps resident, the same warp count
as two register-limited M128 CTAs, but stages/transposes each K-major weight
tile once for 256 rows. At each production projection the transformed weight
tile is 47.8125 MiB; executed B traffic falls from about 0.747 to 0.374 GiB at
M=2K, 2.988 to 1.494 GiB at 8K, and 11.953 to 5.977 GiB at 32K. A and output
traffic do not fall, so the modeled total traffic reduction is about 24% and
actual DRAM savings may be lower with L2 reuse. The first conservative route
starts at M=2048, where gate/up still expose 1,088 CTAs and down 320; shorter
rows retain M128. Its compute-form B tile is double-buffered so transpose of
the next K64 tile does not overwrite the current tile; this collapses the
source-level wait/sync/transpose/sync sequence to wait/transpose/sync and
halves full-CTA barrier sites across the K loop. Promotion requires <=128
compiled registers, zero spills,
bitwise M128 parity for full and tail tiles, and a win on every selected
projection shape before the end-to-end 2K/8K/32K gate.

The raw CUDA microgate is `atlas-spark-bench --bench nvfp4_gemm`. It now feeds
the row-major M64, K-major M128, and K-major M256 symbols identical packed
activations and scales plus byte-equivalent weight layouts and `scale2_ab`;
requires all three BF16 output vectors to match bit-for-bit; and surrounds
each output with 4 KiB canaries before reporting separate Criterion rows. Run
it only under the benchmark GPU reservation. Start with
`M=256,257,511`, `K=64,128,256`, `N=128,256`, then run both Qwen projection
shapes at every selected long-prefill M. A tolerance/cosine pass is not enough
for this implementation-only comparison.

```bash
ATLAS_TARGET_MODEL=qwen3.8-27b \
ATLAS_BENCH_M=2048 ATLAS_BENCH_K=5120 ATLAS_BENCH_N=17408 \
cargo bench -p atlas-spark-bench --bench nvfp4_gemm

ATLAS_TARGET_MODEL=qwen3.8-27b \
ATLAS_BENCH_M=2048 ATLAS_BENCH_K=17408 ATLAS_BENCH_N=5120 \
cargo bench -p atlas-spark-bench --bench nvfp4_gemm
```

`ATLAS_E2M1_SILU_QUANT=1` implements that later boundary without changing the
promoted profile. Its kernel reads the BF16 gate/up projection outputs,
computes the same SiLU expression as `moe_silu_mul`, rounds the result through
BF16 in registers, and feeds those rounded values into the existing E4M3-scale
and E2M1-nibble algorithm. The selected W4A4 down GEMM therefore sees the same
activation-precision boundary as the standalone path while the intermediate
BF16 matrix is never written and reread from global memory. It requires static
checkpoint scales so no fused global-absmax protocol is implied. This is a
compiled, default-off candidate only; qualify it after the dynamic/static and
K-major W4A4 rows with quality, packed-activation parity, and 2K/8K/32K TTFT.
This is not a port of SGLang's CUTE fused kernel: the pinned implementation is
an SM100/tcgen05 DeepSeek shared-expert path that rejects non-major-10 compute
capability and is not selected for dense Qwen. Its semantics are useful review
input, but neither its timings nor the external 2.17K prompt result validate
this Atlas SM121 candidate.

### Why each non-obvious flag is set

| Flag | Effect | Evidence |
|---|---|---|
| `GAMMA=15` | draft width | Optimal **and** maximal: the drafter's `block_size=16` gives `trained_drafts = block_size-1`. gamma 20 is refused by the loader. Measured 10/12/15 = 56.71/59.30/62.92 — monotonic into the ceiling. |
| `ATLAS_SSM_GDN_LAZY=1` | skips 15-of-16 discarded snapshot writes | +5.16 tok/s, bit-exact (replays the identical FP32 recurrence on commit) |
| `ATLAS_SSM_GDN_SEQ_PERSISTENT=1` | keeps H in registers across the block | +1.70 tok/s. Together with lazy commit this cut `ssm_gdn_fp32_seq` from 16.9 to 4.9 ms/step |
| `ATLAS_ATTN_QKV_FUSED=1` | fuses QKV projection | +0.79 tok/s |
| `ATLAS_DFLASH_DRAFT_SPLITK=8` | splits K on occupancy-starved drafter GEMMs | +1.13 tok/s (64.07 vs 62.94). Reassociates the K loop so bit-exactness is unproven; measured byte-identical on the probe |
| `ATLAS_WEIGHT_CACHE=1` | caches post-transform weights | Weight-load phase 17 s vs ~45-60 s. See `docs/weight-cache.md` |
| watchdogs mostly off | see §6 | each was measured killing healthy output |

---

## 3. Weights, drafter, and cache

**Target.** `unsloth/Qwen3.8-27B-NVFP4` @ `7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108`.
Byte-verified against upstream: `model.safetensors` sha256 `c473512c70eace07…`,
`model_mtp.safetensors` `1d8268aa85ace093…`, 2 shards, 21.8 GB. Upstream
super-squashed the repo on 2026-08-15, which is why older revision hashes
(including the one the reference submission cites) now 404.

**Drafter.** `drafter-qwen38-v2-epoch4-step24852`, DFlash family, 69 tensors,
3.96 GiB BF16, `block_size: 16`. The engine quantises 6 layers x 7 dense + fc
to NVFP4 at load; the BF16 sources are retained (~3.3 GB held) because
`gpu.free()` on GB10 UVM posts in-band TLB invalidations that corrupt
neighbouring allocations (BUG #29). That is memory cost, not bandwidth.

Alternatives, both measured worse: the official `incoai` DFlash2 drafter runs
52% acceptance against v2's 66% and lands ~7 tok/s behind; DSpark checkpoints
measured 23.4 and 9.2 tok/s in separate audits.

**Weight cache.** ~13 GB per model variant, LRU-bounded to 32 GB, keyed on a
fingerprint including transform-affecting env vars and weight content samples.
704 slot-verifications with 0 failures. Full contract in `docs/weight-cache.md`.

---

## 4. Where the cycle goes

Cycle census (`ATLAS_DFLASH_SPEC_CYCLE_V2=1`, CUDA graphs on, 765 records,
gamma 15, 1500-token probe):

```
verify_complete   106.56 ms   80.5%
propose_complete   22.75 ms   17.2%
accept              1.84 ms    1.4%
TOTAL             132.42 ms
emitted/cycle       5.87        (accepted 4.87 of 15)
```

Verify cost is linear in draft width. Three-point fit (gamma 15/11/7):

```
verify(k) = 75.53 + 1.890k ms      residuals -0.20 / +0.41 / -0.20
```

The intercept is the weight sweep: 16.327 GB at 232 GB/s achievable = **70.37 ms**,
so verify runs at ~1.5x its floor. The slope was **2.996 ms/node** before
persistent-H and lazy commit landed; those changes cut it 1.59x.

**Acceptance falls with generation length** — the single most important
quality-of-drafter signal we have:

| probe length | accepted / 15 |
|---|---|
| 400 tokens | 7.23 (48.2%) |
| 800 tokens | 5.81 (38.7%) |
| 1500 tokens | 4.87 (32.5%) |

Per-position conditional match rate is **flat** at ~0.87 across 10,999 cycles
(position 1 = 0.88, position 12 = 0.88), and a constant-hazard model with a
single p=0.87 predicts E[L]=5.43 against a measured 5.42. The drafter's errors
are uniformly distributed, i.e. semantic difficulty — not local incoherence.

### What valid 72, 80, and 85 tok/s require

The old 200-cycle `72.2518` census is useful only for locating expensive code;
its delivered-token and acceptance arithmetic belongs to the broken no-op-
attention trajectory and cannot set promotion budgets. Against the current
corrected-tree control near 58.012 tok/s, valid 72 requires 24.1% more delivered
throughput, 80 requires 37.9%, and 85 requires 46.5%. Those gains may come from
shorter corrected-target verification cycles, higher DFlash acceptance after
real prompt attention, or both. A new corrected-trajectory cycle census is
required before translating them into millisecond budgets.

Promotion still requires five canonical Weschera runs at or above the stated
target, deterministic hashes for the corrected target trajectory, exact
drafter/target/binary identities, and no co-tenant contamination. The old
`f51d8358...` hash must never be accepted as the control.

The first exact verifier optimization tested on the corrected tree is the
default-off `ATLAS_ATTN_QKV_EXACT_M17_ASTAGE=1` route (launcher alias
`ATTN_QKV_M17_ASTAGE`). Its raw GPU gate passed 190/190 cases byte-for-byte
against both the parent and serial-K1 oracle. A fresh balanced isolated gate
reduced the 16-layer attention-QKV term by 2.05 ms/step (12.03 to 9.98 ms).
The same corrected binary then moved Weschera from roughly 58.15 to 59.46
tok/s with the same output hash, a 2.25% gain. This is real but far smaller
than the remaining 72/80/85 gap.

Prefill-only FlashInfer FFN, projection, and GDN work cannot raise decode
throughput. Those routes reduce prompt latency; the decode target is a deeper
verifier-family saving and a drafter aligned to the corrected attention
trajectory so more delivered tokens amortize each verification cycle.

---

## 5. Numerics: what is bit-exact and what is not

The engine's stated contract was once "verify commits the scalar-oracle token".
That was deliberately retired on 2026-08-17 (REFREEZE block,
`benchmark/arms/atlas-fork.sh:54-62`). Current measurements are relative to
reference hash `12e0c0ad`, not to the scalar oracle.

| Change | Bit-exact? |
|---|---|
| GDN lazy commit, persistent-H, QKV fuse | **yes** |
| Weight cache | **yes** (704 verifications, 0 failures) |
| `ATLAS_FFN_TC`, `ATLAS_SSM_PROJ_TC`, `ATLAS_LM_HEAD_TC` | **no** — MMA reduction order + BF16 weight rounding |
| split-K (`ATLAS_FFN_DOWN_SPLITK`, `ATLAS_DFLASH_DRAFT_SPLITK`) | **no** — K-loop reassociation |

"Lossless" in older comments meant FP32 partials with no mid-accumulation BF16
rounding. That is true and is a different claim from token-exactness. Witness:
for `[2^24, 1, 1, -2^24]`, left-to-right FP32 gives 0 and a 2+2 split gives 1.

The bit-exact path remains available via `ATLAS_FFN_TC=0 ATLAS_SSM_PROJ_TC=0`
(hash `f376a16e`) and costs 27.2 vs 31.2 tok/s at gamma 6.

---

## 6. Gotchas that have cost real time

1. **`SEQS>8` hard-reboots the host.** Unified memory; the launcher refuses it.
   A `SEQS=16` corpus run caused a global OOM and took the machine down.
2. **`GPU_MEM_UTIL` up = KV pool down.** Measured: 0.55 → 16.6 GB allocatable
   (9338 blocks); 0.68 → 5.7 GB (3960 blocks). Do not "tune" it upward to get
   more batch — you get less.
3. **Kernel builds need the target env.** `ATLAS_TARGET_MODEL=qwen3.8-27b
   ATLAS_TARGET_QUANT=nvfp4 cargo build --release -p spark-server`. Without it
   the build silently targets `qwen3-next-80b-a3b` and the server will not serve.
4. **MODEL.toml is not tracked by the kernel build cache.** `touch
   crates/atlas-kernels/build.rs` after editing it or your change is a no-op
   with an unchanged binary hash.
5. **Model-level vs quant-level MODEL.toml.** `kernels/gb10/qwen3.8-27b/MODEL.toml`
   is read; `.../nvfp4/MODEL.toml` is not.
6. **`target/release/spark` is rebuilt with different kernel targets.** Pin the
   binary into `qwen38/benchmark/bin/` and grep it for the target string before
   trusting a measurement.
7. **`ATLAS_FULL_PROFILE` disables CUDA graphs.** It is still representative
   here (graphs are worth only ~2.5 ms of a 106 ms verify), but do not compare
   profiled absolute times against census times without knowing that.
8. **Dispatch shadowing.** `ATLAS_ATTN_QKV_BATCHED`, `ATLAS_ATTN_QKV_SPLITK`
   are inert on gated Qwen3.8 at the widths we serve — `exact_attention_qkv_route`
   returns early for n=4..17. `ATLAS_FFN_DOWN_SPLITK` is live only because
   `ATLAS_FFN_TC=1` forces the exact route off. Before trusting any flag, find
   the function that reads it and walk up to the first early return.
9. **Client-side kills do not cancel server-side generation.** Killing a corpus
   generator leaves its in-flight requests running; a subsequent 5-token probe
   measured 87 s while queued behind them.

---

## 7. Levers measured and found null

Recorded so they are not re-derived. Each cost real GPU time.

| Lever | Result |
|---|---|
| rt2 register-tiled GEMV (upstream PR 648 port) | +0.6%, inside drift |
| Exact-GEMV route instead of TC tiles | **2.5x slower** (24.98 vs 62.84) |
| `ATLAS_TC_NVFP4_M16` / `_MS_ATTN` on QKV | −30% / −27%; both together −36%. Independently reproduces a 2026-08-19 result (51.48 → 39.73) that was never written up |
| Kernel-launch batching in attention | zero — CUDA graphs already absorb it |
| k16 vs k8 load width | 0.61% of the instruction stream |
| Wave/tail utilisation | ~3% of the layer |
| CTA supply in attention | 3072/512 CTAs, 11x past saturation |
| Register pressure | attention 72 regs vs lm_head 76 — lm_head is worse and 3.6x faster |
| DDTree wide trees (old slope) | needed +45% acceptance at 31 nodes |
| PCTree (arXiv 2608.02123) | our DFlash checkpoint has no Markov head; and at b=1.890 the break-even needs +26.9% against a published +18.6% |
| Markov fixup on the chain | per-position hazard is flat; a perfect head caps at +9.2% |
| FFN dispatch variants | null, inside baseline drift |

**Unresolved:** `attn_qkv_proj` runs at ~55 GB/s against 313 GB/s that `lm_head`
achieves on the same machine with worse register pressure. Every mechanism we
can compute has been eliminated. The remaining hypothesis is that the true
floor is L2-bound on a 15.1x activation re-read (~4.0 ms, not the 2.85 ms
weight-bytes figure). `crates/spark-model/examples/w4a16_attention_qkv_throughput_probe.rs`
settles it in one GPU-minute and **has not been run**.

The default-off `ATLAS_ATTN_QKV_EXACT_M17_ASTAGE=1` candidate now tests that
hypothesis directly. For each 512-column wave it stages the M17 raw-BF16 input
once per CTA, then preserves the parent's `k8=lane+64*wave`, low/high FMA,
shuffle, cross-warp-add, and BF16-store order for all four output groups. At
production M=17 its logical activation requests fall from 2.139 GB to 0.535 GB
for gated Q and from 0.357 GB to 0.089 GB for dual K/V per full-attention
layer; these are source-level requests, not measured L2 transactions. CUDA 13
SM121 compilation reports 48 registers for QG, 44 for dual K/V, 18,016 bytes
shared, and zero spills, versus 72 registers/608 bytes shared for both parents.
The candidate adds 20 full-CTA barriers at K=5120, so resource shape is not a
speed claim. Before enabling it in a measured profile, run
`w4a16_exact_attention_m17_astage_microgate` and require all production,
strided, N-tail, partial-wave, canary, and immutable-input cases to pass; then
run the throughput probe's alternating A/B/B/A parent/candidate schedule.

`ATLAS_ATTN_GATE_BATCHED=1` is the launch-only companion candidate. For gated
M=5..17 attention it calls the existing `sigmoid_gate_mul_batched` once per
layer instead of once per row. The flattened kernel reconstructs the identical
token/channel addresses and retains BF16-to-FP32 conversion, ordered sigmoid,
FP32 multiply, and one BF16 store per independent element. Across 16 attention
layers this changes 80 launches to 16 at M=5 and 272 to 16 at M=17, saving
64–256 launches without reducing traffic or `expf` work. It remains default off
until scalar-versus-batched byte parity and alternating Weschera timing pass;
the v5 route manifest must then require
`ENGAGED ATLAS_ATTN_GATE_BATCHED: multi_seq` exactly once.

---

## 8. What remains for 70+ tok/s

Arithmetic, from §4: `tok/s = emitted_per_cycle / cycle_seconds`. Every kernel
term is at or near its floor. The open lever is acceptance:

- today: p ≈ 0.87 per token, 6.81 emitted/cycle, 62.9 tok/s
- for 70: p ≈ 0.92, ~9.2 emitted/cycle
- champion-class drafter, measured on its own target: p = 0.956

Acceptance is the open lever, and it is a property of the drafter rather than
the engine — every kernel term above is at or near its bandwidth floor, so no
serving-side knob moves it. Drafter work is out of tree.

---

## 9. Reproducing

```bash
cd /path/to/apathy-atlas

# build. Both target vars are mandatory, and nvcc must be on PATH: cudarc's
# build script shells out to a bare `nvcc` and does not consult CUDA_HOME.
export PATH=/usr/local/cuda/bin:$PATH
ATLAS_TARGET_MODEL=qwen3.8-27b ATLAS_TARGET_QUANT=nvfp4 \
  cargo build --release -p spark-server

# serve (speed profile)
MODEL_DIR=/path/to/Qwen3.8-27B-NVFP4 DRAFT=/path/to/dflash-drafter \
  ./bench/qwen38-gb10/serve.sh

# measure
python3 bench/qwen38-gb10/weschera_minheap_repro.py \
  --output /tmp/minheap.json --repetitions 5 --max-tokens 400
```

`arms/atlas-fork.sh`, referenced elsewhere in this document, is a working file
on the measurement box and does not ship. `bench/qwen38-gb10/serve.sh` is the
published equivalent.

Interleave A/B arms rather than running them sequentially — sequential arms on
this box manufacture phantom deltas. Baseline drift is ~±1 tok/s; treat
anything smaller as noise.

### Packaged reproduction

The same recipe, the same probe, and the same floor are packaged as a container
in `qwen38/container/production-v2/` — pinned binary, baked drafter, mounted
target, `make repro`. See
[`QWEN38_PRODUCTION_CONTAINER.md`](QWEN38_PRODUCTION_CONTAINER.md). Use it when
you want the number reproduced rather than the knobs varied; use the launcher
above when you are varying knobs.

---

## 12. What decode speed is actually attributable to

Measured 2026-08-24 on the published container, MinHeap probe, 400 tokens.
Decode rate is one ratio:

    tok/s = emitted_per_cycle / cycle_time

|                  | emitted/cycle | cycle time | tok/s |
|---|---:|---:|---:|
| no speculation   | 1.00 | 72.1 ms | 13.9 |
| γ=7              | 6.06 | 112.3 ms | 54.0 |
| γ=15             | 8.33 | 129.8 ms | 64.2 |

Speculation multiplies tokens-per-cycle by **8.33** while multiplying cycle cost
by only **1.80**. That ratio, 4.62x, is the entire speedup. The **numerator is
the drafter** (acceptance x depth); the **denominator is the engine**. Both are
required and neither alone gets there.

Isolated A/B contributions to the denominator, same drafter, same probe:

| Change | Δ tok/s |
|---|---:|
| GDN lazy commit + persistent-H | **+6.48** (63.80 vs 57.32) |
| split-K draft head | +1.13 (64.07 vs 62.94) |
| tensor-core verify flags | **+0.25** (64.10 vs 63.85) — null |

**The tensor-core result corrects an earlier claim in this document.** The TC
flags were described as load-bearing for speed. They are not: disabling all
three costs 0.25 tok/s. They remain a numerics re-reference (§5) — that part
stands — but they are not where the throughput comes from.

### Depth saturates, and more is not better

Solving the measured γ=15 point (8.33 emitted) under a constant-hazard model
gives per-token acceptance **p ≈ 0.904**. Expected accepted converges to
`p/(1-p)` = **9.4 tokens**, so draft width beyond ~γ12 buys almost nothing while
verify cost keeps growing at 1.890 ms/node:

| γ | emitted/cycle | cycle ms | tok/s |
|---:|---:|---:|---:|
| 15 | 8.33 | 129.8 | **64.2** |
| 23 | 9.47 | 151.3 | 62.6 |
| 31 | 9.98 | 172.8 | **57.7** |

A `block_size: 32` drafter at today's acceptance is a **loss**, not a win.
Depth only pays if acceptance rises with it:

| p | best γ | tok/s at best γ |
|---:|---:|---:|
| 0.904 (today) | 17 | 64.5 |
| 0.924 | 20 | 74.4 |
| 0.944 | 24 | 88.5 |

**70 tok/s needs p ≈ 0.924**, about +0.02 over today. That is a corpus and
training question, not a kernel or draft-width one — and it is why the remaining
work is drafter work.
