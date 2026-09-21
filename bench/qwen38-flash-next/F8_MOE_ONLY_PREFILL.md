# F8: serial core, grouped MoE prefill

2026-09-05. Experimental candidate: **not full-state-qualified or a speed result**.
F11 now adds a fresh native build and8/8 short outputs matching the same-binary
baseline, with all48-layer engagement receipts. See `F11_NATIVE_SMOKE.md`.
Worktree `perf/qwen38-flash-next`, HEAD `11e76f29a68ed5e22be8fd14cc72e318bdcc6111`
plus preserved pre-existing dirty work. No changes to the DeepSeek or GLM branches.

## Lesson applied

The Mia upstream audit is pinned to commit
`203834ca88000c8192112e396b80d886b522caa0` of
[Qwen3.8-Flash-Next-Single-DGX-Spark](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/tree/203834ca88000c8192112e396b80d886b522caa0).
Its client-TTFT-derived filler-prompt results are not Atlas GPU prefill evidence.
The audit is `/var/tmp/atlas-mia-flash-next-sep5.pPZXqG/RELEVANCE.md`.

Recovered local F7 evidence changes that audit's immediate PLE-first priority:
PLE-only batching at 1395 tokens measured 31.419s versus 31.592s TTFT with
identical output, while the layer stack remained token-serial (TEAM_INBOX line
45713). Existing broader batching changed outputs. The Aug31 ROWWISE results
in CHAMPION.md also explicitly admit output drift and baseline nondeterminism.

F8 therefore batches only the stateless FFN across all 48 layers. It preserves
the shipping row-ordered attention/SSM core and both HC prepare/inject routines.
It does not copy vLLM-specific graph settings, FP8 KV policy, vocabulary slicing,
or mmap readahead settings into Atlas.

## Candidate contract

- Explicit opt-in `ATLAS_QWEN4_PREFILL_MOE_BATCH=1`; absent/0 keeps old dispatch.
  Invalid selector values fail. No default promotion.
- Canonical 48-layer Flash-Next, H2560, residual width10240, 512 experts/top10,
  intermediate640, single GPU, eager C1, BF16 residual and FP32 SSM state.
- Batched rows2..2048, complete prompt within the initial 2048-token window.
  Singleton chunks/continuations keep the old decode route and earn no receipt.
- Conflicting PLE, attention, SSM, hyper/QSA GEMM, worklist and H-FP16
  experiments are rejected. Full attention batching and ROWWISE remain off.
- Validate actual original-layout NVFP4 expert tables, required kernels and
  complete router/shared projection bundle before each layer's core effects.
- Finish every core row before staging MLP inputs in the checked QKV tail;
  copy to normal norm_output, call grouped FFN once, inject each row as decode.
- Success-only receipts require all12 attention and36 SSM layers, explicitly
  labelled `ffn=grouped core=serial_token_ordered`. Enqueued is not parity.

The existing loader predequants router/shared NVFP4 weights to FP8 for ordinary
prefill. F8 admits that complete bundle, so router/shared precision and grouped
expert accumulation remain deliberate numerical differences to qualify.
No new CUDA, loader, weight, graph or server changes were made.

The separate DeepSeek broadcast correction was checked, not blindly ported.
Flash-Next has an inline shared array, not the faulty restrict-qualified helper.
Retained PTX SHA `03b973e371006609d3b26d45dc90216b41c03d490669c6bd25e27a4fe0573cbd`
shows publication, barrier, then reload. F11's short model smoke does not replace
a dedicated device-arithmetic parity test for this retained kernel.

## Verification

CPU:615 library tests plus20 focused tests pass; scoped rustfmt and
`git diff --check` pass. Focused tests cover pure layout/selector boundaries,
loaded-weight admission, serial-core/staging wiring, singleton behavior and
48-layer receipts. These are not a numerical oracle.

Exact final CPU command, run with no visible GPU:

```sh
ATLAS_SKIP_BUILD=1 SKIP_ATLAS_BUILD=1 CUDARC_CUDA_VERSION=13000 \
CUDA_VISIBLE_DEVICES= \
RUSTFLAGS='-C target-cpu=native -L native=/usr/local/cuda/lib64 -l dylib=cublasLt -l dylib=cuda' \
CARGO_TARGET_DIR=/var/tmp/atlas-flashnext-f8-cpu.X5ZbhMKW \
cargo test --locked --offline -p spark-model --lib \
  --test qwen4_prefill_moe_plan --test qwen4_prefill_moe_wiring \
  --test qwen4_ssm_prefill_moe_model --test qwen4_attention_prefill_moe_model \
  --test qwen4_prefill_moe_admission_model
```

Log: `/var/tmp/atlas-flashnext-f8-cpu-final.log`. The initial CPU command failed
at link time because the existing skip-build path omits cuBLASLt linkage;
explicit test-only linker flags resolved it. No build-script workaround added.

## Ordered remaining gates

1. Read current coordination/inbox and reclose source/checkpoint. F11 completed
   the first native build; its fingerprinted binary is recorded in the smoke
   receipt. Rebuild under a fresh claim if any build input changes.
2. Fresh root GPU/server claim, pinned same binary for selector0 and1,
   prefix cache off, C1, no speculation and all other prefill experiments off.
3. Establish repeated serial-baseline determinism; compare small varied
   prompts, coding outputs, final KV/SSM/HC state and next-token continuation.
   Include singleton chunks, chunk/reset boundaries and interleaved requests.
4. On mismatch, stop timing. Compare identical layer inputs through router
   logits/IDs/weights, routed/shared outputs and final BF16 blend. Do not
   relax tolerances or select a convenient nondeterministic baseline output.
5. Only after parity: same-binary uncached256/2048 ABBA, both varied coding
   and filler prompts, separate GPU prefill from client TTFT, and verify
   unchanged decode/output hashes. Long-QSA batching needs a separate gate.

No 2000+ prefill, decode improvement, long-context or multimodal qualification
is claimed for F8. Existing CHAMPION.md numbers remain historical, unchanged.
