# Atlas Spark — Qwen3.8-27B NVFP4 (Dense Hybrid SSM+Attn) on GB10
#
# Builds spark-server from source for the (gb10, qwen3.8-27b, nvfp4) kernel
# target. The runtime image carries the binary, the jinja templates and the
# licence — no model weights, no drafter, no credentials. Weights are mounted
# or pulled at run time.
#
# Build from repo root:
#   docker build -f docker/gb10/qwen3.8-27b/nvfp4/Dockerfile -t atlas-gb10:qwen3.8-27b .
#
# Run (target only, no speculative decoding):
#   docker run --gpus all --ipc=host -p 8888:8888 \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     atlas-gb10:qwen3.8-27b serve \
#       --model-from-path /root/.cache/huggingface/Qwen3.8-27B-NVFP4 \
#       --kernel-target qwen3.8-27b --max-seq-len 16384
#
# Run (with a DFlash drafter — the configuration the published decode numbers
# were measured under; see bench/qwen38-gb10/ for the full flag set):
#   docker run --gpus all --ipc=host -p 8888:8888 \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     -v /path/to/drafter:/drafter:ro \
#     atlas-gb10:qwen3.8-27b serve \
#       --model-from-path /root/.cache/huggingface/Qwen3.8-27B-NVFP4 \
#       --kernel-target qwen3.8-27b \
#       --dflash --draft-model /drafter \
#       --dflash-gamma 15 --dflash-quantization nvfp4 --max-seq-len 8192
#
# --kernel-target names a MODEL directory under kernels/<hw>/; passing the
# hardware name (gb10) is rejected by the loader.
#
# STATUS: this image has been built AND SERVED AND MEASURED — a container-served
# instance reaches 64.05 tok/s median on the published decode probe with
# byte-identical output. The other eight per-model Dockerfiles under docker/gb10/
# all build with the correct kernel target, but none has been served.
# See docker/docker-guide.md.

# ── Build stage ──────────────────────────────────────────────────
FROM nvidia/cuda:13.0.0-devel-ubuntu24.04 AS builder

RUN apt-get update -qq && \
    apt-get install -y -qq curl build-essential pkg-config git cmake libclang-dev && \
    rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
ENV CUDA_HOME=/usr/local/cuda
# nvcc must be ON PATH, not merely findable via CUDA_HOME: cudarc 0.19.2's
# build.rs:115 shells out to a bare `nvcc --version` and panics if it is not
# resolvable, regardless of CUDA_HOME/CUDA_PATH/CUDA_ROOT. The nvidia/cuda
# devel images happen to set this themselves, so today the build works by
# INHERITANCE rather than by declaration — which breaks silently on a base
# image bump. Declare it.
ENV PATH="/usr/local/cuda/bin:${PATH}"

WORKDIR /build

COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY crates/ crates/
# No `COPY vendor/`: the vendored C++ xgrammar-rs tree was deliberately deleted
# when the grammar stack went pure-Rust (see Cargo.toml, `[patch.crates-io]`
# note). The directory does not exist, so the line broke every build here.
COPY kernels/ kernels/
# The runtime stage copies /build/jinja-templates/ from this stage, so it has to
# exist here first — docker/gb10/Dockerfile (the one image that has been built)
# does the same.
COPY jinja-templates/ jinja-templates/

ENV ATLAS_TARGET_HW=gb10
ENV ATLAS_TARGET_MODEL=qwen3.8-27b
ENV ATLAS_TARGET_QUANT=nvfp4

RUN cargo build --release -p spark-server

# ── Runtime stage ────────────────────────────────────────────────
FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04

LABEL org.opencontainers.image.licenses="AGPL-3.0-only"

# NCCL for multi-GPU (EP=2), RDMA userspace for RoCE transport
RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends --allow-change-held-packages \
      libnccl2 libibverbs1 librdmacm1 ibverbs-providers \
      libnl-3-200 libnl-route-3-200 && \
    rm -rf /var/lib/apt/lists/* && \
    NCCL_VER=$(dpkg-query -W -f='${Version}' libnccl2 2>/dev/null); \
    if [ -z "$NCCL_VER" ]; then \
      echo "ERROR: libnccl2 is not installed — the install step above failed"; exit 1; \
    fi && \
    echo "Linked NCCL: $NCCL_VER" && \
    if ! dpkg --compare-versions "$NCCL_VER" ge "2.28"; then \
      echo "ERROR: NCCL $NCCL_VER < 2.28 — symmetric memory APIs unavailable"; exit 1; \
    fi

COPY --from=builder /build/target/release/spark /usr/local/bin/spark
# Match docker/gb10/Dockerfile, the only image in this tree that has been
# built and exercised. Qwen3.8 ships its own chat_template.jinja inside the
# checkpoint, but the templates cost little and a model whose checkpoint
# lacks one would otherwise fail at serve time rather than at build time.
COPY --from=builder /build/jinja-templates/ /jinja-templates/
COPY LICENSE /LICENSE

# ── Tuned decode profile ─────────────────────────────────────────────
# Baked in as defaults so a bare `docker run ... serve` reaches the published
# decode rate. Without these the engine runs at roughly 8.8 tok/s against 63.9
# with them — and speculation on an untuned verify path is actually SLOWER
# than no speculation at all (13.9), so a partial configuration is the worst
# of the three. Override any of them with `-e VAR=value`.
# Provenance for every value: bench/qwen38-gb10/FLAGS.md
# Gamma-derived. These MUST track --dflash-gamma: the defaults below are for
# gamma 15 (a block_size=16 drafter, which is what the published figure uses).
# If you pass a different --dflash-gamma, override both — otherwise draft width
# and tree node count disagree with the drafter and acceptance collapses
# silently. For gamma G: DRAFT_CAP=G, DDTREE_MAX_NODES=G+1.
ENV ATLAS_DFLASH_DRAFT_CAP=15
ENV ATLAS_DDTREE_MAX_NODES=16
ENV ATLAS_FFN_TC=1
ENV ATLAS_SSM_PROJ_TC=1
ENV ATLAS_LM_HEAD_TC=1
ENV ATLAS_LM_HEAD_T=1
ENV ATLAS_ACCEPT_FAST_ARGMAX=1
ENV ATLAS_PREFILL_PROJ_FAST=0
ENV ATLAS_PREFILL_FFN_FAST=0
ENV ATLAS_SSM_GDN_SEQ_PERSISTENT=1
ENV ATLAS_SSM_GDN_LAZY=1
ENV ATLAS_ATTN_QKV_FUSED=1
ENV ATLAS_ATTN_QKV_EXACT_STRIDED=1
ENV ATLAS_ATTN_QKV_BATCHED=0
ENV ATLAS_ATTN_QKV_SPLITK=4
ENV ATLAS_DFLASH_DRAFT_SPLITK=8
ENV ATLAS_DFLASH_QUANT=nvfp4
ENV ATLAS_DFLASH_LM_HEAD_NVFP4=1
ENV ATLAS_DFLASH_LM_HEAD_FP8=1
ENV ATLAS_DFLASH_FFN_KGAMMA=1
ENV ATLAS_DFLASH_ATTN_KGAMMA=1
ENV ATLAS_DFLASH_NOISE_ONLY=1
ENV ATLAS_DFLASH_CTX_WINDOW=4096
ENV ATLAS_DFLASH_FREE_SLOTS=0
ENV ATLAS_DFLASH_FREE_SLOTS_TAIL=4
ENV ATLAS_DFLASH_SAM=0
ENV ATLAS_DFLASH_ASYNC=0
ENV ATLAS_DFLASH_RETR_WIDE=31
ENV ATLAS_DFLASH_KERNEL_PROFILE=0
ENV ATLAS_DFLASH_TREE_COMMIT=0
ENV ATLAS_THINK_SPEC=1
ENV ATLAS_FFN_FUSED_GATEUP=1
ENV ATLAS_FFN_KGAMMA_M16=1
ENV ATLAS_FFN_KGAMMA_M128=1
ENV ATLAS_FFN_M16_TRANSPOSED=1
ENV ATLAS_FFN_DOWN_SPLITK=4
ENV ATLAS_NVFP4_GATE_UP_M128=1
ENV ATLAS_NO_GEMV_SW=1
ENV ATLAS_TC_NVFP4_M16=0
ENV ATLAS_TC_NVFP4_M16_MS_ATTN=0
ENV ATLAS_SSM_OUT_SPLITK=4
ENV ATLAS_SSM_QKVZ_SPLITK=4
ENV ATLAS_SSM_BA_BATCH=1
ENV ATLAS_WY17_LAZY=1
ENV ATLAS_WY17_LAZY_COMMIT=0
ENV ATLAS_WY17_SPLIT=2
ENV ATLAS_DISABLE_TREE_WY=1
ENV ATLAS_DSPARK_ASYMMETRIC_ATTN=1
ENV ATLAS_FLASH_ATTN_KGAMMA_SPLITK=1
ENV ATLAS_FA2_KGAMMA=1
ENV ATLAS_PAGED_DECODE_SPLITK=1
ENV ATLAS_LM_HEAD_BATCH3=1
ENV ATLAS_DDTREE_UNCAP=0
ENV ATLAS_DDTREE_TREE_AWARE_VERIFY=0
ENV ATLAS_DDTREE_TREE_TOKENS_VERIFY=0
ENV ATLAS_DDTREE_TREE_CONV_EXACT=0
ENV ATLAS_TREE_AWARE_ATTN=0
ENV ATLAS_MULTISEQ_GRAPHS=0
ENV ATLAS_DFLASH_SPEC_CYCLE_V2=1
ENV ATLAS_WEIGHT_CACHE=1

ENV RUST_LOG=info
EXPOSE 8888

ENTRYPOINT ["spark"]
