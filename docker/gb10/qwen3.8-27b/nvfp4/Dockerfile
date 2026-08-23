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

ENV RUST_LOG=info
EXPOSE 8888

ENTRYPOINT ["spark"]
