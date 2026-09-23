#!/bin/bash
# GPU-lock job (EXCLUSIVE): ncu (full sections) on the prefill attention kernel at T=2072, via the
# production image (ncu works with --cap-add=SYS_ADMIN despite RmProfilingAdminOnly). Timing-free:
# ncu serialises and replays; only the counters/stall reasons are read from this.
Q=/home/flocka/atlas/.gb10-queue
D=/home/flocka/atlas/dsv41-attention/kernels/gb10/deepseek-v4.1/attn
IMAGE="sha256:a148c1ceef4c3101e7ef1d8c7fe98219489260dfea8abe7581b10b5cc11e22fc"
echo "$(date -u +%FT%TZ) dsv41-attention ncu on sparse-attn fixture (no model, <1 GB, ~5 min) QUEUED pid=$$" >> $Q
exec 9>/home/flocka/atlas/.gb10.lock
flock 9
echo "$(date -u +%FT%TZ) dsv41-attention ncu START pid=$$" >> $Q
prof() {  # prof <kernel> <skip> <out>
  docker run --rm --gpus all --network none --cap-add=SYS_ADMIN --entrypoint /usr/local/cuda/bin/ncu \
    --mount type=bind,src=$D,dst=/w -w /w/real "$IMAGE" \
    --kernel-name "$1" --launch-skip "$2" --launch-count 1 --set full --page details --print-units base \
    --log-file "/w/$3" /w/sparse_attn > /dev/null 2>&1; echo $?
}
r1=$(prof dsv41_sparse_attn_mma32h 24 ncu_mma32h.txt)
r2=skipped
echo "$(date -u +%FT%TZ) dsv41-attention ncu END rc=$r1/$r2 pid=$$" >> $Q
flock -u 9
echo "DONE $r1 $r2"
