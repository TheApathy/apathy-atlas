#!/bin/bash
# GPU job: indexer gate. Queue line, flock, START/END, positive DONE line with the exit code.
Q=/home/flocka/atlas/.gb10-queue
DIR=/home/flocka/atlas/dsv41-attention/kernels/gb10/deepseek-v4.1/attn
BIN=$DIR/sparse_index_gate
[ -x "$BIN" ] || { echo "DONE rc=127 (binary missing)"; exit 127; }
echo "$(date -u +%FT%TZ) dsv41-attention indexer gate v2 (12 small fixtures + pool cases, <1 min) QUEUED pid=$$" >> $Q
exec 9>/home/flocka/atlas/.gb10.lock
flock 9
echo "$(date -u +%FT%TZ) dsv41-attention indexer gate START pid=$$" >> $Q
cd $DIR
$BIN idx_fixtures/runD_L20_kernel_L02_0 idx_fixtures/runD_L20_kernel_L02_1 idx_fixtures/runD_L20_kernel_L14_1 \
     idx_fixtures/runD_L20_kernel_L20_0 idx_fixtures/runD_L20_kernel_L20_1 idx_fixtures/runD_L20_kernel_L24_0 \
     idx_fixtures/runE_torch_L02_0 idx_fixtures/runE_torch_L02_1 idx_fixtures/runE_torch_L14_1 \
     idx_fixtures/runE_torch_L20_0 idx_fixtures/runE_torch_L20_1 idx_fixtures/runE_torch_L24_0 > index_gate.log 2>&1
rc=$?
echo "$(date -u +%FT%TZ) dsv41-attention indexer gate END rc=$rc pid=$$" >> $Q
flock -u 9
echo "DONE rc=$rc"
