#!/bin/bash
DIR=/home/flocka/atlas/dsv41-attention/kernels/gb10/deepseek-v4.1/attn
BIN=$DIR/sparse_index_gate
cd $DIR
$BIN idx_fixtures/runD_L20_kernel_L02_0 idx_fixtures/runD_L20_kernel_L02_1 idx_fixtures/runD_L20_kernel_L14_1 \
     idx_fixtures/runD_L20_kernel_L20_0 idx_fixtures/runD_L20_kernel_L20_1 idx_fixtures/runD_L20_kernel_L24_0 \
     idx_fixtures/runE_torch_L02_0 idx_fixtures/runE_torch_L02_1 idx_fixtures/runE_torch_L14_1 \
     idx_fixtures/runE_torch_L20_0 idx_fixtures/runE_torch_L20_1 idx_fixtures/runE_torch_L24_0
