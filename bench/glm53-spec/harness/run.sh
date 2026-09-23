#!/usr/bin/env bash
set -e
cd /home/flocka/atlas/glm-spec-bench/harness
export PATH=/usr/local/cuda-13.0/bin:$PATH
nvcc -arch sm_121f -O3 --fmad false -DTQ_PLUS_SIGNS -std c++17 -I/var/tmp/exllamav3-glm53-r28/exllamav3/exllamav3_ext -Xcudafe --diag_suppress=20012 -o rowbatch_equiv rowbatch_equiv.cu
