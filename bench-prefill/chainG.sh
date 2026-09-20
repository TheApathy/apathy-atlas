#!/usr/bin/env bash
W=/home/flocka/atlas/qwen27b-prefill-work/bench
cd $W/gdn
echo "[$(date -u +%T)] START gdn r1/r2"
flock -w 14400 /home/flocka/atlas/.gb10.lock bash -c '
  echo "=== R1 registers-only (smem 29696) ==="; timeout 600 ./gdn_test_r1 2048 29696
  echo "=== R2 registers + C-arrays in smem (smem 78848) ==="; timeout 600 ./gdn_test_r2 2048 78848
' > gdn_r12.txt 2>&1
echo "[$(date -u +%T)] CHAING-DONE"
