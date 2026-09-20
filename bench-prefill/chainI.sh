#!/usr/bin/env bash
W=/home/flocka/atlas/qwen27b-prefill-work/bench
cd $W/gdn
echo "[$(date -u +%T)] START gdn r4"
flock -w 14400 /home/flocka/atlas/.gb10.lock timeout 900 ./gdn_test4 2048 > gdn_r4.txt 2>&1
echo "[$(date -u +%T)] CHAINI-DONE"
