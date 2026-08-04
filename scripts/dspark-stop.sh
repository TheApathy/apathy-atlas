#!/usr/bin/env bash
# Stop the running spark server and WAIT for it to actually exit.
#
# The GPU allocation is only released on process exit; relaunching before then
# makes the new server OOM. Targets the specific pid (never a blanket pkill).
set -uo pipefail

PIDS=$(pgrep -f "target/release/spark serve" || true)
if [ -z "$PIDS" ]; then
  echo "no spark server running"
  exit 0
fi

for pid in $PIDS; do
  echo "stopping pid=$pid"
  kill "$pid" 2>/dev/null || true
done

for pid in $PIDS; do
  until ! kill -0 "$pid" 2>/dev/null; do sleep 2; done
  echo "pid=$pid exited"
done

# Process exit is not the same as the CUDA context being torn down — the driver
# reclaims the ~100 GB allocation a beat later. Relaunching into that window
# makes the new server size its KV pool against near-zero free memory and die
# with "KV cache can hold at most 0 concurrent sequence(s)". nvidia-smi reports
# N/A on GB10 (integrated memory), so there is nothing to poll: just settle.
sleep 20
echo "gpu settled"
