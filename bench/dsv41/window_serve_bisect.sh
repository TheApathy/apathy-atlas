#!/usr/bin/env bash
# Driver vs serve at the SAME build: (1) driver runG prefill (h taps) + 48-token decode (text-only
# post-merge regression check vs the pre-merge k124_decode2 step lines); (2) serve with the model's
# h taps on, a 1-token runG request (prefill only) then the 48-token one + parity's tool request.
set -uo pipefail
S=/tmp/claude-1000/-home-flocka-atlas/153d8f07-762d-4884-a8f5-8f7a1a13f1ce/scratchpad
B=/home/flocka/atlas/dsv41-integration/bench/dsv41
I=/home/flocka/atlas/DSV41_PORT/integrate
export ABORT_GB=20
rm -rf "$S/drvG" "$S/srvG"
$B/keep124_window.sh k124_drv_postmerge \
  --run runG_replay --layers 40 --path model --moe-real --attn-real --tap-names h --tap-dir $S/drvG --decode 48 \
  > $I/k124_drv_postmerge.log 2>&1
ATLAS_DSV41_TAP_DIR=$S/srvG ATLAS_DSV41_TAP_NAMES=h $B/serve_window.sh serve3 8900 $I/serve3 > $I/serve3.log 2>&1
echo "DONE serve-bisect chain"
