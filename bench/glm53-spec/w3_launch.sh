#!/usr/bin/env bash
# Waits for the speed-matrix window END (only lines after line 2402 count), then runs window w3.
Q=/home/flocka/atlas/.gb10-queue
until tail -n +2403 "$Q" | grep -q "speed-matrix window END" && grep -q "DONE m3 rc=0" /home/flocka/atlas/glm-spec-bench/build-m3.log 2>/dev/null; do sleep 20; done
cd /home/flocka/atlas/glm-spec-bench
R=ATLAS_GLM53_EXL3_ROWBATCH=1; D=ATLAS_GLM53_VERIFY_DSA_PRECOMPUTE=1,ATLAS_GLM53_VERIFY_DSA_BATCH_OUTPUT=1; P=ATLAS_GLM53_PREFIX_COMMIT=1
H=ATLAS_GLM53_STATE_HASH
export PRE_CMD=/home/flocka/atlas/glm-spec-bench/harness/rowbatch_equiv CONTROL_PERTURB=T
exec ./window3.sh w3 $PWD/bin/spark-m3 \
  T:target \
  S4:dflash3:$R,$D,$P \
  S3:dflash3:$R,$D \
  S4g2:dflash2:$R,$D,$P \
  Th@1:target:$H=%RUN%/Th.hash \
  S4h@1:dflash3:$R,$D,$P,$H=%RUN%/S4h.hash \
  S4ctl@1:dflash3:$R,$D,$P,ATLAS_GLM53_PREFIX_COMMIT_CONTROL_LEGACY_RESTAGE=1,$H=%RUN%/S4ctl.hash \
  T2:target
