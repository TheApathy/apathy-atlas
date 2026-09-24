#!/usr/bin/env bash
# State/admission diag window on m4, after my w4 window has ENDed and m4 is built.
Q=/home/flocka/atlas/.gb10-queue
until grep -q "glm-dflash2 window w4 END" "$Q" && grep -q "DONE m4 rc=0" /home/flocka/atlas/glm-spec-bench/build-m4.log 2>/dev/null; do sleep 20; done
cd /home/flocka/atlas/glm-spec-bench
R=ATLAS_GLM53_PREFIX_COMMIT_ROWS=3,ATLAS_GLM53_EXL3_ROWBATCH=1,ATLAS_GLM53_VERIFY_DSA_PRECOMPUTE=1,ATLAS_GLM53_VERIFY_DSA_BATCH_OUTPUT=1,ATLAS_GLM53_PREFIX_COMMIT=1
H=ATLAS_GLM53_STATE_HASH; O=@MAXSEQ=4096
export TRIALS=1 PROMPTS="short long prose code2 think long2k" CONTROL_PERTURB=Th
exec ./window4.sh w5 $PWD/bin/spark-m4 \
  Th:target:$O,$H=%RUN%/Th.hash \
  S3h:dflash3:$O,$R,$H=%RUN%/S3h.hash \
  S2h:dflash2:$O,$R,$H=%RUN%/S2h.hash \
  Cleg:dflash3:$O,$R,ATLAS_GLM53_PREFIX_COMMIT_CONTROL_LEGACY_RESTAGE=1,$H=%RUN%/Cleg.hash \
  Cext:dflash3:$O,$R,ATLAS_GLM53_SPEC_CONTROL_COMMIT_EXTRA_ROW=1,$H=%RUN%/Cext.hash
