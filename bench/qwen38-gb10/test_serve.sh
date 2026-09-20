#!/usr/bin/env bash
# CPU-only behavioral contract for the Qwen3.8 launch wrappers.
set -euo pipefail

if [[ "${ATLAS_SERVE_TEST_FAKE-}" == 1 && "${1-}" == serve ]]; then
  printf 'fake_pid=%s\n' "$$"
  for argument in "$@"; do
    printf 'arg=%s\n' "$argument"
  done
  env | LC_ALL=C sort
  exit 0
fi

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SELF="$HERE/$(basename -- "${BASH_SOURCE[0]}")"
SERVE="$HERE/serve.sh"
SERVE_V3="$HERE/serve-v3-72tps.sh"
SERVE_NO_SPEC="$HERE/serve-no-spec.sh"

invoke() {
  env -i \
    PATH=/usr/bin:/bin \
    MODEL_DIR=/model \
    BIN="$SELF" \
    ATLAS_SERVE_TEST_FAKE=1 \
    "$@"
}

must_fail() {
  if invoke "$@" >/dev/null 2>&1; then
    printf 'expected launcher failure: %s\n' "$*" >&2
    exit 1
  fi
}

legacy="$({ invoke DRAFT=/draft "$SERVE"; } 2>&1)"
[[ "$legacy" == *$'arg=--dflash\n'* ]]
[[ "$legacy" == *$'arg=--draft-model\narg=/draft\n'* ]]
[[ "$legacy" != *"ATLAS_QUALIFICATION_RUN pid="* ]]

explicit="$(invoke RUNTIME_MODE=dflash-v3 DRAFT=/draft "$SERVE")"
[[ "$explicit" == *$'arg=--dflash\n'* ]]
[[ "$explicit" == *$'ATLAS_DFLASH_ECHO=0\n'* ]]

inherited_echo="$(invoke RUNTIME_MODE=dflash-v3 DRAFT=/draft ATLAS_DFLASH_ECHO=1 "$SERVE")"
[[ "$inherited_echo" == *$'ATLAS_DFLASH_ECHO=0\n'* ]]
[[ "$inherited_echo" != *$'ATLAS_DFLASH_ECHO=1\n'* ]]

no_spec="$(invoke RUNTIME_MODE=no-spec PREFILL_QKNORM_ROPE=1 PREFILL_ATTN_GATE_FUSED=1 "$SERVE" --bind 127.0.0.1)"
[[ "$no_spec" != *$'arg=--dflash\n'* ]]
[[ "$no_spec" != *$'arg=--draft-model\n'* ]]
[[ "$no_spec" == *$'arg=--bind\narg=127.0.0.1\n'* ]]
[[ "$no_spec" == *$'ATLAS_PREFILL_QKNORM_ROPE=1\n'* ]]
[[ "$no_spec" == *$'ATLAS_PREFILL_ATTN_GATE_FUSED=1\n'* ]]
if printf '%s\n' "$no_spec" | grep -qx 'PREFILL_QKNORM_ROPE=1'; then
  echo "launcher leaked its short flag alias into the server environment" >&2
  exit 1
fi
if printf '%s\n' "$no_spec" | grep -qx 'PREFILL_ATTN_GATE_FUSED=1'; then
  echo "launcher leaked its attention-gate short flag alias into the server environment" >&2
  exit 1
fi

atlas_spelling="$(invoke RUNTIME_MODE=no-spec ATLAS_PREFILL_ATTN_GATE_FUSED=1 "$SERVE")"
[[ "$atlas_spelling" == *$'ATLAS_PREFILL_ATTN_GATE_FUSED=1\n'* ]]

m17_astage_alias="$(invoke RUNTIME_MODE=no-spec ATTN_QKV_M17_ASTAGE=1 "$SERVE")"
[[ "$m17_astage_alias" == *$'ATLAS_ATTN_QKV_EXACT_M17_ASTAGE=1\n'* ]]
if printf '%s\n' "$m17_astage_alias" | grep -qx 'ATTN_QKV_M17_ASTAGE=1'; then
  echo "launcher leaked its M17 activation-staging short flag alias" >&2
  exit 1
fi
m17_astage_atlas="$(invoke RUNTIME_MODE=no-spec ATLAS_ATTN_QKV_EXACT_M17_ASTAGE=1 "$SERVE")"
[[ "$m17_astage_atlas" == *$'ATLAS_ATTN_QKV_EXACT_M17_ASTAGE=1\n'* ]]

ffn_dual_alias="$(invoke RUNTIME_MODE=no-spec PREFILL_FFN_DUAL_FUSED=1 "$SERVE")"
[[ "$ffn_dual_alias" == *$'ATLAS_PREFILL_FFN_DUAL_FUSED=1\n'* ]]
if printf '%s\n' "$ffn_dual_alias" | grep -qx 'PREFILL_FFN_DUAL_FUSED=1'; then
  echo "launcher leaked its FFN-dual short flag alias into the server environment" >&2
  exit 1
fi

ffn_dual_atlas="$(invoke RUNTIME_MODE=no-spec ATLAS_PREFILL_FFN_DUAL_FUSED=1 "$SERVE")"
[[ "$ffn_dual_atlas" == *$'ATLAS_PREFILL_FFN_DUAL_FUSED=1\n'* ]]

flashinfer="$(invoke RUNTIME_MODE=no-spec PREFILL_FFN_FLASHINFER=1 \
  FLASHINFER_SM121_LIB=/bin/sh "$SERVE")"
[[ "$flashinfer" == *$'ATLAS_PREFILL_FFN_FLASHINFER=1\n'* ]]
[[ "$flashinfer" == *$'ATLAS_FLASHINFER_SM121_LIB=/bin/sh\n'* ]]
if printf '%s\n' "$flashinfer" | grep -qx 'PREFILL_FFN_FLASHINFER=1'; then
  echo "launcher leaked its FlashInfer short flag alias into the server environment" >&2
  exit 1
fi

flashinfer_proj="$(invoke RUNTIME_MODE=no-spec PREFILL_PROJ_FLASHINFER=1 \
  FLASHINFER_SM121_LIB=/bin/sh "$SERVE")"
[[ "$flashinfer_proj" == *$'ATLAS_PREFILL_PROJ_FLASHINFER=1\n'* ]]
[[ "$flashinfer_proj" == *$'ATLAS_FLASHINFER_SM121_LIB=/bin/sh\n'* ]]
if printf '%s\n' "$flashinfer_proj" | grep -qx 'PREFILL_PROJ_FLASHINFER=1'; then
  echo "launcher leaked its FlashInfer projection short flag alias" >&2
  exit 1
fi

flashinfer_ssm="$(invoke RUNTIME_MODE=no-spec PREFILL_SSM_FLASHINFER=1 \
  FLASHINFER_SM121_LIB=/bin/sh "$SERVE")"
[[ "$flashinfer_ssm" == *$'ATLAS_PREFILL_SSM_FLASHINFER=1\n'* ]]
[[ "$flashinfer_ssm" == *$'ATLAS_FLASHINFER_SM121_LIB=/bin/sh\n'* ]]
if printf '%s\n' "$flashinfer_ssm" | grep -qx 'PREFILL_SSM_FLASHINFER=1'; then
  echo "launcher leaked its FlashInfer SSM short flag alias" >&2
  exit 1
fi

flashinfer_ssm_atlas="$(invoke RUNTIME_MODE=no-spec ATLAS_PREFILL_SSM_FLASHINFER=1 \
  ATLAS_FLASHINFER_SM121_LIB=/bin/sh "$SERVE")"
[[ "$flashinfer_ssm_atlas" == *$'ATLAS_PREFILL_SSM_FLASHINFER=1\n'* ]]

gdn_c143="$(invoke RUNTIME_MODE=no-spec GDN_C143_PREFILL=1 \
  GDN_C143_SM121_LIB=/bin/sh "$SERVE")"
[[ "$gdn_c143" == *$'ATLAS_GDN_C143_PREFILL=1\n'* ]]
[[ "$gdn_c143" == *$'ATLAS_GDN_C143_SM121_LIB=/bin/sh\n'* ]]
if printf '%s\n' "$gdn_c143" | grep -Eq '^GDN_C143_(PREFILL|SM121_LIB)='; then
  echo "launcher leaked a GDN c143 short alias into the server environment" >&2
  exit 1
fi

gdn_c143_atlas="$(invoke RUNTIME_MODE=no-spec ATLAS_GDN_C143_PREFILL=1 \
  ATLAS_GDN_C143_SM121_LIB=/bin/sh "$SERVE")"
[[ "$gdn_c143_atlas" == *$'ATLAS_GDN_C143_PREFILL=1\n'* ]]
[[ "$gdn_c143_atlas" == *$'ATLAS_GDN_C143_SM121_LIB=/bin/sh\n'* ]]

wrapped_no_spec="$(invoke "$SERVE_NO_SPEC")"
[[ "$wrapped_no_spec" != *$'arg=--dflash\n'* ]]
[[ "$wrapped_no_spec" == *$'arg=--model-name\narg=qwen38-atlas-fork\n'* ]]
[[ "$wrapped_no_spec" == *$'arg=--kv-cache-dtype\narg=nvfp4\n'* ]]
[[ "$wrapped_no_spec" == *$'arg=--mtp-vocab\narg=248320\n'* ]]
[[ "$wrapped_no_spec" == *$'ATLAS_PREFILL_FFN_FUSED_EPILOGUE=0\n'* ]]
[[ "$wrapped_no_spec" == *$'ATLAS_PREFILL_FFN_DUAL_FUSED=0\n'* ]]
[[ "$wrapped_no_spec" == *$'ATLAS_PREFILL_SSM_FLASHINFER=0\n'* ]]
[[ "$wrapped_no_spec" == *$'ATLAS_GDN_C143_PREFILL=0\n'* ]]

one_million="$(invoke RUNTIME_MODE=no-spec MAX_SEQ_LEN=1000000 KV_CACHE_DTYPE=nvfp4 "$SERVE")"
[[ "$one_million" == *$'arg=--max-seq-len\narg=1000000\n'* ]]
[[ "$one_million" == *$'arg=--rope-yarn-factor\narg=4\n'* ]]
[[ "$one_million" != *$'arg=--rope-theta-override\n'* ]]
[[ "$one_million" != *$'arg=--dflash\n'* ]]

wrapped_ffn_dual="$(invoke DRAFT=/draft PREFILL_FFN_DUAL_FUSED=1 "$SERVE_V3")"
[[ "$wrapped_ffn_dual" == *$'ATLAS_PREFILL_FFN_DUAL_FUSED=1\n'* ]]
wrapped_ffn_dual_atlas="$(invoke DRAFT=/draft ATLAS_PREFILL_FFN_DUAL_FUSED=1 "$SERVE_V3")"
[[ "$wrapped_ffn_dual_atlas" == *$'ATLAS_PREFILL_FFN_DUAL_FUSED=1\n'* ]]
[[ "$wrapped_no_spec" == *$'ATLAS_PREFILL_ATTN_GATE_FUSED=0\n'* ]]

wrapped_v3="$(invoke DRAFT=/draft "$SERVE_V3")"
[[ "$wrapped_v3" == *$'ATLAS_PREFILL_ATTN_GATE_FUSED=0\n'* ]]
wrapped_v3_candidate="$(invoke DRAFT=/draft PREFILL_ATTN_GATE_FUSED=1 "$SERVE_V3")"
[[ "$wrapped_v3_candidate" == *$'ATLAS_PREFILL_ATTN_GATE_FUSED=1\n'* ]]
if printf '%s\n' "$wrapped_v3_candidate" | grep -qx 'PREFILL_ATTN_GATE_FUSED=1'; then
  echo "V3 wrapper leaked its attention-gate short flag alias" >&2
  exit 1
fi
wrapped_v3_atlas_candidate="$(invoke DRAFT=/draft ATLAS_PREFILL_ATTN_GATE_FUSED=1 "$SERVE_V3")"
[[ "$wrapped_v3_atlas_candidate" == *$'ATLAS_PREFILL_ATTN_GATE_FUSED=1\n'* ]]
wrapped_no_spec_atlas_candidate="$(invoke ATLAS_PREFILL_ATTN_GATE_FUSED=1 "$SERVE_NO_SPEC")"
[[ "$wrapped_no_spec_atlas_candidate" == *$'ATLAS_PREFILL_ATTN_GATE_FUSED=1\n'* ]]

must_fail RUNTIME_MODE= DRAFT=/draft "$SERVE"
must_fail RUNTIME_MODE=unknown DRAFT=/draft "$SERVE"
must_fail RUNTIME_MODE=dflash-v3 "$SERVE"
must_fail RUNTIME_MODE=no-spec DRAFT=/draft "$SERVE"
must_fail RUNTIME_MODE=no-spec "$SERVE" --dflash
must_fail RUNTIME_MODE=no-spec "$SERVE" --dflash=true
must_fail RUNTIME_MODE=no-spec "$SERVE" --dflash-gamma=7
must_fail RUNTIME_MODE=no-spec "$SERVE" --draft-model=/draft
must_fail RUNTIME_MODE=no-spec "$SERVE" --speculative
must_fail RUNTIME_MODE=no-spec "$SERVE" --speculative=true
must_fail RUNTIME_MODE=no-spec "$SERVE" --self-speculative
must_fail RUNTIME_MODE=no-spec "$SERVE" --ngram-speculative
must_fail RUNTIME_MODE=no-spec "$SERVE" --mtp-gate=auto
must_fail RUNTIME_MODE=no-spec DRAFT=/draft "$SERVE_V3"
must_fail RUNTIME_MODE=dflash-v3 "$SERVE_NO_SPEC"
must_fail DRAFT=/draft "$SERVE_NO_SPEC"
must_fail DRAFT=/draft ATLAS_QUALIFICATION_RUN_NONCE=bad "$SERVE"
must_fail RUNTIME_MODE=no-spec PREFILL_ATTN_GATE_FUSED= "$SERVE"
must_fail RUNTIME_MODE=no-spec PREFILL_ATTN_GATE_FUSED=true "$SERVE"
must_fail RUNTIME_MODE=no-spec PREFILL_ATTN_GATE_FUSED=0 ATLAS_PREFILL_ATTN_GATE_FUSED=1 "$SERVE"
must_fail RUNTIME_MODE=no-spec PREFILL_FFN_DUAL_FUSED= "$SERVE"
must_fail RUNTIME_MODE=no-spec ATLAS_PREFILL_FFN_DUAL_FUSED= "$SERVE"
must_fail RUNTIME_MODE=no-spec PREFILL_FFN_DUAL_FUSED=true "$SERVE"
must_fail RUNTIME_MODE=no-spec PREFILL_FFN_DUAL_FUSED=0 ATLAS_PREFILL_FFN_DUAL_FUSED=1 "$SERVE"
must_fail RUNTIME_MODE=no-spec PREFILL_FFN_FLASHINFER= "$SERVE"
must_fail RUNTIME_MODE=no-spec PREFILL_FFN_FLASHINFER=0 ATLAS_PREFILL_FFN_FLASHINFER=1 "$SERVE"
must_fail RUNTIME_MODE=no-spec PREFILL_PROJ_FLASHINFER= "$SERVE"
must_fail RUNTIME_MODE=no-spec PREFILL_PROJ_FLASHINFER=true "$SERVE"
must_fail RUNTIME_MODE=no-spec PREFILL_PROJ_FLASHINFER=0 ATLAS_PREFILL_PROJ_FLASHINFER=1 "$SERVE"
must_fail RUNTIME_MODE=no-spec PREFILL_SSM_FLASHINFER= "$SERVE"
must_fail RUNTIME_MODE=no-spec PREFILL_SSM_FLASHINFER=true "$SERVE"
must_fail RUNTIME_MODE=no-spec PREFILL_SSM_FLASHINFER=0 ATLAS_PREFILL_SSM_FLASHINFER=1 "$SERVE"
must_fail RUNTIME_MODE=no-spec ATTN_QKV_M17_ASTAGE= "$SERVE"
must_fail RUNTIME_MODE=no-spec ATLAS_ATTN_QKV_EXACT_M17_ASTAGE= "$SERVE"
must_fail RUNTIME_MODE=no-spec ATTN_QKV_M17_ASTAGE=true "$SERVE"
must_fail RUNTIME_MODE=no-spec ATTN_QKV_M17_ASTAGE=0 ATLAS_ATTN_QKV_EXACT_M17_ASTAGE=1 "$SERVE"
must_fail RUNTIME_MODE=no-spec GDN_C143_PREFILL= "$SERVE"
must_fail RUNTIME_MODE=no-spec ATLAS_GDN_C143_PREFILL=true "$SERVE"
must_fail RUNTIME_MODE=no-spec GDN_C143_PREFILL=0 ATLAS_GDN_C143_PREFILL=1 "$SERVE"
must_fail RUNTIME_MODE=no-spec GDN_C143_PREFILL=1 "$SERVE"
must_fail RUNTIME_MODE=no-spec GDN_C143_PREFILL=1 GDN_C143_SM121_LIB=relative.so "$SERVE"
must_fail RUNTIME_MODE=no-spec GDN_C143_PREFILL=1 GDN_C143_SM121_LIB=/bin/sh \
  ATLAS_GDN_C143_SM121_LIB=/bin/true "$SERVE"
must_fail RUNTIME_MODE=no-spec GDN_C143_PREFILL=1 GDN_C143_SM121_LIB=/bin/sh \
  GDN_PREFILL_GATECACHE=1 "$SERVE"
must_fail RUNTIME_MODE=no-spec PREFILL_FFN_FLASHINFER=1 E2M1_STATIC_SCALE=1 \
  FLASHINFER_SM121_LIB=relative.so "$SERVE"
must_fail RUNTIME_MODE=no-spec PREFILL_FFN_FLASHINFER=1 "$SERVE"
must_fail RUNTIME_MODE=no-spec PREFILL_PROJ_FLASHINFER=1 "$SERVE"
must_fail RUNTIME_MODE=no-spec PREFILL_SSM_FLASHINFER=1 "$SERVE"
must_fail DRAFT=/draft PREFILL_FFN_DUAL_FUSED= "$SERVE_V3"
must_fail DRAFT=/draft ATLAS_PREFILL_FFN_DUAL_FUSED=true "$SERVE_V3"
must_fail DRAFT=/draft PREFILL_FFN_DUAL_FUSED=0 ATLAS_PREFILL_FFN_DUAL_FUSED=1 "$SERVE_V3"
must_fail PREFILL_FFN_DUAL_FUSED= "$SERVE_NO_SPEC"
must_fail PREFILL_ATTN_GATE_FUSED= "$SERVE_NO_SPEC"
must_fail ATLAS_PREFILL_ATTN_GATE_FUSED= "$SERVE_NO_SPEC"
must_fail PREFILL_ATTN_GATE_FUSED=true "$SERVE_NO_SPEC"
must_fail PREFILL_ATTN_GATE_FUSED=0 ATLAS_PREFILL_ATTN_GATE_FUSED=1 "$SERVE_NO_SPEC"
must_fail DRAFT=/draft PREFILL_ATTN_GATE_FUSED= "$SERVE_V3"
must_fail DRAFT=/draft ATLAS_PREFILL_ATTN_GATE_FUSED= "$SERVE_V3"
must_fail DRAFT=/draft PREFILL_ATTN_GATE_FUSED=true "$SERVE_V3"
must_fail DRAFT=/draft PREFILL_ATTN_GATE_FUSED=0 ATLAS_PREFILL_ATTN_GATE_FUSED=1 "$SERVE_V3"
must_fail RUNTIME_MODE=no-spec MAX_SEQ_LEN=1000001 "$SERVE"
must_fail RUNTIME_MODE=no-spec MAX_SEQ_LEN=999999999999999999999999999999 "$SERVE"
must_fail RUNTIME_MODE=no-spec MAX_SEQ_LEN=1000000 ROPE_THETA_OVERRIDE=41800000 "$SERVE"
must_fail RUNTIME_MODE=no-spec MAX_SEQ_LEN=1000000 ROPE_YARN_FACTOR=2 "$SERVE"
must_fail RUNTIME_MODE=dflash-v3 DRAFT=/draft MAX_SEQ_LEN=1000000 "$SERVE"
must_fail RUNTIME_MODE=no-spec MAX_SEQ_LEN=262144 ROPE_YARN_FACTOR=4 "$SERVE"

nonce="$(printf 'a%.0s' {1..64})"
qualified="$({ invoke DRAFT=/draft ATLAS_QUALIFICATION_RUN_NONCE="$nonce" "$SERVE"; } 2>&1)"
binding_line="$(printf '%s\n' "$qualified" | grep '^ATLAS_QUALIFICATION_RUN pid=')"
fake_line="$(printf '%s\n' "$qualified" | grep '^fake_pid=')"
[[ "$(printf '%s\n' "$qualified" | grep -c '^ATLAS_QUALIFICATION_RUN pid=')" == 1 ]]
[[ "${binding_line#ATLAS_QUALIFICATION_RUN pid=}" == "${fake_line#fake_pid=} nonce=$nonce" ]]

printf 'serve launcher contract: PASS\n'
