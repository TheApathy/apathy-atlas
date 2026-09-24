#!/usr/bin/env bash
# usage: run_arms.sh <bin> <outroot> <arm>...   (run inside gpu_window.sh)
# arms: prefill | prefill256 | control256 | decode | control-decode | final
set -uo pipefail
BIN=${1:?bin}; ROOT=${2:?outroot}; shift 2
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
P=$HERE/prompts_1024.json
ESC=ATLAS_GLM53_UNVALIDATED_BRINGUP=1
CTL=ATLAS_GLM53_NEGATIVE_CONTROL=skip-kda-commit
R256=ATLAS_GLM53_LAYER_MAJOR_PREFILL_ROWS=256
# RECIPE=<yaml> selects a recipe other than this worktree's built-in one.
RARG=(); [ -n "${RECIPE:-}" ] && RARG=(--recipe "$RECIPE")
rc=0
for arm in "$@"; do
  echo "=== arm $arm $(date +%T)"
  case $arm in
    prefill)        python3 $HERE/atlas_driver.py "${RARG[@]}" --bin "$BIN" --out "$ROOT/$arm" --prefill $P --env $ESC ;;
    prefill256)     python3 $HERE/atlas_driver.py "${RARG[@]}" --bin "$BIN" --out "$ROOT/$arm" --prefill $P --env $ESC --env $R256 ;;
    control256)     python3 $HERE/atlas_driver.py "${RARG[@]}" --bin "$BIN" --out "$ROOT/$arm" --prefill $P --env $ESC --env $R256 --env $CTL ;;
    decode)         python3 $HERE/atlas_driver.py "${RARG[@]}" --bin "$BIN" --out "$ROOT/$arm" --decode $P --env $ESC ;;
    control-decode) python3 $HERE/atlas_driver.py "${RARG[@]}" --bin "$BIN" --out "$ROOT/$arm" --decode $P --env $ESC --env $CTL ;;
    # Long-context arms: the recipe's own defaults must cover the prompt (RECIPE=16k yaml).
    long4096)       python3 $HERE/atlas_driver.py "${RARG[@]}" --bin "$BIN" --out "$ROOT/$arm" --prefill $HERE/prompts_4096.json ;;
    long8192)       python3 $HERE/atlas_driver.py "${RARG[@]}" --bin "$BIN" --out "$ROOT/$arm" --prefill $HERE/prompts_8192.json ;;
    long16000)      python3 $HERE/atlas_driver.py "${RARG[@]}" --bin "$BIN" --out "$ROOT/$arm" --prefill $HERE/prompts_16000.json ;;
    # No escape, no extra switches: the recipe exactly as shipped.
    final)          python3 $HERE/atlas_driver.py "${RARG[@]}" --bin "$BIN" --out "$ROOT/$arm" --prefill $P ;;
    final-decode)   python3 $HERE/atlas_driver.py "${RARG[@]}" --bin "$BIN" --out "$ROOT/$arm" --decode $P ;;
    *) echo "unknown arm $arm"; exit 2 ;;
  esac
  r=$?; echo "=== arm $arm rc=$r $(date +%T)"; [ $r -ne 0 ] && rc=$r
done
exit $rc
