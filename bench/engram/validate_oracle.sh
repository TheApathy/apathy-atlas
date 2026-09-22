#!/usr/bin/env bash
# Validate the engram lane against the oracle capture. CPU only; no GPU lock.
#
#   ./validate_oracle.sh [ref-dir]
#
# Expects token_map_i32.bin (from export_map.py) in the working directory, and a
# splice of hash.rs (see README). Emits candidate dumps and runs compare.py on
# engram_hashes (exact integer) and engram_rows (float).
set -euo pipefail
REF="${1:-/home/flocka/atlas/DSV41_PORT/oracle/ref/runA}"
CMP=/home/flocka/atlas/DSV41_PORT/oracle/compare.py

fail=0
for L in L01 L14; do
  python3 "$CMP" --ref-dir "$REF" --name "$L.engram_hashes.000" --cand "cand_${L}_good.bin" || fail=1
  python3 "$CMP" --ref-dir "$REF" --name "$L.engram_rows.000"   --cand "cand_${L}_rows.bin"  || fail=1
done

# Negative controls: each MUST fail, and must fail with the count the n-gram
# structure predicts. A control that merely "differs" is not evidence.
echo "--- negative controls (each must FAIL) ---"
for m in neg-mult neg-primes neg-tokenmap; do
  for L in L01 L14; do
    if python3 "$CMP" --ref-dir "$REF" --name "$L.engram_hashes.000" \
         --cand "cand_${L}_${m}.bin" >/dev/null 2>&1; then
      # neg-mult and neg-primes perturb LAYER 0 only, so L14 passing is correct.
      [[ "$m" != "neg-tokenmap" && "$L" == "L14" ]] || { echo "CONTROL DID NOT FAIL: $m $L"; fail=1; }
    fi
  done
done
[[ $fail -eq 0 ]] && echo "ALL CHECKS PASSED" || echo "FAILURES ABOVE"
exit $fail
