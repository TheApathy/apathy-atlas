#!/usr/bin/env bash
# Exact no-speculation control launcher for prefill qualification.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ -v RUNTIME_MODE && "$RUNTIME_MODE" != no-spec ]]; then
  echo "serve-no-spec.sh requires RUNTIME_MODE=no-spec" >&2
  exit 2
fi
if [[ -n "${DRAFT-}" ]]; then
  echo "serve-no-spec.sh requires DRAFT to be unset or empty" >&2
  exit 2
fi
export RUNTIME_MODE=no-spec

# Keep every target-side flag identical to the measured V3 candidate; only the
# canonical DFlash CLI tuple and runtime-only environment entries may differ.
# Dynamic path; the sourced profile is checked separately.
# shellcheck disable=SC1091
source "$HERE/serve-v3-target-profile.sh"

exec "$HERE/serve.sh" "$@"
