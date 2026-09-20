#!/usr/bin/env bash
set -euo pipefail

native_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
fi_data_root=${FLASHINFER_DATA_ROOT:-/home/flocka/.local/lib/python3.12/site-packages/flashinfer/data}
fi_license=${FLASHINFER_LICENSE_FILE:-/home/flocka/.local/lib/python3.12/site-packages/flashinfer_python-0.6.6.dist-info/licenses/LICENSE}

while read -r expected relative; do
  [[ -z "${expected}" || "${expected}" == \#* ]] && continue
  actual=$(sha256sum "${fi_data_root}/${relative}" | awk '{print $1}')
  if [[ "${actual}" != "${expected}" ]]; then
    echo "dependency hash mismatch: ${relative}: expected ${expected}, got ${actual}" >&2
    exit 1
  fi
done < "${native_dir}/DEPENDENCIES.sha256"

expected_fi_license=cb67c224f503e0a063908950b12f89a7280c6e527dcffac972aa114e4bf3c5de
actual_fi_license=$(sha256sum "${fi_license}" | awk '{print $1}')
if [[ "${actual_fi_license}" != "${expected_fi_license}" ]]; then
  echo "FlashInfer license hash mismatch" >&2
  exit 1
fi

cmp "${fi_license}" "${native_dir}/LICENSES/Apache-2.0.txt"
rg -q 'SPDX-License-Identifier: BSD-3-Clause' \
  "${native_dir}/LICENSES/BSD-3-Clause.txt"
