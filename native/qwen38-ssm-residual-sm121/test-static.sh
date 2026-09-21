#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-only
set -eu
unset ENV BASH_ENV PYTHONHOME PYTHONPATH
case $0 in
  */*) script_dir=${0%/*} ;;
  *) script_dir=. ;;
esac
native_dir=$(CDPATH= cd -- "${script_dir}" && pwd -P)
export PYTHONDONTWRITEBYTECODE=1

/usr/bin/python3 -m unittest -v \
  "${native_dir}/test_static.py" \
  "${native_dir}/test_build_provenance.py"
/bin/sh -n "${native_dir}/build.sh" "${native_dir}/test-static.sh"

for file in \
  "${native_dir}/include/atlas_qwen38_ssm_residual.h" \
  "${native_dir}/src/atlas_qwen38_ssm_residual.cu" \
  "${native_dir}/test_static.py"; do
  test "$(/usr/bin/wc -l <"${file}")" -le 250
  /usr/bin/grep -q 'SPDX-License-Identifier:' "${file}"
done

/usr/bin/grep -q -- '--fmad=false' "${native_dir}/build_provenance.py"
/usr/bin/grep -q 'code=sm_121a' "${native_dir}/build_provenance.py"
artifact=$(/usr/bin/find "${native_dir}" -type f \( -name '*.so' -o -name '*.o' -o -name '*.ptx' \) -print -quit)
if test -n "${artifact}"; then
  echo "unexpected compiled artifact in source tree" >&2
  exit 1
fi

echo "PASS: Qwen3.8 SSM two-pass residual source contract"
