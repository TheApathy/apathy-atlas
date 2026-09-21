#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-only
set -eu
unset ENV BASH_ENV PYTHONHOME PYTHONPATH
case $0 in
  */*) script_dir=${0%/*} ;;
  *) script_dir=. ;;
esac
native_dir=$(CDPATH= cd -- "${script_dir}" && pwd -P)
exec /usr/bin/env -i PATH=/usr/bin:/bin LC_ALL=C \
  /usr/bin/python3 "${native_dir}/build_provenance.py" "$@"
