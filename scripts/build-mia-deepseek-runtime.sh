#!/usr/bin/env bash
# Rebuild the audited one-Spark Mia/EXL3 runtime without starting a GPU job.
set -euo pipefail

UPSTREAM_URL="${MIA_UPSTREAM_URL:-https://github.com/tpurtell/ds4-mia-exl3-k2-1spark.git}"
UPSTREAM_COMMIT="${MIA_UPSTREAM_COMMIT:-f20b97dfd7666c00c316f29542e2e53f33cabb19}"
IMAGE="${MIA_IMAGE:-ds4-mia-exl3-k2-1spark:local}"
BUILD_DIR="${MIA_BUILD_DIR:-}"

if [[ -z "$BUILD_DIR" ]]; then
  BUILD_DIR=$(mktemp -d -t atlas-mia-build.XXXXXX)
  trap 'rm -rf -- "$BUILD_DIR"' EXIT
fi

git init -q "$BUILD_DIR"
git -C "$BUILD_DIR" remote remove origin >/dev/null 2>&1 || true
git -C "$BUILD_DIR" remote add origin "$UPSTREAM_URL"
git -C "$BUILD_DIR" fetch -q --depth=1 origin "$UPSTREAM_COMMIT"
git -C "$BUILD_DIR" checkout -q --detach FETCH_HEAD

actual=$(git -C "$BUILD_DIR" rev-parse HEAD)
[[ "$actual" == "$UPSTREAM_COMMIT" ]] || {
  echo "Mia source mismatch: expected $UPSTREAM_COMMIT, got $actual" >&2
  exit 1
}

docker build --progress=plain --label "org.atlas.mia.commit=$actual" -t "$IMAGE" "$BUILD_DIR"
docker image inspect "$IMAGE" \
  --format 'image={{.Id}} bytes={{.Size}} source={{index .Config.Labels "org.opencontainers.image.source"}} mia_commit={{index .Config.Labels "org.atlas.mia.commit"}}'
