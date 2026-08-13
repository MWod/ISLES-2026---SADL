#!/usr/bin/env bash
#
# Saves the image and its 11-member model tarball for Grand Challenge upload.

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DOCKER_IMAGE_TAG="isles2026_sadl"

export DOCKER_CLI_HINTS=false

log() { printf "> %s\n" "$1"; }

log "(Re)build the image"
export DOCKER_QUIET_BUILD=1
source "${SCRIPT_DIR}/do_build.sh"

build_timestamp=$( docker inspect --format='{{ .Created }}' "$DOCKER_IMAGE_TAG")
if [ -z "$build_timestamp" ]; then
    log "ERROR: unable to inspect image $DOCKER_IMAGE_TAG"
    exit 1
fi
container_tarball="${DOCKER_IMAGE_TAG}_${build_timestamp//:/-}.tar.gz"
container_tarball="${SCRIPT_DIR}/${container_tarball}"

log "Save image to ${container_tarball}"
docker save "$DOCKER_IMAGE_TAG" | gzip -c > "$container_tarball"
log "Container tarball: $(du -h "$container_tarball" | awk '{print $1}')  ($container_tarball)"

model_tarball="${SCRIPT_DIR}/model.tar.gz"
model_dir="${SCRIPT_DIR}/model"

if [ ! -d "$model_dir" ]; then
    log "ERROR: ${model_dir} missing. Place the extracted model tree at" \
        "${model_dir} (or symlink it there) before running do_save.sh."
    exit 1
fi

log "Pack 11-member model -> ${model_tarball} (following symlinks with -h)"
tar -C "${model_dir}" -czhf "${model_tarball}" .    # -h dereferences symlinks
log "Model tarball: $(du -h "$model_tarball" | awk '{print $1}')  ($model_tarball)"

log "DONE."
log "  Upload $container_tarball  ->  Algorithm image on Grand Challenge"
log "  Upload $model_tarball      ->  Model resource (11-member) linked to that algorithm"
