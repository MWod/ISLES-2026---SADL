#!/usr/bin/env bash
#
# Build the ISLES 2026 SADL container image.
#
# Build context = PROJECT ROOT (not docker/) because the Dockerfile COPYs
# ``third_party/nnUNet`` and ``Code/src/nnunet_isles`` from the tree above.

set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT=$( cd -- "${SCRIPT_DIR}/.." && pwd )
DOCKER_IMAGE_TAG="isles2026_sadl"

docker build \
  --platform=linux/amd64 \
  --tag "$DOCKER_IMAGE_TAG"  \
  --file "${SCRIPT_DIR}/Dockerfile" \
  ${DOCKER_QUIET_BUILD:+--quiet} \
  "$PROJECT_ROOT" 2>&1
