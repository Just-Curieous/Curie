#!/usr/bin/env bash
# Build the upstream image with the build dependencies pinned to their Feb-Mar 2025
# releases; the floating versions upstream uses no longer build OpenHands 02bc7de.
# Run from benchmark/exp_bench. Extra arguments go to docker build.
set -euo pipefail
cd "$(dirname "$0")/.."
docker build \
  --build-arg BASE_IMAGE=continuumio/miniconda3:25.1.1-2 \
  --build-arg NPM_VERSION=11.1.0 \
  --build-arg CONDA_NODEJS_SPEC=nodejs=22.13.0 \
  --build-arg CONDA_POETRY_SPEC=poetry=2.1.1 \
  -t "${IMAGE:-exp-bench-image}" -f ExpDockerfile_default "$@" .
