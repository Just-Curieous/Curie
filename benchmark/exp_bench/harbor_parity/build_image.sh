#!/usr/bin/env bash
# Build the upstream EXP-Bench image (ExpDockerfile_default) with the versions used for
# Harbor parity. Run from benchmark/exp_bench. Extra arguments go to `docker build`
# (e.g. --platform linux/amd64 on Apple silicon, or --build-arg OPENHANDS_REF=<ref>).
#
# The pins replace upstream's floating versions (miniconda3:latest, npm@latest, latest
# conda nodejs/poetry), which no longer build OpenHands 02bc7de in late 2026; they are
# the releases current when that OpenHands commit was made (Feb-Mar 2025).
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE="${IMAGE:-exp-bench-image}"
OPENHANDS_REF="${OPENHANDS_REF:-02bc7de36dda10d890d39ac05fee16a0d55193f6}"

docker build --progress=plain \
  --build-arg BASE_IMAGE="${BASE_IMAGE:-continuumio/miniconda3:25.1.1-2}" \
  --build-arg NPM_VERSION="${NPM_VERSION:-11.1.0}" \
  --build-arg CONDA_NODEJS_SPEC="${CONDA_NODEJS_SPEC:-nodejs=22.13.0}" \
  --build-arg CONDA_POETRY_SPEC="${CONDA_POETRY_SPEC:-poetry=2.1.1}" \
  --build-arg OPENHANDS_REF="${OPENHANDS_REF}" \
  --build-arg AZURE_ORG_PATCH="${AZURE_ORG_PATCH:-0}" \
  --build-arg OPENHANDS_DOCKER_SOCK_PATCH="${OPENHANDS_DOCKER_SOCK_PATCH:-1}" \
  -t "${IMAGE}" -f ExpDockerfile_default "$@" .

docker run --rm --entrypoint cat "${IMAGE}" /helper_openhands/OPENHANDS_COMMIT
