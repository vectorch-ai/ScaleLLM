#!/bin/bash
#  adapted from https://github.com/tensorflow/serving

# ==============================================================================
#
# Script to run commands (like builds and scripts) in a docker container.
#
#
# Note: This script binds your working directory (via pwd) and /tmp to the
# Docker container. Any scripts or programs you run will need to have its
# output files/dirs written to one of the above locations for persistence.
#
# Typical usage (to build from lastest upstream source):
# $ git clone --recursive https://github.com/vectorch-ai/ScaleLLM.git
# $ cd ScaleLLM
# $ ./tools/run_in_docker.sh cmake -G Ninja -S . -B build
# $ ./tools/run_in_docker.sh cmake --build build --target all

set -e

function usage() {
  local progname=$(basename $0)
  echo "Usage:"
  echo "  ${progname} [-d <docker-image-name>] [-o <docker-run-options>] <command> [args ...]"
  echo ""
  echo "Examples:"
  echo "  ${progname} cmake -G Ninja -S . -B build"
  echo "  ${progname} cmake --build build --target all"
  exit 1
}

function get_switch_user_cmd() {
  local uid=$(id -u)
  local gid=$(id -g)
  local username=$(id -n -u)
  local groupname=$(id -n -g)
  local cmdline="groupadd -f ${groupname} && groupmod -o -g ${gid} ${groupname}"
  cmdline+="; id -u ${username} &>/dev/null || useradd -N ${username} && usermod -o -u ${uid} -g ${gid} ${username}"
  cmdline+="; chroot --userspec=${username} / "
  echo "${cmdline}"
}

(( $# < 1 )) && usage

IMAGE="vectorchai/scalellm:devel"
RUN_OPTS=()
while [[ $# > 1 ]]; do
  case "$1" in
    -d)
      IMAGE="$2"; shift 2;;
    -o)
      RUN_OPTS=($2); shift 2;;
    *)
      break;;
  esac
done

RUN_OPTS+=(--rm -it --network=host)
# Map the working directory and /tmp to allow scripts/binaries to run and also
# output data that might be used by other scripts/binaries
RUN_OPTS+=("-v $(pwd):$(pwd)")
RUN_OPTS+=("-v /tmp:/tmp")
CMD="bash -c 'cd $(pwd); VCPKG_DEFAULT_BINARY_CACHE=$(pwd)/.vcpkg_bincache $@'"

[[ "${CMD}" = "" ]] && usage
[[ ! -x $(command -v docker) ]] && echo "ERROR: 'docker' command missing from PATH." && usage

echo "== Pulling docker image: ${IMAGE}"
if ! docker pull ${IMAGE} ; then
  echo "WARNING: Failed to docker pull image ${IMAGE}"
fi

echo "docker run ${RUN_OPTS[@]} ${IMAGE} bash -c \"$(get_switch_user_cmd) ${CMD}\""
docker run ${RUN_OPTS[@]} ${IMAGE} bash -c "$(get_switch_user_cmd) ${CMD}"
