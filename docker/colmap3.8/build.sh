#!/bin/bash
# - Defines colors and message types:
RED="\033[0;31m"
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
NC="\033[0m" # No Color
INFO="${GREEN}INFO${NC}    "
WARNING="${YELLOW}WARNING${NC} "
ERROR="${RED}ERROR${NC}   "
bold() { echo -e "\e[1m$*\e[0m"; }

# - Default variables
# Image tag to use, 'latest' by default:
vtag="3.8"
# String aggregating the docker build options to use:
docker_opts=""
# Default CUDA Compute Capability is empty (to enable automatic search):
CUDA_CC=""

# Check for required commands:
if ! command -v docker >/dev/null 2>&1; then
  echo -e "${ERROR}Docker is not installed or not found!"
  exit 1
fi

usage() {
  echo -e "$(bold USAGE):"
  echo "  ./docker/build.sh [OPTIONS]"
  echo ""

  echo -e "$(bold DESCRIPTION):"
  echo "  Build a docker image named 'roboticsmicrofarms/colmap' using 'Dockerfile' in the same location.

  It must be run from the 'plant-3d-vision' repository root folder as it is the build context and it will be copied during at image build time!
  Do not forget to initialize or update the sub-modules if necessary!"
  echo ""

  echo -e "$(bold OPTIONS):"
  echo "  -t, --tag
    Image tag to use. Note that a '-cuda_cc\${CUDA_CC}' suffix will be added." \
    "By default, use the '${vtag}' tag."
  echo "  --cuda-cc
    The CUDA Compute Capability value to use to build Colmap." \
    "By default, try to gess it from the system."
  # -- Docker options:
  echo "  --no-cache
    Do not use cache when building the image, (re)start from scratch."
  echo "  --pull
    Always attempt to pull a newer version of the parent image."
  echo "  --plain
    Plain output during docker build."
  # -- General options:
  echo "  -h, --help
    Output a usage message and exit."
}

while [ "$1" != "" ]; do
  case $1 in
  -t | --tag)
    shift
    vtag=$1
    ;;
  --cuda-cc)
    shift
    CUDA_CC=$1
    ;;
  --no-cache)
    docker_opts="${docker_opts} --no-cache"
    ;;
  --pull)
    docker_opts="${docker_opts} --pull"
    ;;
  --plain)
    docker_opts="${docker_opts} --progress=plain"
    ;;
  -h | --help)
    usage
    exit
    ;;
  *)
    usage
    exit 1
    ;;
  esac
  shift
done

# If CUDA_CC is not set, attempt to derive it:
if [ -z "${CUDA_CC}" ]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo -e "${ERROR}nvidia-smi is not installed or not found!"
    exit 1
  fi
  CUDA_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv | awk 'NR==2' | sed -e 's/\.//g')
  if [ -z "${CUDA_CC}" ] || ! [[ "${CUDA_CC}" =~ ^[0-9]+$ ]]; then
    echo -e "${ERROR}Failed to determine CUDA GPU Compute Capability!"
    exit 1
  fi
  echo -e "${INFO}Found CUDA GPU Compute Capability: ${CUDA_CC}"
else
  echo -e "${INFO}Using provided CUDA GPU Compute Capability: ${CUDA_CC}"
fi

# Construct the docker build command
docker_cmd="docker build"
docker_cmd+=" --build-arg CUDA_ARCHITECTURES=\"${CUDA_CC}\""
docker_cmd+=" -t \"roboticsmicrofarms/colmap:${vtag}-cuda_cc${CUDA_CC}\""
docker_cmd+=" ${docker_opts}"  # Additional options like --no-cache, --pull, etc.
docker_cmd+=" -f \"docker/colmap3.8/Dockerfile\""
docker_cmd+=" ."  # Build context

# Print the full command that will be executed
echo -e "${INFO}Executing command: ${docker_cmd}"

# Get the date to estimate docker image build time:
start_time=$(date +%s)
# Start the docker image build:
eval ${docker_cmd}
# Get docker build exit code:
docker_build_status=$?
# Get elapsed time:
elapsed_time=$(($(date +%s) - start_time))

# Print build time if successful (code 0), else print exit code
if [ ${docker_build_status} == 0 ]; then
  echo -e "\n${INFO}Docker build SUCCEEDED in ${elapsed_time}s!"
else
  echo -e "\n${ERROR}Docker build FAILED after ${elapsed_time}s with code ${docker_build_status}!"
fi

# Exit with docker build exit code:
exit ${docker_build_status}
