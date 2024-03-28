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
    Image tag to use." \
    "By default, use the '${vtag}' tag."
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

# Get the CUDA GPU Compute Capability:
CUDA_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv  | awk 'NR==2' | sed -e 's/\.//g')
if [ ${CUDA_CC} != "" ]; then
  echo -e "\n${INFO}Found CUDA GPU Compute Capability: ${CUDA_CC}"
else
  echo -e "\n${ERROR}Could not find CUDA GPU Compute Capability value!"
  exit 1
fi

# Get the date to estimate docker image build time:
start_time=$(date +%s)
# Start the docker image build:
docker build \
  -t roboticsmicrofarms/colmap:${vtag} ${docker_opts} \
  --build-arg CUDA_ARCHITECTURES=${CUDA_CC} \
  -f docker/colmap3.8/Dockerfile .
# Get docker build exit code:
docker_build_status=$?
# Get elapsed time:
elapsed_time=$(expr $(date +%s) - ${start_time})

# Print build time if successful (code 0), else print exit code
if [ ${docker_build_status} == 0 ]; then
  echo -e "\n${INFO}Docker build SUCCEEDED in ${elapsed_time}s!"
else
  echo -e "\n${ERROR}Docker build FAILED after ${elapsed_time}s with code ${docker_build_status}!"
fi
# Exit with docker build exit code:
exit ${docker_build_status}
