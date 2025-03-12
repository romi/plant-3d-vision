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
vtag="latest"
# String aggregating the docker build options to use:
docker_opts=""
# Default Colmap version to use:
COLMAP_VERSION="3.8"
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
  echo "  Build a docker image named 'roboticsmicrofarms/plant-3d-vision' using 'Dockerfile' in the same location.

  It must be run from the 'plant-3d-vision' repository root folder as it is the build context and it will be copied during at image build time!
  Do not forget to initialize or update the sub-modules if necessary!"
  echo ""

  echo -e "$(bold OPTIONS):"
  echo "  -t, --tag
    Image tag to use." \
    "By default, use the '${vtag}' tag."
  echo "  --colmap
    The version of Colmap to use." \
    "By default, use '${COLMAP_VERSION}'."
  echo "  --cuda-cc
    The CUDA Compute Capability value to use." \
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
  --colmap)
    shift
    COLMAP_VERSION=$1
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

# Check if required base image exists or can be pulled
base_image="roboticsmicrofarms/colmap:${COLMAP_VERSION}-cuda_cc${CUDA_CC}"
echo -e "${INFO}Checking for base image: ${base_image}..."

if ! docker image inspect "${base_image}" >/dev/null 2>&1; then
    echo -e "${WARNING}Base image not found locally, attempting to pull..."
    if ! docker pull "${base_image}" >/dev/null 2>&1; then
        echo -e "${ERROR}Failed to pull required base image: ${base_image}"
        echo -e "${ERROR}Please ensure the image exists and you have internet connectivity."
        echo -e "${INFO}Alternatively, you can build it from 'colmap${COLMAP_VERSION}/' directory."
        exit 1
    fi
    echo -e "${INFO}Successfully pulled base image"
else
  echo -e "Done!"
fi


# Get the date to estimate docker image build time:
start_time=$(date +%s)
# Start the docker image build:
docker build \
  --build-arg COLMAP_VERSION="${COLMAP_VERSION}" \
  --build-arg CUDA_CC="${CUDA_CC}" \
  -t "roboticsmicrofarms/plant-3d-vision:${vtag}-cuda_cc${CUDA_CC}" ${docker_opts} \
  -f "docker/Dockerfile" .
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
