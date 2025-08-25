#!/bin/bash

# --------------------------------
# Functions for colors and messages
# --------------------------------
setup_colors() {
  RED="\033[0;31m"    # Define red color code
  GREEN="\033[0;32m"  # Define green color code
  YELLOW="\033[0;33m" # Define yellow color code
  NC="\033[0m"        # No Color code to reset colors
  INFO="${GREEN}INFO${NC}    "    # Prefix for info messages
  WARNING="${YELLOW}WARNING${NC} " # Prefix for warning messages
  ERROR="${RED}$(bold ERROR)${NC}   " # Prefix for error messages using bold function
}

bold() {
  echo -e "\e[1m$*\e[0m" # Make text bold and reset
}

log_info() {
  echo -e "${INFO}$1" # Print info message with INFO prefix
}

log_warning() {
  echo -e "${WARNING}$1" # Print warning message with WARNING prefix
}

log_error() {
  echo -e "${ERROR}$1" # Print error message with ERROR prefix
}

# --------------------------------
# Functions for script initialization
# --------------------------------
initialize_variables() {
  # Image tag to use, 'latest' by default:
  VTAG="latest"
  # String aggregating the docker build options to use:
  DOCKER_OPTS=""
  # Default Colmap version to use:
  COLMAP_VERSION="3.8"
  # Default CUDA Compute Capability is empty (to enable automatic search):
  CUDA_CC=""
  # PYCUDA NVCC flags
  PYCUDA_NVCC_FLAGS=""
}

# --------------------------------
# Check for required dependencies
# --------------------------------
check_dependencies() {
  if ! command -v docker >/dev/null 2>&1; then
    log_error "Docker is not installed or not found!"
    exit 1
  fi
}

# --------------------------------
# Usage information function
# --------------------------------
show_usage() {
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
    "By default, use the '${VTAG}' tag."
  echo "  --colmap
    The version of Colmap to use." \
    "By default, use '${COLMAP_VERSION}'."
  echo "  --cuda-cc
    The CUDA Compute Capability value to use." \
    "By default, try to guess it from the system."
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

# --------------------------------
# Command line parsing function
# --------------------------------
parse_arguments() {
  while [ "$1" != "" ]; do
    case $1 in
    -t | --tag)
      shift
      VTAG=$1
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
      DOCKER_OPTS="${DOCKER_OPTS} --no-cache"
      ;;
    --pull)
      DOCKER_OPTS="${DOCKER_OPTS} --pull"
      ;;
    --plain)
      DOCKER_OPTS="${DOCKER_OPTS} --progress=plain"
      ;;
    -h | --help)
      show_usage
      exit 0
      ;;
    *)
      show_usage
      exit 1
      ;;
    esac
    shift
  done
}

# --------------------------------
# CUDA Compute Capability setup
# --------------------------------
setup_cuda_compute_capability() {
  # If CUDA_CC is not set, attempt to derive it:
  if [ -z "${CUDA_CC}" ]; then
    if ! command -v nvidia-smi >/dev/null 2>&1; then
      log_error "nvidia-smi is not installed or not found!"
      exit 1
    fi
    CUDA_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv | awk 'NR==2' | sed -e 's/\.//g')
    if [ -z "${CUDA_CC}" ] || ! [[ "${CUDA_CC}" =~ ^[0-9]+$ ]]; then
      log_error "Failed to determine CUDA GPU Compute Capability!"
      exit 1
    fi
    log_info "Found CUDA GPU Compute Capability: ${CUDA_CC}"
  else
    log_info "Using provided CUDA GPU Compute Capability: ${CUDA_CC}"
  fi
}

# --------------------------------
# Check and pull base image
# --------------------------------
check_base_image() {
  # Check if required base image exists or can be pulled
  base_image="roboticsmicrofarms/colmap:${COLMAP_VERSION}-cuda_cc${CUDA_CC}"
  log_info "Checking for base image: ${base_image}..."

  if ! docker image inspect "${base_image}" >/dev/null 2>&1; then
    log_warning "Base image not found locally, attempting to pull..."
    if ! docker pull "${base_image}" >/dev/null 2>&1; then
      log_error "Failed to pull required base image: ${base_image}"
      log_error "Please ensure the image exists and you have internet connectivity."
      log_info "Alternatively, you can build it from 'colmap${COLMAP_VERSION}/' directory."
      exit 1
    fi
    log_info "Successfully pulled base image."
  else
    log_info "Found the required base image."
  fi
}

# --------------------------------
# Determine CUDA version and setup NVCC flags
# --------------------------------
setup_cuda_nvcc_flags() {
  base_image="roboticsmicrofarms/colmap:${COLMAP_VERSION}-cuda_cc${CUDA_CC}"

  CUDA_VERSION=$(docker run -t --rm --gpus all --entrypoint bash ${base_image} -c "nvidia-smi -q | grep 'CUDA Version' | awk '{print \$3}'")
  # Check if CUDA_VERSION is a number
  if ! [[ "${CUDA_VERSION}" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    log_warning "Could not parse CUDA version!"
    log_info "Using default PYCUDA_NVCC_FLAGS"
    PYCUDA_NVCC_FLAGS=""
  else
    log_info "Found CUDA version in base image: ${CUDA_VERSION}"
    # Extract major version for comparison
    CUDA_MAJOR_VERSION=$(echo "${CUDA_VERSION}" | cut -d. -f1)

    # If CUDA_VERSION is greater than 12, `nvcc` arch can not be greater than 86
    if [ "${CUDA_MAJOR_VERSION}" -ge "12" ] && [ "${CUDA_CC}" -ge "86" ]; then
      log_info "Setting PYCUDA_NVCC_FLAGS for CUDA CC > 86"
      PYCUDA_NVCC_FLAGS="-arch=sm_86"
    else
      log_info "Using default PYCUDA_NVCC_FLAGS"
      PYCUDA_NVCC_FLAGS=""
    fi
  fi
}

# --------------------------------
# Build Docker image
# --------------------------------
build_docker_image() {
  # Construct the docker build command
  docker_cmd="docker build"
  docker_cmd+=" --build-arg COLMAP_VERSION=\"${COLMAP_VERSION}\""
  docker_cmd+=" --build-arg CUDA_CC=\"${CUDA_CC}\""
  docker_cmd+=" --build-arg PYCUDA_NVCC_FLAGS=\"${PYCUDA_NVCC_FLAGS}\""
  docker_cmd+=" -t \"roboticsmicrofarms/plant-3d-vision:${VTAG}-cuda_cc${CUDA_CC}\""
  docker_cmd+=" ${DOCKER_OPTS}"  # Additional options like --no-cache, --pull, etc.
  docker_cmd+=" -f \"docker/Dockerfile\""
  docker_cmd+=" ."  # Build context

  # Print the full command that will be executed
  log_info "Executing command: ${docker_cmd}"

  # Get the date to estimate docker image build time:
  start_time=$(date +%s)

  # Execute the docker build command
  eval ${docker_cmd}

  # Get docker build exit code:
  docker_build_status=$?

  # Get elapsed time:
  elapsed_time=$(($(date +%s) - start_time))

  # Print build time if successful (code 0), else print exit code
  if [ ${docker_build_status} -eq 0 ]; then
    log_info "Docker build SUCCEEDED in ${elapsed_time}s!"
  else
    log_error "Docker build FAILED after ${elapsed_time}s with code ${docker_build_status}!"
  fi

  # Exit with docker build exit code:
  exit ${docker_build_status}
}

# --------------------------------
# Main script execution
# --------------------------------
main() {
  setup_colors
  check_dependencies
  initialize_variables
  parse_arguments "$@"
  setup_cuda_compute_capability
  check_base_image
  setup_cuda_nvcc_flags
  build_docker_image
}

# Execute main function with all arguments
main "$@"