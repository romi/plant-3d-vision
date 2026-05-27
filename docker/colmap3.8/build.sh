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
  # Image tag to use, '3.8' by default:
  VTAG="3.8"
  # String aggregating the docker build options to use:
  DOCKER_OPTS=""
  # Default Ubuntu version
  UBUNTU_VERSION="22.04"
  # Default CUDA Compute Capability is empty (to enable automatic detection):
  CUDA_CC=""
  # Default NVIDIA CUDA Version is empty (to enable automatic detection):
  NVIDIA_CUDA_VERSION=""
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
  echo "  Build a docker image named 'roboticsmicrofarms/colmap' using 'Dockerfile' in the same location.

  It must be run from the 'plant-3d-vision' repository root folder as it is the build context and it will be copied during at image build time!
  Do not forget to initialize or update the sub-modules if necessary!"
  echo ""

  echo -e "$(bold OPTIONS):"
  echo "  -t, --tag
    Image tag to use. Note that a '-cuda_cc\${CUDA_CC}' suffix will be added." \
    "By default, use the '${VTAG}' tag."
  echo "  --cuda-cc
    The CUDA Compute Capability value to use to build Colmap." \
    "By default, try to guess it from the system."
  echo "  --cuda-version
    The CUDA version to use to build Colmap." \
    "By default, try to guess it from the system."
  echo "  --ubuntu-version
    The Ubuntu version to use to build Colmap." \
    "By default, use '${UBUNTU_VERSION}'."
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

  echo "$(bold DOCKER OPTIONS):"
  echo "Any other option will be passed to the 'docker buildx build' command."
}

# --------------------------------
# Command line parsing function
# --------------------------------
parse_arguments() {
  docker_option=""
  while [ "$1" != "" ]; do
    case $1 in
    -t | --tag)
      shift
      VTAG=$1
      ;;
    --cuda-cc)
      shift
      CUDA_CC=$1
      ;;
    --cuda-version)
      shift
      NVIDIA_CUDA_VERSION=$1
      ;;
    --ubuntu-version)
      shift
      UBUNTU_VERSION=$1
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
      DOCKER_OPTS="${DOCKER_OPTS} $1"
      ;;
    esac
    shift
  done
}

# --------------------------------
# CUDA setup functions
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

setup_cuda_version() {
  # If NVIDIA_CUDA_VERSION is not set, attempt to derive it:
  if [ -z "${NVIDIA_CUDA_VERSION}" ]; then
    # Check if nvidia-smi exists
    if ! command -v nvidia-smi &> /dev/null; then
      log_error "nvidia-smi command not found. Please install NVIDIA drivers."
      NVIDIA_CUDA_VERSION="11.8.0" # Default fallback version
      log_warning "Assuming default CUDA version: ${NVIDIA_CUDA_VERSION}."
    else
      # Extract CUDA version from nvidia-smi output
      NVIDIA_CUDA_VERSION=$(nvidia-smi -q 2>/dev/null | grep 'CUDA Version' | awk '{print $4}')
      if [ -z "${NVIDIA_CUDA_VERSION}" ]; then
        log_error "Failed to determine host NVIDIA CUDA Version using nvidia-smi!"
        NVIDIA_CUDA_VERSION="11.8.0" # Default fallback version
        log_warning "Assuming default host CUDA version: ${NVIDIA_CUDA_VERSION}."
      else
        log_info "Found host NVIDIA CUDA Version: ${NVIDIA_CUDA_VERSION}"
      fi
    fi
  else
    log_info "Using provided NVIDIA CUDA Version: ${NVIDIA_CUDA_VERSION}"
  fi

  # Check if the detected version is above 11.8.0
  IFS='.' read -r -a cuda_version_array <<< "$NVIDIA_CUDA_VERSION"
  IFS='.' read -r -a max_cuda_version_array <<< "11.8.0"

  for ((i=0; i<${#cuda_version_array[@]}; i++)); do
    if [ "${cuda_version_array[i]}" -gt "${max_cuda_version_array[i]}" ]; then
      log_info "Colmap3.8 works with a max CUDA version of 11.8.0."
      NVIDIA_CUDA_VERSION="11.8.0"
      break
    elif [ "${cuda_version_array[i]}" -lt "${max_cuda_version_array[i]}" ]; then
      break
    fi
  done

  # Properly format version to ensure major.minor.patch format
  # Count the number of dots in the version string
  dot_count=$(echo "${NVIDIA_CUDA_VERSION}" | tr -cd '.' | wc -c)

  if [ "$dot_count" -eq 0 ]; then
    # Only major version (e.g., "11")
    NVIDIA_CUDA_VERSION="${NVIDIA_CUDA_VERSION}.0.0"
  elif [ "$dot_count" -eq 1 ]; then
    # Only major.minor (e.g., "11.8")
    NVIDIA_CUDA_VERSION="${NVIDIA_CUDA_VERSION}.0"
  fi

  log_info "Final NVIDIA CUDA Version: ${NVIDIA_CUDA_VERSION}"
}


check_and_fix_base_image() {
  local ubuntu_version="${UBUNTU_VERSION}"
  local cuda_version="${NVIDIA_CUDA_VERSION}"
  local base_image="nvidia/cuda:${cuda_version}-devel-ubuntu${ubuntu_version}"

  log_info "Checking if base image ${base_image} exists..."

  if docker manifest inspect "${base_image}" >/dev/null 2>&1; then
    log_info "Base image ${base_image} found."
    return 0
  else
    log_warning "Base image ${base_image} not found in registry!"
    log_info "Searching for alternative images..."

    log_error "No suitable alternative found. Please check available images at https://hub.docker.com/r/nvidia/cuda/tags"
    return 1
  fi
}

# --------------------------------
# Docker build function
# --------------------------------
build_docker_image() {
  # Construct the docker build command
  docker_cmd="docker buildx build"
  docker_cmd+=" --build-arg NVIDIA_CUDA_VERSION=\"${NVIDIA_CUDA_VERSION}\""
  docker_cmd+=" --build-arg CUDA_ARCHITECTURES=\"${CUDA_CC}\""
  docker_cmd+=" --build-arg UBUNTU_VERSION=\"${UBUNTU_VERSION}\""
  docker_cmd+=" -t \"roboticsmicrofarms/colmap:${VTAG}-cuda_cc${CUDA_CC}\""
  docker_cmd+=" ${DOCKER_OPTS}"  # Additional options like --no-cache, --pull, etc.
  docker_cmd+=" -f \"docker/colmap3.8/Dockerfile\""
  docker_cmd+=" ."  # Build context

  # Print the full command that will be executed
  log_info "Executing command: ${docker_cmd}"

  # Get the date to estimate docker image build time:
  start_time=$(date +%s)
  
  # Start the docker image build:
  eval ${docker_cmd}
  
  # Get docker build exit code:
  docker_build_status=$?
  
  # Get elapsed time:
  elapsed_time=$(($(date +%s) - start_time))

  # Print build time if successful (code 0), else print exit code
  if [ ${docker_build_status} -eq 0 ]; then
    log_info "Docker build SUCCEEDED in ${elapsed_time}s!"
    # Export the tag for GitHub Actions (if running in CI)
    if [ -n "${GITHUB_ENV}" ]; then
        echo "TAG=${VTAG}-cuda_cc${CUDA_CC}" >> "${GITHUB_ENV}"
    fi
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
  setup_cuda_version
  check_and_fix_base_image || exit 1
  build_docker_image
}

# Execute main function with all arguments
main "$@"