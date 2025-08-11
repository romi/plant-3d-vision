
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
  vtag="3.8"
  # String aggregating the docker build options to use:
  docker_opts=""
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
    "By default, use the '${vtag}' tag."
  echo "  --cuda-cc
    The CUDA Compute Capability value to use to build Colmap." \
    "By default, try to guess it from the system."
  echo "  --cuda-version
    The CUDA version to use to build Colmap." \
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
      vtag=$1
      ;;
    --cuda-cc)
      shift
      CUDA_CC=$1
      ;;
    --cuda-version)
      shift
      NVIDIA_CUDA_VERSION=$1
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
    # Extract CUDA version from nvidia-smi output
    NVIDIA_CUDA_VERSION=$(nvidia-smi -q | grep 'CUDA Version' | awk '{print $4}')
    if [ -z "${NVIDIA_CUDA_VERSION}" ]; then
      log_error "Failed to determine NVIDIA CUDA Version using nvidia-smi!"
      NVIDIA_CUDA_VERSION="12.9.1" # Default fallback version
      log_warning "Assuming default CUDA version: ${NVIDIA_CUDA_VERSION}."
    else
      log_info "Found NVIDIA CUDA Version: ${NVIDIA_CUDA_VERSION}"
    fi
  else
    log_info "Using provided NVIDIA CUDA Version: ${NVIDIA_CUDA_VERSION}"
  fi

  # Assuming NVIDIA_CUDA_VERSION contains the detected version
  if [[ "${NVIDIA_CUDA_VERSION}" != *.*[*]* ]]; then
    # If only major or major.minor is present, append .0 to make it major.minor.release format
    NVIDIA_CUDA_VERSION="${NVIDIA_CUDA_VERSION}.0"
  fi

  # Split the version into components for comparison
  IFS='.' read -r -a version_parts <<< "$NVIDIA_CUDA_VERSION"
  # Extract major and minor versions
  MAJOR=${version_parts[0]}
  MINOR=${version_parts[1]}
  # Compare with max allowed CUDA version (11.8)
  if (( 10#$MAJOR < 11 || ( 10#$MAJOR == 11 && 10#$MINOR <= 8 ) )); then
    log_info "CUDA version $NVIDIA_CUDA_VERSION is supported."
  else
    log_warning "CUDA version $NVIDIA_CUDA_VERSION exceeds the maximum allowed version of 11.8 for Colmap3.8."
    NVIDIA_CUDA_VERSION="11.8.0"
  fi

  log_info "Final NVIDIA CUDA Version: ${NVIDIA_CUDA_VERSION}"
}

check_and_fix_base_image() {
  ubuntu_version="24.04"
  local cuda_version="${NVIDIA_CUDA_VERSION}"
  local base_image="nvidia/cuda:${cuda_version}-devel-ubuntu${ubuntu_version}"

  log_info "Checking if base image ${base_image} exists..."

  if docker manifest inspect "${base_image}" >/dev/null 2>&1; then
    log_info "Base image ${base_image} found."
    return 0
  else
    log_warning "Base image ${base_image} not found in registry!"
    log_info "Searching for alternative images..."

    # Try finding alternatives with same Ubuntu version but similar CUDA version
    # Extract major.minor from CUDA version (e.g., 12.2.0 -> 12.2)
    local cuda_major_minor=$(echo "${cuda_version}" | cut -d'.' -f1,2)
    local cuda_major=$(echo "${cuda_version}" | cut -d'.' -f1)

    # Try similar minor versions
    for minor in {0..9}; do
      local alt_cuda="${cuda_major}.${minor}"
      if [[ "${alt_cuda}" != "${cuda_major_minor}" ]]; then
        local alt_image="nvidia/cuda:${alt_cuda}.0-devel-ubuntu${ubuntu_version}"
        if docker manifest inspect "${alt_image}" >/dev/null 2>&1; then
          log_info "Found alternative with similar CUDA version: ${alt_image}"
          NVIDIA_CUDA_VERSION="${alt_cuda}.0"
          log_info "Automatically selecting ${alt_image}"
          return 0
        fi
      fi
    done

    log_error "No suitable alternative found. Please check available images at https://hub.docker.com/r/nvidia/cuda/tags"
    return 1
  fi
}

# --------------------------------
# Docker build function
# --------------------------------
build_docker_image() {
  # Construct the docker build command
  docker_cmd="docker build"
  docker_cmd+=" --build-arg NVIDIA_CUDA_VERSION=\"${NVIDIA_CUDA_VERSION}\""
  docker_cmd+=" --build-arg CUDA_ARCHITECTURES=\"${CUDA_CC}\""
  docker_cmd+=" -t \"roboticsmicrofarms/colmap:${vtag}-cuda_cc${CUDA_CC}\""
  docker_cmd+=" ${docker_opts}"  # Additional options like --no-cache, --pull, etc.
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