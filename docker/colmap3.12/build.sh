#!/usr/bin/env bash

# Load shared helpers
source "$(dirname "$0")/utils.sh"

# --------------------------------
# Functions for script initialization
# --------------------------------
initialize_variables() {
  # Image tag to use, '3.12.4' by default:
  VTAG="3.12.4"
  # String aggregating the docker build options to use:
  DOCKER_OPTS=""
  # Default Ubuntu version
  UBUNTU_VERSION="24.04"
  # Default CUDA Compute Capability is empty (to enable automatic detection):
  CUDA_CC=""
  # Default NVIDIA CUDA Version is empty (to enable automatic detection):
  NVIDIA_CUDA_VERSION=""
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
# Docker build function
# --------------------------------
build_docker_image() {
  # Construct the docker build command
  docker_cmd="docker build"
  docker_cmd+=" --build-arg NVIDIA_CUDA_VERSION=\"${NVIDIA_CUDA_VERSION}\""
  docker_cmd+=" --build-arg CUDA_ARCHITECTURES=\"${CUDA_CC}\""
  docker_cmd+=" --build-arg UBUNTU_VERSION=\"${UBUNTU_VERSION}\""
  docker_cmd+=" -t \"roboticsmicrofarms/colmap:${VTAG}-cuda_cc${CUDA_CC}\""
  docker_cmd+=" ${DOCKER_OPTS}"  # Additional options like --no-cache, --pull, etc.
  docker_cmd+=" -f \"docker/colmap3.12/Dockerfile\""
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
  check_docker_availability
  initialize_variables
  parse_arguments "$@"
  setup_cuda_compute_capability
  setup_cuda_version
  check_nvidia_base_image
  build_docker_image
}

# Execute main function with all arguments
main "$@"