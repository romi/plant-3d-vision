#!/bin/bash

# Load shared helpers
source "$(dirname "$0")/../utils.sh"

# --------------------------------
# Functions for script initialization
# --------------------------------
initialize_variables() {
  # Image tag to use, 'colmapX.X-cuda_ccXX' by default:
  VTAG=""
  # String aggregating the docker build options to use:
  DOCKER_OPTS=""
  # Default Colmap version to use:
  COLMAP_VERSION="3.8"
  # Default CUDA Compute Capability is empty (to enable automatic search):
  CUDA_CC=""
  # PYCUDA NVCC flags
  PYCUDA_NVCC_FLAGS=""
  # Debug mode is disabled by default
  DEBUG_MODE=false
}

# --------------------------------
# Usage information function
# --------------------------------
show_usage() {
  echo -e "$(bold USAGE):"
  echo "  ./docker/build.sh [OPTIONS]"
  echo ""

  echo -e "$(bold DESCRIPTION):"
  echo "  Build a docker image named 'roboticsmicrofarms/p3dv-base' using 'Dockerfile' in the same location.

  It must be run from the 'plant-3d-vision' repository root folder as it is the build context and it will be copied during at image build time!
  Do not forget to initialize or update the sub-modules if necessary!"
  echo ""

  echo -e "$(bold OPTIONS):"
  echo "  -t, --tag
    Image tag to use." \
    "By default, use the 'colmapX.X-cuda_ccXX' tag."
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
  # -- Debug option:
  echo "  --debug
    Enable debug mode to print additional debug information."
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
    --debug)
      DEBUG_MODE=true
      log_debug "Debug mode enabled"
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
# Build Docker image
# --------------------------------
build_docker_image() {
  if [ -z "${VTAG}" ]; then
    VTAG="colmap${COLMAP_VERSION}-cuda_cc${CUDA_CC}"
  fi

  # Construct the docker build command
  docker_cmd="docker build"
  docker_cmd+=" --build-arg COLMAP_VERSION=\"${COLMAP_VERSION}\""
  docker_cmd+=" --build-arg CUDA_CC=\"${CUDA_CC}\""
  docker_cmd+=" --build-arg PYCUDA_NVCC_FLAGS=\"${PYCUDA_NVCC_FLAGS}\""
  docker_cmd+=" -t \"roboticsmicrofarms/p3dv-base:${VTAG}\""
  docker_cmd+=" ${DOCKER_OPTS}"  # Additional options like --no-cache, --pull, etc.
  docker_cmd+=" -f \"docker/p3dv-base/Dockerfile\""
  docker_cmd+=" ."  # Build context

  # Print the build configuration options
  log_debug "Build configuration:"
  log_debug "- COLMAP_VERSION: ${COLMAP_VERSION}"
  log_debug "- CUDA_CC: ${CUDA_CC}"
  log_debug "- PYCUDA_NVCC_FLAGS: ${PYCUDA_NVCC_FLAGS}"
  log_debug "- Docker tag: roboticsmicrofarms/p3dv-base:${VTAG}"
  log_debug "- Docker options: ${DOCKER_OPTS}"
  # Print the full command that will be executed
  log_debug "Executing command: ${docker_cmd}"

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
    log_debug "Docker image successfully created with tag: roboticsmicrofarms/p3dv-base:${VTAG}"
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
  check_colmap_base_image
  setup_cuda_nvcc_flags
  build_docker_image
}

# Execute main function with all arguments
main "$@"