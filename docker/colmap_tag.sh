#!/usr/bin/env bash

# Load shared helpers
source "$(dirname "$0")/utils.sh"

# --------------------------------
# Functions for script initialization
# --------------------------------
initialize_variables() {
  # Image tag to use, '3.8' by default:
  COLMAP_VERSION="3.8"
  # Default CUDA Compute Capability is empty (to enable automatic detection):
  CUDA_CC=""
}

# --------------------------------
# Usage information function
# --------------------------------
show_usage() {
  echo -e "$(bold USAGE):"
  echo "  ./docker/colmap_tag.sh [VERSION] [OPTIONS]"
  echo ""

  echo -e "$(bold DESCRIPTION):"
  echo "  Compute the Docker image tag using the specified COLMAP version."
  echo "  By default, use COLMAP '${COLMAP_VERSION}'."
  echo ""

  echo -e "$(bold OPTIONS):"
  echo "  --cuda-cc"
  echo "    The CUDA Compute Capability value to use to build COLMAP."
  echo "    By default, try to guess it from the system."
  # -- General options:
  echo "  -h, --help"
  echo "    Output a usage message and exit."
}

# --------------------------------
# Command line parsing function
# --------------------------------
parse_arguments() {
  # If the first argument looks like a version (doesn't start with a dash),
  # treat it as the colmap version and shift it away.
  if [[ "$1" != "" && "$1" != "-"* ]]; then
    COLMAP_VERSION="$1"
    shift
  fi

  while [ "$1" != "" ]; do
    case $1 in
    --cuda-cc)
      shift
      CUDA_CC=$1
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
# Main script execution
# --------------------------------
main() {
  setup_colors
  initialize_variables
  parse_arguments "$@"
  setup_cuda_compute_capability
  if [ -n "${GITHUB_ENV}" ]; then
    # Export the tag for GitHub Actions (if running in CI)
    echo "TAG=${VTAG}-cuda_cc${CUDA_CC}" >> "${GITHUB_ENV}"
  else
    # Output the colmap docker image with cuda compute capability
    echo "${COLMAP_VERSION}-cuda_cc${CUDA_CC}"
  fi
}

# Execute main function with all arguments
main "$@"