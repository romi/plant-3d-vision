#!/usr/bin/env bash

# -------------------------------------------------
# utils.sh – shared helper functions for scripts
# -------------------------------------------------

# ---------------------------------
# Functions for colors and messages
# ---------------------------------

# Make text bold
bold() {
  echo -e "\e[1m$*\e[0m"
}

# Initialise colour variables and log‑prefixes
setup_colors() {
  RED="\033[0;31m"    # Define red color code
  GREEN="\033[0;32m"  # Define green color code
  YELLOW="\033[0;33m" # Define yellow color code
  BLUE="\033[0;34m"   # Define blue color code for debug messages
  NC="\033[0m"        # No Color code to reset colors
  INFO="${GREEN}INFO${NC}    "    # Prefix for info messages
  WARNING="${YELLOW}WARNING${NC} " # Prefix for warning messages
  ERROR="${RED}$(bold ERROR)${NC}   " # Prefix for error messages using bold function
  DEBUG="${BLUE}DEBUG${NC}   "   # Prefix for debug messages
}

# Logging helpers
log_info() {
  echo -e "${INFO}$1"
}

log_warning() {
  echo -e "${WARNING}$1"
}

log_error() {
  echo -e "${ERROR}$1"
}

log_debug() {
  if [ "${DEBUG_MODE}" = true ]; then
    echo -e "${DEBUG}$1" # Print debug message with DEBUG prefix if debug mode is enabled
  fi
}

# --------------------------------
# Docker related functions
# --------------------------------
check_docker_availability() {
  if ! command -v docker >/dev/null 2>&1; then
    log_error "Docker is not installed or not found!"
    exit 1
  fi
}

check_nvidia_base_image() {
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

check_colmap_base_image() {
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

setup_cuda_nvcc_flags() {
  # Check if required variables are set
  if [ -z "${COLMAP_VERSION}" ] || [ -z "${CUDA_CC}" ]; then
    log_error "Required variables COLMAP_VERSION or CUDA_CC not set!"
    return 1
  fi

  base_image="roboticsmicrofarms/colmap:${COLMAP_VERSION}-cuda_cc${CUDA_CC}"

  # The base image is built on top of an nvidia/cuda image that echo a message with the 'CUDA Version'
  log_info "Running Docker container to detect CUDA version in base image..."
  docker_output=$(docker run --rm --gpus all "${base_image}" 2>/dev/null)
  log_debug "Docker output: ${docker_output}"

  # Extract CUDA version with robust parsing
  CUDA_VERSION=$(echo "$docker_output" | grep -o 'CUDA Version [0-9.]*' | awk '{print $3}' | tr -d '[:space:]')
  log_debug "Extracted CUDA version: '${CUDA_VERSION}'"

  # Check if CUDA_VERSION is a number
  if [ -z "${CUDA_VERSION}" ] || ! [[ "${CUDA_VERSION}" =~ ^[0-9]+(\.[0-9]+)*$ ]]; then
    log_warning "Could not parse base image CUDA version!"
    log_debug "Got CUDA version from base image output: ${CUDA_VERSION}"
    log_debug "Using default PYCUDA_NVCC_FLAGS"
  else
    log_info "Found valid CUDA version in base image: ${CUDA_VERSION}"
    # Extract major version for comparison
    CUDA_MAJOR_VERSION=$(echo "${CUDA_VERSION}" | cut -d. -f1)
    log_debug "CUDA major version: ${CUDA_MAJOR_VERSION}"

    # If CUDA_VERSION is lower than 12, `nvcc` arch can not be greater than 86
    if [ "${CUDA_MAJOR_VERSION}" -lt "12" ] && [ "${CUDA_CC}" -ge "86" ]; then
      log_debug "Setting PYCUDA_NVCC_FLAGS for CUDA CC > 86"
      PYCUDA_NVCC_FLAGS="-arch=sm_86"
      log_debug "Using PYCUDA_NVCC_FLAGS=${PYCUDA_NVCC_FLAGS}"
    else
      log_debug "Using default PYCUDA_NVCC_FLAGS"
    fi
  fi
}
