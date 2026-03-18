#!/bin/bash

# --------------------------------
# Functions for colors and messages
# --------------------------------
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

log_debug() {
  if [ "${DEBUG_MODE}" = true ]; then
    echo -e "${DEBUG}$1" # Print debug message with DEBUG prefix if debug mode is enabled
  fi
}

# --------------------------------
# Functions for script initialization
# --------------------------------
initialize_variables() {
  # Name of the conda environment to create:
  ENV_NAME="plant3dvision"
  # Python version to use when creating a conda environment:
  py_version="3.10"
  # Boolean flag to update pip tools:
  update_pip_tools=0
  # Options to use with `pip`:
  pip_opt=""
  # Boolean flag to install webterm requirements:
  webterm=0
  # Boolean flag to install documentation requirements:
  doc=0
  # Boolean flag to install notebook requirements:
  notebook=0
  # Boolean flag to control environment creation:
  create_env=1
  # Debug mode is disabled by default
  DEBUG_MODE=false
}

# --------------------------------
# Check for required dependencies
# --------------------------------
# Function to check numpy version
check_numpy_version() {
  numpy_version=$(python3 -c "import numpy; print(numpy.__version__)" 2>/dev/null)
  if [ $? -eq 0 ]; then
    log_info "Numpy version installed: ${numpy_version}"
    echo "Numpy version installed: ${numpy_version}" >> numpy_versions.log
  else
    log_warning "Numpy is not installed or could not be detected."
    echo "Numpy is not installed or could not be detected." >> numpy_versions.log
  fi
}

# Check if conda is installed & available
check_conda(){
  if ! command -v conda &>/dev/null; then
    log_error "Conda is not installed or not found in PATH. Please install Conda before running this script."
    exit 1
  fi
}

get_installed_setuptools() {
  # Try to get the version of setuptools & return an empty string without raising an error
  python3 -c 'import setuptools; print(setuptools.__version__)' 2>/dev/null || echo ""
}

get_required_setuptools() {
  local toml_path="${1:-pyproject.toml}"  # Use provided path or default to "pyproject.toml"

  return $(python3 -c '
import re
import os
from pathlib import Path
try:
  toml_path = Path(os.getenv("toml_path", None))
  if not toml_path.exists():
      print("-1")
  with open(toml_path, "r") as f:
      content = f.read()
  # Look for setuptools requirements with version specifications
  matches = re.findall(r"setuptools\s?[>=<]+([0-9]+(\.[0-9]+)*)", content)
  if matches:
      print(matches[0][0])  # First match, first group
  else:
      print("0")
except Exception as e:
    print("-2")  # Default version if file not found or parsing error
')
}

# Function to create and activate conda environment
create_conda_environment() {
  check_conda  # Check if conda is installed & available
  local env_name="$1"
  local python_version="$2"

  # Get the path to the environment to create:
  CONDA_BASE_PATH=$(dirname "$(dirname $CONDA_EXE)")
  CONDA_ENV_PATH=${CONDA_BASE_PATH}/envs/${env_name}

  if [ -d "$CONDA_ENV_PATH" ]; then
    log_warning "# - Using existing '${env_name}' conda environment..."
  else
    log_info "# - Creating '${env_name}' conda environment..."
    start_time=$(date +%s)
    # conda create -y -n "${env_name}" python="${python_version}" "numpy<2"  # add "numpy<2" for compatibility with pytorch < 2.6
    conda create -y -n "${env_name}" python="${python_version}"
    if [ $? -ne 0 ]; then
      log_error "Failed to create conda environment '${env_name}'."
      return 1
    fi
    log_info "Conda environment creation done in $(($(date +%s) - start_time)) s."
  fi
  return 0
}

# Function to install package sources
install_package_source() {
  local package_name="$1"
  local source_path="$2"
  local extra_args="$3"  # Optional extra arguments like ".[io]"

  log_info "# - Installing '${package_name}' sources..."

  # Debug information about the current Python environment
  log_debug "Python environment debug information:"
  log_debug "- Python executable: $(which python3)"
  log_debug "- Python version: $(python3 --version)"
  # Check if we're in a conda environment and print its ENV_NAME
  if [[ -n "${CONDA_PREFIX}" ]]; then
    log_debug "- Active conda environment: $(basename "${CONDA_PREFIX}")"
  else
    log_debug "- No conda environment is active"
  fi
  # Print pip path to verify which pip is being used
  log_debug "- Pip executable: $(which pip)"
  log_debug "- Pip version: $(python3 -m pip --version)"
  log_debug "- setuptools `get_installed_setuptools`"

  # Check required and installed setuptools if a pyproject.toml file exists
  if [[ -f "${source_path}/pyproject.toml" ]]; then
    installed_setuptools=$(get_installed_setuptools)
    required_setuptools=$(get_required_setuptools "${source_path}/pyproject.toml")

    # Check if both versions are available
    if [[ -n "${required_setuptools}" && -n "${installed_setuptools}" ]]; then
      # Check if both are valid numbers and compare them
      if [[ "${required_setuptools}" =~ ^[0-9]+$ && "${installed_setuptools}" =~ ^[0-9]+$ ]]; then
        if [[ ${required_setuptools} -gt ${installed_setuptools} ]]; then
          log_warning "Required setuptools (${required_setuptools}) is greater than the one installed (${installed_setuptools})!"
          log_info "Consider updating it with 'python3 -m pip install --upgrade setuptools'"
        else
          log_info "Found version of setuptools ${installed_setuptools} >= ${required_setuptools} (required)"
        fi
      else
        log_warning "Could not compare setuptools versions. Required: ${required_setuptools}, Installed: ${installed_setuptools}"
      fi
    # Handle cases where one or both versions are missing
    elif [[ -z "${required_setuptools}" && -n "${installed_setuptools}" ]]; then
      log_warning "Could not detect required version of setuptools (got ${required_setuptools})! Installed version: ${installed_setuptools}"
    elif [[ -n "${required_setuptools}" && -z "${installed_setuptools}" ]]; then
      log_warning "Could not detect installed version of setuptools (got ${installed_setuptools})! Required version: ${required_setuptools}"
    else
      log_warning "Could not detect required or installed version of setuptools!"
    fi
  else
    log_error "Could not find TOML file at: ${source_path}/pyproject.toml"
  fi

  start_time=$(date +%s)
  log_debug "Running: python3 -m pip install ${pip_opt} \"${source_path}${extra_args}\""
  python3 -m pip install ${pip_opt} "${source_path}${extra_args}"
  build_status=$?

  if [ ${build_status} == 0 ]; then
    log_info "'${package_name}' sources installed in $(($(date +%s) - start_time)) s."
    # Check numpy version after installation.
    check_numpy_version
    # Test package installation
    log_info "Testing '${package_name}' package installation with Python import..."
    python3 -c "import ${package_name}" 2>/dev/null
    test_import_status=$?
    if [ ${test_import_status} -gt 0 ]; then
      log_warning "Failure!"
      # Re-run to show failure message:
      python3 -c "import ${package_name}"
      exit ${test_import_status}
    else
      log_info "Successful!"
    fi
  else
    log_error "'${package_name}' sources install failed with code '${build_status}'!"
    exit ${build_status}
  fi
}


# --------------------------------
# Usage information function
# --------------------------------
show_usage() {
  echo -e "$(bold USAGE):"
  echo -e "  ./install.sh [OPTIONS]"
  echo ""

  echo -e "$(bold DESCRIPTION):"
  echo -e "  Install the sources and dependencies for the 'plant-3d-vision' ROMI library in a conda environment."
  echo ""

  echo -e "$(bold OPTIONS):"
  echo "  -n, --name
    Name of the conda environment to use, defaults to '${ENV_NAME}'."
  echo "  -e, --dev
    Install the sources in developer mode."
  echo "  -u, --user
    Install to the Python user install directory for your platform."
  echo "  --webterm
    Install the packages required to run WebTerm."
  echo "  --doc
    Install the packages required to build documentation."
  echo "  --notebook
    Install the packages required to run jupyter notebooks."
  echo "  --python
    Set the version of python to use, defaults to '${py_version}'.
    Only used if the conda environment is created."
  echo "  -U, --update-tools
    Update pip tools, like setuptools and packaging."
  echo "  --no-env
    Skip conda environment creation and activation."
  echo "  --no-cache-dir
    Deactivate pip cache directory."
  # -- Debug option:
  echo "  --debug
    Enable debug mode to print additional debug information."
  # General options:
  echo "  -h, --help
    Output a usage message and exit."
  echo ""

  echo -e "$(bold EXAMPLES):"
  echo "  1. Create a 'plant3dvision' conda environment & install the sources in 'develop' mode."
  echo "  $ ./install.sh --dev"
  echo "  2 Install the sources in an existing 'romi' environment."
  echo "  $ ./install.sh -n romi"
}

# --------------------------------
# Command line parsing function
# --------------------------------
  parse_arguments() {
  while [ "$1" != "" ]; do
    case $1 in
    -n | --name)
      shift
      ENV_NAME=$1
      ;;
    -e | --dev)
      pip_opt="${pip_opt} -e"  # editable mode should always be the last pip option
      ;;
    -u | --user)
      pip_opt="--user ${pip_opt}"
      ;;
    --update-tools)
      update_pip_tools=1
      ;;
    --no-cache-dir)
      pip_opt="--no-cache-dir ${pip_opt}"
      ;;
    --gui)
      gui=1
      ;;
    --webterm)
      webterm=1
      ;;
    --doc)
      doc=1
      ;;
    --notebook)
      notebook=1
      ;;
    --python)
      shift
      py_version=$1
      ;;
    --no-env)
      create_env=0
      ;;
    --debug)
      DEBUG_MODE=true
      log_debug "Debug mode enabled"
      ;;
    -h | --help)
      show_usage
      exit
      ;;
    *)
      show_usage
      exit 1
      ;;
    esac
    shift
  done
}

p3dv_optional_deps (){
  p3dv_opt_deps=""
  if [ "${gui}" -eq 1 ]; then
    log_info "Using GUI requirements..."
    p3dv_opt_deps="${p3dv_opt_deps}gui,"
  fi

  if [ "${webterm}" -eq 1 ]; then
    log_info "Using WebTerm requirements..."
    p3dv_opt_deps="${p3dv_opt_deps}webterm,"
  fi

  if [ "${doc}" -eq 1 ]; then
    log_info "Using documentation requirements..."
    p3dv_opt_deps="${p3dv_opt_deps}doc,"
  fi

  if [ "${notebook}" -eq 1 ]; then
    log_info "Using notebook requirements..."
    p3dv_opt_deps="${p3dv_opt_deps}nb,"
  fi
  # Remove trailing comma if present
  p3dv_opt_deps="${p3dv_opt_deps%,}"
  # Add surrounding braces if not empty
  if [ "${p3dv_opt_deps}" != "" ]; then
    p3dv_opt_deps="[${p3dv_opt_deps%}]"
  fi
}

update_pip_tools(){
  if [ ${update_pip_tools} == 1 ]; then
    # Upgrade pip to the latest version
    log_info "Upgrading 'pip' to the latest version..."
    python3 -m pip install --upgrade pip
    # Upgrade setuptools to the latest version
    log_info "Upgrading 'setuptools' to the latest version..."
    python3 -m pip install --upgrade setuptools
    # Upgrade packaging to the latest version
    log_info "Upgrading 'packaging' to the latest version..."
    python3 -m pip install --upgrade packaging
  fi
}

# --------------------------------
# Main script execution
# --------------------------------
main() {
  setup_colors
  initialize_variables
  parse_arguments "$@"
  p3dv_optional_deps

  # Handle conda environment
  if [ ${create_env} == 1 ]; then
    # Create a conda environment
    create_conda_environment "${ENV_NAME}" "${py_version}"
    if [ $? -ne 0 ]; then
      exit 1  # exit on failure
    fi
    # Ensure conda is available in the script context
    eval "$(conda shell.bash hook)"
    # Activate the conda environment
    conda activate ${ENV_NAME}
    if [ $? -ne 0 ]; then
      log_error "Failed to activate conda environment '${ENV_NAME}'."
      return 1
    fi
  else
    log_info "# - Skipping conda environment creation..."
  fi

  update_pip_tools

  # Debug information about the current Python environment
  log_debug "Python environment debug information:"
  log_debug "- Python executable: $(which python3)"
  log_debug "- Python version: $(python3 --version)"
  # Check if we're in a conda environment and print its ENV_NAME
  if [[ -n "${CONDA_PREFIX}" ]]; then
    log_debug "- Active conda environment: $(basename "${CONDA_PREFIX}")"
  else
    log_debug "- No conda environment is active"
  fi
  # Print pip path to verify which pip is being used
  log_debug "- Pip executable: $(which pip)"
  log_debug "- Pip version: $(python3 -m pip --version)"
  log_debug "- setuptools `get_installed_setuptools`"

  # Check numpy version after installation.
  check_numpy_version

  # Define packages to install as an array of arrays
  declare -a packages=(
    "plantdb.commons|plantdb/src/commons/|[io]"
    "plantdb.client|plantdb/src/client/|"
    "plantdb.server|plantdb/src/server/|"
    "romitask|romitask/|"
    "skeleton_refinement|skeleton_refinement/|"
    "romiseg|romiseg/|"
    "romicgal|romicgal/|"
    "dtw|dtw/|"
    "plant3dvision|.|${p3dv_opt_deps}"
  )

  # Special pre-installation steps for some packages
  echo "Installing the following packages:"
  for package_info in "${packages[@]}"; do
    IFS="|" read -r package_name source_path extra_args <<< "${package_info}"
    echo "  - '${package_name}' from '${source_path}' with optional arguments '${extra_args}'"
  done

  # Special pre-installation steps for some packages
  for package_info in "${packages[@]}"; do
    IFS="|" read -r package_name source_path extra_args <<< "${package_info}"

    # Special pre-installation steps for specific packages
    if [[ "${package_name}" == "romiseg" ]]; then
      log_info "# - Installing PyTorch dependencies for 'romiseg'..."
      # python3 -m pip install torch==1.13.1 torchvision==0.14.1 --extra-index-url https://download.pytorch.org/whl/cu118
      python3 -m pip install 'torch>=2.0.0' 'torchvision>=0.15.0' --extra-index-url 'https://download.pytorch.org/whl/cu118'
    elif [[ "${package_name}" == "romicgal" ]]; then
      log_info "# - Installing pybind11 dependency for 'romicgal'..."
      python3 -m pip install pybind11
    elif [[ "${package_name}" == "dtw" ]]; then
      log_info "# - Installing requirements for 'dtw'..."
      python3 -m pip install -r dtw/requirements.txt
    fi

    # Install the package
    install_package_source "${package_name}" "${source_path}" "${extra_args}"
  done
}

# Execute main function with all arguments
main "$@"