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

# Name of the conda environment to create:
name="plant3dvision"
# Python version to use when creating a conda environment:
py_version="3.9"
# Options to use with `pip`:
pip_opt=""
# Boolean to install webterm requirements:
webterm=0
# Boolean to install documentation requirements:
doc=0
# Boolean to install notebook requirements:
notebook=0
# Boolean to control environment creation:
create_env=1

# Function to check numpy version
check_numpy_version() {
  numpy_version=$(python3 -c "import numpy; print(numpy.__version__)" 2>/dev/null)
  if [ $? -eq 0 ]; then
    echo -e "${INFO}Numpy version installed: ${numpy_version}"
    echo "Numpy version installed: ${numpy_version}" >> numpy_versions.log
  else
    echo -e "${WARNING}Numpy is not installed or could not be detected."
    echo "Numpy is not installed or could not be detected." >> numpy_versions.log
  fi
}

# Check if conda is installed & available
check_conda(){
  if ! command -v conda &>/dev/null; then
    echo -e "${ERROR}Conda is not installed or not found in PATH. Please install Conda before running this script."
    exit 1
  fi
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
    echo -e "${WARNING}# - Using existing '${env_name}' conda environment..."
  else
    echo -e "${INFO}# - Creating '${env_name}' conda environment..."
    start_time=$(date +%s)
    # conda create -y -n "${env_name}" python="${python_version}" "numpy<2"  # add "numpy<2" for compatibility with pytorch < 2.6
    conda create -y -n "${env_name}" python="${python_version}"
    if [ $? -ne 0 ]; then
      echo -e "${ERROR}Failed to create conda environment '${env_name}'."
      return 1
    fi
    echo -e "${INFO}Conda environment creation done in $(($(date +%s) - start_time)) s."
  fi

  eval "$(conda shell.bash hook)"
  conda activate ${env_name}
  if [ $? -ne 0 ]; then
    echo -e "${ERROR}Failed to activate conda environment '${env_name}'."
    return 1
  fi

  return 0
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

# Function to install package sources
install_package_source() {
  local package_name="$1"
  local source_path="$2"
  local extra_args="$3"  # Optional extra arguments like ".[io]"

  echo -e "\n\n${INFO}# - Installing '${package_name}' sources..."

  # Check required and installed setuptools if a pyproject.toml file exists
  if [[ -f "${source_path}/pyproject.toml" ]]; then
    installed_setuptools=$(get_installed_setuptools)
    required_setuptools=$(get_required_setuptools "${source_path}/pyproject.toml")

    # Check if both versions are available
    if [[ -n "${required_setuptools}" && -n "${installed_setuptools}" ]]; then
      # Check if both are valid numbers and compare them
      if [[ "${required_setuptools}" =~ ^[0-9]+$ && "${installed_setuptools}" =~ ^[0-9]+$ ]]; then
        if [[ ${required_setuptools} -gt ${installed_setuptools} ]]; then
          echo -e "${WARNING}Required setuptools (${required_setuptools}) is greater than the one installed (${installed_setuptools})!"
          echo -e "${INFO}Consider updating it with 'python3 -m pip install --upgrade setuptools'"
        else
          echo -e "${INFO}Found version of setuptools ${installed_setuptools} >= ${required_setuptools} (required)"
        fi
      else
        echo -e "${WARNING}Could not compare setuptools versions. Required: ${required_setuptools}, Installed: ${installed_setuptools}"
      fi
    # Handle cases where one or both versions are missing
    elif [[ -z "${required_setuptools}" && -n "${installed_setuptools}" ]]; then
      echo -e "${WARNING}Could not detect required version of setuptools (got ${required_setuptools})! Installed version: ${installed_setuptools}"
    elif [[ -n "${required_setuptools}" && -z "${installed_setuptools}" ]]; then
      echo -e "${WARNING}Could not detect installed version of setuptools (got ${installed_setuptools})! Required version: ${required_setuptools}"
    else
      echo -e "${WARNING}Could not detect required or installed version of setuptools!"
    fi
  else
    echo -e "${ERROR}Could not find TOML file at: ${source_path}/pyproject.toml"
  fi

  start_time=$(date +%s)
  python3 -m pip install ${pip_opt} "${source_path}/${extra_args}"
  build_status=$?

  if [ ${build_status} == 0 ]; then
    echo -e "${INFO}'${package_name}' sources installed in $(($(date +%s) - start_time)) s."
    # Check numpy version after installation.
    check_numpy_version

    # Test import if there's a package to import (skip for some packages that may not have direct imports)
    if [[ -n "${package_name}" && "${package_name}" != "." ]]; then
      python3 -c "import ${package_name}" 2>/dev/null
      test_import_status=$?
      if [ ${test_import_status} -gt 0 ]; then
        echo -e "${WARNING}'${package_name}' test import failed!"
        python3 -c "import ${package_name}"
      fi
    fi
  else
    echo -e "${ERROR}'${package_name}' sources install failed with code '${build_status}'!"
    exit ${build_status}
  fi
}


usage() {
  echo -e "$(bold USAGE):"
  echo -e "  ./install.sh [OPTIONS]"
  echo ""

  echo -e "$(bold DESCRIPTION):"
  echo -e "  Install the sources and dependencies for the 'plant-3d-vision' ROMI library in a conda environment."
  echo ""

  echo -e "$(bold OPTIONS):"
  echo "  -n, --name
    Name of the conda environment to use, defaults to '${name}'."
  echo "  --dev
    Install the sources in developer mode."
  echo "  --user
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
  echo "  --no-env
    Skip conda environment creation and activation."
  echo "  --no-cache-dir
    Deactivate pip cache directory."
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

while [ "$1" != "" ]; do
  case $1 in
  -n | --name)
    shift
    name=$1
    ;;
  --dev)
    pip_opt="${pip_opt} -e"
    ;;
  --user)
    pip_opt="${pip_opt} --user"
    ;;
  --no-cache-dir)
    pip_opt="${pip_opt} --no-cache-dir"
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

# Handle conda environment
if [ ${create_env} == 1 ]; then
  # Create and activate conda environment
  create_conda_environment "${name}" "${py_version}"
  if [ $? -ne 0 ]; then
    exit 1
  fi
else
  echo -e "${INFO}# - Skipping conda environment creation..."
fi


echo -e "\n\n${INFO}Using `python3 --version`"
echo -e "${INFO}Using `python3 -m pip --version`"
echo -e "${INFO}Using setuptools `get_installed_setuptools`"


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
  "plant3dvision|.|"
)

# Special pre-installation steps for some packages
for package_info in "${packages[@]}"; do
  IFS="|" read -r package_name source_path extra_args <<< "${package_info}"

  # Special pre-installation steps for specific packages
  if [[ "${package_name}" == "romiseg" ]]; then
    echo -e "\n\n${INFO}# - Installing PyTorch dependencies for 'romiseg'..."
    # python3 -m pip install torch==1.13.1 torchvision==0.14.1 --extra-index-url https://download.pytorch.org/whl/cu118
    python3 -m pip install 'torch>=2.0.0' 'torchvision>=0.15.0' --extra-index-url 'https://download.pytorch.org/whl/cu118'
  elif [[ "${package_name}" == "romicgal" ]]; then
    echo -e "\n\n${INFO}# - Installing pybind11 dependency for 'romicgal'..."
    python3 -m pip install pybind11
  elif [[ "${package_name}" == "dtw" ]]; then
    echo -e "\n\n${INFO}# - Installing requirements for 'dtw'..."
    python3 -m pip install -r dtw/requirements.txt
  fi

  # Install the package
  install_package_source "${package_name}" "${source_path}" "${extra_args}"
done


if [ "${webterm}" -eq 1 ]; then
  echo -e "\n\n${INFO}# - Installing WebTerm requirements..."
  start_time=$(date +%s)
  python3 -m pip install .[webterm]

  build_status=$?
  if [ ${build_status} == 0 ]; then
    echo -e "${INFO}WebTerm requirements installed in $(($(date +%s) - start_time)) s."
  else
    echo -e "${ERROR}WebTerm requirements install failed with code '${build_status}'!"
    exit ${build_status}
  fi
fi

if [ "${doc}" -eq 1 ]; then
  echo -e "\n\n${INFO}# - Installing documentation requirements..."
  start_time=$(date +%s)
  python3 -m pip install .[doc]

  build_status=$?
  if [ ${build_status} == 0 ]; then
    echo -e "${INFO}Documentation requirements installed in $(($(date +%s) - start_time)) s."
  else
    echo -e "${ERROR}Documentation requirements install failed with code '${build_status}'!"
    exit ${build_status}
  fi
fi

if [ "${notebook}" -eq 1 ]; then
  echo -e "\n\n${INFO}# - Installing notebook requirements..."
  start_time=$(date +%s)
  python3 -m pip install .[nb]

  build_status=$?
  if [ ${build_status} == 0 ]; then
    echo -e "${INFO}Notebook requirements installed in $(($(date +%s) - start_time)) s."
  else
    echo -e "${ERROR}Notebook requirements install failed with code '${build_status}'!"
    exit ${build_status}
  fi
fi
