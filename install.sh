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
# Python version to use:
py_version="3.9"
# Options to use with `pip`:
pip_opt=""
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
    conda create -y -n "${env_name}" python="${python_version}" "numpy<2"
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
  --no-cache-dir)
    pip_opt="${pip_opt} --no-cache-dir"
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

# Check numpy version after installation.
check_numpy_version

# Install `plantdb.commons` sources:
echo -e "\n\n${INFO}# - Installing 'plantdb.commons' sources..."
start_time=$(date +%s)
python3 -m pip install ${pip_opt} plantdb/src/commons/.[io]
build_status=$?
if [ ${build_status} == 0 ]; then
  echo -e "${INFO}'plantdb.commons' sources installed in $(($(date +%s) - start_time)) s."
  # Check numpy version after installation.
  check_numpy_version
else
  echo -e "${ERROR}'plantdb.commons' sources install failed with code '${build_status}'!"
  exit ${build_status}
fi

# Install `plantdb.commons` sources:
echo -e "\n\n${INFO}# - Installing 'plantdb.client' sources..."
start_time=$(date +%s)
python3 -m pip install ${pip_opt} plantdb/src/client/.[io]
build_status=$?
if [ ${build_status} == 0 ]; then
  echo -e "${INFO}'plantdb.client' sources installed in $(($(date +%s) - start_time)) s."
  # Check numpy version after installation.
  check_numpy_version
else
  echo -e "${ERROR}'plantdb.client' sources install failed with code '${build_status}'!"
  exit ${build_status}
fi

# Install `romitask` sources:
echo -e "\n\n${INFO}# - Installing 'romitask' sources..."
start_time=$(date +%s)
python3 -m pip install ${pip_opt} romitask/
build_status=$?
if [ ${build_status} == 0 ]; then
  echo -e "${INFO}'romitask' sources installed in $(($(date +%s) - start_time)) s."
  # Check numpy version after installation.
  check_numpy_version
else
  echo -e "${ERROR}'romitask' sources install failed with code '${build_status}'!"
  exit ${build_status}
fi

# Install `skeleton_refinement` sources:
echo -e "\n\n${INFO}# - Installing 'skeleton_refinement' sources..."
start_time=$(date +%s)
python3 -m pip install ${pip_opt} skeleton_refinement/
build_status=$?
if [ ${build_status} == 0 ]; then
  echo -e "${INFO}'skeleton_refinement' sources installed in $(($(date +%s) - start_time)) s."
  # Check numpy version after installation.
  check_numpy_version
else
  echo -e "${ERROR}'skeleton_refinement' sources install failed with code '${build_status}'!"
  exit ${build_status}
fi

# Install `romiseg` sources:
echo -e "\n\n${INFO}# - Installing 'romiseg' sources..."
start_time=$(date +%s)
python3 -m pip install torch==1.12.1+cu102 torchvision==0.13.1+cu102 --extra-index-url https://download.pytorch.org/whl/cu102
python3 -m pip install ${pip_opt} romiseg/
build_status=$?
if [ ${build_status} == 0 ]; then
  echo -e "${INFO}'romiseg' sources installed in $(($(date +%s) - start_time)) s."
  # Check numpy version after installation.
  check_numpy_version
else
  echo -e "${ERROR}'romiseg' sources install failed with code '${build_status}'!"
  exit ${build_status}
fi

# Install `romicgal` sources:
echo -e "\n\n${INFO}# - Installing 'romicgal' sources..."
start_time=$(date +%s)
python3 -m pip install pybind11
python3 -m pip install romicgal/
build_status=$?
if [ ${build_status} == 0 ]; then
  echo -e "${INFO}'romicgal' sources installed in $(($(date +%s) - start_time)) s."
  # Check numpy version after installation.
  check_numpy_version
else
  echo -e "${ERROR}'romicgal' sources install failed with code '${build_status}'!"
  exit ${build_status}
fi

# Install `dtw` sources:
echo -e "\n\n${INFO}# - Installing 'dtw' sources..."
start_time=$(date +%s)
python3 -m pip install -r dtw/requirements.txt
python3 -m pip install ${pip_opt} dtw/
build_status=$?
if [ ${build_status} == 0 ]; then
  echo -e "${INFO}'dtw' sources installed in $(($(date +%s) - start_time)) s."
  # Check numpy version after installation.
  check_numpy_version
else
  echo -e "${ERROR}'dtw' sources install failed with code '${build_status}'!"
  exit ${build_status}
fi

# Install `plant-3d-vision` sources:
echo -e "\n\n${INFO}# - Installing 'plant-3d-vision' sources..."
start_time=$(date +%s)
python3 -m pip install ${pip_opt} .
build_status=$?
if [ ${build_status} == 0 ]; then
  echo -e "${INFO}'plant-3d-vision' sources installed in $(($(date +%s) - start_time)) s."
  # Check numpy version after installation.
  check_numpy_version
else
  echo -e "${ERROR}'plant-3d-vision' sources install failed with code '${build_status}'!"
  exit ${build_status}
fi

if [ "${doc}" -eq 1 ]; then
  echo -e "\n\n${INFO}# - Installing documentation requirements..."
  start_time=$(date +%s)
  python3 -m pip install -U "Sphinx>5" sphinx-material sphinx-argparse sphinx-copybutton sphinx-panels sphinx-prompt myst-nb myst-parser

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
  python3 -m pip install -U jupyter notebook ipywidgets plotly

  build_status=$?
  if [ ${build_status} == 0 ]; then
    echo -e "${INFO}Notebook requirements installed in $(($(date +%s) - start_time)) s."
  else
    echo -e "${ERROR}Notebook requirements install failed with code '${build_status}'!"
    exit ${build_status}
  fi
fi
