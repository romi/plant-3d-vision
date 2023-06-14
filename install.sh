#!/bin/bash

# - Defines colors and message types:
RED="\033[0;31m"
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
NC="\033[0m" # No Color
INFO="${GREEN}INFO${NC}    "
WARNING="${YELLOW}WARNING${NC} "
ERROR="${ERROR}ERROR${NC}   "
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

usage() {
  echo -e "$(bold USAGE):"
  echo -e "  ./install.sh [OPTIONS]"
  echo ""

  echo -e "$(bold DESCRIPTION):"
  echo -e "  Install the sources for the 'plant-3d-vision' ROMI library in a conda environment."
  echo ""

  echo -e "$(bold OPTIONS):"
  echo "  -n, --name
    Name of the conda environment to use, defaults to '${name}'."
  echo "  --dev
    Install sources in developer mode."
  echo "  --doc
    Install packages required to build documentation."
  echo "  --notebook
    Install packages required to run jupyter notebooks."
  echo "  --python
    Set version of python to use, defaults to '${py_version}'.
    Only used it the conda environment is created."
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

# source ${HOME}/miniconda3/bin/activate
# Get the path to the environment to create:
CONDA_BASE_PATH=$(dirname "$(dirname $CONDA_EXE)")
CONDA_ENV_PATH=${CONDA_BASE_PATH}/envs/${name}

if [ -d $CONDA_ENV_PATH ]; then
  echo -e "${WARNING}# - Using existing '${name}' conda environment..."
  conda activate ${name}
else
  echo -e "${INFO}# - Creating '${name}' conda environment..."
  start_time=$(date +%s)
  conda create -n ${name} python=${py_version}
  echo -e "${INFO}Conda environment creation done in $(expr $(date +%s) - ${start_time}) s."
  conda activate ${name}
fi

# Install `plantdb` sources:
echo -e "\n\n${INFO}# - Installing 'plantdb' sources..."
start_time=$(date +%s)
python3 -m pip install -r plantdb/requirements.txt
python3 -m pip install ${pip_opt} plantdb/
build_status=$?
if [ ${build_status} == 0 ]; then
  echo -e "${INFO}'plantdb' sources installed in $(expr $(date +%s) - ${start_time}) s."
else
  echo -e "${ERROR}'plantdb' sources install failed with code '${build_status}'!"
  exit ${build_status}
fi

# Install `romitask` sources:
echo -e "\n\n${INFO}# - Installing 'romitask' sources..."
start_time=$(date +%s)
python3 -m pip install -r romitask/requirements.txt
python3 -m pip install ${pip_opt} romitask/
build_status=$?
if [ ${build_status} == 0 ]; then
  echo -e "${INFO}'romitask' sources installed in $(expr $(date +%s) - ${start_time}) s."
else
  echo -e "${ERROR}'romitask' sources install failed with code '${build_status}'!"
  exit ${build_status}
fi

# Install `romiseg` sources:
echo -e "\n\n${INFO}# - Installing 'romiseg' sources..."
start_time=$(date +%s)
python3 -m pip install torch==1.12.1+cu102 torchvision==0.13.1+cu102 --extra-index-url https://download.pytorch.org/whl/cu102
python3 -m pip install ${pip_opt} romiseg/
build_status=$?
if [ ${build_status} == 0 ]; then
  echo -e "${INFO}'romiseg' sources installed in $(expr $(date +%s) - ${start_time}) s."
else
  echo -e "${ERROR}'romiseg' sources install failed with code '${build_status}'!"
  exit ${build_status}
fi

# Install `romicgal` sources:
echo -e "\n\n${INFO}# - Installing 'romicgal' sources..."
start_time=$(date +%s)
python3 -m pip install pybind11
python3 -m pip install ${pip_opt} romicgal/
build_status=$?
if [ ${build_status} == 0 ]; then
  echo -e "${INFO}'romicgal' sources installed in $(expr $(date +%s) - ${start_time}) s."
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
  echo -e "${INFO}'dtw' sources installed in $(expr $(date +%s) - ${start_time}) s."
else
  echo -e "${ERROR}'dtw' sources install failed with code '${build_status}'!"
  exit ${build_status}
fi

# Install `plant-3d-vision` sources:
echo -e "\n\n${INFO}# - Installing 'plant-3d-vision' sources..."
start_time=$(date +%s)
python3 -m pip install -r requirements.txt
python3 -m pip install ${pip_opt} .
build_status=$?
if [ ${build_status} == 0 ]; then
  echo -e "${INFO}'plant-3d-vision' sources installed in $(expr $(date +%s) - ${start_time}) s."
else
  echo -e "${ERROR}'plant-3d-vision' sources install failed with code '${build_status}'!"
  exit ${build_status}
fi

if [ ${doc} != 0 ]; then
  echo -e "\n\n${INFO}# - Installing documentation requirements..."
  start_time=$(date +%s)
  python3 -m pip install -U "Sphinx>5" sphinx-material sphinx-argparse sphinx-copybutton sphinx-panels sphinx-prompt myst-nb myst-parser

  build_status=$?
  if [ ${build_status} == 0 ]; then
    echo -e "${INFO}Documentation requirements installed in $(expr $(date +%s) - ${start_time}) s."
  else
    echo -e "${ERROR}Documentation requirements install failed with code '${build_status}'!"
    exit ${build_status}
  fi
fi

if [ ${notebook} != 0 ]; then
  echo -e "\n\n${INFO}# - Installing notebook requirements..."
  start_time=$(date +%s)
  python3 -m pip install -U jupyter notebook ipywidgets plotly

  build_status=$?
  if [ ${build_status} == 0 ]; then
    echo -e "${INFO}Notebook requirements installed in $(expr $(date +%s) - ${start_time}) s."
  else
    echo -e "${ERROR}Notebook requirements install failed with code '${build_status}'!"
    exit ${build_status}
  fi
fi
