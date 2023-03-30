#!/bin/bash

name="plant3dvision"
py_version="3.9"

pip_opt=""
doc=0
notebook=0

usage() {
  echo "USAGE:"
  echo -e "  ./install.sh [OPTIONS]\n"

  echo "DESCRIPTION:"
  echo -e "  Install the sources for the 'plant-3d-vision' ROMI library in a conda environment.\n"

  echo "OPTIONS:"
  echo "  -n, --name
    Name of the conda environment to use, defaults to '$name'."
  echo "  --dev
    Install sources in developer mode."
  echo  "  --doc
    Install packages required to build documentation."
  echo  "  --notebook
    Install packages required to run jupyter notebooks."
  echo "  --python
    Set version of python to use, defaults to '$py_version'."
  # General options:
  echo "  -h, --help
    Output a usage message and exit."
}

while [ "$1" != "" ]; do
  case $1 in
  -n | --name)
    shift
    name=$1
    ;;
  --dev)
    pip_opt="$pip_opt -e"
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

RED='\033[0;31m'
NC='\033[0m' # No Color

source ~/miniconda3/bin/activate

echo -e "${RED}# - Creating '$name' conda environment...${NC}"
start_time=$(date +%s)
conda create -n $name python=$py_version
echo -e "${RED}Conda environment creation done in $(expr $(date +%s) - $start_time) s.${NC}"
conda activate $name

# Install `plantdb` sources:
echo -e "\n\n${RED}# - Installing 'plantdb' sources...${NC}"
start_time=$(date +%s)
python3 -m pip install -r plantdb/requirements.txt
python3 -m pip install $pip_opt plantdb/
build_status=$?
if [ $build_status == 0 ]; then
  echo -e "${RED}'plantdb' sources installed in $(expr $(date +%s) - $start_time) s.${NC}"
else
  echo -e "${RED}'plantdb' sources install failed with code '${build_status}'!${NC}"
  exit $build_status
fi

# Install `romitask` sources:
echo -e "\n\n${RED}# - Installing 'romitask' sources...${NC}"
start_time=$(date +%s)
python3 -m pip install -r romitask/requirements.txt
python3 -m pip install $pip_opt romitask/
build_status=$?
if [ $build_status == 0 ]; then
  echo -e "${RED}'romitask' sources installed in $(expr $(date +%s) - $start_time) s.${NC}"
else
  echo -e "${RED}'romitask' sources install failed with code '${build_status}'!${NC}"
  exit $build_status
fi

# Install `romiseg` sources:
echo -e "\n\n${RED}# - Installing 'romiseg' sources...${NC}"
start_time=$(date +%s)
python3 -m pip install torch==1.12.1+cu102 torchvision==0.13.1+cu102 --extra-index-url https://download.pytorch.org/whl/cu102
python3 -m pip install $pip_opt romiseg/
build_status=$?
if [ $build_status == 0 ]; then
  echo -e "${RED}'romiseg' sources installed in $(expr $(date +%s) - $start_time) s.${NC}"
else
  echo -e "${RED}'romiseg' sources install failed with code '${build_status}'!${NC}"
  exit $build_status
fi

# Install `romicgal` sources:
echo -e "\n\n${RED}# - Installing 'romicgal' sources...${NC}"
start_time=$(date +%s)
python3 -m pip install pybind11
python3 -m pip install $pip_opt romicgal/
build_status=$?
if [ $build_status == 0 ]; then
  echo -e "${RED}'romicgal' sources installed in $(expr $(date +%s) - $start_time) s.${NC}"
else
  echo -e "${RED}'romicgal' sources install failed with code '${build_status}'!${NC}"
  exit $build_status
fi

# Install `dtw` sources:
echo -e "\n\n${RED}# - Installing 'dtw' sources...${NC}"
start_time=$(date +%s)
python3 -m pip install -r dtw/requirements.txt
python3 -m pip install $pip_opt dtw/
build_status=$?
if [ $build_status == 0 ]; then
  echo -e "${RED}'dtw' sources installed in $(expr $(date +%s) - $start_time) s.${NC}"
else
  echo -e "${RED}'dtw' sources install failed with code '${build_status}'!${NC}"
  exit $build_status
fi

# Install `plant-3d-vision` sources:
echo -e "\n\n${RED}# - Installing 'plant-3d-vision' sources...${NC}"
start_time=$(date +%s)
python3 -m pip install -r requirements.txt
python3 -m pip install $pip_opt .
build_status=$?
if [ $build_status == 0 ]; then
  echo -e "${RED}'plant-3d-vision' sources installed in $(expr $(date +%s) - $start_time) s.${NC}"
else
  echo -e "${RED}'plant-3d-vision' sources install failed with code '${build_status}'!${NC}"
  exit $build_status
fi

if [ $doc != 0 ]; then
  echo -e "\n\n${RED}# - Installing documentation requirements...${NC}"
  start_time=$(date +%s)
  python3 -m pip install -U "Sphinx>5" sphinx-material sphinx-argparse sphinx-copybutton sphinx-panels sphinx-prompt myst-nb myst-parser

  build_status=$?
  if [ $build_status == 0 ]; then
    echo -e "${RED}Documentation requirements installed in $(expr $(date +%s) - $start_time) s.${NC}"
  else
    echo -e "${RED}Documentation requirements install failed with code '${build_status}'!${NC}"
    exit $build_status
  fi
fi

if [ $notebook != 0 ]; then
  echo -e "\n\n${RED}# - Installing notebook requirements...${NC}"
  start_time=$(date +%s)
  python3 -m pip install -U jupyter notebook ipywidgets plotly

  build_status=$?
  if [ $build_status == 0 ]; then
    echo -e "${RED}Notebook requirements installed in $(expr $(date +%s) - $start_time) s.${NC}"
  else
    echo -e "${RED}Notebook requirements install failed with code '${build_status}'!${NC}"
    exit $build_status
  fi
fi
