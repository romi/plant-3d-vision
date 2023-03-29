#!/bin/bash

name="plant3dvision"
py_version="3.9"

pip_opt=""
doc=0
notebook=0

usage() {
  echo "USAGE:"
  echo "  ./install.sh [OPTIONS]
    "

  echo "DESCRIPTION:"
  echo "  Install the sources for the 'plant-3d-vision' ROMI library in a conda environment.
  "

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
    shift
    pip_opt="$pip_opt -e"
    ;;
  --doc)
    shift
    doc=1
    ;;
  --notebook)
    shift
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

echo -e "

${RED}Creating '$name' conda environment...${NC}"
start_time=$(date +%s)
conda create -n $name python=$py_version
conda activate $name
echo -e "${RED}Done in $(expr $(date +%s) - $start_time) s.${NC}"

# Install `plantdb` sources:
echo -e "

${RED}Installing 'plantdb' sources...${NC}"
start_time=$(date +%s)
python3 -m pip install -r plantdb/requirements.txt
python3 -m pip install $pip_opt plantdb/
build_status=$?
if [ $build_status == 0 ]; then
  echo -e "${RED}Done in $(expr $(date +%s) - $start_time) s.${NC}"
else
  echo "${RED}Source install failed with $build_status code!${NC}"
  exit $build_status
fi

# Install `romitask` sources:
echo -e "

${RED}Installing 'romitask' sources...${NC}"
start_time=$(date +%s)
python3 -m pip install -r romitask/requirements.txt
python3 -m pip install $pip_opt romitask/
build_status=$?
if [ $build_status == 0 ]; then
  echo -e "${RED}Done in $(expr $(date +%s) - $start_time) s.${NC}"
else
  echo "${RED}Source install failed with $build_status code!${NC}"
  exit $build_status
fi

# Install `romiseg` sources:
echo -e "

${RED}Installing 'romiseg' sources...${NC}"
start_time=$(date +%s)
python3 -m pip install torch==1.12.1+cu102 torchvision==0.13.1+cu102 --extra-index-url https://download.pytorch.org/whl/cu102
python3 -m pip install $pip_opt romiseg/
build_status=$?
if [ $build_status == 0 ]; then
  echo -e "${RED}Done in $(expr $(date +%s) - $start_time) s.${NC}"
else
  echo "${RED}Source install failed with $build_status code!${NC}"
  exit $build_status
fi

# Install `romicgal` sources:
echo -e "

${RED}Installing 'romicgal' sources...${NC}"
start_time=$(date +%s)
python3 -m pip install pybind11
python3 -m pip install $pip_opt romicgal/
build_status=$?
if [ $build_status == 0 ]; then
  echo -e "${RED}Done in $(expr $(date +%s) - $start_time) s.${NC}"
else
  echo "${RED}Source install failed with $build_status code!${NC}"
  exit $build_status
fi

# Install `dtw` sources:
echo -e "

${RED}Installing 'dtw' sources...${NC}"
start_time=$(date +%s)
python3 -m pip install -r dtw/requirements.txt
python3 -m pip install $pip_opt dtw/
build_status=$?
if [ $build_status == 0 ]; then
  echo -e "${RED}Done in $(expr $(date +%s) - $start_time) s.${NC}"
else
  echo "${RED}Source install failed with $build_status code!${NC}"
  exit $build_status
fi

# Install `plant-3d-vision` sources:
echo -e "

${RED}Installing 'plant-3d-vision' sources...${NC}"
start_time=$(date +%s)
python3 -m pip install -r requirements.txt
python3 -m pip install $pip_opt .
build_status=$?
if [ $build_status == 0 ]; then
  echo -e "${RED}Done in $(expr $(date +%s) - $start_time) s.${NC}"
else
  echo "${RED}Source install failed with $build_status code!${NC}"
  exit $build_status
fi

if [ $doc != 0 ]; then
  echo -e "

${RED}Installing documentation requirements...${NC}"
  start_time=$(date +%s)
  python3 -m pip install -U "Sphinx>5" sphinx-material sphinx-argparse sphinx-copybutton sphinx-panels sphinx-prompt myst-nb myst-parser

  build_status=$?
  if [ $build_status == 0 ]; then
    echo -e "${RED}Done in $(expr $(date +%s) - $start_time) s.${NC}"
  else
    echo "${RED}Requirements install failed with $build_status code!${NC}"
    exit $build_status
  fi
fi

if [ $notebook != 0 ]; then
  echo -e "

${RED}Installing notebooks requirements...${NC}"
  start_time=$(date +%s)
  python3 -m pip install -U notebook ipywidgets plotly

  build_status=$?
  if [ $build_status == 0 ]; then
    echo -e "${RED}Done in $(expr $(date +%s) - $start_time) s.${NC}"
  else
    echo "${RED}Requirements install failed with $build_status code!${NC}"
    exit $build_status
  fi
fi