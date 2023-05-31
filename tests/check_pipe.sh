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

# Default database location:
db='tests/testdata'
# Default test dataset for CNN based pipeline is the "real_plant":
dataset="${db}/real_plant/"
# Virtual test dataset:
v_dataset="${db}/virtual_plant/"
# Default tested ROMI task for CNN based pipeline is "AnglesAndInternodes":
task='AnglesAndInternodes'

usage(){
  echo -e "$(bold USAGE):"
  echo "  ./tests/check_pipe.sh [OPTIONS]"
  echo ""

  echo -e "$(bold DESCRIPTION):"
  echo "  Run both reconstruction pipelines with predefined datasets and configurations.

  Clean the dataset then run 'romi_run_task <task> <dataset> --config <config>'.
  Some information are printed for required task.
  By default report for tasks 'Colmap', 'PointCloud' & 'AnglesAndInternodes'.

  For the ML pipeline, it may download a trained organ segmentation model if missing from the database."
  echo ""

  echo -e "$(bold OPTIONS):"
  echo "  -d, --dataset
    Dataset to use, default to '${dataset}'."
  echo "  -t, --task
    Task to test, default to '${task}'."
  echo "  --virtual
    Use the 'virtual_plant' dataset."
  echo "  --tmp
    Clone the dataset to the temporary folder '/tmp' first." \
    "Use this to avoid messing up your repository '${db}' directory."
  echo "  -h, --help
    Output a usage message and exit."
  echo ""

  echo -e "$(bold EXAMPLES):"
  echo "  #1 - Run both geometric & CNN reconstruction pipelines on default 'real plant' test dataset (safe mode):
  $ ./tests/check_pipe.sh --tmp"
  echo "  #2 - Run both geometric & CNN reconstruction pipelines up to the 'PointCloud' task on 'virtual plant' test dataset (safe mode):
  $ ./tests/check_pipe.sh -t PointCloud --virtual --tmp"
  echo "  #3 - Run both geometric & CNN reconstruction pipelines with another dataset (safe mode):
  $ ./tests/check_pipe.sh --dataset /data/ROMI/DB/arabido_test2/ --tmp"
}
opts=""
while [ "$1" != "" ]; do
  case $1 in
  -d | --dataset)
    shift
    opts="${opts} -d $1"
    ;;
  -t | --task)
    shift
    opts="${opts} -t $1"
    ;;
  --virtual)
    opts="${opts} --virtual"
    ;;
  --tmp)
    opts="${opts} --tmp"
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

# - Check geometric pipeline
echo ""
echo "${INFO}Testing the GEOMETRICAL pipeline..."
./check_geom_pipe.sh ${opts}

# - Check machine learning pipeline
echo ""
echo "${INFO}Testing the CNN based pipeline..."
./check_ml_pipe.sh ${opts}
