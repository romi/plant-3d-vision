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

# - Default variables
# Default test database location:
db='tests/testdata'
# Defaults for 'real_plant' dataset & config:
r_dataset="${db}/real_plant/"
r_cfg='tests/testcfg/geom_pipe_real.toml'
# Defaults for 'virtual_plant' dataset & config:
v_dataset="${db}/virtual_plant/"
v_cfg='tests/testcfg/geom_pipe_virtual.toml'

# Default tested ROMI task for GEOMETRIC based pipeline is "AnglesAndInternodes":
task='AnglesAndInternodes'
# Default test is 'real_plant' dataset & config:
dataset=${r_dataset}
cfg=${r_cfg}
# Default logging level is "INFO":
log_level="INFO"
# Default is NOT to copy to '/tmp' directory
tmp=0

usage() {
  echo -e "$(bold USAGE):"
  echo "  ./tests/check_geom_pipe.sh [OPTIONS]"
  echo ""

  echo -e "$(bold DESCRIPTION):"
  echo "  Run the geometric reconstruction pipeline with predefined datasets and configurations.

  Clean the dataset then run 'romi_run_task <task> <dataset> --config <config>'.
  Some information are printed for required task.
  By default report for tasks 'Colmap', 'PointCloud' & 'AnglesAndInternodes'."
  echo ""

  echo -e "$(bold OPTIONS):"
  echo "  -c, --config
    Pipeline configuration file to use, default to '${cfg}'."
  echo "  -d, --dataset
    Dataset to use, default to '${dataset}'."
  echo "  -t, --task
    Task to test, default to '${task}'."
  echo "  --virtual
    Use the 'virtual_plant' dataset."
  echo "  --tmp
    Clone the dataset to the temporary folder '/tmp' first." \
    "Use this to avoid messing up your repository '${db}' directory."
  echo "  --debug
    Use the 'DEBUG' log-level."
  echo "  -h, --help
    Output a usage message and exit."
  echo ""

  echo -e "$(bold EXAMPLES):"
  echo "  #1 - Run the geometric reconstruction pipeline on default 'real plant' test dataset (safe mode):
  $ ./tests/check_geom_pipe.sh --tmp"
  echo "  #2 - Run the geometric pipeline up to the 'PointCloud' task on 'virtual plant' test dataset (safe mode):
  $ ./tests/check_geom_pipe.sh -t PointCloud --virtual --tmp"
  echo "  #3 - Run a geometric reconstruction pipeline with another config & test dataset (safe mode):
  $ ./check_geom_pipe.sh --config config/geom_pipe_real.toml --dataset /data/ROMI/DB/arabido_test2/ --tmp"
}

while [ "$1" != "" ]; do
  case $1 in
  -c | --config)
    shift
    cfg=$1
    ;;
  -d | --dataset)
    shift
    dataset=$1
    ;;
  -t | --task)
    shift
    task=$1
    ;;
  --virtual)
    dataset=${v_dataset}
    cfg=${v_cfg}
    ;;
  --tmp)
    tmp=1
    ;;
  --debug)
    log_level="DEBUG"
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

# If not defined, set 'PYOPENCL_CTX' to 'O'
if [ -z ${PYOPENCL_CTX+x} ]; then
  export PYOPENCL_CTX='0'
  echo -e "${WARNING}Missing 'PYOPENCL_CTX' environment variable."
  echo -e "${INFO}Set 'PYOPENCL_CTX' to '0'."
fi

# Create the copy to temporary folder
if [ "${tmp}" = 1 ]; then
  echo -e "${INFO}Creation of a temporary copy is requested."
  tmp_db="/tmp/romidb"
  # Create `romidb` folder as root to temporary database...
  mkdir -p ${tmp_db}
  # Add the romidb marker file
  touch "${tmp_db}/romidb"
  # Get the directory name (last in hierarchy):
  data_dir=$(basename "${dataset}")
  # Add a date prefix to make folder unique and a 'GEOM' tag to further explicit test folder name
  data_dir="$(date +%F_%H-%M-%S)_GEOM_${data_dir}"
  # Creates the temporary directory path variable with it
  tmp_dataset="${tmp_db}/${data_dir}"
  # Make sure it does not exist or remove it:
  if [[ -d ${tmp_dataset} ]]; then
    rm -rf ${tmp_dataset}
  fi
  # Copy the dataset to new temporary folder
  echo -e "${INFO}Copying '${dataset}' to '${tmp_dataset}'..."
  cp -R ${dataset} ${tmp_dataset}
  # Finally replace the database & dataset locations by the temporary ones
  dataset="${tmp_dataset}"
  db="${tmp_db}"
fi

# - Run the pipeline, up to the selected task, using the geometric workflow:
# 1. Clean the dataset:
romi_run_task Clean ${dataset} --config ${cfg}

# 2. Run the pipeline, up to the selected task:
echo "romi_run_task ${task} ${dataset} --config ${cfg} --log-level ${log_level}"
romi_run_task ${task} ${dataset} --config ${cfg} --log-level ${log_level}

# 3. Print information about tested task(s):
if [ "${task}" = "AnglesAndInternodes" ]; then
  # Also inform about Colmap and PointCloud tasks if tested task is "AnglesAndInternodes"
  echo ""
  print_task_info Colmap ${dataset}
  echo "
  "
  print_task_info PointCloud ${dataset}
  echo "
  "
fi
print_task_info ${task} ${dataset}
