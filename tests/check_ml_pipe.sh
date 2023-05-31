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
# Default configuration file used for CNN based pipeline is:
cfg='config/ml_pipe_real.toml'
# Default database location:
db='tests/testdata'
# Default test dataset for CNN based pipeline is the "real_plant":
dataset="${db}/real_plant/"
# Virtual test dataset:
v_dataset="${db}/virtual_plant/"
# Default tested ROMI task for CNN based pipeline is "AnglesAndInternodes":
task='AnglesAndInternodes'
# Directory with the trained organ segmentation models:
MODEL_DIRECTORY="models/models"

usage() {
  echo -e "$(bold USAGE):"
  echo "  ./tests/check_ml_pipe.sh [OPTIONS]"
  echo ""

  echo -e "$(bold DESCRIPTION):"
  echo "  Run the CNN reconstruction pipeline with predefined datasets and configurations.

  Clean the dataset then run 'romi_run_task <task> <dataset> --config <config>'.
  Some information are printed for required task.
  By default report for tasks 'Colmap', 'PointCloud' & 'AnglesAndInternodes'.

  It may download a trained organ segmentation model if missing from the database."
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
  echo "  #1 - Run the CNN reconstruction pipeline on default 'real plant' test dataset (safe mode):
  $ ./tests/check_ml_pipe.sh --tmp"
  echo "  #2 - Run the CNN reconstruction pipeline up to the 'SegmentedPointCloud' task on 'virtual plant' test dataset (safe mode):
  $ ./tests/check_ml_pipe.sh -t SegmentedPointCloud --virtual --tmp"
  echo "  #3 - Run the CNN reconstruction pipeline with another config & test dataset (safe mode):
  $ ./tests/check_ml_pipe.sh --config config/ml_pipe_real.toml --dataset /data/ROMI/DB/arabido_test2/ --tmp"
}

tmp=0
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
    cfg='config/ml_pipe_virtual.toml'
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
  # Add a date prefix to make folder unique and a 'ML' tag to further explicit test folder name
  data_dir="$(date +%F_%H-%M-%S)_ML_${data_dir}"
  # Creates the temporary directory path variable with it
  tmp_dataset="${tmp_db}/${data_dir}"
  # Make sure it does not exist or remove it:
  if [[ -d ${tmp_dataset} ]]; then
    rm -rf ${tmp_dataset}
  fi
  # Copy the dataset to new temporary folder
  echo -e "${INFO}Copying '${dataset}' to '${tmp_dataset}'..."
  cp -R ${dataset} ${tmp_dataset}
  # Copy the models fileset to new temporary folder
  echo -e "${INFO}Copying '${db}/models' to '${tmp_db}/models'..."
  cp -R "${db}/models" "${tmp_db}/."
  # Finally replace the database & dataset locations by the temporary ones
  dataset="${tmp_dataset}"
  db="${tmp_db}"
fi

# - Run the pipeline, up to the selected task, using the machine-learning workflow:
# 0. Check the presence of the trained CNN model:
# Create the target directory if missing:
MODEL_DIRECTORY="${db}/${MODEL_DIRECTORY}"
if [ ! -d ${MODEL_DIRECTORY} ]; then
  mkdir -p ${MODEL_DIRECTORY}
  echo -e "${INFO}Created missing models directory: ${MODEL_DIRECTORY}."
fi
# Download the trained CNN model if missing:
MODEL_EPOCH_896_896_50="${MODEL_DIRECTORY}/Resnet_896_896_epoch50.pt"
if [ ! -f ${MODEL_EPOCH_896_896_50} ]; then
  echo -e "${INFO}Downloading missing trained CNN models..."
  wget -P ${MODEL_DIRECTORY} https://media.romi-project.eu/data/Resnet_896_896_epoch50.pt
fi

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
