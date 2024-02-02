#!/bin/bash

# - Defines colors and message types:
RED="\033[0;31m"
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
NC="\033[0m" # No Color
bold() { echo -e "\e[1m$*\e[0m"; }
INFO="${GREEN}INFO${NC}    "
WARNING="${YELLOW}WARNING${NC} "
ERROR="${RED}$(bold ERROR)${NC}   "

# - Default variables
# Default group id to use when starting the container:
gid=2020
# Docker image tag to use, 'latest' by default:
vtag="latest"
# Command to execute after starting the docker container:
cmd=''
# Volume mounting options:
mount_option=""
# If the `ROMI_DB` variable is set, use it as default database location, else set it to empty:
if [ -z ${ROMI_DB+x} ]; then
  host_db=''
else
  host_db=${ROMI_DB}
fi

# - Test commands:
unittest_cmd="python -m unittest discover -s plant-3d-vision/tests/unit/"
integration_test_cmd="python -m unittest discover -s plant-3d-vision/tests/integration/"
pipeline_cmd="cd plant-3d-vision/tests/ && ./check_pipe.sh"
geom_pipeline_cmd="cd plant-3d-vision/tests/ && ./check_geom_pipe.sh"
ml_pipeline_cmd="cd plant-3d-vision/tests/ && ./check_ml_pipe.sh"
gpu_cmd="nvidia-smi"

usage() {
  echo -e "$(bold USAGE):"
  echo "  ./docker/run.sh [OPTIONS] [TEST OPTION]"
  echo ""

  echo -e "$(bold DESCRIPTION):"
  echo "  Start a docker container using the 'roboticsmicrofarms/plant-3d-vision' image.

  It must be run from the 'plant-3d-vision' repository root folder if you wish to use one of the self-testing option!"
  echo ""

  echo -e "$(bold OPTIONS):"
  echo "  -t, --tag
    Image tag to use." \
    "By default, use the '${vtag}' tag."
  echo "  -db, --database
    Path to the host database to mount inside the docker container." \
    "By default, use the 'ROMI_DB' environment variable (if defined)."
  echo "  -v, --volume
    Volume mapping between host and container to mount a local directory in the container." \
    "Absolute paths are required and multiple use of this option is allowed." \
    "For example '-v /host/dir:/container/dir' makes the '/host/dir' directory accessible under '/container/dir' within the container."
  echo "  -c, --cmd
    Defines the command to run at container startup." \
    "By default, start an interactive container with a bash shell."
  echo "  -h, --help
    Output a usage message and exit."
  echo ""

  echo "$(bold TEST OPTIONS):"
  echo "You may select ONE of the test option below to execute this test instead of accessing the terminal or running a command."
  echo "  --unittest
    Run the unit tests defined in 'plant-3d-vision/tests/unit'."
  echo "  --integration_test
    Run the integration tests defined in 'plant-3d-vision/tests/integration'."
  echo "  --pipeline_test
    Run the reconstruction & quantification pipelines (geometric & machine-learning based) on the 'real_plant test dataset."
  echo "  --geom_pipeline_test
    Run the reconstruction & quantification pipeline using the geometric based workflow on the 'real_plant test dataset." \
    "Test dataset are located under 'tests/testdata'."
  echo "  --ml_pipeline_test
    Run the reconstruction & quantification pipeline using the machine-learning based workflow on the 'real_plant test dataset." \
    "Test dataset are located under 'tests/testdata'."
  echo "  --gpu_test
    Test correct access to NVIDIA GPU resources from docker container."
}

docker_option=""
self_test=0
while [ "$1" != "" ]; do
  case $1 in
  -t | --tag)
    shift
    vtag=$1
    ;;
  -db | --database)
    shift
    host_db=$1
    ;;
  -c | --cmd)
    shift
    cmd=$1
    ;;
  --unittest)
    cmd=${unittest_cmd}
    self_test=1
    echo -e "${INFO}Running unitary tests..."
    ;;
  --integration_test)
    cmd=${integration_test_cmd}
    self_test=1
    echo -e "${INFO}Running integration tests..."
    ;;
  --pipeline_test)
    cmd=${pipeline_cmd}
    self_test=1
    echo -e "${INFO}Running reconstruction pipeline self-tests (geometric & machine-learning based)..."
    ;;
  --geom_pipeline_test)
    cmd=${geom_pipeline_cmd}
    self_test=1
    echo -e "${INFO}Running reconstruction pipeline self-test using geometric based workflow..."
    ;;
  --ml_pipeline_test)
    cmd=${ml_pipeline_cmd}
    self_test=1
    echo -e "${INFO}Running reconstruction pipeline self-test using machine-learning based workflow..."
    ;;
  --gpu_test)
    cmd=${gpu_cmd}
    self_test=1
    echo -e "${INFO}Running GPU self-test procedure..."
    ;;
  -v | --volume)
    shift
    if [ "${mount_option}" == "" ]; then
      mount_option="-v $1"
    else
      mount_option="${mount_option} -v $1" # append
    fi
    ;;
  -h | --help)
    usage
    exit
    ;;
  *)
    docker_option="${docker_option} $1" # append
    shift
    docker_option="${docker_option} $1" # append
    ;;
  esac
  shift
done

# If the `ROMI_DB` variable is set, use it as default database location, else set it to empty:
if [ -z ${ROMI_DB+x} ] && [ ${self_test} == 0 ]; then
  echo -e "${WARNING}Environment variable 'ROMI_DB' is not defined, set it to use as default database location!"
fi

# Use local database path `$host_db` to create a bind mount to '/myapp/db':
if [ "${host_db}" != "" ]; then
  mount_option="${mount_option} -v ${host_db}:/myapp/db"
  echo -e "${INFO}Automatic bind mount of '${host_db}' (host) to '/myapp/db' (container)!"
else
  # Only raise next ERROR message if not a SELF-TEST:
  if [ ${self_test} == 0 ]; then
    echo -e "${ERROR}No local host database defined!"
    echo -e "${INFO}Set 'ROMI_DB' or use the '-db' | '--database' option to define it."
    exit 1
  fi
fi

# If a 'host database path' is provided, get the name of the group and its id to, later used with the `--user` option
if [ "${host_db}" != "" ]; then
  group_name=$(stat -c "%G" ${host_db})                              # get the name of the group for the 'host database path'
  gid=$(getent group ${group_name} | cut --delimiter ':' --fields 3) # get the 'gid' of this group
  echo -e "${INFO}Automatic group id definition to '$gid'!"
else
  # Only raise next WARNING message if not a SELF-TEST:
  if [ ${self_test} == 0 ]; then
    echo -e "${WARNING}Using default group id '${gid}'."
  fi
fi

# Check if we have a TTY or not
if [ -t 1 ]; then
  USE_TTY="-it"
else
  USE_TTY=""
fi

if [ "${docker_option}" != "" ]; then
  echo -e "${INFO}Extra docker arguments: '${docker_option}'!"
fi

if [ "${cmd}" = "" ]; then
  # Start in interactive mode. ~/.bashrc will be loaded.
  docker run --rm --gpus all ${mount_option} \
    --user romi:${gid} \
    --env PYOPENCL_CTX='0' \
    ${docker_option} \
    ${USE_TTY} roboticsmicrofarms/plant-3d-vision:${vtag} \
    bash
else
  # Get the date to estimate command execution time:
  start_time=$(date +%s)
  # Start in non-interactive mode (run the command):
  docker run --rm --gpus all ${mount_option} \
    --user romi:${gid} \
    --env PYOPENCL_CTX='0' \
    ${docker_option} \
    ${USE_TTY} roboticsmicrofarms/plant-3d-vision:${vtag} \
    bash -c "${cmd}"
  # Get command exit code:
  cmd_status=$?
  # Print build time if successful (code 0), else print command exit code
  if [ ${cmd_status} == 0 ]; then
    echo -e "\n${INFO}Command SUCCEEDED in $(expr $(date +%s) - ${start_time})s!"
  else
    echo -e "\n${ERROR}Command FAILED after $(expr $(date +%s) - ${start_time})s with code ${cmd_status}!"
  fi
  # Exit with status code:
  exit ${cmd_status}
fi
