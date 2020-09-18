#!/bin/bash

#Important to CI/CD pipeline : It makes the script return a non-zero code if one command fails
set -euo pipefail

###############################################################################
# Example usages:
###############################################################################
# 1. Default run starts an interactive shell:
# $ ./run.sh
#
# 2. Run container using 'debug' image tag & start GPU test:
# $ ./run.sh --gpu_test
#
# 3. Run container using 'debug' image tag & start CI tests:
# $ ./run.sh --pipeline_test
#
# 4. Run container using 'debug' image tag & start a ROMI Task:
# $ ./run.sh -t debug -p "/data/ROMI/DB" -c "romi_run_task AnglesAndInternodes ~/db/2019-02-01_10-56-33 --config ~/romiscan/config/original_pipe_0.toml"

user=$USER
use_uidgid_mapping="false"
db_path=''
vtag="latest"
cmd=''
unittest_cmd="cd romiscan/tests/ && python -m unittest"
pipeline_cmd="cd romiscan/tests/ && ./check_pipe.sh"
geom_pipeline_cmd="cd romiscan/tests/ && ./check_geom_pipe.sh"
ml_pipeline_cmd="cd romiscan/tests/ && ./check_ml_pipe.sh"
gpu_cmd="nvidia-smi"
mount_option=""

usage() {
  echo "USAGE:"
  echo "  ./run.sh [OPTIONS]
    "

  echo "DESCRIPTION:"
  echo "  Run 'romiscan:<vtag>' container with a mounted local (host) database.
    "

  echo "OPTIONS:"
  echo "  -t, --tag
    Docker image tag to use, default to '$vtag'.
    "
  echo "  -p, --database_path
    Path to the host database to mount inside docker container, requires '--user' if not defautl.
    "
  echo "  -u, --user
    User used during docker image build, default to '$user'.
    "
  echo " --use_uidgid_mapping
    If true, the docker image will be run with local user privilege.
    "
  echo "  -c, --cmd
    Defines the command to run at docker startup, by default start an interactive container with a bash shell.
    "
  echo " --unittest_cmd
    Runs unit tests defined in romiscan.
    "
  echo "  --pipeline_test
    Test pipelines (geometric & ML based) in docker container with CI test.
    "
  echo "  --geom_pipeline_test
    Test geometric pipeline in docker container with CI test & test dataset.
    "
  echo "  --ml_pipeline_test
    Test ML pipeline in docker container with CI test & test dataset.
    "
  echo "  --gpu_test
    Test correct access to NVIDIA GPU resources from docker container.
    "

  echo "  -h, --help
    Output a usage message and exit.
    "
}

while [ "$1" != "" ]; do
  case $1 in
  -t | --tag)
    shift
    vtag=$1
    ;;
  -u | --user)
    shift
    user=$1
    ;;
  --use_uidgid_mapping)
    shift
    use_uidgid_mapping="true"
    ;;
  -p | --database_path)
    shift
    db_path=$1
    ;;
  -c | --cmd)
    shift
    cmd=$1
    ;;
  --unittest_cmd)
    cmd=$unittest_cmd
    ;;
  --pipeline_test)
    cmd=$pipeline_cmd
    ;;
  --geom_pipeline_test)
    cmd=$geom_pipeline_cmd
    ;;
  --ml_pipeline_test)
    cmd=$ml_pipeline_cmd
    ;;
  --gpu_test)
    cmd=$gpu_cmd
    ;;
  -v)
    shift
    mount_option=$1
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

# Check if there is another volume to mount
if [ "$mount_option" != ""]
then
  mount_option="-v $mount_option"
fi

# Use 'host database path' & 'docker user' to create a bind mount:
if [ "$db_path" != "" ]
then
  mount_option="$mount_option -v $db_path:/home/$user/db"
fi

# Use uer id ang group id mapping (host <-> image)
if [ "$use_uidgid_mapping" = "true" ]
then
  uidgid=$(id -u):$(id -g)
else
  uidgid=""
fi

if [ "$cmd" = "" ]
then
  # Start in interactive mode:
  docker run --runtime=nvidia --gpus all $mount_option \
    --env PYOPENCL_CTX='0' \
    $uidgid \
    -it romiscan:$vtag
else
  # Start in non-interactive mode (run the command):
  docker run --runtime=nvidia --gpus all $mount_option \
    --env PYOPENCL_CTX='0' \
    $uidgid \
    --rm \
    romiscan:$vtag \
    bash -c "$cmd"
fi