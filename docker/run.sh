#!/bin/bash

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
# $ ./run.sh -t debug -p "/data/ROMI/DB" -c "romi_run_task AnglesAndInternodes ~/db/2019-02-01_10-56-33 --config ~/plant-3d-vision/config/geom_pipe_real.toml"

user=$USER
db_path=''
vtag="latest"
cmd=''
unittest_cmd="python -m unittest discover -s plant-3d-vision/tests/unit/"
integrationtest_cmd="python -m unittest discover -s plant-3d-vision/tests/integration/"
pipeline_cmd="cd plant-3d-vision/tests/ && ./check_pipe.sh"
geom_pipeline_cmd="cd plant-3d-vision/tests/ && ./check_geom_pipe.sh"
ml_pipeline_cmd="cd plant-3d-vision/tests/ && ./check_ml_pipe.sh"
gpu_cmd="nvidia-smi"
mount_option=""

usage() {
  echo "USAGE:"
  echo "  ./run.sh [OPTIONS]
    "

  echo "DESCRIPTION:"
  echo "  Run 'plant3dvision:<vtag>' container with a mounted local (host) database.
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
  echo "  -v, --volume
    Volume mapping for docker, e.g. '/abs/host/dir:/abs/container/dir'. Multiple use is allowed.
    "
  echo "  -c, --cmd
    Defines the command to run at docker startup, by default start an interactive container with a bash shell.
    "
  echo " --unittest_cmd
    Runs unit tests defined in plant3dvision.
    "
  echo " --integrationtest_cmd
    Runs integration tests defined in plant3dvision.
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
  --integrationtest_cmd)
    cmd=$integrationtest_cmd
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
  -v | --volume)
    shift
    if [ "$mount_option" == "" ]
    then
      mount_option="-v $1"
    else
      mount_option="$mount_option -v $1"  # append
    fi
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

# Use 'host database path' & 'docker user' to create a bind mount:
if [ "$db_path" != "" ]
then
  mount_option="$mount_option -v $db_path:/home/$user/database/"
fi

if [ "$cmd" = "" ]
then
  # Start in interactive mode:
  docker run --gpus all $mount_option \
    --env PYOPENCL_CTX='0' \
    -it plant3dvision:$vtag /bin/bash
else
  # Start in non-interactive mode (run the command):
  docker run --gpus all $mount_option \
    --env PYOPENCL_CTX='0' \
    --rm \
    plant3dvision:$vtag \
    bash -c "$cmd"
fi