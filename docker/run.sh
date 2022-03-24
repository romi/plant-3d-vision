#!/bin/bash

vtag="latest"
host_db=$DB_LOCATION
cmd=''
mount_option=""
# Test commands:
unittest_cmd="python -m unittest discover -s tests/unit/"
integration_test_cmd="python -m unittest discover -s tests/integration/"
pipeline_cmd="./tests/check_pipe.sh"
geom_pipeline_cmd="./tests/check_geom_pipe.sh"
ml_pipeline_cmd="./tests/check_ml_pipe.sh"
gpu_cmd="nvidia-smi"

usage() {
  echo "USAGE:"
  echo "  ./docker/run.sh [OPTIONS]
    "

  echo "DESCRIPTION:"
  echo "  Run 'roboticsmicrofarms/plant-3d-vision' container with a mounted local (host) database.
  It must be run from the 'plant-3d-vision' repository root folder!
  "

  echo "OPTIONS:"
  echo "  -t, --tag
    Docker image tag to use, default to '$vtag'."
  echo "  -db, --database
    Path to the host database that will be mounted inside docker container, default to '$host_db'."
  echo "  -v, --volume
    Volume mapping for docker, e.g. '/abs/host/dir:/abs/container/dir'.
    Multiple use is allowed."
  echo "  -c, --cmd
    Defines the command to run at container startup.
    By default, return the usage message and exit."
  echo "  --unittest
    Runs unit tests defined in plant-3d-vision/tests/unit."
  echo "  --integration_test
    Runs integration tests defined in plant-3d-vision/tests/integration."
  echo "  --pipeline_test
    Test pipelines (geometric & ML based) in docker container with CI test & test dataset."
  echo "  --geom_pipeline_test
    Test geometric pipeline in docker container with CI test & test dataset."
  echo "  --ml_pipeline_test
    Test ML pipeline in docker container with CI test & test dataset."
  echo "  --gpu_test
    Test correct access to NVIDIA GPU resources from docker container."
  echo "  -h, --help
    Output a usage message and exit."
}

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
    cmd=$unittest_cmd
    ;;
  --integration_test)
    cmd=$integration_test_cmd
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
    if [ "$mount_option" == "" ]; then
      mount_option="-v $1"
    else
      mount_option="$mount_option -v $1" # append
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

# Use 'host database path' $host_db' to create a bind mount to '/myapp/db':
if [ "$host_db" != "" ]; then
  mount_option="$mount_option -v $host_db:/myapp/db"
fi

# Check if we have a TTY or not
if [ -t 1 ]; then
  USE_TTY="-it"
else
  USE_TTY=""
fi

if [ "$cmd" = "" ]; then
  # Start in interactive mode:
  docker run --gpus all $mount_option \
    --env PYOPENCL_CTX='0' \
    $USE_TTY roboticsmicrofarms/plant-3d-vision:$vtag \
    bash
else
  # Start in non-interactive mode (run the command):
  docker run --gpus all $mount_option \
    --env PYOPENCL_CTX='0' \
    $USE_TTY roboticsmicrofarms/plant-3d-vision:$vtag \
    bash -c ". /venv/bin/activate && $cmd"
fi
