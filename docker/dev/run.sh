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
# $ ./run.sh -t debug -p "/data/ROMI/DB" -c "romi_run_task AnglesAndInternodes ~/db/2019-02-01_10-56-33 --config ~/romiscan/config/original_pipe_0.toml"
# $ ./run.sh -t debug -p "/data/ROMI/DB/" -v "/data/ROMI/configs/:/home/jonathan/configs" --update -c "romi_run_task AnglesAndInternodes db/arabido_test4 --config configs/original_pipe_0.toml && print_task_info AnglesAndInternodes db/arabido_test4"

user=$USER
db_path=''
vtag="latest"
cmd=''
unittest_cmd="cd romiscan/tests/ && python -m unittest"
pipeline_cmd="cd romiscan/tests/ && ./check_pipe.sh"
geom_pipeline_cmd="cd romiscan/tests/ && ./check_geom_pipe.sh"
ml_pipeline_cmd="cd romiscan/tests/ && ./check_ml_pipe.sh"
gpu_cmd="nvidia-smi"
mount_option=""
update=''

usage() {
  echo "USAGE:"
  echo "  ./run.sh [OPTIONS]
    "

  echo "DESCRIPTION:"
  echo "  Run 'roboticsmicrofarms/romiscan_dev:<vtag>' container with a mounted local (host) database.
    "

  echo "OPTIONS:"
  echo "  -t, --tag
    Docker image tag to use, default to '$vtag'.
    "
  echo "  -p, --database_path
    Path to the host database to mount inside docker container, requires '--user' if not default.
    "
  echo "  -u, --user
    User used during docker image build, default to '$user'.
    "
  echo "  -v, --volume
    Volume mapping for docker, e.g. '/abs/host/dir:/abs/container/dir'. Multiple use is allowed.
    "
  echo "  --update
    Update the git branches for ROMI repositories before running the command.
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
  -p | --database_path)
    shift
    db_path=$1
    ;;
  -c | --cmd)
    shift
    cmd=$1
    ;;
  --update)
    update="cd romiscan && git pull && cd ../romiscanner  && git pull  && cd ../romidata && git pull && cd .. && "
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
    echo
    echo "UNKNOWN option '$1'!"
    echo
    usage
    exit 1
    ;;
  esac
  shift
done

# Use 'host database path' & 'docker user' to create a bind mount:
if [ "$db_path" != "" ]
then
  mount_option="$mount_option -v $db_path:/home/$user/db"
fi

if [ "$cmd" = "" ]
then
  # Start in interactive mode:
  docker run --runtime=nvidia --gpus all $mount_option \
    --env PYOPENCL_CTX='0' \
    -it roboticsmicrofarms/romiscan_dev:$vtag bash
else
  # Start in non-interactive mode (run the command):
  docker run --runtime=nvidia --gpus all $mount_option \
    --env PYOPENCL_CTX='0' \
    --rm \
    roboticsmicrofarms/romiscan_dev:$vtag \
    bash -c "$update""$cmd"
fi