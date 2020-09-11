#!/bin/bash

###############################################################################
# Example usages:
###############################################################################
# 1. Default build options will create `roboticsmicrofarms/romiscan:latest`:
# $ ./build.sh
#
# 2. Build image with 'debug' image tag & another 'romiscan' branch options:
# $ ./build.sh -t debug -b 'feature/faster_docker'

vtag="latest"
romidata_branch='dev'
romiscan_branch='dev'
romiscanner_branch='master'
docker_opts=""

usage() {
  echo "USAGE:"
  echo "  ./build.sh [OPTIONS]
    "

  echo "DESCRIPTION:"
  echo "  Build a docker image named 'roboticsmicrofarms/romiscan' using Dockerfile in same location.
    "

  echo "OPTIONS:"
  echo "  -t, --tag
    Docker image tag to use, default to '$vtag'.
    "
  echo "  -b, --romiscan
    Git branch to use for cloning 'romiscan' inside docker image, default to '$romiscan_branch'.
    "
  echo "  --romidata
    Git branch to use for cloning 'romidata' inside docker image, default to '$romidata_branch'.
    "
  echo "  --romiscanner
    Git branch to use for cloning 'romiscanner' inside docker image, default to '$romiscanner_branch'.
    "
  # Docker options:
  echo "  --no-cache
    Do not use cache when building the image, (re)start from scratch.
    "
  echo "  --pull
    Always attempt to pull a newer version of the parent image.
    "
  # General options:
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
  -b | --romiscan)
    shift
    romiscan_branch=$1
    ;;
  --romidata)
    shift
    romidata_branch=$1
    ;;
  --romiscanner)
    shift
    romiscanner_branch=$1
    ;;
  --no-cache)
    shift
    docker_opts="$docker_opts --no-cache"
    ;;
  --pull)
    shift
    docker_opts="$docker_opts --pull"
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

# Get the date to estimate docker image build time:
start_time=`date +%s`

# Start the docker image build:
docker build -t roboticsmicrofarms/romiscan:$vtag $docker_opts \
  --build-arg ROMISCAN_BRANCH=$romiscan_branch \
  --build-arg ROMIDATA_BRANCH=$romidata_branch \
  --build-arg ROMISCANNER_BRANCH=$romiscanner_branch \
  .

# Print docker image build time:
echo
echo Build time is $(expr `date +%s` - $start_time) s
