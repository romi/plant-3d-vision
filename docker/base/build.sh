#!/bin/bash

###############################################################################
# Example usages:
###############################################################################
# 1. Default build options will create `roboticsmicrofarms/romiscan_base:latest`:
# $ ./build.sh
#
# 2. Build image for a 'githubrunner' user and specify user & group id value:
# $ ./build.sh -u githubrunner --uid 1005 --gid 1005

user=$USER
uid=$(id -u)
gid=$(id -g)
vtag="latest"
docker_opts=""

usage() {
  echo "USAGE:"
  echo "  ./build.sh [OPTIONS]
    "

  echo "DESCRIPTION:"
  echo "  Build a docker image named 'roboticsmicrofarms/romiscan_base' using Dockerfile in same location.
    "

  echo "OPTIONS:"
  echo "  -t, --tag
    Docker image tag to use, default to '$vtag'.
    "
  echo "  -u, --user
    User name to create inside docker image, default to '$user'.
    "
  echo "  --uid
    User id to use with 'user' inside docker image, default to '$uid'.
    "
  echo "  --gid
    Group id to use with 'user' inside docker image, default to '$gid'.
    "
  # Docker options:
  echo "  --no-cache
    Do not use cache when building the image, (re)start from scratch.
    "
  echo "  --pull
    Always attempt to pull a newer version of the image.
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
  -u | --user)
    shift
    user=$1
    ;;
  --uid)
    shift
    uid=$1
    ;;
  --gid)
    shift
    gid=$1
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

# Start the docker build:
docker build -t roboticsmicrofarms/romiscan_base:$vtag $docker_opts \
  --build-arg USER_NAME=$user \
  --build-arg USER_ID=$uid \
  --build-arg GROUP_ID=$gid \
  .

# Print docker image build time:
echo
echo Build time is $(expr `date +%s` - $start_time) s
