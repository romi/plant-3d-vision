#!/bin/bash

###############################################################################
# Example usages:
###############################################################################
# 1. Default build options:
# $ ./build.sh
#
# 2. Build image with 'debug' image tag & another 'romiscan' branch options:
# $ ./build.sh -t debug -b 'feature/faster_docker'

user=$USER
uid=$(id -u)
gid=$(id -g)
vtag="latest"
romidata_branch='dev'
romiscan_branch='dev'
romiscanner_branch='master'

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
  echo "  -u, --user
    User name to create inside docker image, default to '$user'.
    "
  echo "  --uid
    User id to use with 'user' inside docker image, default to '$uid'.
    "
  echo "  --gid
    Group id to use with 'user' inside docker image, default to '$gid'.
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

docker build -t roboticsmicrofarms/romiscan:$vtag \
  --build-arg USER_NAME=$user --build-arg USER_ID=$uid --build-arg GROUP_ID=$gid \
  --build-arg ROMISCAN_BRANCH=$romiscan_branch --build-arg ROMIDATA_BRANCH=$romidata_branch --build-arg ROMISCANNER_BRANCH=$romiscanner_branch \
  .
