#!/bin/bash

#Important to CI/CD pipeline : It makes the script return a non-zero code if one command fails
set -eo pipefail

###############################################################################
# Example usages:
###############################################################################
# 1. Default build options will create `romiscan:latest`:
# $ ./build.sh
#
# 2. Build image for a 'githubrunner' user and specify user & group id value:
# $ ./build.sh -u githubrunner --uid 1003 -g scanner --gid 1003
#
# 3. Build image with 'debug' image tag & another 'romiscan' branch options:
# $ ./build.sh -t debug -b 'my_branch'

vtag="latest"
romiscan_branch='dev'
user=$USER
uid=$(id -u)
group=$(id -g -n)
gid=$(id -g)
docker_opts=""

usage() {
  echo "USAGE:"
  echo "  ./build.sh [OPTIONS]
    "

  echo "DESCRIPTION:"
  echo "  Build a docker image named 'romiscan' using Dockerfile in same location.
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
  echo "  -g, --group
    Group name to create inside docker image, default to 'group'.
    "
  echo "  --gid
    Group id to use with 'user' inside docker image, default to '$gid'.
    "
  echo "  -b, --romiscan
    Git branch to use for cloning 'romiscan' inside docker image, default to '$romiscan_branch'.
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
  -u | --user)
    shift
    user=$1
    ;;
  --uid)
    shift
    uid=$1
    ;;
  -g | --group)
    shift
    group=$1
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

# Start the docker image build:
docker build -t romiscan:$vtag $docker_opts \
  --build-arg ROMISCAN_BRANCH=$romiscan_branch \
  --build-arg USER_NAME=$user \
  --build-arg USER_ID=$uid \
  --build-arg GROUP_NAME=$group \
  --build-arg GROUP_ID=$gid \
  .

# Print docker image build time:
echo
echo Build time is $(expr `date +%s` - $start_time) s
