#!/bin/bash

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
docker_opts=""

romidata_branch='dev'
romiscan_branch='dev'
romiscanner_branch='dev_lyon'

usage() {
  echo "USAGE:"
  echo "  ./build.sh [OPTIONS]
    "

  echo "DESCRIPTION:"
  echo "  Build a docker image named 'roboticsmicrofarms/romiscan_dev' using Dockerfile in same location.
    "

  echo "OPTIONS:"
  echo "  -t, --tag
    Docker image tag to use, default to '$vtag'.
    "
  echo "  --romidata
    Git branch to use for cloning 'romiscan' inside docker image, default to '$romiscan_branch'.
    "
  echo "  --romiscan
    Git branch to use for cloning 'romiscan' inside docker image, default to '$romiscan_branch'.
    "
  echo "  --romiscanner
    Git branch to use for cloning 'romiscan' inside docker image, default to '$romiscanner_branch'.
    "
  # Docker options:
  echo "  -nc, --no-cache
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
  --romidata)
    shift
    romidata_branch=$1
    ;;
  --romiscan)
    shift
    romiscan_branch=$1
    ;;
  --romiscanner)
    shift
    romiscanner_branch=$1
    ;;
  -nc | --no-cache)
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
docker build -t roboticsmicrofarms/romiscan_dev:$vtag $docker_opts \
  --build-arg ROMIDATA_BRANCH=$romidata_branch \
  --build-arg ROMISCAN_BRANCH=$romiscan_branch \
  --build-arg ROMISCANNER_BRANCH=$romiscanner_branch \
  .

docker_build_status=$?

# Important to CI/CD pipeline to track docker build failure
if  [ $docker_build_status != 0 ]
then
  echo "docker build failed with $docker_build_status code"
fi

# Print docker image build time:
echo
echo Build time is $(expr `date +%s` - $start_time) s

exit $docker_build_status
