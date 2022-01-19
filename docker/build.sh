#!/bin/bash

vtag="latest"
docker_opts=""

usage() {
  echo "USAGE:"
  echo "  ./docker/build.sh [OPTIONS]
    "

  echo "DESCRIPTION:"
  echo "  Build a docker image named 'roboticsmicrofarms/plant-3d-vision' using 'Dockerfile' in the same location.
  It must be run from the 'plant-3d-vision' repository root folder as it is the build context and it will be copied during at image build time!
  Do not forget to initialize or update the sub-modules if necessary!
  "

  echo "OPTIONS:"
  echo "  -t, --tag
    Docker image tag to use, default to '$vtag'."
  # Docker options:
  echo "  --no-cache
    Do not use cache when building the image, (re)start from scratch."
  echo "  --pull
    Always attempt to pull a newer version of the parent image."
  # General options:
  echo "  -h, --help
    Output a usage message and exit."
}

while [ "$1" != "" ]; do
  case $1 in
  -t | --tag)
    shift
    vtag=$1
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
start_time=$(date +%s)

# Start the docker image build:
docker build -t roboticsmicrofarms/plant-3d-vision:$vtag $docker_opts -f docker/Dockerfile .

# Important to CI/CD pipeline to track docker build failure
docker_build_status=$?
if [ $docker_build_status != 0 ]; then
  echo "docker build failed with $docker_build_status code"
fi

# Print docker image build time:
echo
echo Build time is $(expr $(date +%s) - $start_time) s

exit $docker_build_status
