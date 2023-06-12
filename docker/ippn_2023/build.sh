#!/bin/bash

# As the `COPY` instruction in the Dockerfile use a build context and it cannot "see" outside this context,
# we have to create a temporary copy.
# - Defines where the database and notebooks to copy are:
HOST_DB_LOCATION="/Data/ROMI/evaluation_real"
HOST_NOTEBOOK="/Data/ROMI/notebooks/HowTo*"

# Get the script location
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
# Create a temporary location
TMP_DB=${SCRIPT_DIR}/data
TMP_NB=${SCRIPT_DIR}/notebooks

# Copy:
echo "Creating a temporary copy of the database directory '${TMP_DB}'..."
mkdir "${TMP_DB}/"
cp -R ${HOST_DB_LOCATION}/* "${TMP_DB}"
rm "${TMP_DB}/lock"
echo "Creating a temporary copy of the notebooks directory '${TMP_NB}'..."
mkdir "${TMP_NB}/"
cp -R ${HOST_NOTEBOOK} "${TMP_NB}"

# Build docker image:
docker build \
  --build-arg HOST_DB_LOCATION="$(realpath -m --relative-to=. ${TMP_DB}/)" \
  --build-arg HOST_NOTEBOOK="$(realpath -m --relative-to=. ${TMP_NB}/)" \
  -t roboticsmicrofarms/plant-3d-vision:ippn \
  -f ${SCRIPT_DIR}/Dockerfile .

# Clean-up:
echo "Cleaning the temporary copy of the database directory '${TMP_DB}'..."
rm -rf "${TMP_DB:?}"
echo "Cleaning the temporary copy of the notebooks directory '${TMP_NB}'..."
rm -rf "${TMP_NB:?}"