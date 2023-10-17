#!/bin/bash

# - Defines colors and message types:
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
NC="\033[0m" # No Color
INFO="${GREEN}INFO${NC}    "
WARNING="${YELLOW}WARNING${NC} "

model="Resnet_896_896_epoch50.pt"
url="https://media.romi-project.eu/data/${model}"
path="tests/testdata/models/models"

if [[ -f "${path}/${model}" ]]; then
  echo -e "${INFO}Found trained CNN model file '${model}'."
else
  echo -e "${WARNING}Could not find trained CNN model file '${model}'!"
  echo -e "${INFO}Downloading it..."
  wget -nv --show-progress --progress=bar:force:noscroll ${url}
  mkdir -p ${path}
  mv ${model} ${path}
fi
