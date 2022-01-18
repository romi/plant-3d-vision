#!/bin/bash

model="Resnet_896_896_epoch50.pt"
url="https://media.romi-project.eu/data/$model"
path="plant-3d-vision/tests/testdata/models/models"

if [[ -d "$path/$model" ]]; then
  wget -nv --show-progress --progress=bar:force:noscroll $url
  mkdir -p $path
  mv $model $path
fi
