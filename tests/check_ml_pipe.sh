#!/bin/bash

# Set up the environment variable PYOPENCL_CTX
export PYOPENCL_CTX='0'

##### Check machine learning pipeline
# 1. download models
MODEL_DIRECTORY="testdata/models/models"

if [ ! -d ${MODEL_DIRECTORY} ]; then
	mkdir ${MODEL_DIRECTORY}
fi

MODEL_EPOCH_896_896_50="${MODEL_DIRECTORY}/Resnet_896_896_epoch50.pt"

if [ ! -f ${MODEL_EPOCH_896_896_50} ]; then
	wget https://media.romi-project.eu/data/Resnet_896_896_epoch50.pt
	mv Resnet_896_896_epoch50.pt ${MODEL_DIRECTORY}
fi

# 2. clean
romi_run_task --config ../config/ml_pipe_real.toml Clean testdata/real_plant/ --local-scheduler
# 3. run pipeline
romi_run_task --config ../config/ml_pipe_real.toml AnglesAndInternodes testdata/real_plant/ --local-scheduler
print_task_info PointCloud testdata/real_plant/
print_task_info AnglesAndInternodes testdata/real_plant/

# 4. clean
romi_run_task --config ../config/ml_pipe_virtual.toml Clean testdata/virtual_plant/ --local-scheduler
# 5. run pipeline
romi_run_task --config ../config/ml_pipe_virtual.toml AnglesAndInternodes testdata/virtual_plant/ --local-scheduler
print_task_info PointCloud testdata/virtual_plant/
print_task_info AnglesAndInternodes testdata/virtual_plant/
