#!/bin/bash

##### Check machine learning pipeline
# 1. download models
MODEL_DIRECTORY="testdata/models/models"

if [ ! -d ${MODEL_DIRECTORY} ]; then
	mkdir ${MODEL_DIRECTORY}
fi

MODEL_EPOCH_896_896_50="${MODEL_DIRECTORY}/Resnetdataset_gl_png_896_896_epoch50.pt"

if [ ! -f ${MODEL_EPOCH_896_896_50} ]; then
	wget https://media.romi-project.eu/data/Resnetdataset_gl_png_896_896_epoch50.pt
	mv Resnetdataset_gl_png_896_896_epoch50.pt ${MODEL_DIRECTORY}
fi

#MODEL_TMP_40="${MODEL_DIRECTORY}/tmp_epoch40.pt"
#if [ ! -f ${MODEL_TMP_40} ]; then
#	wget https://media.romi-project.eu/data/tmp_epoch40.pt
#	mv tmp_epoch40.pt ${MODEL_DIRECTORY}
#fi

# 2. clean
romi_run_task --config ../config/ml_pipe_vplants_4.toml Clean testdata/virtual_plant/ --local-scheduler
# 3. run pipeline
romi_run_task --config ../config/ml_pipe_vplants_4.toml AnglesAndInternodes testdata/virtual_plant/ --local-scheduler
print_task_info PointCloud testdata/real_plant/
print_task_info AnglesAndInternodes testdata/virtual_plant/
