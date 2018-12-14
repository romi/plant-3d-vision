#!/bin/sh
# The project folder must contain a folder "images" with all the images.
DATASET_PATH=$1
MATCHER=$2
USE_GPU=0

if [ "x$MATCHER" == "x" ]; then
    MATCHER="sequential"
fi

echo "DATASET PATH = $DATASET_PATH"

colmap feature_extractor --database_path $DATASET_PATH/database.db \
       --image_path $DATASET_PATH/images \
       --ImageReader.camera_model "SIMPLE_RADIAL" \
       --ImageReader.single_camera 1 \
       --SiftExtraction.use_gpu $USE_GPU \
       > log.txt 2>&1

if [ $MATCHER == "exhaustive" ]; then
    colmap exhaustive_matcher \
	   --database_path $DATASET_PATH/database.db \
           --SiftMatching.use_gpu $USE_GPU \
        >> log.txt 2>&1
else 
    colmap sequential_matcher \
	   --database_path $DATASET_PATH/database.db \
           --SiftMatching.use_gpu $USE_GPU \
	   >> log.txt 2>&1
fi
    
mkdir -p $DATASET_PATH/sparse

colmap mapper \
   --database_path $DATASET_PATH/database.db \
   --image_path $DATASET_PATH/images \
   --output_path $DATASET_PATH/sparse \
   --Mapper.ba_refine_focal_length 0 \
   --Mapper.ba_refine_extra_params 0 \
   >> log.txt 2>&1

colmap model_aligner \
       --input_path $DATASET_PATH/sparse/0 \
       --output_path $DATASET_PATH/sparse/0 \
       --ref_images_path /tmp/poses.txt \
       --robust_alignment_max_error 10 \
       >> log.txt 2>&1

colmap model_converter \
       --input_path $DATASET_PATH/sparse/0 \
       --output_type 'TXT' \
       --output_path $DATASET_PATH \
       >> log.txt 2>&1
       
