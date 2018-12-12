#!/bin/sh
# The project folder must contain a folder "images" with all the images.
exec 2> /dev/null

DATASET_PATH=$1

MATCHER=$2
if [ "x$MATCHER" == "x" ]; then
    MATCHER="sequential"
fi

colmap feature_extractor --database_path $DATASET_PATH/database.db \
       --image_path $DATASET_PATH/images \
       --ImageReader.camera_params "430.12,300,200,-0.012" \
       --ImageReader.camera_model "SIMPLE_RADIAL" \
       --ImageReader.single_camera 1 \
       > log.txt 2>&1

if [ $MATCHER == "exhaustive" ]; then
    colmap exhaustive_matcher \
	   --database_path $DATASET_PATH/database.db \
        >> log.txt 2>&1
else 
    colmap sequential_matcher \
	   --database_path $DATASET_PATH/database.db \
	   >> log.txt 2>&1
fi
    
mkdir -p $DATASET_PATH/sparse

FILE=$DATASET_PATH/sparse/0/images.bin
if [ -f $FILE ]; then
    colmap mapper \
       --database_path $DATASET_PATH/database.db \
       --image_path $DATASET_PATH/images \
       --input_path $DATASET_PATH/sparse/0 \
       --output_path $DATASET_PATH/sparse/0 \
       --Mapper.ba_refine_focal_length 0 \
       --Mapper.ba_refine_extra_params 0 \
       >> log.txt 2>&1
else
    colmap mapper \
       --database_path $DATASET_PATH/database.db \
       --image_path $DATASET_PATH/images \
       --output_path $DATASET_PATH/sparse \
       --Mapper.ba_refine_focal_length 0 \
       --Mapper.ba_refine_extra_params 0 \
       >> log.txt 2>&1
fi

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
       
