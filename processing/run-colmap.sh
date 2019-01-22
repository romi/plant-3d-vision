#!/bin/sh
# The project folder must contain a folder "images" with all the images.
DATASET_PATH=$1
MATCHER=$2
DENSE=$3
USE_GPU=1

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
   --Mapper.ba_refine_focal_length 1 \
   --Mapper.ba_refine_extra_params 1 \
   --Mapper.filter_max_reproj_error 1 \
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

mkdir $DATASET_PATH/dense
colmap image_undistorter \
    --image_path $DATASET_PATH/images \
    --input_path $DATASET_PATH/sparse/0 \
    --output_path $DATASET_PATH/dense \
    --output_type COLMAP \
    --max_image_size 2000 \
    >> log.txt 2>&1

       
if [ $DENSE == 1 ]; then
    colmap patch_match_stereo \
        --workspace_path $DATASET_PATH/dense \
        --workspace_format COLMAP \
        --PatchMatchStereo.geom_consistency true \
    >> log.txt 2>&1

    colmap stereo_fusion \
        --workspace_path $DATASET_PATH/dense \
        --workspace_format COLMAP \
        --input_type geometric \
        --output_path $DATASET_PATH/dense/fused.ply \
    >> log.txt 2>&1
fi
