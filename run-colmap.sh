# The project folder must contain a folder "images" with all the images.

exec 2> /dev/null

DATASET_PATH="~/.local/share/lettucethink/scanner/data"

DATASET_ID=$1
if [ "x$DATASET_ID" == "x" ]; then
    DATASET_ID=.
fi

MATCHER=$2
if [ "x$MATCHER" == "x" ]; then
    MATCHER="sequential"
fi

python3 generate_poses.py -i "2018-11-26_15-05-57"

colmap feature_extractor --database_path $DATASET_PATH/$DATASET_ID/database.db \
       --image_path $DATASET_PATH/$DATASET_ID/images \
       --ImageReader.camera_params "430.12,300,200,-0.012" \
       --ImageReader.camera_model "SIMPLE_RADIAL" \
       --ImageReader.single_camera 1 \
       > log.txt 2>&1

if [ $MATCHER == "exhaustive" ]; then
    colmap exhaustive_matcher \
	   --database_path $DATASET_PATH/$DATASET_ID/database.db \
        >> log.txt 2>&1
else 
    colmap sequential_matcher \
	   --database_path $DATASET_PATH/$DATASET_ID/database.db \
	   >> log.txt 2>&1
fi
    
mkdir -p $DATASET_PATH/$DATASET_ID/sparse

FILE=$DATASET_PATH/$DATASET_ID/sparse/0/images.bin
if [ -f $FILE ]; then
    colmap mapper \
       --database_path $DATASET_PATH/$DATASET_ID/database.db \
       --image_path $DATASET_PATH/$DATASET_ID/images \
       --input_path $DATASET_PATH/$DATASET_ID/sparse/0 \
       --output_path $DATASET_PATH/$DATASET_ID/sparse/0 \
       --Mapper.ba_refine_focal_length 0 \
       --Mapper.ba_refine_extra_params 0 \
       >> log.txt 2>&1
else
    colmap mapper \
       --database_path $DATASET_PATH/$DATASET_ID/database.db \
       --image_path $DATASET_PATH/$DATASET_ID/images \
       --output_path $DATASET_PATH/$DATASET_ID/sparse \
       --Mapper.ba_refine_focal_length 0 \
       --Mapper.ba_refine_extra_params 0 \
       >> log.txt 2>&1
fi

colmap model_aligner \
       --input_path $DATASET_PATH/$DATASET_ID/sparse/0 \
       --output_path $DATASET_PATH/$DATASET_ID/sparse/0 \
       --ref_images_path /tmp/poses.txt \
       --robust_alignment_max_error 10 \
       >> log.txt 2>&1

colmap model_converter \
       --input_path $DATASET_PATH/$DATASET_ID/sparse/0 \
       --output_type 'TXT' \
       --output_path $DATASET_PATH/$DATASET_ID \
       >> log.txt 2>&1
       
