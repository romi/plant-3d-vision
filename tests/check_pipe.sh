##### Check geometric pipeline
romi_run_task --config ../config/original_pipe_0.toml AnglesAndInternodes real_plant/ --local-scheduler

##### Check machine learning pipeline
# 1. download models
wget https://media.romi-project.eu/data/Resnetdataset_gl_png_896_896_epoch50.pt
cp Resnetdataset_gl_png_896_896_epoch50.pt models/models/

wget https://media.romi-project.eu/data/tmp_epoch40.pt
cp tmp_epoch40.pt models/models/

# 2. run pipeline
romi_run_task --config ../config/ml_pipe_vplants_3.toml PointCloud virtual_plant/ --local-scheduler
