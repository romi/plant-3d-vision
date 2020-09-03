#!/bin/bash

##### Check geometric pipeline
# 1. clean
romi_run_task --config ../config/original_pipe_1.toml Clean testdata/real_plant/ --local-scheduler
# 2. run pipeline
romi_run_task --config ../config/original_pipe_1.toml AnglesAndInternodes testdata/real_plant/ --local-scheduler
print_task_info PointCloud testdata/real_plant/
print_task_info AnglesAndInternodes testdata/real_plant/