#!/bin/bash

# Set up the environment variable PYOPENCL_CTX
export PYOPENCL_CTX='0'

##### Check geometric pipeline
# 1. clean
romi_run_task --config ../config/geom_pipe_real.toml Clean testdata/real_plant/ --local-scheduler
# 2. run pipeline
romi_run_task --config ../config/geom_pipe_real.toml AnglesAndInternodes testdata/real_plant/ --local-scheduler

echo ""
print_task_info Colmap testdata/real_plant/
echo "
"
print_task_info PointCloud testdata/real_plant/
echo "
"
print_task_info AnglesAndInternodes testdata/real_plant/
