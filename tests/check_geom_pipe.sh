#!/bin/bash

##### Check geometric pipeline
# 1. clean
romi_run_task --config ../config/test_geom_pipe.toml Clean testdata/real_plant/ --local-scheduler
# 2. run pipeline
romi_run_task --config ../config/test_geom_pipe.toml AnglesAndInternodes testdata/real_plant/ --local-scheduler

echo ""
print_task_info Colmap testdata/real_plant/
echo "
"
print_task_info PointCloud testdata/real_plant/
echo "
"
print_task_info AnglesAndInternodes testdata/real_plant/
