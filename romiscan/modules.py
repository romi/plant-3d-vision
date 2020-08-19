#!/usr/bin/env python
# -*- coding: utf-8 -*-

MODULES = {
    "Scan": "romiscanner.scan",
    "VirtualScan": "romiscanner.scan",
    "CalibrationScan": "romiscanner.scan",
    "Colmap": "romiscan.tasks.colmap",
    "Undistorted": "romiscan.tasks.proc2d",
    "Masks": "romiscan.tasks.proc2d",
    "Segmentation2D": "romiscan.tasks.proc2d",
    "Voxels": "romiscan.tasks.cl",
    "PointCloud": "romiscan.tasks.proc3d",
    "SegmentedPointCloud": "romiscan.tasks.proc3d",
    "TriangleMesh": "romiscan.tasks.proc3d",
    "CurveSkeleton": "romiscan.tasks.proc3d",
    "TreeGraph": "romiscan.tasks.arabidopsis",
    "AnglesAndInternodes": "romiscan.tasks.arabidopsis",
    "Visualization": "romiscan.tasks.visualization",
    "Clean": "romidata.task",
    "VirtualPlant": "romiscanner.lpy"
}

TASKS = list(MODULES.keys())

EVAL_MODULES = {
    "VoxelGroundTruth": "romiscan.tasks.eval",
    "VoxelsEvaluation": "romiscan.tasks.eval",
    "PointCloudGroundTruth": "romiscan.tasks.eval",
    "PointCloudEvaluation": "romiscan.tasks.eval",
    "ClusteredMeshGroundTruth": "romiscan.tasks.eval",
    "PointCloudSegmentationEvaluation": "romiscan.tasks.eval",
    "Segmentation2DEvaluation": "romiscan.tasks.eval"
}

EVAL_TASKS = list(EVAL_MODULES.keys())