#!/usr/bin/env python
# -*- coding: utf-8 -*-

MODULES = {
    # Scanning modules:
    "Scan": "romiscanner.scan",
    "VirtualPlant": "romiscanner.lpy",
    "VirtualScan": "romiscanner.scan",
    "CalibrationScan": "romiscanner.scan",
    # Geometric reconstruction modules:
    "Colmap": "romiscan.tasks.colmap",
    "Undistorted": "romiscan.tasks.proc2d",
    "Masks": "romiscan.tasks.proc2d",
    "Voxels": "romiscan.tasks.cl",
    "PointCloud": "romiscan.tasks.proc3d",
    "TriangleMesh": "romiscan.tasks.proc3d",
    "CurveSkeleton": "romiscan.tasks.proc3d",
    # Machine learning reconstruction modules:
    "Segmentation2D": "romiscan.tasks.proc2d",
    "SegmentedPointCloud": "romiscan.tasks.proc3d",
    "ClusteredMesh": "romiscan.tasks.proc3d",
    # Quantification modules:
    "TreeGraph": "romiscan.tasks.arabidopsis",
    "AnglesAndInternodes": "romiscan.tasks.arabidopsis",
    # Visu modules:
    "Visualization": "romiscan.tasks.visualization",
    # Database modules:
    "Clean": "romidata.task"
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