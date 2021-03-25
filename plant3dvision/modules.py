#!/usr/bin/env python
# -*- coding: utf-8 -*-

MODULES = {
    # Scanning modules:
    "Scan": "plantimager.tasks.scan",
    "VirtualPlant": "plantimager.tasks.lpy",
    "VirtualScan": "plantimager.tasks.scan",
    "CalibrationScan": "plantimager.tasks.scan",
    # Geometric reconstruction modules:
    "Colmap": "plant3dvision.tasks.colmap",
    "Undistorted": "plant3dvision.tasks.proc2d",
    "Masks": "plant3dvision.tasks.proc2d",
    "Voxels": "plant3dvision.tasks.cl",
    "PointCloud": "plant3dvision.tasks.proc3d",
    "TriangleMesh": "plant3dvision.tasks.proc3d",
    "CurveSkeleton": "plant3dvision.tasks.proc3d",
    # Machine learning reconstruction modules:
    "Segmentation2D": "plant3dvision.tasks.proc2d",
    "SegmentedPointCloud": "plant3dvision.tasks.proc3d",
    "ClusteredMesh": "plant3dvision.tasks.proc3d",
    "OrganSegmentation": "plant3dvision.tasks.proc3d",
    # Quantification modules:
    "TreeGraph": "plant3dvision.tasks.arabidopsis",
    "AnglesAndInternodes": "plant3dvision.tasks.arabidopsis",
    # Visu modules:
    "Visualization": "plant3dvision.tasks.visualization",
    # Database modules:
    "Clean": "plantdb.task"
}

TASKS = list(MODULES.keys())

EVAL_MODULES = {
    "VoxelGroundTruth": "plant3dvision.tasks.eval",
    "VoxelsEvaluation": "plant3dvision.tasks.eval",
    "PointCloudGroundTruth": "plant3dvision.tasks.eval",
    "PointCloudEvaluation": "plant3dvision.tasks.eval",
    "ClusteredMeshGroundTruth": "plant3dvision.tasks.eval",
    "PointCloudSegmentationEvaluation": "plant3dvision.tasks.eval",
    "Segmentation2DEvaluation": "plant3dvision.tasks.eval"
}

EVAL_TASKS = list(EVAL_MODULES.keys())