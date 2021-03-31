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
    # Evaluation tasks:
    "VoxelsGroundTruth": "plant3dvision.tasks.evaluation",
    "VoxelsEvaluation": "plant3dvision.tasks.evaluation",
    "PointCloudGroundTruth": "plant3dvision.tasks.evaluation",
    "PointCloudEvaluation": "plant3dvision.tasks.evaluation",
    "ClusteredMeshGroundTruth": "plant3dvision.tasks.evaluation",
    "PointCloudSegmentationEvaluation": "plant3dvision.tasks.evaluation",
    "Segmentation2DEvaluation": "plant3dvision.tasks.evaluation",
    # Visu modules:
    "Visualization": "plant3dvision.tasks.visualization",
    # Database modules:
    "Clean": "plantdb.task"
}

TASKS = list(MODULES.keys())
