#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import datetime
import filecmp
import json
import logging
import os
import pathlib
import shutil
import subprocess
import tempfile
from collections import Counter
from os.path import splitext

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from matplotlib import cm
from plant3dvision.metrics import chamfer_distance
from plant3dvision.metrics import point_cloud_registration_fitness
from plant3dvision.metrics import set_metrics
from plant3dvision.metrics import surface_ratio
from plant3dvision.metrics import volume_ratio

from plantdb import FSDB
from plantdb.fsdb import LOCK_FILE_NAME
from plantdb.fsdb import MARKER_FILE_NAME
from plantdb.io import read_image
from plantdb.io import read_json
from plantdb.io import read_npz
from plantdb.io import read_point_cloud
from plantdb.io import read_triangle_mesh

dirname, filename = os.path.split(os.path.abspath(__file__))
logger = logging.getLogger(f'{filename}')

CONF_FILE = os.path.join(dirname, "conf_robustness_comparison.json")


def save_data_repartition(data, data_type, db):
    """Save repartition plots.

    Parameters
    ----------
    data : list
        list of data to plot
    data_type : string
        angles or internodes
    db : pathlib.Path
        folder in which register the graphs

    """
    fig, ax = plt.subplots()
    fig.suptitle(f'{data_type} distribution for same scan same pipe', fontsize=16)
    fig.set_size_inches(10, 6)
    data = np.array(data).transpose()
    # n_scans = data.shape()[1]
    ax.set_title(f'Number of scans: {data.shape[1]}')
    bxp = ax.boxplot(list(data), patch_artist=True)

    # fill with colors
    colors = cm.get_cmap("magma", len(data))
    for patch, color in zip(bxp['boxes'], colors.colors):
        patch.set_facecolor(color)

    # adding horizontal grid lines
    ax.yaxis.grid(True)
    ax.set_xlabel(f'{data_type}')
    ax.set_ylabel('Observed values')

    plt.tight_layout()
    fig.savefig(db / f"repeat_results_{data_type}.png")


def pairwise_heatmap(pw_dict, scans_list, task_name, metrics, db, **kwargs):
    """Save a PNG of the pairwise heatmap.

    Parameters
    ----------
    pw_dict : dict
        Pairwise dictionary with a float value and a pair of scan as keys.
    scans_list : list
        List of scan to use in pairwise heatmap representation.
    task_name : str
        Name of the task, used in title and filename.
    metrics : str
        Name of the metric, used in title and filename.
    db : FSDB
        Database containing the scan, used for PNG file location.

    Other Parameters
    ----------------
    fname : str
        PNG file location, override the automatic location.

    Examples
    --------
    >>> from plantdb import FSDB
    >>> # - Connect to a ROMI databse to access an 'images' fileset to reconstruct with COLMAP:
    >>> db = FSDB("/data/ROMI/repeat_test_organseg")
    >>> db.connect()
    >>> scans_list = [scan for scan in db.get_scans() if scan.id != 'models']
    >>> import json
    >>> pw_json = json.load(open('/data/ROMI/repeat_test_organseg/PointCloud_comparison.json', 'r'))
    >>> metrics = 'chamfer distances'
    >>> pw_dict = unjsonify_tuple_keys(pw_json[metrics])
    >>> pairwise_heatmap(pw_dict, scans_list, 'PointCloud', metrics, db)

    """
    scans_list = sorted([scan.id for scan in scans_list])
    n_scans = len(scans_list)
    pw_mat = np.zeros((n_scans, n_scans))
    for i, scan_i in enumerate(scans_list):
        for j, scan_j in enumerate(scans_list):
            if j <= i:
                continue  # only scan half of the pairwise matrix
            try:
                data = pw_dict[(scan_i, scan_j)]
            except:
                data = pw_dict[(scan_j, scan_i)]
            pw_mat[i, j] = data
            pw_mat[j, i] = data

    fig, ax = plt.subplots()
    fig.set_size_inches(n_scans / 2., n_scans / 2.)
    im = ax.imshow(pw_mat)

    # We want to show all ticks...
    ax.set_xticks(np.arange(n_scans))
    ax.set_yticks(np.arange(n_scans))
    # ... and label them with the respective list entries
    ax.set_xticklabels(scans_list)
    ax.set_yticklabels(scans_list)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(n_scans):
        for j in range(i + 1, n_scans):
            text = ax.text(i, j, round(pw_mat[i, j], 1),
                           ha="center", va="center", color="w", size=7)

    ax.set_title(f"Pairwise heatmap for {task_name} {metrics}")
    plt.tight_layout()
    fig.colorbar(im)
    fname = kwargs.get("fname", pathlib.Path(db.basedir) / f'{task_name}-{metrics}_heatmap.png')
    fig.savefig(fname)


def _get_task_fileset(scan_dataset, task_name):
    """Returns the `Fileset` object produced by `task_name` in given `scan_dataset`.

    Parameters
    ----------
    scan_dataset : plantdb.FSDB.Scan
        scan dataset that should contain the unique output of `task_name`
    task_name : string
        name of the task to test

    Returns
    -------
    plantdb.FSDB.Scan.Fileset
        The `Fileset` object produced by `task_name` in given `scan_dataset`

    Raises
    ------
    AssertionError
        If there is more than one `Fileset` object matching the task.
        This mean that this task was run several times with different parameters.

    """
    task_filesets = [fs for fs in scan_dataset.filesets if fs.id.startswith(task_name)]
    try:
        assert len(task_filesets) == 1
    except AssertionError:
        msg = f"There is more than one occurrence of the task {task_name}:\n"
        msg += "  - " + "  - ".join([tfs.id for tfs in task_filesets])
        raise AssertionError(msg)
    return task_filesets[0]


def _get_files(scan_dataset, task_name, unique=False):
    """Returns the `File` object produced by `task_name` in given `scan_dataset`.

    Parameters
    ----------
    scan_dataset : plantdb.FSDB.Scan
        scan dataset that should contain the unique output of `task_name`
    task_name : string
        name of the task to test
    unique : bool, optional
        if ``True``, assert that there is only one `File` in the `Fileset`

    Returns
    -------
    list(plantdb.FSDB.Scan.Fileset.File)
        The `File` objects produced by `task_name` in given `scan_dataset`

    Raises
    ------
    AssertionError
        If there is more than one `File` object in the task `Fileset`, requires ``unique=True``.

    """
    # Get the `Fileset` corresponding to the replicated task:
    fs = _get_task_fileset(scan_dataset, task_name)
    # Get the list of `File` ids (task outputs)
    f_list = [f.id for f in fs.files]
    if unique:
        try:
            assert len(f_list) == 1
        except AssertionError:
            raise AssertionError("This method can only compare tasks that output a single point-cloud!")
    # - Return the `File` objects:
    return [fs.get_file(f) for f in f_list]


def compare_binary_mask(db, task_name):
    """Use set metrics to compare binary masks for each unique pairs of repetition.

    Parameters
    ----------
    db : plantdb.FSDB
        Local ROMI database instance with the replicated scan datasets
    task_name : string
        name of the task to test

    """
    precision, recall = {}, {}
    mean_precision, mean_recall = {}, {}
    median_precision, median_recall = {}, {}
    # List the duplicated `Scan` datasets:
    scans_list = [scan for scan in db.get_scans() if scan.id != 'models']
    logger.info(f"Comparing {task_name} output for replicated scans list: {[s.id for s in scans_list]}")
    # - Double loop to compare all unique pairs of repeats:
    for t, ref_scan in enumerate(scans_list):
        # Get `Fileset` with mask files:
        ref_fileset = _get_task_fileset(ref_scan, task_name)
        if t + 1 > len(scans_list):
            break  # all unique pairs to compare are done! 
        for flo_scan in scans_list[t + 1:]:
            # Get `Fileset` with mask files:
            flo_fileset = _get_task_fileset(flo_scan, task_name)
            # Compare the two mask files list to get the list of 'same files" (ie. with same name)
            ref_dir = pathlib.Path(db.basedir, ref_scan.id, ref_fileset.id)
            flo_dir = pathlib.Path(db.basedir, flo_scan.id, flo_fileset.id)
            common_files = filecmp.dircmp(ref_dir, flo_dir).common_files
            # Remove the extension as the `Fileset.get_file()` method does not accept it!
            common_files = [splitext(sf)[0] for sf in common_files]
            # Dictionary key made of the two repeat names:
            k = f"{ref_scan.id} - {flo_scan.id}"
            precision[k], recall[k] = {}, {}
            # For each file with the same name compute some set metrics:
            for sf in common_files:
                # Read the two mask files:
                ref_mask = read_image(ref_fileset.get_file(sf))
                flo_mask = read_image(flo_fileset.get_file(sf))
                # Compute set metrics for each pair of masks:
                _, _, _, _, p, r = set_metrics(ref_mask, flo_mask)
                precision[k][sf] = p
                recall[k][sf] = r
            # Compute an average & the median for the metrics:
            mean_precision[k] = np.mean(list(precision[k].values()))
            mean_recall[k] = np.mean(list(recall[k].values()))
            median_precision[k] = np.median(list(precision[k].values()))
            median_recall[k] = np.median(list(recall[k].values()))

    global_mean_precision = average_pairwise_comparison(mean_precision)
    global_mean_recall = average_pairwise_comparison(mean_recall)
    global_median_precision = average_pairwise_comparison(median_precision)
    global_median_recall = average_pairwise_comparison(median_recall)
    print("\nAverage precision (closer to 1.0 is better) between pairs of repetitions:")
    print(mean_precision)
    print("\nAverage recall (closer to 1.0 is better) between pairs of repetitions:")
    print(mean_recall)
    print("\nMedian precision (closer to 1.0 is better) between pairs of repetitions:")
    print(median_precision)
    print("\nMedian recall (closer to 1.0 is better) between pairs of repetitions:")
    print(median_recall)
    print(f"Global average precision between pairs of repetitions: {global_mean_precision}")
    print(f"Global average recall between pairs of repetitions: {global_mean_recall}")
    print(f"Global median precision between pairs of repetitions: {global_median_precision}")
    print(f"Global median recall: between pairs of repetitions {global_median_recall}")

    # Write a JSON summary file of the comparison:
    with open(pathlib.Path(db.basedir) / f'{task_name}_comparison.json', 'w') as out_file:
        json.dump({
            'mean precision': mean_precision, 'mean recall': mean_recall,
            'median precision': median_precision, 'median recall': median_recall,
            'precision': precision, 'recall': recall,
            'global mean precision': global_mean_precision, 'global mean recall': global_mean_recall,
            'global median precision': global_median_precision, 'global median recall': global_median_recall
        }, out_file, indent=2)

    return


def jsonify_tuple_keys(json_dict: dict) -> dict:
    return {f'{k[0]} - {k[1]}': v for k, v in json_dict.items()}


def unjsonify_tuple_keys(json_dict: dict) -> dict:
    return {tuple(k.split(' - ')): v for k, v in json_dict.items()}


def average_pairwise_comparison(fbf_comp_dict: dict) -> float:
    if len(fbf_comp_dict.values()):
        return sum(fbf_comp_dict.values()) / len(fbf_comp_dict.values())
    else:
        return None


def compare_pointcloud(db, task_name):
    """Use the chamfer distance to compare point-cloud for each unique pairs of repetition.

    Parameters
    ----------
    db : plantdb.FSDB
        Local ROMI database instance with the replicated scan datasets
    task_name : string
        name of the task to test

    """
    chamfer_dist = {}
    fitness, residuals = {}, {}
    # List the duplicated `Scan` datasets:
    scans_list = [scan for scan in db.get_scans() if scan.id != 'models']
    logger.info(f"Comparing {task_name} output for replicated scans list: {[s.id for s in scans_list]}")
    # - Double loop to compare all unique pairs of repeats:
    for t, ref_scan in enumerate(scans_list):
        # - Read the `PointCloud.ply` file
        ref_pcd_file = _get_files(ref_scan, task_name, unique=True)[0]
        ref_pcd = read_point_cloud(ref_pcd_file)
        if t + 1 > len(scans_list):
            break  # all unique pairs to compare are done! 
        for flo_scan in scans_list[t + 1:]:
            # - Read the `PointCloud.ply` file
            flo_pcd_file = _get_files(flo_scan, task_name, unique=True)[0]
            flo_pcd = read_point_cloud(flo_pcd_file)
            k = (ref_scan.id, flo_scan.id)
            # - Compute the Chamfer distance
            chamfer_dist[k] = chamfer_distance(ref_pcd, flo_pcd)
            # - Compute the registration fitness & residuals
            fitness[k], residuals[k] = point_cloud_registration_fitness(ref_pcd, flo_pcd)

    # Print result to terminal:
    print("\nChamfer distance (lower is better) between pairs of repetitions:")
    print(chamfer_dist)
    # Print result to terminal:
    print("\nFitness (closer to 1.0 is better) between pairs of repetitions:")
    print(fitness)
    print("\nResiduals (inliner RMSE) between pairs of repetitions:")
    print(residuals)
    average_chamfer_dist = average_pairwise_comparison(chamfer_dist)
    print(f"\nAverage Chamfer distance (lower is better) between pairs of repetitions: {average_chamfer_dist}")
    average_fitness = average_pairwise_comparison(fitness)
    print(f"Average fitness (closer to 1.0 is better) between pairs of repetitions: {average_fitness}")
    average_residuals = average_pairwise_comparison(residuals)
    print(f"Average residuals: {average_residuals}")

    # Write a JSON summary file of the comparison:
    with open(pathlib.Path(db.basedir) / f'{task_name}_comparison.json', 'w') as out_file:
        json.dump({
            'chamfer distances': jsonify_tuple_keys(chamfer_dist),
            'fitness': jsonify_tuple_keys(fitness),
            'residuals': jsonify_tuple_keys(residuals),
            'average chamfer distances': average_chamfer_dist,
            'average fitness': average_fitness,
            'average residuals': average_residuals,
        }, out_file)
    # Creates a heatmap of the pariwise comparisons:
    pairwise_heatmap(chamfer_dist, scans_list, 'chamfer distances', task_name, db)
    pairwise_heatmap(fitness, scans_list, 'fitness', task_name, db)
    pairwise_heatmap(residuals, scans_list, 'residuals', task_name, db)

    return


def compare_voxels(db, task_name):
    # List the duplicated `Scan` datasets:
    scans_list = [scan for scan in db.get_scans() if scan.id != 'models']
    logger.info(f"Comparing {task_name} output for replicated scans list: {[s.id for s in scans_list]}")
    # Initialize the list of labels with the first NPZ file in the list
    ref_scan = scans_list[0]
    ref_voxel_file = _get_files(ref_scan, task_name, unique=True)[0]
    ref_voxel = read_npz(ref_voxel_file)
    labels = list(ref_voxel.keys())
    # Initialize returned dictionary with labels as keys!
    r2 = {label: {} for label in labels}
    mean_abs_dev = {label: {} for label in labels}
    # - Double loop to compare all unique pairs of repeats:
    for t, ref_scan in enumerate(scans_list):
        ref_voxel_file = _get_files(ref_scan, task_name, unique=True)[0]
        ref_voxel = read_npz(ref_voxel_file)
        if t + 1 > len(scans_list):
            break  # all unique pairs to compare are done!
        for flo_scan in scans_list[t + 1:]:
            flo_voxel_file = _get_files(flo_scan, task_name, unique=True)[0]
            flo_voxel = read_npz(flo_voxel_file)
            k = (ref_scan.id, flo_scan.id)
            for label in labels:
                r2[label][k] = np.sum(np.sqrt(ref_voxel[label] - flo_voxel[label]))
                mean_abs_dev[label][k] = np.mean(np.abs(ref_voxel[label] - flo_voxel[label]))

    # Print result to terminal:
    print("\nVoxels sum of square difference (lower is better) between pairs of repetitions:")
    print(r2)
    print("\nVoxels mean absolute deviation (lower is better) between pairs of repetitions:")
    print(mean_abs_dev)
    average_r2 = {}
    average_mean_abs_dev = {}
    for label in labels:
        average_r2[label] = average_pairwise_comparison(r2[label])
        average_mean_abs_dev[label] = average_pairwise_comparison(mean_abs_dev[label])
        print(f"Average voxels sum of square difference (lower is better) between pairs of repetitions for label "
              f"{label}: {average_r2[label]}")
        print(f"Average voxels mean absolute deviation (lower is better) between pairs of repetitions for label "
              f"{label}: {average_mean_abs_dev[label]}")

    # Write a JSON summary file of the comparison:
    with open(pathlib.Path(db.basedir) / f'{task_name}_comparison.json', 'w') as out_file:
        json.dump({
            'sum of square difference': {label: jsonify_tuple_keys(r2[label]) for label in labels},
            'mean absolute deviation': {label: jsonify_tuple_keys(mean_abs_dev[label]) for label in labels},
            'average pairwise sum of square difference': {label: average_r2[label] for label
                                                          in labels},
            'average pairwise mean absolute deviation': {label: average_mean_abs_dev[label] for label
                                                         in labels}
        }, out_file)

    # Creates a heatmap of the pariwise comparisons:
    for label in labels:
        metric_str = f'{label} sum of square difference'
        pairwise_heatmap(r2[label], scans_list, metric_str, task_name, db,
                         fname=pathlib.Path(db.basedir) / f"{task_name}-{metric_str.replace(' ', '_')}_heatmap.png")
        metric_str = f'{label} mean absolute deviation'
        pairwise_heatmap(mean_abs_dev[label], scans_list, f'{label} mean absolute deviation', task_name, db,
                         fname=pathlib.Path(db.basedir) / f"{task_name}-{metric_str.replace(' ', '_')}_heatmap.png")


def compare_labelled_pointcloud(db, task_name):
    # List the duplicated `Scan` datasets:
    scans_list = [scan for scan in db.get_scans() if scan.id != 'models']
    logger.info(f"Comparing {task_name} output for replicated scans list: {[s.id for s in scans_list]}")
    # Initialize the list of labels with the first PLY file in the list
    ref_scan = scans_list[0]
    ref_voxel_file = _get_files(ref_scan, task_name, unique=True)[0]
    unique_labels = list(set(ref_voxel_file.get_metadata('labels')))

    precision = {ulabel: {} for ulabel in unique_labels}
    recall = {ulabel: {} for ulabel in unique_labels}

    # - Double loop to compare all unique pairs of repeats:
    for ref_idx, ref_scan in enumerate(scans_list):
        # - Read the `PointCloud.ply` file
        ref_pcd_file = _get_files(ref_scan, task_name, unique=True)[0]
        ref_pcd = read_point_cloud(ref_pcd_file)
        ref_labels = ref_pcd_file.get_metadata('labels')
        ref_pcd_tree = o3d.geometry.KDTreeFlann(ref_pcd)
        if ref_idx + 1 > len(scans_list):
            break  # all unique pairs to compare are done!
        for flo_scan in scans_list[ref_idx + 1:]:
            # - Read the `PointCloud.ply` file
            flo_pcd_file = _get_files(flo_scan, task_name, unique=True)[0]
            flo_pcd = read_point_cloud(flo_pcd_file)
            flo_labels = flo_pcd_file.get_metadata('labels')
            res = {ulabel: {"tp": 0, "fp": 0, "tn": 0, "fn": 0} for ulabel in unique_labels}
            # For each point of the floating point-cloud...
            for i, pts in enumerate(flo_pcd.points):
                # ... get its label
                label_i = flo_labels[i]
                # ... search for the closest point in reference neighborhood tree
                [k, idx, _] = ref_pcd_tree.search_knn_vector_3d(pts, 1)
                for ulabel in unique_labels:
                    if label_i == ulabel:  # Positive cases
                        if label_i == ref_labels[idx[0]]:
                            res[ulabel]["tp"] += 1
                        else:
                            res[ulabel]["fp"] += 1
                    else:  # Negative cases
                        if label_i == ref_labels[idx[0]]:
                            res[ulabel]["tn"] += 1
                        else:
                            res[ulabel]["fn"] += 1
            k = (ref_scan.id, flo_scan.id)
            for ulabel in unique_labels:
                precision[ulabel][k] = res[ulabel]["tp"] / (res[ulabel]["tp"] + res[ulabel]["fp"])
                recall[ulabel][k] = res[ulabel]["tp"] / (res[ulabel]["tp"] + res[ulabel]["fn"])

    # Print result to terminal:
    print("\nPrecision (closer to 1.0 is better) between pairs of repetitions:")
    print(precision)
    # Print result to terminal:
    print("\nRecall (closer to 1.0 is better) between pairs of repetitions:")
    print(recall)

    average_precision = {}
    average_recall = {}
    for label in unique_labels:
        average_precision[label] = average_pairwise_comparison(precision[label])
        average_recall[label] = average_pairwise_comparison(recall[label])
        print(f"Average precision (closer to 1.0 is better) between pairs of repetitions for label "
              f"{label}: {average_precision[label]}")
        print(f"Average recall (closer to 1.0 is better) between pairs of repetitions for label "
              f"{label}: {average_recall[label]}")

    # Write a JSON summary file of the comparison:
    with open(pathlib.Path(db.basedir) / f'{task_name}_comparison.json', 'w') as out_file:
        json.dump({
            'precision': {ulabel: jsonify_tuple_keys(precision[ulabel]) for ulabel in unique_labels},
            'recall': {ulabel: jsonify_tuple_keys(recall[ulabel]) for ulabel in unique_labels},
            'average pairwise precision': {label: average_precision[label] for label in unique_labels},
            'average pairwise recall': {label: average_recall[label] for label in unique_labels}
        }, out_file)

    # Creates a heatmap of the pariwise comparisons:
    for ulabel in unique_labels:
        metric_str = f'{ulabel} precision'
        pairwise_heatmap(precision[ulabel], scans_list, metric_str, task_name, db,
                         fname=pathlib.Path(db.basedir) / f"{task_name}-{metric_str.replace(' ', '_')}_heatmap.png")
        metric_str = f'{ulabel} recall'
        pairwise_heatmap(recall[ulabel], scans_list, f'{ulabel} recall', task_name, db,
                         fname=pathlib.Path(db.basedir) / f"{task_name}-{metric_str.replace(' ', '_')}_heatmap.png")


def compare_trianglemesh_points(db, task_name):
    """Use the chamfer distance to compare mesh vertices point-cloud for each unique pairs of repetition.

    Parameters
    ----------
    db : plantdb.FSDB
        Local ROMI database instance with the replicated scan datasets
    task_name : string
        name of the task to test

    """
    chamfer_dist = {}
    surf_ratio = {}
    vol_ratio = {}
    # List the duplicated `Scan` datasets:
    scans_list = [scan for scan in db.get_scans() if scan.id != 'models']
    logger.info(f"Comparing {task_name} output for replicated scans list: {[s.id for s in scans_list]}")
    # - Double loop to compare all unique pairs of repeats:
    for t, ref_scan in enumerate(scans_list):
        # - Read the `TriangleMesh.ply` file
        ref_mesh_file = _get_files(ref_scan, task_name, unique=True)[0]
        ref_mesh = read_triangle_mesh(ref_mesh_file)
        # - Extract a PointCloud from the mesh vertices
        ref_pcd = o3d.geometry.PointCloud(ref_mesh.vertices)
        if t + 1 > len(scans_list):
            break  # all unique pairs to compare are done! 
        for flo_scan in scans_list[t + 1:]:
            # - Read the `TriangleMesh.ply` file
            flo_mesh_file = _get_files(flo_scan, task_name, unique=True)[0]
            flo_mesh = read_triangle_mesh(flo_mesh_file)
            # - Extract a PointCloud from the mesh vertices
            flo_pcd = o3d.geometry.PointCloud(flo_mesh.vertices)
            k = (ref_scan.id, flo_scan.id)
            # - Compute the Chamfer distance
            chamfer_dist[k] = chamfer_distance(ref_pcd, flo_pcd)
            surf_ratio[k] = surface_ratio(ref_mesh, flo_mesh)
            vol_ratio[k] = volume_ratio(ref_mesh, flo_mesh)

    print("\nChamfer distance (lower is better) between pairs of repetitions:")
    print(chamfer_dist)
    print("\nSurface ratio (closer to 1.0 is better) between pairs of repetitions:")
    print(surf_ratio)
    print("\nVolume ratio (closer to 1.0 is better) between pairs of repetitions:")
    print(vol_ratio)
    average_chamfer_dist = average_pairwise_comparison(chamfer_dist)
    print(f"\nAverage Chamfer distance (lower is better) between pairs of repetitions: {average_chamfer_dist}")
    average_surf_ratio = average_pairwise_comparison(surf_ratio)
    print(f"Average surface ratio (closer to 1.0 is better) between pairs of repetitions: {average_surf_ratio}")
    average_vol_ratio = average_pairwise_comparison(vol_ratio)
    print(f"Average volume ratio (closer to 1.0 is better) between pairs of repetitions: {average_vol_ratio}")

    # Write a JSON summary file of the comparison:
    with open(pathlib.Path(db.basedir) / f'{task_name}_comparison.json', 'w') as out_file:
        json.dump({
            'chamfer distances': jsonify_tuple_keys(chamfer_dist),
            'surface ratio': jsonify_tuple_keys(surf_ratio),
            'volume ratio': jsonify_tuple_keys(vol_ratio),
            'average chamfer distances': average_chamfer_dist,
            'average surface ratio': average_surf_ratio,
            'average volume ratio': average_vol_ratio
        }, out_file)

    return


def compare_curveskeleton_points(db, task_name):
    """Use the chamfer distance to compare point-cloud for each unique pairs of repetition.

    Parameters
    ----------
    db : plantdb.FSDB
        Local ROMI database instance with the replicated scan datasets
    task_name : string
        name of the task to test

    """
    chamfer_dist = {}
    # List the duplicated `Scan` datasets:
    scans_list = [scan for scan in db.get_scans() if scan.id != 'models']
    logger.info(f"Comparing {task_name} output for replicated scans list: {[s.id for s in scans_list]}")
    # - Double loop to compare all unique pairs of repeats:
    for t, ref_scan in enumerate(scans_list):
        # - Read the `CurveSkeleton.json` file
        ref_json_file = _get_files(ref_scan, task_name, unique=True)[0]
        ref_json = read_json(ref_json_file)
        # - Extract a PointCloud from the skeleton vertices
        ref_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ref_json["points"]))
        if t + 1 > len(scans_list):
            break  # all unique pairs to compare are done!
        for flo_scan in scans_list[t + 1:]:
            # - Read the `CurveSkeleton.json` file
            flo_json_file = _get_files(flo_scan, task_name, unique=True)[0]
            flo_json = read_json(flo_json_file)
            # - Extract a PointCloud from the skeleton vertices
            flo_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(flo_json["points"]))
            k = f"{ref_scan.id} - {flo_scan.id}"
            # - Compute the Chamfer distance
            chamfer_dist[k] = chamfer_distance(ref_pcd, flo_pcd)

    print("\nChamfer distance (lower is better) between pairs of repetitions:")
    print(chamfer_dist)
    average_chamfer_dist = average_pairwise_comparison(chamfer_dist)
    print(f"\nAverage Chamfer distance (lower is better) between pairs of repetitions: {average_chamfer_dist}")

    # Write a JSON summary file of the comparison:
    with open(pathlib.Path(db.basedir) / f'{task_name}_comparison.json', 'w') as out_file:
        json.dump({'chamfer distances': chamfer_dist,
                   'average chamfer distance': average_chamfer_dist}, out_file)

    return


def angles_and_internodes_comparison(db, task_name):
    """Coarse comparison of sequences of angles and internodes.

    Prints a dict containing the number of scans for each number of organs indexed, ex : {12:4, 14:6} (4 scans with 12
    organs recognized and 6 with 14 organs)
    For the biggest number of scans with same number of organs, creates plots of the repartition of angles (and
    internodes) for each organ id

    Parameters
    ----------
    db : plantdb.FSDB
        Local ROMI database instance with the replicated scan datasets
    task_name : string
        name of the task to test

    """
    angles_and_internodes = {}
    nb_organs = []
    # List the duplicated `Scan` datasets:
    scans_list = [scan for scan in db.get_scans() if scan.id != 'models']
    logger.info(f"Comparing {task_name} output for replicated scans list: {[s.id for s in scans_list]}")
    # - Load all JSON files with angles and internodes values:
    for t, scan in enumerate(scans_list):
        # - Read the `AnglesAndInternodes.json` file
        json_file = _get_files(scan, task_name, unique=True)[0]
        ae_dict = read_json(json_file)
        angles_and_internodes[scan.id] = {
            "angles": np.array(ae_dict["angles"]) * 180 / np.pi,
            "internodes": ae_dict["internodes"],
            "nb_organs": len(ae_dict["angles"])
        }
        nb_organs.append(len(ae_dict["angles"]))

    counter_nb_organs = Counter(nb_organs)
    print(" ** comparison results ** ")
    print("number of scans with the same nb of organs: ", counter_nb_organs)
    print("Min number of organs detected: ", min(nb_organs))
    print("Max number of organs detected: ", max(nb_organs))
    print("Average number of organs detected: ", np.mean(nb_organs))

    max_occurrence_nb_organs = counter_nb_organs.most_common()[0][0]
    angles = [angles_and_internodes[scan_num]["angles"] for scan_num in angles_and_internodes
              if angles_and_internodes[scan_num]["nb_organs"] == max_occurrence_nb_organs]
    internodes = [angles_and_internodes[scan_num]["internodes"] for scan_num in angles_and_internodes
                  if angles_and_internodes[scan_num]["nb_organs"] == max_occurrence_nb_organs]
    save_data_repartition(angles, "angles", pathlib.Path(db.basedir))
    save_data_repartition(internodes, "internodes", pathlib.Path(db.basedir))

    # Write a JSON summary file of the comparison:
    with open(pathlib.Path(db.basedir) / f'{task_name}_comparison.json', 'w') as out_file:
        json.dump({
            'Min number of organs detected': min(nb_organs),
            'Max number of organs detected': max(nb_organs),
            'Average number of organs detected': np.mean(nb_organs),
            'number of scans with the same nb of organs': counter_nb_organs
        }, out_file)


def file_by_file_comparison(db, task_name):
    """
    Compares task folder output file by file, print result

    Parameters
    ----------
    db : plantdb.FSDB
        Local ROMI database instance with the replicated scan datasets
    task_name : string
        name of the task to test

    """
    logger.info("Performing file-by-file comparisons...")
    fcmp = {}
    scans_list = [scan for scan in db.get_scans() if scan.id != 'models']
    for t, ref_scan in enumerate(scans_list):
        if t + 1 > len(scans_list):
            break
        for flo_scan in scans_list[t + 1:]:
            ref_dir = pathlib.Path(db.basedir, ref_scan.id, _get_task_fileset(ref_scan, task_name).id)
            flo_dir = pathlib.Path(db.basedir, flo_scan.id, _get_task_fileset(flo_scan, task_name).id)
            n_diff = len(filecmp.dircmp(ref_dir, flo_dir).diff_files)
            n_same = len(filecmp.dircmp(ref_dir, flo_dir).same_files)
            if n_same == 0:
                similarity = 0
            else:
                similarity = (n_same / (n_diff + n_same)) * 100
            fcmp[f"{ref_scan.id} - {flo_scan.id}"] = similarity

    # Print result to terminal:
    print("File-by-file comparison (similarity percentage) between pairs of repetitions:")
    print(fcmp)
    average_similarity = average_pairwise_comparison(fcmp)
    print(f"Mean Similarity: {average_similarity:.2f} %")
    print()

    # Write a JSON summary file of the comparison:
    with open(pathlib.Path(db.basedir) / 'filebyfile_comparison.json', 'w') as out_file:
        json.dump({'Similarity (%)': fcmp, 'Mean similarity (%)': average_similarity}, out_file)


def compare_task_output(db, task_name, independant_tests=False):
    """
    Method to compare outputs of a task on replicated datasets.

    Parameters
    ----------
    db : plantdb.FSDB
        Local ROMI database instance with the replicated scan datasets
    task_name : string
        name of the task to test
    independant_tests : bool
        if ``True`` indicate all tasks were performed independently and will be evaluated, else only evaluate `task_name`

    Notes
    -----
    The behaviour of this method is configured by the JSON file `CONF_FILE`.

    """
    # List paths where tested task output(s) should be located
    folder_task_list = [pathlib.Path(db.basedir, scan.id, _get_task_fileset(scan, task_name).id) for scan in
                        db.get_scans() if scan.id != 'models']
    if len(folder_task_list) == 0:
        raise IOError(f"Output files of task {task_name} for db {db.basedir} are missing")
    # - Performs file-by-file comparisons for the selected task:
    file_by_file_comparison(db, task_name)
    # - Load the metric for the selected task from the JSON configuration file:
    c = json.load(open(CONF_FILE))
    # - Evaluate all tasks if independent tests, else evaluate only tested task:
    if independant_tests:
        for task in c.keys():
            try:
                eval(c[task]["comp_func"] + "(db, task)")
                print()  # make blocks of evaluation task summary clearer!
            except:
                pass
    else:
        try:
            eval(c[task_name]["comp_func"] + "(db, task_name)")
        except NameError:
            logger.warning(f"The comparison function {c[task_name]['comp_func']} provided in the "
                           f"conf_robustness_comparison.json doesn't exist")


def fill_test_db(test_db, init_scan_path, task_cfg, nb, models_path):
    """
    From an initial scan, copy it in temporary folder, cleans it,
    runs the pipe to the comparison point task and copy the scan a certain
    number of times in a test folder

    Parameters
    ----------
    test_db : pathlib.Path
        name of the folder into which copy the tests scans
    init_scan_path : pathlib.Path
        path of the initial scan dataset to use for repeatability test
    task_cfg : dict
        name of the configuration file to run the pipeline
    nb : int
        number time to repeat the same task to evaluate
    models_path : pathlib.Path
        name of models folder for segmentation2D

    Returns
    -------
    list
        list of pathlib.Path of created scan datasets

    """
    scan_name = init_scan_path.name
    created_copied_scans = []
    # Initialize a temporary directory use to store scan dataset before cleaning and running previous tasks to task to analyse:
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Get the path of the temporary ROMI database
        tmp_path = pathlib.Path(tmp_dir)
        tmp_scan_folder = pathlib.Path(tmp_path / f"tmp_{scan_name}")
        logger.info(f"Created a temporary folder '{tmp_scan_folder}'")
        # Creates the `romidb` marker for active FSDB
        with open(pathlib.Path(tmp_path / MARKER_FILE_NAME), 'w'):
            pass
        logger.info("Duplicating the scan dataset to a temporary folder...")
        # if models are needed, copies models folder to temporary folder
        if models_path:
            try:
                shutil.copytree(str(models_path), str(tmp_path / "models"))
                shutil.copytree(str(models_path), str(test_db / "models"))
            except:
                pass

        # Duplicate the initial scan dataset to the temporary folder
        shutil.copytree(str(init_scan_path), str(tmp_scan_folder))
        # Check for previous analysis & if yes, clean the dataset
        init_configuration_file = pathlib.Path(tmp_scan_folder / "pipeline.toml")
        if init_configuration_file.is_file():
            logger.info("Using detected configuration pipeline backup file to clean the scan dataset!")
            print(str(init_configuration_file))
            run_pipe(tmp_scan_folder, "Clean", str(init_configuration_file))
        # Run the previous task on the temporary copy
        if task_cfg["previous_task"]:
            logger.info(f"Running {task_cfg['previous_task']} pipeline on the scan dataset!")
            run_pipe(tmp_scan_folder, task_cfg["previous_task"], task_cfg["config_file"])
        logger.info(f"Copying previous run of {task_cfg['previous_task']} pipeline...")
        # Duplicate the results of the previous task to the test database
        for i in range(nb):
            logger.info(f"Copy #{i}...")
            copied_scan_name = pathlib.Path(test_db / f"{scan_name}_{i}")
            shutil.copytree(str(tmp_scan_folder), str(copied_scan_name))
            created_copied_scans.append(copied_scan_name)
    return created_copied_scans


def create_test_db_path(root_location, task_name):
    """
    Generate test database name, ex: 20200803124841_rep_test_TriangleMesh

    Parameters
    ----------
    root_location : pathlib.Path
        Root path where to create the test database.
    task_name : str
        Name of the task to test.

    Returns
    -------
    pathlib.Path
        Path to the test database.
    """
    now = datetime.datetime.now()
    now_str = now.strftime("%Y%m%d%H%M%S")
    return pathlib.Path(root_location / f"{now_str}_rep_test_{task_name}")


def create_test_db(root_location, task_name):
    """
    Creates test db with `romidb` marker file in it.

    Parameters
    ----------
    root_location : pathlib.Path
        Root path where to create the test database.
    task_name : str
        Name of the task to test.

    Returns
    -------
    pathlib.Path
        Path to the test folder.

    """
    # Get the path of the ROMI database used to perform the test and creates the directory
    test_db = create_test_db_path(root_location, task_name)
    test_db.mkdir(exist_ok=True)
    # Creates the marker file to get an active plantdb.FSDB:
    marker_file = test_db / MARKER_FILE_NAME
    marker_file.touch()
    return test_db


def run_pipe(scan_path, task_name, cfg_file):
    """
    run configured pipeline for given task on a scan

    Parameters
    ----------
    scan_path : pathlib.Path
        contains information on database
    task_name : string
        task name, must be a key in the CONF_FILE
    cfg_file : string
        name of the configuration file to run the pipeline

    Returns
    -------

    """
    logger.info(f"Executing tasks '{task_name}' on scan dataset '{str(scan_path)}'.")
    # TODO: use luigi.build() instead of subprocess.run call ?
    cmd = ["romi_run_task", "--config", cfg_file, task_name, str(scan_path), "--local-scheduler"]
    subprocess.run(cmd, check=True)
    return


def compute_repeatability_test(db, task_cfg):
    """
    Creates test folder with copies of the input scan and runs repeatability tests on it

    Parameters
    ----------
    db : plantdb.FSDB
        Local ROMI database instance with the replicated scan datasets
    task_cfg : dict
        configuration dict of the task to test

    """
    scans_list = [scan.id for scan in db.get_scans() if scan.id != 'models']
    logger.info(f"Repeating task {task_cfg['name']} on replicated scans list: {scans_list}")
    test_db.disconnect()  # to allow the luigi pipeline to run!
    # Execute the task to test for all scans in the test database:
    for copied_scan in scans_list:
        copied_scan_path = pathlib.Path(db.basedir, copied_scan)
        run_pipe(copied_scan_path, task_cfg["name"], task_cfg["config_file"])
    # Reconnect to the DB before leaving...
    test_db.connect()
    return


def config_task(task_name, cfg_file, full_pipe, previous_task):
    """
    Setup task configuration dictionary.

    Parameters
    ----------
    task_name : string
        task name, must be a key in the CONF_FILE
    cfg_file : string
        name of the configuration file to run the pipeline
    full_pipe : bool
        whether or not compute repeatability from the start
    previous_task : string
        name of the previous class in the pipeline, either None for the full pipe or the one linked to the class name
        in CONF_FILE (comparison point task)

    Returns
    -------
    dict
        configuration dict of the task to test

    """
    task_cfg = {
        "name": task_name,
        "config_file": cfg_file
    }
    if full_pipe:
        task_cfg["previous_task"] = None
    else:
        task_cfg["previous_task"] = previous_task
    return task_cfg


def run():
    """
    creates a test db at the root of the db linked to the scan to analyze
    """
    DESC = """ROMI reconstruction & analysis pipeline repeatability test procedure.

    Analyse the repeatability of a reconstruction & analysis pipeline by:
    1. duplicating the scan in a temporary folder (and cleaning it if necessary)
    2. running the pipeline up to the previous task of the task to test
    3. copying this result to a new database and replicate the dataset
    4. repeating the task to test for each replicate
    5. comparing the results pair by pair.

    Comparison can be done at the scale of the files but also with metrics if a reference can be set.
    
    To create fully independent tests, we run the pipeline up to the task to test on each replicate.
    
    Note that in order to use the ML pipeline, you will first have to:
    1. create an output directory
    2. use the `--models` argument to copy the CNN trained models required to run the pipeline.
    """
    # - Load the JSON config file of the script:
    c = json.load(open(CONF_FILE))
    valid_tasks = list(c.keys())

    # TODO: an option to set a reference (or would that be the job of the Evaluation tasks?!)
    # TODO: use the `pipeline.toml` to defines the previous tasks ?

    parser = argparse.ArgumentParser(description=DESC)

    parser.add_argument("scan",
                        help="scan to use for repeatability analysis")
    parser.add_argument("task", default="AnglesAndInternodes",
                        choices=valid_tasks,
                        help=f"task to test, should be in: {', '.join(valid_tasks)}")
    parser.add_argument("config_file",
                        help="path to the TOML config file of the analysis pipeline")
    parser.add_argument("-n", "--replicate_number", default=2,
                        help="number of replicate to use for repeatability analysis")
    parser.add_argument("-f", "--full_pipe", action="store_true", default=False,
                        help="run the analysis pipeline on each replicate independently")
    parser.add_argument("-np", "--no_pipeline", action="store_true", default=False,
                        help="do not run the pipeline, only compare tasks outputs")
    parser.add_argument("-db", '--test_database',
                        help="test database location to use. Use at your own risks!")
    parser.add_argument("--models", default=False,
                        help="models database location to use with ML pipeline.")
    # ->Test database path exists:
    #   - NO: create the directory & use it instead of auto-generated path
    #   - YES:
    #     ->Test database contain scan datasets
    #       - NO: use it instead of auto-generated path
    #       - YES: use number of scans as replicate number & compute repeatability test

    # - Parse the input arguments to variables:
    args = parser.parse_args()
    task2test = args.task
    replicate_number = int(args.replicate_number)
    init_scan_path = pathlib.Path(args.scan).expanduser()
    db_location = init_scan_path.parent
    root_location = db_location.parent
    if args.models:
        models_path = pathlib.Path(args.models).expanduser()
    else:
        models_path = None
    logger.info(f"Got scan path: {init_scan_path}")

    # - Task configuration
    task_cfg = config_task(task2test, args.config_file, args.full_pipe, c[task2test]["prev_task"])
    # - 'test database' setup & dataset duplication:
    if args.test_database is not None:
        logger.info(f"Got a test database location as argument: '{args.test_database}'.")
        test_db_path = pathlib.Path(args.test_database)
        test_db_path.mkdir(exist_ok=True)
    else:
        # test db is created next to db containing original scan dataset
        test_db_path = create_test_db(root_location, task_cfg["name"])

    scan_replicates = list([idir.stem for idir in test_db_path.iterdir() if idir.is_dir() and idir.stem != 'models'])
    n_replicates = len(scan_replicates)
    # Case without/with scan datasets in the test database root folder
    if n_replicates == 0:
        # Duplicate the scan dataset as many times as requested
        test_scans = fill_test_db(test_db_path, init_scan_path, task_cfg, replicate_number, models_path)
    else:
        # If there is something in the root folder that mean you passed a value to `--test_database`
        # And we then consider that it already contains the replicated scans!
        logger.info(
            f"Found {n_replicates} scan replicates in existing test database: {', '.join(map(str, scan_replicates))}")
        # Make sure the lock is OFF!
        marker_file = test_db_path / MARKER_FILE_NAME
        lock_file = test_db_path / LOCK_FILE_NAME
    try:
        marker_file.touch()
    except:
        pass
    try:
        lock_file.unlink()
        # lock_file.unlink(missing_ok=True)  # missing_ok only available since Python3.8
    except:
        pass

    # - Instantiate ROMI database
    test_db = FSDB(str(test_db_path))
    test_db.connect()

    if not args.no_pipeline:
        # - Repeat task on all replicated scans dataset of the test database:
        compute_repeatability_test(test_db, task_cfg)

    # - Compare the output(s) of the task across replicated scans dataset:
    compare_task_output(test_db, task_cfg["name"], independant_tests=args.full_pipe)


if __name__ == "__main__":
    run()