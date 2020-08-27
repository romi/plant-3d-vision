#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Python script to call after a ROMI task to print a task summary.
"""

import argparse
import json
import os

import numpy as np
import toml
from romiscan.modules import TASKS

HELP_URL = "https://docs.romi-project.eu/Scanner/metadata/tasks_metadata/"


def parsing():
    parser = argparse.ArgumentParser(
        description='Print a summary of a ROMI task.',
        epilog="""See {} for help with configuration files.
                                     """.format(HELP_URL))

    parser.add_argument('task', metavar='task', type=str,
                        choices=TASKS,
                        help=f"Choose a ROMI task in: {', '.join(TASKS)}.")

    parser.add_argument('db_path', metavar='dataset_path', type=str,
                        help='FSDB scan dataset to process (path).')

    return parser


def get_dataset_size(db_path):
    return len([f for f in os.listdir(os.path.join(db_path, 'images')) if os.path.splitext(f)[1] in ['.jpg', '.png']])


def json_metadata(task, task_id, db_path):
    """ Get task metadata JSON.

    Parameters
    ----------
    task : str
        Name of the task.
    task_id : str
        Id of the tasks.
    db_path : str
        Path to scan dataset.

    Returns
    -------
    str
        JSON location.
    dict
        Loaded JSON dictionary

    """
    json_f = os.path.join(db_path, task_id, f"{task}.json")
    return json_f, json.load(open(json_f, 'r'))


def metadata_info(task, task_id, db_path):
    """Dump a JSON metadata file for a given task & dataset. """
    json_f, md_json = json_metadata(task, task_id, db_path)
    print(json.dumps(md_json, sort_keys=True, indent=2))


def angles_and_internodes_info(task, task_id, db_path):
    """Print info about AnglesAndInternodes task output."""
    json_f, md_json = json_metadata(task, task_id, db_path)
    for md_info in ["angles", "internodes"]:
        try:
            data = md_json[md_info]
            print("Found the following {} between successive organs:".format(md_info))
            print([round(d, 1) for d in data])
        except KeyError as e:
            print("Could not find {} entry in {}".format(e, json_f))


def curve_skeleton_info(task, task_id, db_path):
    json_f, md_json = json_metadata(task, task_id, db_path)
    print(f"Found a skeleton of {len(md_json['points'])} points.")


def pointcloud_info(task, task_id, db_path):
    """Print info about PointCloud task output."""
    from romidata.io import read_point_cloud, dbfile_from_local_file
    ply_f = os.path.join(db_path, task_id, f"{task}.ply")
    ply = read_point_cloud(dbfile_from_local_file(ply_f))
    if ply.is_empty():
        print(f"PLY file for task '{task_id}' is empty!")
    else:
        print(f"Found PLY file for task '{task_id}':")
    print(f" - {len(ply.points)} points")
    print(f" - pointcloud dimensions (x, y, z): {ply.get_max_bound() - ply.get_min_bound()}")


def segmented_pointcloud_info(task, task_id, db_path):
    """Print info about SegmentedPointCloud task output."""
    from romidata.io import read_point_cloud, dbfile_from_local_file
    ply_f = os.path.join(db_path, task_id, f"{task}.ply")
    ply = read_point_cloud(dbfile_from_local_file(ply_f))
    if ply.is_empty():
        print(f"PLY file for task '{task_id}' is empty!")
    else:
        print(f"Found PLY file for task '{task_id}':")
    print(f" - {len(ply.points)} points")
    print(f" - {len(np.unique(np.asarray(ply.colors), axis=0))} unique colors")
    print(f" - pointcloud dimensions (x, y, z): {ply.get_max_bound() - ply.get_min_bound()}")


def triangle_mesh_info(task, task_id, db_path):
    """Print info about TriangleMesh task output."""
    from romidata.io import read_triangle_mesh, dbfile_from_local_file
    ply_f = os.path.join(db_path, task_id, f"{task}.ply")
    ply = read_triangle_mesh(dbfile_from_local_file(ply_f))
    if ply.is_empty():
        print(f"PLY file for task '{task_id}' is empty!")
    else:
        print(f"Found PLY file for task '{task_id}':")
    print(f" - {len(ply.vertices)} vertices")
    print(f" - {len(ply.triangles)} triangles")
    print(f" - mesh dimensions (x, y, z): {ply.get_max_bound() - ply.get_min_bound()}")


def segmentation2d_info(task, task_id, db_path):
    """Print info about ClusteredMesh task output."""
    out_dir = os.path.join(db_path, task_id)
    files = os.listdir(out_dir)
    png_files = [f for f in files if f.endswith('.png')]
    organs = {}
    for png in png_files:
        organ = os.path.splitext(png.split('_')[1])[0]
        if organ in organs.keys():
            organs[organ] += 1
        else:
            organs[organ] = 1

    n_imgs = get_dataset_size(db_path)
    print(f"{task_id} detected the following organs in RGB images: {', '.join(list(organs.keys()))}")
    print(f"{task_id} organ detected:\n - ", end="")
    print("\n - ".join([f"{v}/{n_imgs} for {k}" for k, v in organs.items()]))


def clustered_mesh_info(task, task_id, db_path):
    """Print info about ClusteredMesh task output."""
    from romidata.io import read_triangle_mesh, dbfile_from_local_file
    out_dir = os.path.join(db_path, task_id)
    files = os.listdir(out_dir)
    ply_files = [f for f in files if f.endswith('.ply')]
    organs = {}
    for ply in ply_files:
        ply_f = os.path.join(db_path, task_id, ply)
        ply_mesh = read_triangle_mesh(dbfile_from_local_file(ply_f))
        if ply_mesh.is_empty():
            continue  # don't count empty PLY as valid organs!
        organ = ply.split('_')[0]
        if organ in organs.keys():
            organs[organ] += 1
        else:
            organs[organ] = 1

    print(f"{task_id} organ reconstruction:\n - ", end="")
    print("\n - ".join([f"{v}x {k}" for k, v in organs.items()]))


def masks_info(task, task_id, db_path):
    out_dir = os.path.join(db_path, task_id)
    files = os.listdir(out_dir)
    img_files = [f for f in files if f.endswith('.jpg')]
    print(f"Found {len(img_files)} mask image files.")


def undistorted_info(task, task_id, db_path):
    out_dir = os.path.join(db_path, task_id)
    files = os.listdir(out_dir)
    img_files = [f for f in files if f.endswith('.jpg')]
    print(f"Found {len(img_files)} undistorted image files.")


def colmap_info(task, task_id, db_path):
    out_dir = os.path.join(db_path, task_id)
    try:
        cam_json = json.load(open(os.path.join(out_dir, "cameras.json"), 'r'))
    except FileNotFoundError:
        print("Could not find COLMAP estimated camera parameters!")
    else:
        print("Estimated camera parameters:")
        print(json.dumps(cam_json["1"], sort_keys=True, indent=2))

    try:
        imgs_json = json.load(open(os.path.join(out_dir, "images.json"), 'r'))
    except FileNotFoundError:
        print("Could not find COLMAP estimated camera parameters!")
    else:
        n_xys = [len(img_json["xys"]) for _, img_json in imgs_json.items()]
        av_n_xys = round(sum(n_xys) / len(imgs_json), 2)
        print(f"Average number of 2D keypoints per image: {av_n_xys}")
        print(f"Min number of 2D keypoints per image: {min(n_xys)}")
        print(f"Max number of 2D keypoints per image: {max(n_xys)}")


def info_from_task(task, task_id, db_path):
    if task == "AnglesAndInternodes":
        return angles_and_internodes_info(task, task_id, db_path)
    elif task == "TreeGraph":
        return NotImplementedError
    elif task == "CurveSkeleton":
        return curve_skeleton_info(task, task_id, db_path)
    elif task == "TriangleMesh":
        return triangle_mesh_info(task, task_id, db_path)
    elif task == "PointCloud":
        return pointcloud_info(task, task_id, db_path)
    elif task == "Voxels":
        # Heavy NPZ file to read, not very informative...
        return NotImplementedError
    elif task == "Masks":
        return masks_info
    elif task == "Colmap":
        return colmap_info(task, task_id, db_path)
    elif task == "Undistorted":
        return undistorted_info(task, task_id, db_path)
    ## - ML pipeline specific tasks:
    elif task == "Segmentation2D":
        return segmentation2d_info(task, task_id, db_path)
    elif task == "SegmentedPointCloud":
        return segmented_pointcloud_info(task, task_id, db_path)
    elif task == "ClusteredMesh":
        return clustered_mesh_info(task, task_id, db_path)
    else:
        return NotImplementedError


def list_configured_tasks(toml_conf):
    """List the configured tasks in a toml file.

    Parameters
    ----------
    toml_conf : dict
        The TOML tasks dictionary.

    Returns
    -------
    list(str)
        List of tasks names.

    """
    return list(toml_conf.keys())


if __name__ == "__main__":
    """"""
    args = parsing().parse_args()

    config = toml.load(os.path.join(args.db_path, "pipeline.toml"))
    # conf_tasks = list_configured_tasks(config)

    print("# - Used TOML configuration:")
    try:
        print(config[args.task])
    except KeyError:
        print(f"Task '{args.task}' is not defined in the configuration file!")

    print("\n# - Generated metadata:")
    md_path = os.path.join(args.db_path, 'metadata')
    json_list = [f for f in os.listdir(md_path) if f.startswith(args.task) and f.endswith('.json')]
    if json_list == []:
        print("Could not find the JSON metadata file associated to task '{}' in dataset '{}'!".format(args.task, args.db_path))
        raise IOError
    elif len(json_list) == 1:
        md_json = json_list[0]
        print("Found a JSON metadata file associated to task '{}' in dataset '{}'!".format(args.task, args.db_path))
        md_json = os.path.join(md_path, md_json)
    else:
        print("Found more than one JSON metadata file associated to task '{}' in dataset '{}':".format(args.task, args.db_path))
        [print(" - {}".format(json_f)) for json_f in json_list]
        md_json = max([os.path.join(md_path, json_f) for json_f in json_list], key=os.path.getctime)
        print("The most recent one is '{}'".format(os.path.split(md_json)[-1]))

    task_id = os.path.splitext(os.path.split(md_json)[-1])[0]
    print("{} task recorded the following parameters metadata:".format(task_id))
    md_json = json.load(open(md_json, 'r'))
    print(json.dumps(md_json, sort_keys=True, indent=2))

    print("\n# - Task outputs:")
    try:
        info_from_task(args.task, task_id, args.db_path)
    except FileNotFoundError as e:
        print(e)
        print("ERROR: No task output file found! Maybe it did not finish ?!")
