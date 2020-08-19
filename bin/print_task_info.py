#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Python script to call after a ROMI task to print a task summary.
"""

import argparse
import json
import os

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


def angles_and_internodes_info(task_id, db_path):
    json_f = os.path.join(db_path, os.path.splitext(task_id)[0], "AnglesAndInternodes.json")
    md_json = json.load(open(json_f, 'r'))
    for md_info in ["angles", "internodes"]:
        try:
            data = md_json[md_info]
            print("Found the following {} between successive organs:".format(md_info))
            print([round(d, 1) for d in data])
        except KeyError as e:
            print("Could not find {} entry in {}".format(e, json_f))


def info_from_task(task, task_id, db_path):
    if task == "AnglesAndInternodes":
        return angles_and_internodes_info(task_id, db_path)
    elif task == "TreeGraph":
        return NotImplementedError
    elif task == "CurveSkeleton":
        return NotImplementedError
    elif task == "TriangleMesh":
        return NotImplementedError
    elif task == "PointCloud":
        return NotImplementedError
    elif task == "Voxels":
        return NotImplementedError
    elif task == "Masks":
        return NotImplementedError
    elif task == "Colmap":
        return NotImplementedError
    elif task == "Undistorted":
        return NotImplementedError
    ## - ML pipeline specific tasks:
    elif task == "Segmentation2D":
        return NotImplementedError
    elif task == "SegmentedPointCloud":
        return NotImplementedError
    elif task == "ClusteredMesh":
        return NotImplementedError
    else:
        return NotImplementedError


if __name__ == "__main__":
    """"""
    args = parsing().parse_args()

    config = toml.load(os.path.join(args.db_path, "pipeline.toml"))
    print("# - Used TOML configuration:")
    print(config[args.task])

    print("\n# - Generated metadata:")
    md_path = os.path.join(args.db_path, 'metadata')
    json_list = [f for f in os.listdir(md_path) if f.startswith(args.task) and f.endswith('.json')]
    if json_list == []:
        print("Could not find the JSON metadata file associated to task '{}' in dataset '{}'!".format(args.task, args.db_path))
        raise IOError
    elif len(json_list) == 1:
        md_json = json_list[0]
        print("Found a JSON metadata file associated to task '{}' in dataset '{}':".format(args.task, args.db_path))
        print("{}".format(md_json))
        md_json = os.path.join(md_path, md_json)
    else:
        print("Found more than one JSON metadata file associated to task '{}' in dataset '{}':".format(args.task, args.db_path))
        [print(" - {}".format(json_f)) for json_f in json_list]
        md_json = max([os.path.join(md_path, json_f) for json_f in json_list], key=os.path.getctime)
        print("The most recent one is '{}'".format(os.path.split(md_json)[-1]))

    task_id = os.path.split(md_json)[-1]
    print("{} task recorded the following metadata:".format(task_id))
    md_json = json.load(open(md_json, 'r'))
    print(json.dumps(md_json, sort_keys=True, indent=2))

    print("\n# - Task outputs:")
    info_from_task(args.task, task_id, args.db_path)
