#!/usr/bin/env python3

import os
import json
import tempfile
import datetime
import shutil
import subprocess
import filecmp
import numpy as np
import pathlib
from optparse import OptionParser
from collections import Counter
import matplotlib.pyplot as plt

CONF_FILE = "conf_robustness_comparison.json"


def save_data_repartition(data, data_type, db):
    """
    Save repartition plots

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
    ax.set_title(f'{data_type} distribution for same scan same pipe')
    ax.boxplot(list(np.array(data).transpose()))
    fig.savefig(db / f"repeat_results_{data_type}.png")


def angles_and_internodes_comparison(task_folder_output_list, results_folder):
    """
    Coarse comparison of sequences of angles and internodes
    Prints a dict containing the number of scans for each number of organs indexed, ex : {12:4, 14:6} (4 scans with 12
    organs recognized and 6 with 14 organs)
    For the biggest number of scans with same number of organs, creates plots of the repartition of angles (and
    internodes) for each organ id

    Parameters
    ----------
    task_folder_output_list : pathlib.Path list
        list of task output path to compare

    results_folder : pathlib.Path
        path to put the potential results files

    """
    angles_and_internodes = {}
    nb_organs = []
    ae_file_list = [f / "AnglesAndInternodes.json" for f in task_folder_output_list]
    for ae_file in ae_file_list:
        with open(ae_file) as f:
            ae_list = json.load(f)
            angles_and_internodes[str(ae_file.parent.parent)[-1]] = {
                "angles": np.array(ae_list["angles"]) * 180 / np.pi,
                "internodes": ae_list["internodes"],
                "nb_organs": len(ae_list["angles"])
            }
            nb_organs.append(len(ae_list["angles"]))
    counter_nb_organs = Counter(nb_organs)
    print(" ** comparison results ** ")
    print("number of scans with the same nb of organs: ", counter_nb_organs)
    max_occurrence_nb_organs = counter_nb_organs.most_common()[0][0]
    angles = [angles_and_internodes[scan_num]["angles"] for scan_num in angles_and_internodes
              if angles_and_internodes[scan_num]["nb_organs"] == max_occurrence_nb_organs]
    internodes = [angles_and_internodes[scan_num]["internodes"] for scan_num in angles_and_internodes
                  if angles_and_internodes[scan_num]["nb_organs"] == max_occurrence_nb_organs]
    save_data_repartition(angles, "angles", results_folder)
    save_data_repartition(internodes, "internodes", results_folder)


def file_by_file_comparison(task_folder_output_list):
    """
    Compares task folder output file by file, print result

    Parameters
    ----------
    task_folder_output_list : pathlib.Path list
        list of task output path to compare

    """
    other_folders = task_folder_output_list[:]
    identical_folders_list = []
    for current_task_output in task_folder_output_list:
        if current_task_output in other_folders:
            current_identical_folder_list = [f for f in other_folders
                                             if not len(filecmp.dircmp(str(current_task_output), str(f)).diff_files)]
            identical_folders_list.append(current_identical_folder_list)
            other_folders = list(set(other_folders) - set(current_identical_folder_list))
        else:
            pass
    print(" ** comparison results ** ")
    print(f"compare_file {identical_folders_list}")
    print(np.array(identical_folders_list).shape)


def get_task_folder(scan, task_name):
    """
    Look for the output folder of a task in the files.json file of a scan
    returns name as a string of a task output folder
    """
    files = json.load(open(str(scan/"files.json")))
    task_folders = [d["id"] for d in files["filesets"] if task_name in d["id"]]
    return task_folders[0] # shame ...


def compare_task_output(scan_list, task_name, previous_task, results_folder):
    """
    Method to compare outputs of a task
    TODO: one method per task ? in config file?

    Parameters
    ----------
    path_db : string
        path of the database

    scan_list : list
        list of scans path to compare

    task_name : string
        name of the task to test

    previous_task : string
        name of the task which is the comparison point

    """
    task_folder = get_task_folder(scan_list[0], task_name)
    folder_task_list = [scan / task_folder for scan in scan_list]
    if previous_task is None and task_name == "AnglesAndInternodes" and len(folder_task_list):
        angles_and_internodes_comparison(folder_task_list, results_folder)
    elif len(folder_task_list):
        file_by_file_comparison(folder_task_list)
    else:
        print(f"Output files of task {task_name} for db {str(scan_list[0].parent)} are missing")


def fill_test_db(test_folder, scan, previous_task, config_file, nb):
    """
    From an initial scan, copy it in temporary folder, cleans it,
    runs the pipe to the comparison point task and copy the scan a certain
    number of times in a test folder

    Parameters
    ----------
    test_folder : pathlib.Path
        name of the folder into which copy the tests scans

    scan : pathlib.Path
        path of the initial scan


    previous_task : string
        comparison point task

    config_file : string
        path of the config file

    nb : int
        number of time the scan is replicated

    Returns
    ----------
    created_copied_scans : list
        list of pathlib.Path of created scans

    """
    scan_name = scan.name
    created_copied_scans = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)
        tmp_scan_folder = tmp_path / f"tmp_{scan_name}"
        with open(tmp_path / "romidb", 'w'):
            pass
        shutil.copytree(str(scan), str(tmp_scan_folder))
        init_configuration_file = tmp_scan_folder / "pipeline.toml"
        if init_configuration_file.is_file():
            print(str(init_configuration_file))
            run_pipe(tmp_scan_folder, "Clean", str(init_configuration_file))
        if previous_task:
            run_pipe(tmp_scan_folder, previous_task, config_file)
        for i in range(nb):
            copied_scan_name = test_folder / f"{scan_name}_{i}"
            shutil.copytree(str(tmp_scan_folder), str(copied_scan_name))
            created_copied_scans.append(copied_scan_name)
    return created_copied_scans


def get_test_folder_name(db_path, task_name):
    """
    Generate output folder's name, ex: 20200803124841_rep_test_TriangleMesh
    """
    now = datetime.datetime.now()
    now_str = now.strftime("%Y%m%d%H%M%S")
    return pathlib.Path(db_path / f"{now_str}_rep_test_{task_name}")


def create_test_db(db_path, scan, task_name, previous_task, config_file, nb):
    """
    Creates test db with romdb file in it
    """
    test_folder = get_test_folder_name(db_path, task_name)
    test_folder.mkdir(exist_ok=True)
    with open(test_folder / "romidb", mode='w'):
        pass
    test_scans = fill_test_db(test_folder, scan, previous_task, config_file, nb)
    return test_folder, test_scans


def run_pipe(scan_path, task_to_run, configuration_file):
    """
    run pipeline on a scan
    """
    print(" &&&& ")
    print(task_to_run)
    scan = str(scan_path)
    print(scan)
    cmd = ["romi_run_task", "--config", configuration_file, task_to_run, scan, "--local-scheduler"]
    subprocess.run(cmd, check=True)


def compute_repeatability_test(scan, task, nb):
    """
    Creates test folder with copies of the input scan and runs repeatability tests on it

    Parameters
    ----------
    scan : pathlib.Path
        contains information on database

    task : task object

    nb : int
        number of time the scan in replicated

    """
    initial_db_root = scan.parent.parent
    test_db, test_scans_list = create_test_db(initial_db_root, scan, task["name"], task["previous_task"], task["config_file"], nb)
    for copied_scan in test_scans_list:
        run_pipe(copied_scan, task["name"], task["config_file"])
    compare_task_output(test_scans_list, task["name"], task["previous_task"], test_db)


def set_task(name, config, full_pipe, previous_task):
    """
    initialize task features

    Parameters
    ----------
    name : string
        class name, must be a key in the CONF_FILE

    config : string
        name of the configuration file to run the pipeline

    full_pipe : bool
        whether or not compute repeatability from the start

    previous_task : string
        name of the previous class in the pipeline, either None for the full pipe or the one linked to the class name
        in CONF_FILE (comparison point task)

    Returns
    ----------
    task : dict
        dict containing features of the task to test
    """
    task = {
        "name": name,
        "config_file": config
    }
    if full_pipe:
        task["previous_task"] = None
    else:
        task["previous_task"] = previous_task
    return task


if __name__ == "__main__":
    """
    creates a test db at the root of the db linked to the scan to analyze
    """
    usage = "usage: %prog db"
    parser = OptionParser(usage=usage)

    parser.add_option("-s", "--scan", dest="scan",
                      help="scan to analyze", default=1)
    parser.add_option("-c", "--config_file", dest="config_file",
                      help="path to the config file")
    parser.add_option("-n", "--replicate_number", dest="replicate_number",
                      help="number of replicate for a scan", default=2)
    parser.add_option("-t", "--task", dest="task",
                      help="task to test", default="AnglesAndInternodes")
    parser.add_option("-f", "--full_pipe", dest="full_pipe",
                      action="store_true",
                      help="run the test for the whole pipe", default=False)

    (options, args) = parser.parse_args()

    c = json.load(open(CONF_FILE))

    task_to_test = options.task
    test_full_pipe = options.full_pipe
    replicate_number = int(options.replicate_number)
    initial_scan_path = pathlib.Path(options.scan).expanduser()

    if task_to_test in c.keys():
        task_dict = set_task(task_to_test, options.config_file, test_full_pipe, c[task_to_test]["prev_task"])
        compute_repeatability_test(initial_scan_path, task_dict, replicate_number)
    else:
        print("unknown task")
