#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Task Robustness Evaluation

Evaluate the robustness of a Reconstruction & Quantification (R&Q) pipeline task by
replicating a scan dataset, running the task on each replicate, and comparing the outputs.

The module exposes the functions used to build the evaluation database,
duplicate scan datasets, run the task, and compare its results across replicates.
It is also a runnable CLI (`robustness_evaluation`) that wires these steps together.

## Key Features

- Build and populate a temporary database to prepare the reference scan.
- Duplicate the reference scan into an evaluation database as replicates.
- Run the task to evaluate on each replicate, optionally on the whole pipeline.
- Compare task outputs file-by-file and with task-specific metrics.
"""

import datetime
import filecmp
import glob
import json
import logging
import os
import shutil
import sys
import tempfile
from itertools import combinations
from pathlib import Path

import click
import tomlkit
from click_option_group import optgroup
from plantdb.commons.fsdb.core import FSDB
from plantdb.commons.fsdb.path_helpers import MARKER_FILE_NAME
from romitask.cli.romi_run_task import run_task
from romitask.log import get_logger
from tqdm import tqdm

from plant3dvision.colmap import estimate_rotation_translation_mat
from plant3dvision.compare import _get_task_fileset
from plant3dvision.compare import average_pairwise_comparison

dirname, filename = os.path.split(os.path.abspath(__file__))

CONF_FILE = os.path.join(dirname, "robustness_evaluation.json")

DATETIME_FMT = "%Y.%m.%d_%H.%M"


def file_by_file_comparison(eval_db_path: FSDB, task_name: str, scan_replicates: list[Path]) -> None:
    """Compare task folders file by file and dump the results to a JSON file.

    Parameters
    ----------
    eval_db_path : plantdb.commons.fsdb.FSDB
        Local ROMI database instance with the replicated scan datasets.
    task_name : str
        Name of the task to evaluate.
    scan_replicates : list of pathlib.Path
        List of `Path` to the replicated scans.

    See Also
    --------
    filecmp.dircmp : Recursively compares the files and directories of two trees.

    Notes
    -----
    Produce a JSON file that summarizes the comparison.
    """
    logger.info(f"Performing file-by-file comparison for the outputs of task '{task_name}'...")
    fcmp = {}
    scan_pairs = combinations(scan_replicates, 2)
    for ref_scan, flo_scan in scan_pairs:
        # Compare the files' `os.stat()` signatures (file type, size, and modification time):
        cmp = filecmp.dircmp(ref_scan, flo_scan)
        # Compute the replicated task folders similarity percentage:
        n_diff = len(cmp.diff_files)
        n_same = len(cmp.same_files)
        if n_same == 0:
            similarity = 0
        else:
            similarity = (n_same / (n_diff + n_same)) * 100
        fcmp[f"{ref_scan.name} - {flo_scan.name}"] = similarity

    average_similarity = average_pairwise_comparison(fcmp)
    # Log result to terminal:
    if len(scan_replicates) <= 3:
        logger.info(f"File-by-file comparison (similarity percentage) between pairs of repetitions:{fcmp}")
        logger.info(f"Mean Similarity: {average_similarity:.2f} %")

    # Write a JSON summary file of the comparison:
    with open(Path(eval_db_path.basedir) / 'filebyfile_comparison.json', 'w') as out_file:
        json.dump({'Similarity (%)': fcmp, 'Mean similarity (%)': average_similarity}, out_file)
    return


def compare_task_output(eval_db_path: FSDB, task_name: str, scan_replicates: list[Path],
                        independent_runs: bool = False) -> None:
    """Compare task outputs on a series of replicated datasets.

    Parameters
    ----------
    eval_db_path : plantdb.commons.fsdb.FSDB
        Local ROMI database instance with the replicated scan datasets.
    task_name : str
        Name of the task to evaluate.
    scan_replicates : list of pathlib.Path
        List of `Path` to the replicated scans.
    independent_runs : bool, default False
        When ``True``, indicates that the chained tasks were performed
        independently for each replicate, so each task downstream of
        `task_name` (and `task_name` itself) is evaluated. Otherwise only the
        evaluation defined for `task_name` is performed.

    Notes
    -----
    The behaviour of this method is configured by the JSON file `CONF_FILE`.
    It is possible to define more than one comparison method per task.
    """
    eval_db_path.connect()
    # - Check the list of scan replicates for successful call of `romi_run_task`
    # (scan dataset should contain the `task_name` output fileset):
    valid_replicates = []
    for scan in scan_replicates:
        try:
            _ = _get_task_fileset(eval_db_path.get_scan(scan.name), task_name).id
        except AssertionError as e:
            logger.error(e)
        else:
            valid_replicates.append(scan)
    n_actual_replicates = len(valid_replicates)
    logger.info(f"Got {n_actual_replicates} valid replicates of task '{task_name}' (with an output fileset).")

    # - Performs file-by-file comparisons for the selected task:
    file_by_file_comparison(eval_db_path, task_name, valid_replicates)
    # - Load the metric for the selected task from the JSON configuration file:
    comparison_config = json.load(open(CONF_FILE))

    # - Evaluate all tasks if independent tests, else evaluate only tested task:
    if independent_runs:
        for task in comparison_config.keys():
            comp_func = comparison_config[task]["comp_func"]
            # If no comparison function is defined, skip this!
            if comp_func == "":
                continue
            # Check the task has been called by pipeline (should produce output fileset), otherwise continue the loop
            try:
                _ = _get_task_fileset(eval_db_path.get_scan(valid_replicates[0].name), task)
            except AssertionError:
                continue
            _perform_comparisons(eval_db_path, comp_func, task, valid_replicates)
    else:
        comp_func = comparison_config[task_name]["comp_func"]
        _perform_comparisons(eval_db_path, comp_func, task_name, valid_replicates)

    return


def _perform_comparisons(eval_db_path: FSDB, comp_func: str | list[str], task_name: str,
                         valid_replicates: list[Path]) -> None:
    """Call the task comparison function(s) on the scan replicates that ran the task.

    Parameters
    ----------
    eval_db_path : plantdb.commons.fsdb.FSDB
        Local ROMI database instance with the replicated scan datasets.
    comp_func : str or list of str
        Name(s) of the comparison function(s) to call for `task_name`.
    task_name : str
        Name of the task to evaluate.
    valid_replicates : list of pathlib.Path
        List of `Path` to the scan replicates that successfully ran `task_name`.
    """
    if isinstance(comp_func, str):
        comp_func = [comp_func]  # make a list to work in following loop
    # Get the list of `plantdb.commons.fsdb.Scan` instances to compare from list of Path:
    valid_replicates = [eval_db_path.get_scan(scan.name) for scan in valid_replicates]
    # - More than one comparison function can be applied to the same task!
    for func in comp_func:
        logger.info(f"Comparing '{task_name}' outputs with '{func}'...")
        if func == "estimated_pose_variability":
            for axes in ["xyz", "xy", "z", "pan"]:
                eval(func + "(eval_db_path, task_name, valid_replicates, axes=axes)")
        else:
            eval(func + "(eval_db_path, task_name, valid_replicates)")
        # try:
        #     eval(func + "(db, task_name)")
        # except NameError:
        #     logger.warning(f"The comparison function `{func}` "
        #                    "(defined in robustness_evaluation.json) does NOT exist!")
        # except Exception as e:
        #     logger.error(f"The comparison function `{func}` failed!")
        #     logger.error(e)
    return


def _copy_tree(origin: Path | str, dest: Path | str) -> None:
    """Copy a whole directory from `origin` to `dest`.

    Parameters
    ----------
    origin : pathlib.Path or str
        Source directory to copy.
    dest : pathlib.Path or str
        Destination directory for the copy.
    """
    try:
        shutil.copytree(str(origin), str(dest))
    except:
        logger.error(f"Failed to copy '{origin}' to '{dest}'!")
    return


def fill_tmp_db(dataset_path: Path | str, models_path: Path | None = None,
                extra_scans: list[Path] | None = None, clean: bool = False) -> Path:
    """Fill the temporary DB with replicates of `ref_scan` and run the pipeline up to the previous task.

    Parameters
    ----------
    dataset_path : pathlib.Path or str
        Path to the 'reference scan dataset' to use for evaluation.
    models_path : pathlib.Path
        Name of models directory to copy in evaluation DB, to use with task ``Segmentation2D``.
    extra_scans : list of pathlib.Path
        Name of other scan dataset directories to copy in evaluation DB.
    clean : bool, optional
        Force cleaning the 'reference scan dataset' to use for evaluation.

    Returns
    -------
    pathlib.Path
        The path to the temporary 'reference scan dataset'.

    Notes
    -----
    The evaluation database is created as follows:
      - create a temporary directory (`/tmp/tmp_{ref_scan.name}`)
      - copy the 'models' dataset, if any
      - copy the extra scan dataset, if any
      - copy the reference scan dataset (`ref_scan`) to the temporary directory

    """
    # Initialize a temporary directory use to store scan dataset before cleaning and running previous tasks to task to analyse:
    tmp_dir = tempfile.mkdtemp()
    # - Get the path of the temporary ROMI database
    tmp_path = Path(tmp_dir)
    logger.info(f"Create a temporary directory '{tmp_path}'.")
    # - Create the `MARKER_FILE_NAME` marker for an active `plantdb.commons.fsdb.FSDB`
    with open(tmp_path / MARKER_FILE_NAME, 'w'):
        logger.info(f"Create the `{MARKER_FILE_NAME}` marker for an active `plantdb.commons.fsdb.FSDB`.")

    # - If ML models are needed, copy them to evaluation DB directory:
    if models_path is not None:
        logger.info("Copy the required ML models dataset...")
        _copy_tree(models_path, tmp_path / "models")
    # - If extra scans are needed, copy them to evaluation DB directory:
    if extra_scans is not None:
        for x_scan in extra_scans:
            logger.info(f"Copy the required calibration dataset '{x_scan.name}'...")
            _copy_tree(x_scan, tmp_path / x_scan.name)

    # Copy the 'reference scan dataset' to the temporary directory
    logger.info("Copy the 'reference scan dataset' to the temporary directory...")
    tmp_scan_dir = tmp_path / f"{dataset_path.name}"
    _copy_tree(dataset_path, tmp_scan_dir)

    _run_clean_if_requested(tmp_scan_dir, clean)

    return tmp_scan_dir


def _run_clean_if_requested(tmp_scan_dir: Path, clean: bool) -> None:
    """Clean the input dataset if requested.

    Parameters
    ----------
    tmp_scan_dir : pathlib.Path
        The path to the scan dataset directory to clean.
    clean : bool
        Whether to run the ``Clean`` task on `tmp_scan_dir`.
    """
    bak_cfg_file = tmp_scan_dir / "pipeline.toml"
    if clean:
        # Create a quiet Clean task configuration:
        with open(bak_cfg_file, 'w') as cfg_f:
            cfg = {"Clean": {"no_confirm": True}}
            tomlkit.dump(cfg, cfg_f)
        run_task(tmp_scan_dir, "Clean", str(bak_cfg_file), no_auth=True)
    else:
        logger.info("No cleaning of the reference scan dataset!")
    return


def fill_eval_db(eval_db_path: Path, dataset_path: Path, n_replicate: int,
                 models_path: Path | None = None, extra_scans: list[Path] | None = None) -> list[Path]:
    """Fill the evaluation DB with replicates of `ref_scan` and run the pipeline up to the previous task.

    Parameters
    ----------
    eval_db_path : pathlib.Path
        The path to the evaluation DB.
    dataset_path : pathlib.Path
        The path to the dataset to use for evaluation.
    n_replicate : int
        Number time to duplicate the reference dataset.
    models_path : pathlib.Path
        Name of models directory to copy in evaluation DB, to use with task ``Segmentation2D``.
    extra_scans : list of pathlib.Path
        Name of other dataset directories to copy in evaluation DB.

    Returns
    -------
    list
        A list of path to the created scan datasets.

    Notes
    -----
    The evaluation database is created as follows:
      - create a temporary directory (`/tmp/tmp_{ref_scan.name}`)
      - copy the 'models' dataset, if any
      - copy the extra scan dataset, if any
      - copy the reference scan dataset (`ref_scan`) to the temporary directory
      - clean
      - runs the pipeline up to the `previous_task` using the `config`, if any
      - replicate the previous scan directory `nb` times to the evaluation DB (`eval_db`)

    """
    scan_replicates = []
    # - If ML models are needed, copy them to evaluation DB directory:
    if models_path is not None:
        logger.info("Copy the required ML models dataset...")
        _copy_tree(models_path, eval_db_path / "models")
    # - If extra scans are needed, copy them to evaluation DB directory:
    if extra_scans is not None:
        for x_scan in extra_scans:
            logger.info(f"Copy the required calibration dataset '{x_scan.name}'...")
            _copy_tree(x_scan, eval_db_path / x_scan.name)

    n_fill = len(str(n_replicate))
    for i in range(n_replicate):
        copied_scan_name = eval_db_path / f"{dataset_path.name}_{str(i).zfill(n_fill)}"
        shutil.copytree(str(dataset_path), str(copied_scan_name))
        scan_replicates.append(copied_scan_name.expanduser())

    return scan_replicates


def get_eval_db_path(root_location: Path, task_name: str, suffix: str = "", date_fmt: str = DATETIME_FMT,
                     no_date: bool = False, **kwargs: object) -> Path:
    """Generate the "evaluation database" name, for example: '20200803124841_eval_TriangleMesh'.

    Parameters
    ----------
    root_location : pathlib.Path
        Root path where to create the evaluation database.
    task_name : str
        Name of the task to evaluate.
    suffix : str, optional
        Suffix to append to the created database directory.
    date_fmt : str, optional
        Datetime format to append as evaluation database prefix. Defaults to ``DATETIME_FMT``.
    no_date : bool, optional
        Do not add the datetime as prefix. Defaults to ``False``.

    Returns
    -------
    pathlib.Path
        The path to the evaluation database.
    """
    if no_date:
        folder_name = f"Eval_{task_name}"
    else:
        now = datetime.datetime.now()
        now_str = now.strftime(date_fmt)
        folder_name = f"{now_str}_Eval_{task_name}"

    if suffix != "":
        if not suffix.startswith("_"):
            suffix = f"_{suffix}"
        folder_name += suffix

    return Path(root_location / folder_name)


def initialize_eval_db(eval_db_path: str | Path = "", **kwargs: object) -> Path:
    """Initialize the local FSDB "evaluation database".

    Parameters
    ----------
    eval_db_path : str or pathlib.Path
        Full path to the evaluation database directory.

    Other Parameters
    ----------------
    root_location : pathlib.Path
        Root path where to create the evaluation database.
    task_name : str
        Name of the task to evaluate.
    suffix : str
        Suffix to append to the created database directory.
    date_fmt : str
        Datetime format to append as evaluation database prefix. Defaults to ``DATETIME_FMT``.
    no_date : bool
        Do not add the datetime as prefix. Defaults to ``False``.

    Returns
    -------
    pathlib.Path
        The path to the "evaluation database" directory.

    Notes
    -----
    A valid ROMI database (`plantdb`) is a directory with a `romidb` marker file in it.
    We use the tested task name and datetime to create the name of the directory.
    """
    # - Get the path to the evaluation DB to create:
    if eval_db_path == "":
        eval_db_path = get_eval_db_path(**kwargs).absolute()
    else:
        eval_db_path = Path(eval_db_path).absolute()

    # Check if directory already exists and is valid
    if eval_db_path.exists():
        # If it's not a directory, remove it
        if not eval_db_path.is_dir():
            raise ValueError(f"Path {eval_db_path} already exists but is not a directory.")
        # Check if it already has a marker file (valid ROMI DB)
        marker_file = eval_db_path / MARKER_FILE_NAME
        if marker_file.exists():
            return eval_db_path

    # Create the evaluation DB directory if it doesn't exist
    eval_db_path.mkdir(parents=True, exist_ok=True)

    # - Creates the marker file to be able to get a `plantdb.commons.fsdb.FSDB` instance:
    marker_file = eval_db_path / MARKER_FILE_NAME
    marker_file.touch()
    return eval_db_path


def get_replicates_list(eval_db_path: Path, ds_path: Path) -> list[Path]:
    """Get the list of scan replicates.

    Parameters
    ----------
    eval_db_path : pathlib.Path
        Full path to evaluation database.
    ds_path : pathlib.Path
        Full path to scan dataset to use for evaluation.

    Returns
    -------
    list of str
        List of full path to the replicates of the reference scan found in the evaluation database.

    Examples
    --------
    >>> from pathlib import Path
    >>> eval_db_path = Path('~/romi_db/calibration_robustness/20220905155852_eval_ExtrinsicCalibration_opencv')
    >>> ds_path = Path('~/romi_db/calibration_robustness/db/extrinsic_calib_1')
    >>> get_replicates_list(eval_db_path.expanduser(),ds_path.expanduser())
    """
    return sorted(
        list([idir.expanduser() for idir in eval_db_path.iterdir() if idir.is_dir() and ds_path.name in idir.name]))


def _check_markers(path: Path) -> None:
    """Make sure the ``romidb`` marker file exists at `path`.

    Parameters
    ----------
    path : pathlib.Path
        Path to the directory where the marker file should be created.
    """
    # - Make sure the `romidb` marker file exists:
    marker_file = path / MARKER_FILE_NAME
    try:
        marker_file.touch()
    except:
        pass
    return


def check_extra_dataset(db_location: Path, config_file: Path) -> list[Path]:
    """Use the configuration file to search for defined calibration dataset id(s) to copy.

    Parameters
    ----------
    db_location : pathlib.Path
        The path to the database hosting the input dataset used for robustness evaluation.
    config_file : pathlib.Path
        The path to the pipeline configuration file.

    Returns
    -------
    list
        The list of extra dataset required to run the robustness evaluation with this pipeline configuration.
    """
    pipeline_cfg = tomlkit.load(open(config_file, 'r'))
    extra_ds = []
    for section, cfg_dict in pipeline_cfg.items():
        for k, v in cfg_dict.items():
            if "calibration_scan_id" in k and v != "":
                extra_ds.append(db_location / v)
    nx = len(extra_ds)
    if nx != 0:
        extra_scans_str = ', '.join([xs.name for xs in extra_ds])
        logger.info(f"Found '{nx}' calibration scan{'s' if nx > 1 else ''} to copy: {extra_scans_str}")
    return extra_ds


def run_task_eval(dataset_path: Path | str, task: str, config: str = "", n_replicates: int = 30,
                  clean: bool = False, full_pipe: bool = False, no_pipeline: bool = False, suffix: str = "",
                  eval_db: str = "", date_fmt: str = DATETIME_FMT, no_date: bool = False, models: str = "",
                  log_level: str = "INFO") -> None:
    """Run the robustness evaluation of a task on a dataset.

    Parameters
    ----------
    dataset_path : pathlib.Path or str
        Path to the reference scan dataset to use for the evaluation.
    task : str
        Name of the task to evaluate.
    config : str, default ""
        Path to the pipeline TOML configuration file.
    n_replicates : int, default 30
        Number of replicates to create for the robustness evaluation.
    clean : bool, default False
        Run a ``Clean`` task on the scan dataset prior to duplication.
    full_pipe : bool, default False
        Run the whole pipeline independently for each replicate. Otherwise the
        task to evaluate is run on clones of the results from the upstream task.
    no_pipeline : bool, default False
        Do not run the pipeline, only compare task outputs.
    suffix : str, default ""
        Suffix to append to the evaluation database directory to create.
    eval_db : str, default ""
        Existing evaluation database location to use. If empty, a new
        evaluation database is automatically created in the parent directory
        of the input dataset with a timestamp prefix.
    date_fmt : str, default DATETIME_FMT
        Datetime format to use as prefix for the evaluation database name.
    no_date : bool, default False
        Do not add the datetime prefix to the evaluation database name.
    models : str, default ""
        Models database location to use with the ML pipeline.
    log_level : str, default "INFO"
        Message logging level.
    """
    # - Get the path to the reference scan dataset:
    dataset_path = Path(dataset_path).expanduser()
    # - Path to the local database that contains the reference scan dataset:
    db_location = dataset_path.parent
    # - Load the JSON config file `robustness_evaluation.json`:
    config_dict = json.load(open(CONF_FILE))
    # - Get the task to execute prior to the one to evaluate, if any, from the JSON evaluation configuration file:
    if full_pipe:
        prev_task = ""  # we want independent replicates, so no common previous task!
    else:
        prev_task = config_dict[task]["prev_task"]

    # - Initialize the "evaluation database" and get its path:
    #   If --eval-db is not specified, a new database will be created in the parent directory
    #   of the input dataset with a timestamp prefix
    eval_db_path = initialize_eval_db(eval_db, task_name=task, root_location=db_location.parent,
                                      suffix=suffix, date_fmt=date_fmt, no_date=no_date)

    # - Initialize the logger
    global logger
    logger = get_logger(__name__, eval_db_path / "robustness_evaluation.log", log_level)

    # Check if the evaluation database directory exists and is not empty
    # This situation can happen when using auto-generated database path and it already exists
    if eval_db == "" and eval_db_path.exists():
        # Count the entries in the directory (excluding the '.' and '..' entries)
        dir_entries = [entry for entry in eval_db_path.iterdir() if Path(entry).is_dir()]
        if len(dir_entries) > 0:
            from plant3dvision.utils import yes_no_choice
            logger.warning(f"The directory '{eval_db_path}' is not EMPTY!")
            logger.info(
                "Here is the list of directories found:\n" + "\n".join([f"  - {entry}" for entry in dir_entries]))
            empty_dir = yes_no_choice(f"The directory '{eval_db_path}' is not EMPTY! Do you want to clean it?")
            if empty_dir:
                # Remove the existing eval DB:
                shutil.rmtree(eval_db_path)
                # Re-initialize the "evaluation database" and get its path:
                eval_db_path = initialize_eval_db(eval_db, task_name=task, root_location=db_location.parent,
                                                  suffix=suffix, date_fmt=date_fmt, no_date=no_date)
    logger.info(f"Robustness evaluation of task {task} is running on dataset '{Path(dataset_path).name}'...")
    logger.info(f"Path to the 'input scan dataset' is: '{dataset_path}'")
    # TODO: check if the scan is "raw" or not using call to some Clean task methods to get insight?
    if eval_db != "":
        logger.info(f"Got a manually defined evaluation database location: '{eval_db}'.")
    else:
        logger.info(f"Created an evaluation database at location: '{eval_db_path}'.")

    # - Check if some extra dataset are required (search in pipeline config, e.g. for '*calibration_scan_id'):
    extra_dataset = check_extra_dataset(db_location, config)
    # - Get the path to the CNN model
    path2cnn_model = Path(models).expanduser() if models else None

    # - Scan evaluation DB to check if it's already populated with replicates:
    scan_replicates = get_replicates_list(eval_db_path, dataset_path)
    found_replicates = len(scan_replicates)
    has_tmp_db = False
    # Case without/with scan datasets in the evaluation database root directory
    if found_replicates == 0:
        n_replicates = int(n_replicates)
        # If there is a previous task to run:
        if prev_task != "":
            # Create & populate the temporary DB:
            dataset_path = fill_tmp_db(dataset_path, path2cnn_model, extra_dataset, clean=clean)
            has_tmp_db = True
            # Run the previous task on the temporary DB, if any:
            logger.info(f"Call to '{prev_task}' task on the 'temporary scan dataset'...")
            run_task(dataset_path, prev_task, config, no_auth=True)
            # Duplicate the results of the previous task to the evaluation database:
            logger.info(f"Duplicate the 'temporary scan dataset' {n_replicates} times...")
        else:
            bak_cfg_file = dataset_path / "pipeline.toml"
            # If a previous analysis is found (directory has a `pipeline.toml`) or a clean was requested:
            if bak_cfg_file.is_file() or clean:
                # Copy to a temporary DB, clean it, replace reference scan path by clean one
                dataset_path = fill_tmp_db(dataset_path, clean=clean)
                has_tmp_db = True
                logger.info(f"Duplicate the 'temporary scan dataset' {n_replicates} times...")
            else:
                logger.info(f"Duplicate the 'reference scan dataset' {n_replicates} times...")
        # Duplicate the scan dataset as many times as requested
        scan_replicates = fill_eval_db(eval_db_path, dataset_path, n_replicates, path2cnn_model, extra_dataset)
    else:
        # If there is something in the root directory that mean you gave a value to `--eval_db`
        # And we thus consider that it should already contain the replicated scans!
        logger.info(f"Found {found_replicates} replicate scans in evaluation database!")
        if found_replicates != int(n_replicates):
            logger.warning(
                f"Not the same number of replicates found ({found_replicates}) and requested ({int(n_replicates)})!")

    # Clean the temporary DB:
    tmp_db = dataset_path.parent
    if has_tmp_db and tmp_db.is_dir():
        shutil.rmtree(dataset_path.parent)

    # - Instantiate `FSDB` local "evaluation database":
    _check_markers(eval_db_path)
    test_db = FSDB(str(eval_db_path), no_auth=True)

    # - Execute the task to evaluate for all replicated datasets:
    if not no_pipeline:
        # -- Set the `PYOPENCL_CTX` environment variable:
        if os.getenv('PYOPENCL_CTX') == None:
            os.environ["PYOPENCL_CTX"] = '0'

        if prev_task == "":
            logger.info(f"Independent calls to task '{task}' on {n_replicates} replicate scans...")
        else:
            logger.info(f"Repeat calls to task '{task}' on {n_replicates} replicate scans...")
        for replicated_scan in tqdm(scan_replicates, unit='replicate'):
            run_task(replicated_scan, task, config, no_auth=True)
    else:
        logger.info(f"Using existing outputs of task '{task}' on {n_replicates} replicate scans...")

    # - Compare the output(s) of the task(s) between replicated datasets:
    compare_task_output(test_db, task, scan_replicates, independent_runs=full_pipe)


def _get_scan_id(dataset_path: Path) -> str:
    """Get the base scan id of a scan dataset from its configuration file.

    Parameters
    ----------
    dataset_path : pathlib.Path
        Path to a scan dataset directory.

    Returns
    -------
    str
        The scan id, e.g. ``"ahp6-1_E2_23"``.
    """
    cfg_file = dataset_path / "scan.toml"
    if cfg_file.is_file():
        try:
            cfg = tomlkit.load(open(cfg_file, 'r'))
            return cfg["Scan"]["metadata"]["object"]["dataset_id"]
        except Exception:
            pass
    # Fallback: strip any trailing `_<digits>` replicate suffix from the folder name:
    import re
    return re.sub(r"_\d+$", "", dataset_path.name)


def _write_json_file(file_obj, data: dict) -> None:
    """Write a JSON dictionary to a ``File``, creating/overwriting its content."""
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", newline="", encoding="utf-8", delete=False) as f:
        json.dump(data, f, indent=1)
        tmp_name = f.name
    try:
        file_obj.import_file(tmp_name)
    finally:
        os.remove(tmp_name)


def _find_replicate(eval_dir: Path, scan_id: str) -> Path | None:
    """Return the first replicate ``f"{scan_id}_NN"`` directory found in `eval_dir`, else ``None``."""
    replicas = sorted(
        d for d in eval_dir.iterdir()
        if d.is_dir() and d.name.startswith(f"{scan_id}_")
        and d.name[len(scan_id) + 1:].isdigit()
    )
    return replicas[0] if replicas else None


EXCLUDED_FILES = ("colmap", "model_analyzer", "points3d", "sparse", "keypoints_per_image", "matches_per_image_pair")


def _copy_replicate_colmap(replicate_dir: Path, median_scan) -> dict | None:
    """Copy a replicate's ``Colmap*`` fileset into `median_scan` (if absent) and return its camera model.

    Parameters
    ----------
    replicate_dir : pathlib.Path
        Path to a replicate scan dataset (in the evaluation database holding the `comparison_json`).
    median_scan : plantdb.commons.fsdb.Scan
        The median scan dataset to host the copied ``Colmap*`` fileset.

    Returns
    -------
    dict or None
        The camera intrinsics (``camera_model``) of the replicate, or ``None`` if it could not be read.

    Notes
    -----
    Some Colmap output files are not copied as they are meaningless for the faked task, see `EXCLUDED_FILES`.
    """
    rep_db = FSDB(str(replicate_dir.parent), no_auth=True)
    rep_db.connect()
    try:
        rep_scan = rep_db.get_scan(replicate_dir.name)
        rep_colmap = [fs for fs in rep_scan.get_filesets() if fs.id.startswith("Colmap")]
        assert len(rep_colmap) == 1, f"Found {len(rep_colmap)} 'Colmap*' filesets in '{replicate_dir.name}'."
        rep_colmap = rep_colmap[0]
        camera_model = rep_scan.get_fileset("images").get_files()[0].get_metadata("colmap_camera")["camera_model"]

        if not any(fs.id.startswith("Colmap") for fs in median_scan.get_filesets()):
            colmap_fs = median_scan.create_fileset(rep_colmap.id)
            for file_id in rep_colmap.list_files():
                rep_file_path = rep_colmap.get_file(file_id).path()
                if file_id in EXCLUDED_FILES:
                    colmap_fs.create_file(file_id).write("", ext=rep_file_path.suffix)  # create empty file
                else:
                    colmap_fs.create_file(file_id).import_file(rep_file_path)
        return camera_model
    finally:
        rep_db.disconnect()


def _fake_colmap_fileset(scan, image_files) -> None:
    """Fake a ``Colmap*`` fileset on the median scan so it looks like the Colmap task estimated the median poses.

    Parameters
    ----------
    scan : plantdb.commons.fsdb.Scan
        The median scan dataset. It hosts a ``Colmap*`` fileset when copied from a
        replicate; otherwise one is created.
    image_files : list[plantdb.commons.fsdb.File]
        The 'images' fileset files, whose ``estimated_pose``/``colmap_camera`` metadata
        have already been replaced with the estimated median poses.

    Notes
    -----
    * Reuses an existing ``Colmap*`` fileset if present, else creates one.
    * Some Colmap output files are not copied as they are meaningless for the faked task, see `EXCLUDED_FILES`.
    * Regenerates ``cnc_vs_colmap_poses_estimated.png`` with `CameraPoseQC.plot_pose_estimation_figure`.
    * Writes ``ref2pred_pose_distances.json`` & ``ref2pred_pose_distances_stats.json`` and ``cameras.json``
     from the images metadata.
    """
    import numpy as np
    from plant3dvision.tasks.colmap import CameraPoseQC

    colmap_filesets = [fs for fs in scan.get_filesets() if fs.id.startswith("Colmap")]
    if len(colmap_filesets) == 1:
        colmap_fs = colmap_filesets[0]
    elif len(colmap_filesets) == 0:
        colmap_fs = scan.create_fileset("Colmap_True_10_5_0_median")
        logger.info(f"Created the fake '{colmap_fs.id}' fileset in '{scan.id}'.")
    else:
        raise AssertionError(f"More than one 'Colmap*' fileset in '{scan.id}': "
                             f"{[fs.id for fs in colmap_filesets]}.")

    def _get_or_create(file_id):
        return colmap_fs.get_file(file_id) if colmap_fs.file_exists(file_id) else colmap_fs.create_file(file_id)

    # - Recompute the pose estimation quality against the (median) estimated poses:
    camera_pose_qc = CameraPoseQC(image_files, mad_factor=3., metrics=["xy", "z", "pan"])

    # - Regenerate the pose estimation figure:
    pose_fig_fpath = f"{colmap_fs.path()}/cnc_vs_colmap_poses_estimated.png"
    camera_pose_qc.plot_pose_estimation_figure(figname=pose_fig_fpath)

    # - Write the reference->predicted pose distances & their statistics:
    dist_json = {}
    dist_stats_json = {}
    for dist_name, dist_values in camera_pose_qc.dist_dict.items():
        dist_stats_json.update({
            f"mean_{dist_name}_distance": float(np.nanmean(list(dist_values.values()))),
            f"median_{dist_name}_distance": float(np.nanmedian(list(dist_values.values()))),
            f"std_{dist_name}_distance": float(np.nanstd(list(dist_values.values()))),
        })
        dist_json[f"{dist_name}_distances"] = dist_values
    _write_json_file(_get_or_create("ref2pred_pose_distances"), dist_json)
    _write_json_file(_get_or_create("ref2pred_pose_distances_stats"), dist_stats_json)

    # - Write the camera intrinsics from the images metadata:
    camera_model = next(
        fi.get_metadata("colmap_camera")["camera_model"]
        for fi in image_files
        if fi.get_metadata("colmap_camera", default=None) is not None
    )
    _write_json_file(_get_or_create("cameras"), {str(camera_model["id"]): camera_model})


def create_median_pose_dataset(dataset_path: Path | str, eval_db: Path | str,
                               out_db_path: Path | str | None = None) -> Path:
    """Create a ``f"{scan_id}_median"`` scan dataset from the 'estimated median pose'.

    Copies the provided scan dataset and replaces the 'images' fileset
    ``estimated_pose`` and ``colmap_camera`` metadata with the estimated median
    poses read from a ``Colmap_cnc_poses_comparison.json`` output. It then fakes
    the copied ``Colmap*`` fileset so it looks like the Colmap task estimated the
    median poses (see `_fake_colmap_fileset`).

    Parameters
    ----------
    dataset_path : pathlib.Path or str
        Path to the scan dataset to use as a template (e.g. one of the evaluation replicates).
    eval_db : pathlib.Path or str
        Path to the ``Colmap_cnc_poses_comparison.json`` output hosting the 'estimated median pose'.
    out_db_path : pathlib.Path or str, optional
        Path to the evaluation database where to create the median dataset.
        Defaults to the parent directory of `dataset_path`.

    Returns
    -------
    pathlib.Path
        Path to the created ``{scan_id}_median`` scan dataset.
    """
    dataset_path = Path(dataset_path).expanduser()
    eval_db = Path(eval_db).expanduser()
    if eval_db.is_dir():
        eval_db = eval_db / "Colmap_cnc_poses_comparison.json"
        assert eval_db.exists(), f"'{eval_db}' does not exist!"
    else:
        assert eval_db.name == "Colmap_cnc_poses_comparison.json", f"Expected '{eval_db.name}' to be 'Colmap_cnc_poses_comparison.json'"
        assert eval_db.exists(), f"'{eval_db}' does not exist!"

    if out_db_path is None:
        out_db_path = eval_db.parent
    else:
        out_db_path = Path(eval_db).expanduser()

    # - Get the base 'scan_id' from the scan configuration:
    scan_id = _get_scan_id(dataset_path)

    # - Find a replicate (in the comparison JSON directory) to source the Colmap task outputs from:
    replicate_dir = _find_replicate(eval_db.parent, scan_id)
    if replicate_dir is None:
        logger.warning(f"Could not find any '{scan_id}_NN' replicate next to '{eval_db}'.")

    # - Load the estimated median poses `{image_id: [x, y, z, pan, tilt, roll]}`:
    comparison = json.load(open(eval_db, 'r'))
    median_poses = comparison["estimated median pose"]

    # - Copy the provided scan dataset as a template for the median dataset:
    median_scan_path = out_db_path / f"{scan_id}_median"
    if median_scan_path.exists():
        shutil.rmtree(median_scan_path)
    shutil.copytree(str(dataset_path), str(median_scan_path))
    _check_markers(median_scan_path)
    logger.info(f"Created the median pose scan dataset '{median_scan_path}'.")

    # - Replace the 'images' fileset metadata with the estimated median poses:
    out_db = FSDB(str(out_db_path), no_auth=True)
    out_db.connect()
    scan = out_db.get_scan(f"{scan_id}_median")

    # - Copy the Colmap fileset & camera intrinsics from a replicate if the template lacks them:
    camera_model = None
    if replicate_dir is not None:
        camera_model = _copy_replicate_colmap(replicate_dir, scan)

    image_files = scan.get_fileset("images").get_files()
    for fi in image_files:
        median_pose = median_poses[fi.id]
        x, y, z, pan, tilt, roll = median_pose
        rotmat, tvec = estimate_rotation_translation_mat(x, y, z, pan, tilt, roll)
        # - Replace the 'estimated_pose' metadata:
        fi.set_metadata("estimated_pose", median_pose)
        # - Replace the 'rotmat' & 'tvec' entries of the 'colmap_camera' metadata:
        colmap_camera = fi.get_metadata("colmap_camera")
        if not colmap_camera:
            if camera_model is None:
                raise ValueError(f"No 'colmap_camera' metadata for '{fi.id}' and no replicate "
                                 f"to source the camera model from.")
            colmap_camera = {"camera_model": camera_model}
        colmap_camera["rotmat"] = rotmat.tolist()
        colmap_camera["tvec"] = tvec.tolist()
        fi.set_metadata("colmap_camera", colmap_camera)
    # - Fake the 'Colmap*' fileset so it reflects the (median) estimated poses:
    _fake_colmap_fileset(scan, image_files)
    out_db.disconnect()

    logger.info(f"Replaced 'images' metadata with {len(median_poses)} estimated median poses.")
    return median_scan_path


@click.command("create-median-dataset", context_settings=dict(help_option_names=["-h", "--help"]))
@click.argument("eval_db", type=click.Path(exists=True, dir_okay=True, file_okay=True))
@click.argument("dataset_path", type=click.Path(exists=True, file_okay=False))
@click.option("--out-db", "out_db_path", default="", type=str,
              help="Database location where to create the median dataset. "
                   "Defaults to the parent directory of `dataset_path`.")
@click.option("--log-level", "log_level", default="INFO",
              type=click.Choice(list(logging._nameToLevel.keys())),
              help="Set message logging level. Defaults to `INFO`.")
def main_create_median_dataset(eval_db: str, dataset_path: str, out_db_path: str, log_level: str) -> None:
    """Create a ``{scan_id}_median`` scan dataset using the 'estimated median pose'.

    Reads the 'estimated median pose' from a ``Colmap_cnc_poses_comparison.json``
    output, copies the provided scan dataset to ``f"{scan_id}_median"``, replaces
    the 'images' fileset ``estimated_pose`` and ``colmap_camera`` metadata with the
    estimated median pose values, and fakes the copied ``Colmap*`` fileset so it
    looks like the Colmap task estimated the median poses.
    """
    global logger
    logger = get_logger(__name__, None, log_level)
    create_median_pose_dataset(dataset_path, eval_db, out_db_path or None)


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.argument("task", type=str, default="AnglesAndInternodes")
@click.argument("dataset_path", nargs=-1, type=str)
@click.option("--config", default="", type=str, help="Path to the pipeline TOML configuration file.")
@optgroup.group("Evaluation options")
@optgroup.option("-n", "--n_replicates", default=30, type=int,
                 help="Number of replicates to create for task robustness evaluation. Defaults to `30`.")
@optgroup.option("-c", "--clean", is_flag=True, default=False,
                 help="Run a Clean task on scan dataset prior to duplication.")
@optgroup.option("-f", "--full-pipe", "full_pipe", is_flag=True, default=False,
                 help="Use this to run the whole pipeline independently for each replicate. "
                      "Else the task to evaluate is run on clones of the results from the upstream task, if any.")
@optgroup.option("-np", "--no-pipeline", "no_pipeline", is_flag=True, default=False,
                 help="Do not run the pipeline, only compare tasks outputs. "
                      "Use with `--eval-db` to rerun this code on an existing test evaluation database!")
@optgroup.group("Database options")
@optgroup.option("--suffix", default="", type=str,
                 help="Suffix to append to the evaluation database directory to create.")
@optgroup.option("--eval-db", "eval_db", default="", type=str,
                 help="Existing evaluation database location to use. "
                      "If not set, a new evaluation database will be automatically created in the parent "
                      "directory of the input dataset with a timestamp prefix.")
@optgroup.option("--date-fmt", "date_fmt", default=DATETIME_FMT, type=str,
                 help="Datetime format to use as prefix for the name of the evaluation database directory to create. "
                      "Defaults to `DATETIME_FMT`.")
@optgroup.option("--no-date", "no_date", is_flag=True, default=False,
                 help="Do not add the datetime as prefix to the name of the evaluation database directory to create.")
@optgroup.group("Other options")
@optgroup.option("--median", is_flag=True, help="Create the median dataset from the Colmap median poses per image.")
@optgroup.option("--models", default="", help="Models database location to use with ML pipeline.")
@optgroup.option("--log-level", "log_level", default="INFO",
                 type=click.Choice(list(logging._nameToLevel.keys())),
                 help="Set message logging level. Defaults to `INFO`.")
def main(task: str, dataset_path: tuple[str, ...], config: str, n_replicates: int, clean: bool,
         full_pipe: bool, no_pipeline: bool, suffix: str, eval_db: str, date_fmt: str, no_date: bool,
         median: bool, models: str, log_level: str) -> None:
    """Task robustness evaluation.

    Evaluate the robustness of a Reconstruction & Quantification (R&Q) pipeline task by
    replicating a scan dataset, running the task on each replicate, and comparing the outputs.

    https://docs.romi-project.eu/plant_imager/developer/pipeline_repeatability/
    """
    # - If only one path in the list, get the first one:
    if len(dataset_path) == 1:
        dataset_path = dataset_path[0]

    if isinstance(dataset_path, str):
        # Process the input string `dataset_path` with ``glob``:
        #   - check existence of path:
        #   - may contain UNIX matching symbols (like '*' or '?'):
        folders = glob.glob(dataset_path)
        # Resolve path (make it absolute & normalize):
        folders = [Path(path).resolve() for path in folders]
        # Check that globed paths are directory (and exist, implied):
        folders = sorted([path for path in folders if path.is_dir()])
    elif isinstance(dataset_path, tuple):
        # Resolve path (make it absolute & normalize):
        folders = [Path(path).resolve() for path in dataset_path]
        # Check that listed paths are directory (and exist, implied):
        folders = sorted([path for path in folders if path.is_dir()])
    else:
        sys.exit(f"Error with input dataset path for '{task}' module!")

    for folder in folders:
        print("\n")  # to facilitate the search in the console by separating the datasets
        try:
            run_task_eval(dataset_path=folder, task=task, config=config, n_replicates=n_replicates,
                          clean=clean, full_pipe=full_pipe, no_pipeline=no_pipeline, suffix=suffix,
                          eval_db=eval_db, date_fmt=date_fmt, no_date=no_date, models=models, log_level=log_level)
        except Exception as e:
            print(e)

        if median:
            create_median_pose_dataset(dataset_path=folder, eval_db=eval_db)


if __name__ == '__main__':
    main()
