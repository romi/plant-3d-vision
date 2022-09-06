import filecmp
import json
from collections import Counter
from collections import OrderedDict
from os.path import splitext
from pathlib import Path

import numpy as np
import open3d as o3d
from matplotlib import cm
from matplotlib import pyplot as plt

import plantdb
from plant3dvision.metrics import SetMetrics
from plant3dvision.metrics import chamfer_distance
from plant3dvision.metrics import point_cloud_registration_fitness
from plant3dvision.metrics import surface_ratio
from plant3dvision.metrics import volume_ratio
from plant3dvision.tasks.colmap import compute_colmap_poses_from_camera_json
from plant3dvision.tasks.colmap import get_calibrated_poses
from plant3dvision.tasks.colmap import get_cnc_poses
from plantdb import FSDB
from plantdb.io import read_image
from plantdb.io import read_json
from plantdb.io import read_npz
from plantdb.io import read_point_cloud
from plantdb.io import read_triangle_mesh
from romitask.log import configure_logger

logger = configure_logger(__name__)


def jsonify_tuple_keys(json_dict: dict) -> dict:
    return {f'{k[0]} - {k[1]}': v for k, v in json_dict.items()}


def unjsonify_tuple_keys(json_dict: dict) -> dict:
    return {tuple(k.split(' - ')): v for k, v in json_dict.items()}


def dict_sort_by_values(dico) -> OrderedDict:
    """Returns an OrderedDict instance from value sorted dictionary."""
    import operator
    sorted_tuples = sorted(dico.items(), key=operator.itemgetter(1))
    sorted_dict = OrderedDict()
    for k, v in sorted_tuples:
        sorted_dict[k] = v

    return sorted_dict


def average_pairwise_comparison(fbf_comp_dict: dict) -> float:
    if len(fbf_comp_dict.values()):
        return sum(fbf_comp_dict.values()) / len(fbf_comp_dict.values())
    else:
        return None


def save_data_repartition(data, data_type, db):
    """Save repartition plots.

    Parameters
    ----------
    data : list
        Data to represent as boxplots.
    data_type : str
        Name of the data represented on th x-axis, usually "angles" or "internodes".
    db : pathlib.Path
        Directory to use to save the graphs.

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
    return


def pairwise_heatmap(pw_dict, scans_list, task_name, metrics, db, **kwargs):
    """Save a PNG of the pairwise heatmap.

    Parameters
    ----------
    pw_dict : dict
        Pairwise dictionary with a float value and a pair of scan as keys.
    scans_list : list
        Scans to use in pairwise heatmap representation.
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
    fig.set_size_inches(max(n_scans / 2., 7.), max(n_scans / 2., 7.))
    im = ax.imshow(pw_mat)

    # We want to show all ticks...
    ax.set_xticks(np.arange(n_scans))
    ax.set_yticks(np.arange(n_scans))
    # ... and label them with the respective list entries
    ax.set_xticklabels(scans_list)
    ax.set_yticklabels(scans_list)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(n_scans):
        for j in range(i + 1, n_scans):
            text = ax.text(i, j, "{:.3e}".format(pw_mat[i, j]),
                           ha="center", va="center", color="w", size=7)

    ax.set_title(f"Pairwise heatmap for {task_name} {metrics}")
    plt.tight_layout()
    cbar = fig.colorbar(im, shrink=0.7, format="%.2e")
    fname = kwargs.get("fname", Path(db.basedir) / f'{task_name}-{metrics}_heatmap.png')
    fig.savefig(fname)
    return


def _get_task_fileset(scan_dataset, task_name):
    """Returns the `Fileset` object produced by `task_name` in given `scan_dataset`.

    Parameters
    ----------
    scan_dataset : plantdb.FSDB.Scan
        Dataset where to take the `Fileset` related to `task_name`.
    task_name : str
        Name of the task that generated the `Fileset`.

    Returns
    -------
    plantdb.FSDB.Scan.Fileset
        A `Fileset` instance produced by `task_name` in given `scan_dataset`

    Raises
    ------
    AssertionError
        If there is NO `Fileset` object matching the task.
        If there is more than one `Fileset` object matching the task.
        This would mean that this task was run several times (with different parameters).

    """
    # - List fileset potentially related to the task (the name of the fileset should start with the task name):
    if task_name == "IntrinsicCalibration":
        # This is a special case: task `IntrinsicCalibration` generates a fileset named "camera_model"
        task_filesets = [scan_dataset.get_fileset("camera_model")]
    else:
        task_filesets = [fs for fs in scan_dataset.filesets if fs.id.startswith(task_name)]

    # - Assert fileset unicity:
    try:
        assert len(task_filesets) == 1
    except AssertionError:
        if len(task_filesets) == 0:
            msg = f"There is NO occurrence of the task {task_name}!\n"
        else:
            msg = f"There is more than one occurrence of the task {task_name}:\n"
            msg += "  - " + "  - ".join([tfs.id for tfs in task_filesets])
        raise AssertionError(msg)

    return task_filesets[0]


def _get_files(scan_dataset, task_name, unique=False):
    """Returns the `File` instance(s) produced by `task_name` in given `scan_dataset`.

    Parameters
    ----------
    scan_dataset : plantdb.FSDB.Scan
        Dataset where to take the `Fileset` related to `task_name`.
    task_name : str
        Name of the task that generated the `File` in the `Fileset`.
    unique : bool, optional
        If ``True``, assert that there is only one `File` in the `Fileset`.

    Returns
    -------
    list(plantdb.FSDB.Scan.Fileset.File)
        The `File` objects produced by `task_name` in given `scan_dataset`

    Raises
    ------
    AssertionError
        If there is more than one `File` object in the task `Fileset`, requires ``unique=True``.

    """
    # - Get the `Fileset` corresponding to the replicated task:
    fs = _get_task_fileset(scan_dataset, task_name)
    # - Get the list of `File` ids (task outputs)
    f_list = [f.id for f in fs.files]
    if unique:
        # - Assert file unicity:
        try:
            assert len(f_list) == 1
        except AssertionError:
            raise AssertionError("This method can only compare tasks that output a single point cloud!")
    # - Return the `File` objects:
    return [fs.get_file(f) for f in f_list]


def compare_intrinsic_params(db, task_name, scans_list):
    """Compare the estimated camera intrinsic parameters.

    Parameters
    ----------
    db : plantdb.FSDB
        Local ROMI database instance with the replicated scan datasets
    task_name : str
        name of the task to test
    scans_list : list of plantdb.FSDB.Scan
        List of `Scan` instance to compare.

    References
    ----------
    https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

    """
    from plantdb import io

    def _get_intrinsic_calibration_params(scan, model_name):
        from plant3dvision.camera import get_camera_arrays_from_params
        # - Get the camera model JSON file and load it:
        camera_file = scan.get_fileset("camera_model").get_file("camera_model")
        camera_params = io.read_json(camera_file)[model_name]
        # - Get the camera matrix & the distortion parameters:
        camera, distortion = get_camera_arrays_from_params(model_name, **camera_params)
        return camera, distortion

    def _get_colmap_camera_params_from_images_metadata(scan, model_name):
        # FIXME: will not work with more than one camera model!
        # - Get the 'images' `Fileset`:
        images_fs = scan.get_fileset("images")
        # - Get one image `File`:
        image_f = images_fs.get_files()[0]
        # - Get the camera model dictionary from the metadata:
        camera_dict = image_f.get_metadata()["colmap_camera"]["camera_model"]
        camera_params = camera_dict['params']
        # - Get the camera matrix :
        camera = np.array([[camera_params[0], 0, camera_params[2]],
                           [0, camera_params[1], camera_params[3]],
                           [0, 0, 1]], dtype='float32')
        # - Get the distortion parameters (k1, k2, p1, p2):
        distortion = np.array(camera_params[4:])
        return camera, distortion

    def _get_colmap_camera_params_from_cameras_json(scan, model_name):
        # FIXME: will not work with more than one camera model!
        # - Get the camera model JSON file and load it:
        camera_file = _get_task_fileset(scan, task_name).get_file("cameras")
        camera_dict = io.read_json(camera_file)
        camera_params = camera_dict['1']['params']
        # - Get the camera matrix :
        camera = np.array([[camera_params[0], 0, camera_params[2]],
                           [0, camera_params[1], camera_params[3]],
                           [0, 0, 1]], dtype='float32')
        # - Get the distortion parameters (k1, k2, p1, p2):
        distortion = np.array(camera_params[4:])
        return camera, distortion

    def _compare_models(model):
        camera = {}
        distortion = {}
        for scan in scans_list:
            # - Get the camera matrix and distortion parameters (k1, k2, p1, p2 as k3 is set to 0):
            camera[scan.id], distortion[scan.id] = _get_intrinsic_params(scan, model)

        # - Compute average camera matrix:
        mean_camera = np.mean(list(camera.values()), axis=0)
        logger.debug(f"Average camera matrix:\n{mean_camera}")
        # - Compute average distortion parameters:
        mean_distortion = np.mean(list(distortion.values()), axis=0)
        logger.debug(f"Average distortion parameters:\n{mean_distortion}")

        # - Compute deviation to average:
        camera_mean_dev = {}
        distortion_mean_dev = {}
        for scan in scans_list:
            # - Compute deviation of each camera from average camera:
            camera_mean_dev[scan.id] = np.array(mean_camera - camera[scan.id]).tolist()
            # - Compute deviation of each distortion parameters from average distortion parameters:
            distortion_mean_dev[scan.id] = np.array(mean_distortion - distortion[scan.id]).tolist()

        def _get_fx_fy_cx_cy(camera_mtx):
            return camera_mtx[0, 0], camera_mtx[1, 1], camera_mtx[0, 2], camera_mtx[1, 2]

        mean_fx, mean_fy, mean_cx, mean_cy = _get_fx_fy_cx_cy(mean_camera)
        # - Compute the ABSOLUTE deviation of camera parameters (fx, fy, cx, cy) from their respective average:
        cam_params_dev_to_mean = []
        for scan in scans_list:
            fx, fy, cx, cy = _get_fx_fy_cx_cy(camera[scan.id])
            cam_params_dev_to_mean.append([abs(mean_fx - fx), abs(mean_fy - fy), abs(mean_cx - cx), abs(mean_cy - cy)])
        # - Compute the mean deviation of camera parameters (fx, fy, cx, cy) from their respective average:
        mean_fx_dev, mean_fy_dev, mean_cx_dev, mean_cy_dev = np.mean(np.array(cam_params_dev_to_mean), axis=0)

        mean_k1, mean_k2, mean_p1, mean_p2 = mean_distortion
        # - Compute the ABSOLUTE deviation of distortion parameters (k1, k2, p1, p2) from their respective average:
        dist_params_dev_to_mean = []
        for scan in scans_list:
            k1, k2, p1, p2 = distortion[scan.id]
            dist_params_dev_to_mean.append([abs(mean_k1 - k1), abs(mean_k2 - k2), abs(mean_p1 - p1), abs(mean_p2 - p2)])
        # - Compute the mean deviation of distortion parameters (k1, k2, p1, p2) from their respective average:
        mean_k1_dev, mean_k2_dev, mean_p1_dev, mean_p2_dev = np.mean(np.array(dist_params_dev_to_mean), axis=0)

        json_fname = f'{task_name}{"_" + model if model != "" else ""}_intrinsic_params_comparison.json'
        with open(Path(db.basedir) / json_fname, 'w') as out_file:
            json.dump({
                'mean fx deviation from average fx': float(mean_fx_dev),
                'mean fy deviation from average fy': float(mean_fy_dev),
                'mean cx deviation from average cx': float(mean_cx_dev),
                'mean cy deviation from average cy': float(mean_cy_dev),
                'mean k1 deviation from average k1': float(mean_k1_dev),
                'mean k2 deviation from average k2': float(mean_k2_dev),
                'mean p1 deviation from average p1': float(mean_p1_dev),
                'mean p2 deviation from average p2': float(mean_p2_dev),
                'deviation to average camera': camera_mean_dev,
                'deviation to average distortion': distortion_mean_dev,
            }, out_file, indent=2)

    logger.info(f"Comparing '{task_name}' outputs for {len(scans_list)} replicated scans!")

    # Defines the method to use to get the estimated intrinsic parameters:
    if task_name == "Colmap":
        _get_intrinsic_params = _get_colmap_camera_params_from_cameras_json
    elif task_name == "IntrinsicCalibration":
        _get_intrinsic_params = _get_intrinsic_calibration_params
    elif task_name == "ExtrinsicCalibration":
        _get_intrinsic_params = _get_colmap_camera_params_from_images_metadata
    else:
        raise ValueError(f"Unknown task `{task_name}` for comparison of estimated intrinsic camera parameters!")

    if task_name == "IntrinsicCalibration":
        for model in ["OPENCV", "RADIAL", "SIMPLE_RADIAL"]:
            _compare_models(model)
    else:
        _compare_models("")

    return


def compare_colmap_poses(db, task_name, scans_list):
    """Compare the estimated poses by COLMAP.

    Parameters
    ----------
    db : plantdb.FSDB
        Local ROMI database instance with the replicated scan datasets
    task_name : str
        name of the task to test
    scans_list : list of plantdb.FSDB.Scan
        List of `Scan` instance to compare.

    """
    from scipy.spatial.distance import euclidean
    logger.info(f"Comparing '{task_name}' outputs for {len(scans_list)} replicated scans!")

    # - Get CNC poses indexed by image ids:
    cnc_poses = {}
    n = 0
    while cnc_poses == {} and n < len(scans_list):
        cnc_poses = get_cnc_poses(scans_list[n])
        n_imgs = len(scans_list[n].get_fileset('images').get_files())
        n_poses = len(cnc_poses)
        if n_poses != n_imgs:
            cnc_poses = {}
        n += 1
    if cnc_poses == {}:
        logger.critical(f"Could not get CNC poses for any of the duplicated scan!")
        return

    # - Get image ids:
    image_ids = list(cnc_poses.keys())

    # - Get all poses estimated by COLMAP indexed by image id and by replicate id:
    colmap_poses = {}  # {scan_id: {img_id: [x, y, z]}}
    poses_array = []
    for scan in scans_list:
        if task_name.startswith("Colmap"):
            colmap_poses[scan.id] = compute_colmap_poses_from_camera_json(scan)  # {img_id: [x, y, z]}
        elif "Calibration" in task_name:
            colmap_poses[scan.id] = get_calibrated_poses(scan)
        else:
            logger.critical(f"Nothing defined here for a task named '{task_name}'!")
        poses_array.append([p for im_id, p in colmap_poses[scan.id].items()])

    # - Get the list of all colmap poses (XYZ) indexed by image id:
    colmap_poses_by_image = {im: [] for im in colmap_poses[scan.id].keys()}
    for scan_id, scan_poses in colmap_poses.items():
        for im_id, pose in scan_poses.items():
            colmap_poses_by_image[im_id].append(pose)

    # - Compute the distance of estimated colmap poses from the CNC pose for each replicate and image:
    dist2cnc_pose_by_image = {}
    for im_id, poses in colmap_poses_by_image.items():
        xyz_cnc = cnc_poses[im_id][:3]
        dist2cnc_pose_by_image[im_id] = [euclidean(pose, xyz_cnc) for pose in poses]
    # Compute the mean distance of estimated colmap poses from the CNC pose for each image:
    mean_dist_to_cnc = {im_id: np.mean(d2cnc) for im_id, d2cnc in dist2cnc_pose_by_image.items()}
    mean_dist_to_cnc_rep = {scan.id: np.mean([dist2cnc_pose_by_image[im_id][scan_idx] for im_id in image_ids]) for
                            scan_idx, scan in enumerate(scans_list)}

    # Now compute mean pose per image:
    mean_pose_by_image = {}
    for im_id, pose in colmap_poses_by_image.items():
        mean_pose_by_image[im_id] = np.mean(np.array(pose), axis=0)

    # - Then compute the distance from the mean pose for each replicate:
    dist2mean_pose_by_image = {}
    for im_id, mean_pose in mean_pose_by_image.items():
        dist2mean_pose_by_image[im_id] = [euclidean(colmap_poses[scan.id][im_id], mean_pose) for scan in scans_list]
    # Compute deviation statistics per image:
    mean_dist = {im_id: np.mean(d2m) for im_id, d2m in dist2mean_pose_by_image.items()}
    mean_dist_rep = {scan.id: np.mean([dist2mean_pose_by_image[im_id][scan_idx] for im_id in image_ids]) for
                     scan_idx, scan in enumerate(scans_list)}
    std_dist = {im_id: np.std(d2m) for im_id, d2m in dist2mean_pose_by_image.items()}
    # Compute global deviation statistics:
    global_mean_dist = np.mean([d2m for d2m in dist2mean_pose_by_image.values()])
    global_mean_dist_to_cnc = np.mean([d2cnc for d2cnc in dist2cnc_pose_by_image.values()])

    with open(Path(db.basedir) / f'{task_name}_colmap_poses_comparison.json', 'w') as out_file:
        json.dump({
            'global mean distance to CNC pose': global_mean_dist_to_cnc,
            'replicate mean distance to CNC pose': dict_sort_by_values(mean_dist_to_cnc_rep),
            'mean distance to CNC pose': mean_dist_to_cnc,
            'global mean distance to mean pose': global_mean_dist,
            'replicate mean distance to mean pose': dict_sort_by_values(mean_dist_rep),
            'mean distance to mean pose': mean_dist,
            'std distance to mean pose': std_dist,
            'distance to mean pose': dist2mean_pose_by_image
        }, out_file, indent=2)

    return


def compare_binary_mask(db, task_name, scans_list):
    """Use set metrics to compare binary masks for each unique pairs of repetition.

    Parameters
    ----------
    db : plantdb.FSDB
        Local ROMI database instance with the replicated scan datasets
    task_name : str
        name of the task to test
    scans_list : list of plantdb.FSDB.Scan
        List of `Scan` instance to compare.

    """
    logger.info(f"Comparing '{task_name}' outputs for {len(scans_list)} replicated scans!")
    precision, recall = {}, {}
    mean_precision, mean_recall = {}, {}
    median_precision, median_recall = {}, {}
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
            ref_dir = Path(db.basedir, ref_scan.id, ref_fileset.id)
            flo_dir = Path(db.basedir, flo_scan.id, flo_fileset.id)
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
                _, _, _, _, p, r = SetMetrics(ref_mask, flo_mask)
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
    # Write a JSON summary file of the comparison:
    with open(Path(db.basedir) / f'{task_name}_binary_mask_comparison.json', 'w') as out_file:
        json.dump({
            'mean precision': mean_precision, 'mean recall': mean_recall,
            'median precision': median_precision, 'median recall': median_recall,
            'precision': precision, 'recall': recall,
            'global mean precision': global_mean_precision, 'global mean recall': global_mean_recall,
            'global median precision': global_median_precision, 'global median recall': global_median_recall
        }, out_file, indent=2)

    return


def compare_pointcloud(db, task_name, scans_list):
    """Use the chamfer distance to compare point cloud for each unique pairs of repetition.

    Parameters
    ----------
    db : plantdb.FSDB
        Local ROMI database instance with the replicated scan datasets
    task_name : str
        name of the task to test
    scans_list : list of plantdb.FSDB.Scan
        List of `Scan` instance to compare.

    """
    logger.info(f"Comparing '{task_name}' outputs for {len(scans_list)} replicated scans!")
    chamfer_dist = {}
    fitness, residuals = {}, {}
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

    average_chamfer_dist = average_pairwise_comparison(chamfer_dist)
    average_fitness = average_pairwise_comparison(fitness)
    average_residuals = average_pairwise_comparison(residuals)
    # Write a JSON summary file of the comparison:
    with open(Path(db.basedir) / f'{task_name}_pointcloud_comparison.json', 'w') as out_file:
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


def compare_voxels(db, task_name, scans_list):
    """Compare voxel matrices.

    Parameters
    ----------
    db : plantdb.FSDB
        Local ROMI database instance with the replicated scan datasets
    task_name : str
        name of the task to test
    scans_list : list of plantdb.FSDB.Scan
        List of `Scan` instance to compare.

    """
    logger.info(f"Comparing '{task_name}' outputs for {len(scans_list)} replicated scans!")
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

    average_r2 = {}
    average_mean_abs_dev = {}
    for label in labels:
        average_r2[label] = average_pairwise_comparison(r2[label])
        average_mean_abs_dev[label] = average_pairwise_comparison(mean_abs_dev[label])

    # Write a JSON summary file of the comparison:
    with open(Path(db.basedir) / f'{task_name}_voxels_comparison.json', 'w') as out_file:
        json.dump({
            'sum of square difference': {label: jsonify_tuple_keys(r2[label]) for label in labels},
            'mean absolute deviation': {label: jsonify_tuple_keys(mean_abs_dev[label]) for label in labels},
            'average pairwise sum of square difference': {label: average_r2[label] for label
                                                          in labels},
            'average pairwise mean absolute deviation': {label: average_mean_abs_dev[label] for label
                                                         in labels}
        }, out_file)

    # Creates a heatmap of the pairwise comparisons:
    for label in labels:
        metric_str = f'{label} sum of square difference'
        pairwise_heatmap(r2[label], scans_list, metric_str, task_name, db,
                         fname=Path(db.basedir) / f"{task_name}-{metric_str.replace(' ', '_')}_heatmap.png")
        metric_str = f'{label} mean absolute deviation'
        pairwise_heatmap(mean_abs_dev[label], scans_list, f'{label} mean absolute deviation', task_name, db,
                         fname=Path(db.basedir) / f"{task_name}-{metric_str.replace(' ', '_')}_heatmap.png")

    return


def compare_labelled_pointcloud(db, task_name, scans_list):
    """Use the closed point in neighborhood trees to compare labelled point clouds.

    Parameters
    ----------
    db : plantdb.FSDB
        Local ROMI database instance with the replicated scan datasets
    task_name : str
        name of the task to test
    scans_list : list of plantdb.FSDB.Scan
        List of `Scan` instance to compare.

    """
    logger.info(f"Comparing '{task_name}' outputs for {len(scans_list)} replicated scans!")
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
            # For each point of the floating point cloud...
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

    average_precision = {}
    average_recall = {}
    for label in unique_labels:
        average_precision[label] = average_pairwise_comparison(precision[label])
        average_recall[label] = average_pairwise_comparison(recall[label])

    # Write a JSON summary file of the comparison:
    with open(Path(db.basedir) / f'{task_name}_labelled_pointcloud_comparison.json', 'w') as out_file:
        json.dump({
            'precision': {ulabel: jsonify_tuple_keys(precision[ulabel]) for ulabel in unique_labels},
            'recall': {ulabel: jsonify_tuple_keys(recall[ulabel]) for ulabel in unique_labels},
            'average pairwise precision': {label: average_precision[label] for label in unique_labels},
            'average pairwise recall': {label: average_recall[label] for label in unique_labels}
        }, out_file)

    # Creates a heatmap of the pairwise comparisons:
    for ulabel in unique_labels:
        metric_str = f'{ulabel} precision'
        pairwise_heatmap(precision[ulabel], scans_list, metric_str, task_name, db,
                         fname=Path(db.basedir) / f"{task_name}-{metric_str.replace(' ', '_')}_heatmap.png")
        metric_str = f'{ulabel} recall'
        pairwise_heatmap(recall[ulabel], scans_list, f'{ulabel} recall', task_name, db,
                         fname=Path(db.basedir) / f"{task_name}-{metric_str.replace(' ', '_')}_heatmap.png")
    return


def compare_trianglemesh_points(db, task_name, scans_list):
    """Use the chamfer distance to compare mesh vertices point cloud for each unique pairs of repetition.

    Parameters
    ----------
    db : plantdb.FSDB
        Local ROMI database instance with the replicated scan datasets
    task_name : str
        name of the task to test
    scans_list : list of plantdb.FSDB.Scan
        List of `Scan` instance to compare.

    """
    logger.info(f"Comparing '{task_name}' outputs for {len(scans_list)} replicated scans!")
    chamfer_dist = {}
    surf_ratio = {}
    vol_ratio = {}
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

    average_chamfer_dist = average_pairwise_comparison(chamfer_dist)
    average_surf_ratio = average_pairwise_comparison(surf_ratio)
    average_vol_ratio = average_pairwise_comparison(vol_ratio)
    # Write a JSON summary file of the comparison:
    with open(Path(db.basedir) / f'{task_name}_trianglemesh_points_comparison.json', 'w') as out_file:
        json.dump({
            'chamfer distances': jsonify_tuple_keys(chamfer_dist),
            'surface ratio': jsonify_tuple_keys(surf_ratio),
            'volume ratio': jsonify_tuple_keys(vol_ratio),
            'average chamfer distances': average_chamfer_dist,
            'average surface ratio': average_surf_ratio,
            'average volume ratio': average_vol_ratio
        }, out_file)

    return


def compare_curveskeleton_points(db, task_name, scans_list):
    """Use the chamfer distance to compare point cloud for each unique pairs of repetition.

    Parameters
    ----------
    db : plantdb.FSDB
        Local ROMI database instance with the replicated scan datasets
    task_name : str
        name of the task to test
    scans_list : list of plantdb.FSDB.Scan
        List of `Scan` instance to compare.

    """
    logger.info(f"Comparing '{task_name}' outputs for {len(scans_list)} replicated scans!")
    chamfer_dist = {}
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

    average_chamfer_dist = average_pairwise_comparison(chamfer_dist)
    # Write a JSON summary file of the comparison:
    with open(Path(db.basedir) / f'{task_name}_curveskeleton_points_comparison.json', 'w') as out_file:
        json.dump({'chamfer distances': chamfer_dist,
                   'average chamfer distance': average_chamfer_dist}, out_file)

    return


def compare_angles_and_internodes(db, task_name, scans_list):
    """Coarse comparison of sequences of angles and internodes.

    Prints a dict containing the number of scans for each number of organs indexed, ex : {12:4, 14:6} (4 scans with 12
    organs recognized and 6 with 14 organs)
    For the biggest number of scans with same number of organs, creates plots of the repartition of angles (and
    internodes) for each organ id

    Parameters
    ----------
    db : plantdb.FSDB
        Local ROMI database instance with the replicated scan datasets
    task_name : str
        name of the task to test
    scans_list : list of plantdb.FSDB.Scan
        List of `Scan` instance to compare.

    """
    logger.info(f"Comparing '{task_name}' outputs for {len(scans_list)} replicated scans!")
    angles_and_internodes = {}
    nb_organs = []
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
    max_occurrence_nb_organs = counter_nb_organs.most_common()[0][0]
    angles = [angles_and_internodes[scan_num]["angles"] for scan_num in angles_and_internodes
              if angles_and_internodes[scan_num]["nb_organs"] == max_occurrence_nb_organs]
    internodes = [angles_and_internodes[scan_num]["internodes"] for scan_num in angles_and_internodes
                  if angles_and_internodes[scan_num]["nb_organs"] == max_occurrence_nb_organs]
    save_data_repartition(angles, "angles", Path(db.basedir))
    save_data_repartition(internodes, "internodes", Path(db.basedir))

    # Write a JSON summary file of the comparison:
    with open(Path(db.basedir) / f'{task_name}_comparison.json', 'w') as out_file:
        json.dump({
            'Min number of organs detected': min(nb_organs),
            'Max number of organs detected': max(nb_organs),
            'Average number of organs detected': np.mean(nb_organs),
            'number of scans with the same nb of organs': counter_nb_organs
        }, out_file)
