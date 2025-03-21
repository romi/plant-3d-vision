#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

from plantdb.commons import io
from romitask.log import get_logger

logger = get_logger(__name__)

#: The list of valid camera models.
VALID_MODELS = ["OPENCV", "RADIAL", "SIMPLE_RADIAL"]


def get_opencv_params_from_arrays(mtx, dist):
    """Return a dictionary for an 'OPENCV' model with parameters: fx, fy, cx, cy, k1, k2, p1, p2.

    Parameters
    ----------
    mtx : numpy.ndarray
        3x3 floating-point camera matrix.
    dist : numpy.ndarray
        Vector of distortion coefficients (k1, k2, p1, p2, k3).

    Returns
    -------
    dict
        A dictionary with named parameters as keys and their values.

    """
    params = {
        "fx": float(mtx[0, 0]),
        "fy": float(mtx[1, 1]),
        "cx": float(mtx[0, 2]),
        "cy": float(mtx[1, 2]),
        "k1": float(dist[0]),
        "k2": float(dist[1]),
        "p1": float(dist[2]),
        "p2": float(dist[3])
    }
    return params


def get_radial_params_from_arrays(mtx, dist):
    """Return a dictionary for a 'RADIAL' model with parameters: f, cx, cy, k1, k2.

    Parameters
    ----------
    mtx : numpy.ndarray
        3x3 floating-point camera matrix.
    dist : numpy.ndarray
        Vector of distortion coefficients (k1, k2, p1, p2, k3).

    Returns
    -------
    dict
        A dictionary with named parameters as keys and their values.

    """
    params = {
        "f": float(mtx[0, 0]),
        "cx": float(mtx[0, 2]),
        "cy": float(mtx[1, 2]),
        "k1": float(dist[0]),
        "k2": float(dist[1])
    }
    return params


def get_simple_radial_params_from_arrays(mtx, dist):
    """Return a dictionary for a 'SIMPLE RADIAL' model with parameters: f, cx, cy, k.

    Parameters
    ----------
    mtx : numpy.ndarray
        3x3 floating-point camera matrix.
    dist : numpy.ndarray
        Vector of distortion coefficients (k1, k2, p1, p2, k3).

    Returns
    -------
    dict
        A dictionary with named parameters as keys and their values.

    """
    params = {
        "f": float(mtx[0, 0]),
        "cx": float(mtx[0, 2]),
        "cy": float(mtx[1, 2]),
        "k": float(dist[0])
    }
    return params


def get_camera_params_from_arrays(model, **params):
    """Return a camera matrix and distortion vector for a given model from parameters."""
    if model.lower() == 'opencv':
        return get_opencv_params_from_arrays(**params)
    if model.lower() == 'radial':
        return get_radial_params_from_arrays(**params)
    if model.lower() == 'simple_radial':
        return get_simple_radial_params_from_arrays(**params)


def get_opencv_model_from_params(fx, fy, cx, cy, k1, k2, p1, p2, **kwargs):
    """Return a camera matrix and distortion vector for an 'OPENCV' model from parameters."""
    camera = np.array([[fx, 0, cx],
                       [0, fy, cy],
                       [0, 0, 1]], dtype='float32')
    distortion = np.array([k1, k2, p1, p2], dtype='float32')
    return camera, distortion


def get_radial_model_from_params(f, cx, cy, k1, k2, **kwargs):
    """Return a camera matrix and distortion vector for a 'RADIAL' model from parameters."""
    camera = np.array([[f, 0, cx],
                       [0, f, cy],
                       [0, 0, 1]], dtype='float32')
    distortion = np.array([k1, k2, 0., 0.], dtype='float32')
    return camera, distortion


def get_simple_radial_model_from_params(f, cx, cy, k, **kwargs):
    """Return a camera matrix and distortion vector for a 'SIMPLE RADIAL' model from parameters."""
    camera = np.array([[f, 0, cx],
                       [0, f, cy],
                       [0, 0, 1]], dtype='float32')
    distortion = np.array([k, 0., 0., 0.], dtype='float32')
    return camera, distortion


def get_camera_arrays_from_params(model, **params):
    """Return a camera matrix and distortion vector for a given model from parameters."""
    if model.lower() == 'opencv':
        return get_opencv_model_from_params(**params)
    if model.lower() == 'radial':
        return get_radial_model_from_params(**params)
    if model.lower() == 'simple_radial':
        return get_simple_radial_model_from_params(**params)


def get_camera_model_from_intrinsic(dataset, model="OPENCV"):
    """Get the camera parameters for selected model from intrinsic calibration.

    Parameters
    ----------
    dataset : plantdb.db.Scan
        Get the camera parameters for this scan dataset.
    model : {"OPENCV", "RADIAL", "SIMPLE_RADIAL"}, optional
        Get the parameter for this model.

    Returns
    -------
    dict
        A dictionary with named parameters as keys and their values.

    Raises
    ------
    ValueError
        If the camera model is not valid.

    See Also
    --------
    plant3dvision.camera.VALID_MODELS
    plant3dvision.tasks.calibration.IntrinsicCalibration

    """
    try:
        assert model.upper() in VALID_MODELS
    except AssertionError:
        raise ValueError(f"Selected model '{model}' is not valid!")

    camera_model_file = dataset.get_fileset('camera_model').get_file('camera_model')
    camera_models = io.read_json(camera_model_file)
    return camera_models[model.upper()]


def colmap_str_params(model, **kwargs):
    """Convert a camera model dictionary into a COLMAP string of parameters.

    Parameters
    ----------
    model : {'SIMPLE_RADIAL', 'RADIAL', 'OPENCV'}
        The name of the camera model.

    Other Parameters
    ----------------
    f : float
        The focal length, used with 'radial' & 'simple_radial' models.
    fx, fy : float
        The focal length in x and y, used with 'opencv' model.
    cx, cy : float
        The optical center in x and y, used with all models.
    k : float
        The radial distortion coefficients, used with 'simple_radial' models.
    k1, k2 : float
        The two radial distortion coefficients, used with 'opencv' & 'radial' models.
    p1, p2 : float
        The tangential distortion coefficients, used with 'opencv' model.

    Examples
    --------
    >>> from plant3dvision.camera import colmap_str_params
    >>> params = {'fx': 1200, 'fy': 1300, 'cx': 720, 'cy': 540, 'k1': 0.1, 'k2': 0.11, 'p1': 0.001, 'p2': 0.0011}
    >>> colmap_str_params('opencv', **params)
    '1200,1300,720,540,0.1,0.11,0.001,0.0011'
    >>> params = {'f': 1200, 'cx': 720, 'cy': 540, 'k1': 0.1, 'k2': 0.11}
    >>> colmap_str_params('radial', **params)
    '1200,1200,720,540,0.1,0.11,0.,0.'
    >>> params = {'f': 1200, 'cx': 720, 'cy': 540, 'k': 0.1}
    >>> colmap_str_params('simple_radial', **params)
    '1200,1200,720,540,0.1,0.,0.,0.'

    """
    if model.lower() == 'opencv':
        return f"{kwargs['fx']},{kwargs['fy']},{kwargs['cx']},{kwargs['cy']},{kwargs['k1']},{kwargs['k2']},{kwargs['p1']},{kwargs['p2']}"
    if model.lower() == 'radial':
        return f"{kwargs['f']},{kwargs['f']},{kwargs['cx']},{kwargs['cy']},{kwargs['k1']},{kwargs['k2']},0.,0."
    if model.lower() == 'simple_radial':
        return f"{kwargs['f']},{kwargs['f']},{kwargs['cx']},{kwargs['cy']},{kwargs['k']},0.,0.,0."


def get_camera_kwargs_from_params_list(model, params):
    """Get the kwargs from the list of parameters.

    Parameters
    ----------
    model : {'SIMPLE_RADIAL', 'RADIAL', 'OPENCV'}
        The name of the camera model.
    params : list
        The list of camera model parameters. The lenght and ordering depends on the `camera_model`.

    Returns
    -------
    dict
        A camera model dictionary with its parameter names as keys.

    Examples
    --------
    >>> from plant3dvision.camera import get_camera_kwargs_from_params_list
    >>> get_camera_kwargs_from_params_list('simple_radial', [1200, 720, 540, 0.1])  # params: f, cx, cy, k
    {'model': 'SIMPLE_RADIAL', 'f': 1200, 'cx': 720, 'cy': 540, 'k': 0.1}
    >>> get_camera_kwargs_from_params_list('radial', [1200, 720, 540, 0.1, 0.11])  # params: f, cx, cy, k1, k2
    {'model': 'RADIAL', 'f': 1200, 'cx': 720, 'cy': 540, 'k1': 0.1, 'k2': 0.11}
    >>> get_camera_kwargs_from_params_list('opencv', [1200, 1300, 720, 540, 0.1, 0.11, 0.001, 0.0011])  # params: fx, fy, cx, cy, k1, k2, p1, p2
    {'model': 'OPENCV', 'fx': 1200, 'fy': 1300, 'cx': 720, 'cy': 540, 'k1': 0.1, 'k2': 0.11, 'p1': 0.001, 'p2': 0.0011}
    >>> # As 'fx==fy' & 'p1==p2==0.', the returned model is "RADIAL":
    >>> get_camera_kwargs_from_params_list('opencv', [1200, 1200, 720, 540, 0.1, 0.11, 0.000, 0.0000])  # params: fx, fy, cx, cy, k1, k2, p1, p2
    {'model': 'RADIAL', 'f': 1200, 'cx': 720, 'cy': 540, 'k1': 0.1, 'k2': 0.11}
    >>> # As 'fx==fy' & 'p1==p2==0.' & 'k1==k2', the returned model is "SIMPLE_RADIAL":
    >>> get_camera_kwargs_from_params_list('opencv', [1200, 1200, 720, 540, 0.1, 0.10, 0.000, 0.0000])  # params: fx, fy, cx, cy, k1, k2, p1, p2
    {'model': 'SIMPLE_RADIAL', 'f': 1200, 'cx': 720, 'cy': 540, 'k': 0.1}

    """
    def _simple_radial(camera_params):
        """Parameter list is expected in the following order: f, cx, cy, k."""
        cam_dict = {'model': "SIMPLE_RADIAL"}
        cam_dict.update(dict(zip(['f', 'cx', 'cy', 'k'], camera_params)))
        return cam_dict

    def _radial(camera_params):
        """Parameter list is expected in the following order: f, cx, cy, k1, k2."""
        cam_dict = {'model': "RADIAL"}
        cam_dict.update(dict(zip(['f', 'cx', 'cy', 'k1', 'k2'], camera_params)))
        return cam_dict

    def _opencv(camera_params):
        """Parameter list is expected in the following order: fx, fy, cx, cy, k1, k2, p1, p2."""
        cam_dict = {'model': "OPENCV"}
        cam_dict.update(dict(zip(['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2'], camera_params)))
        return cam_dict

    camera_kwargs = {}
    if model.upper() == 'SIMPLE_RADIAL':
        camera_kwargs = _simple_radial(params)
    elif model.upper() == 'RADIAL':
        camera_kwargs = _radial(params)
    elif model.upper() == 'OPENCV':
        camera_kwargs = _opencv(params)
        # Check if this is a RADIAL model:
        if camera_kwargs['fx'] == camera_kwargs['fy'] and camera_kwargs['p1'] == camera_kwargs['p2'] == 0.:
            if camera_kwargs['k1'] == camera_kwargs['k2']:
                camera_kwargs["model"] = "SIMPLE_RADIAL"
                camera_kwargs['f'] = camera_kwargs.pop('fx')
                camera_kwargs.pop('fy')
                camera_kwargs.pop('k2')
                camera_kwargs.pop('p1')
                camera_kwargs.pop('p2')
                # The next two lines are a bit silly but useful to get correct key ordering...
                camera_kwargs['cx'] = camera_kwargs.pop('cx')
                camera_kwargs['cy'] = camera_kwargs.pop('cy')
                camera_kwargs['k'] = camera_kwargs.pop('k1')
            else:
                camera_kwargs["model"] = "RADIAL"
                camera_kwargs['f'] = camera_kwargs.pop('fx')
                camera_kwargs.pop('fy')
                camera_kwargs.pop('p1')
                camera_kwargs.pop('p2')
                # The next four lines are a bit silly but useful to get correct key ordering...
                camera_kwargs['cx'] = camera_kwargs.pop('cx')
                camera_kwargs['cy'] = camera_kwargs.pop('cy')
                camera_kwargs['k1'] = camera_kwargs.pop('k1')
                camera_kwargs['k2'] = camera_kwargs.pop('k2')

    return camera_kwargs


def get_camera_kwargs_from_images_metadata(img_f):
    """Get the dictionary of camera model parameters from an image file metadata.

    Parameters
    ----------
    img_f : plantdb.commons.fsdb.File
        An image `File` instance with a defined 'colmap_camera' metadata.

    Returns
    -------
    dict
        A camera model dictionary with its parameter names as keys.

    See Also
    --------
    plant3dvision.camera.get_camera_kwargs_from_params_list

    Notes
    -----
    The 'colmap_camera' metadata is a JSON style dictionary of camera parameters in OPENCV format.

    Examples
    --------
    >>> from plantdb.commons.test_database import test_database
    >>> from plant3dvision.camera import get_camera_kwargs_from_images_metadata
    >>> db = test_database()
    >>> db.connect()
    >>> scan = db.get_scan('real_plant_analyzed')
    >>> image_files = scan.get_fileset('images').get_files()
    >>> img_f = image_files[0]
    >>> get_camera_kwargs_from_images_metadata(img_f)
    {'model': 'SIMPLE_RADIAL', 'f': 1166.9518889440105, 'cx': 720.0, 'cy': 540.0, 'k': -0.0013571157486977348}
    >>> db.disconnect()
    """
    camera_model = img_f.get_metadata('colmap_camera')
    if camera_model is None:
        return None
    else:
        camera_model = camera_model['camera_model']
        return get_camera_kwargs_from_params_list(camera_model["model"], camera_model["params"])


def get_camera_kwargs_from_colmap_json(colmap_cameras):
    """Extract camera parameters from COLMAP JSON format and convert to named parameters.

    Processes a dictionary of COLMAP camera parameters in JSON format, converting camera IDs
    to integers if needed, and returns a dictionary of named camera parameters based on the
    camera model.

    Parameters
    ----------
    colmap_cameras : dict
        A dictionary containing COLMAP camera parameters where:
        - Keys are camera IDs (either str or int)
        - Values are dictionaries containing:
            - 'model': str, the camera model name
            - 'params': list, the camera parameters

    Returns
    -------
    dict
        A dictionary containing named camera parameters specific to the camera model.
        Keys are parameter names (e.g., 'fx', 'fy', 'cx', 'cy', 'k1', 'k2', etc.)
        and values are their corresponding numerical values.

    Raises
    ------
    KeyError
        If camera ID 1 is not found in the input dictionary
    IndexError
        If the camera parameters dictionary is empty

    Notes
    -----
    - Currently only processes camera ID 1 and will not work with multiple cameras
    - Input dictionary is expected to be in OPENCV camera model format
    - String camera IDs are automatically converted to integers

    Examples
    --------
    >>> import json
    >>> from plant3dvision.camera import get_camera_kwargs_from_colmap_json
    >>> from plantdb.commons.test_database import test_database
    >>> from plantdb.utils import locate_task_filesets
    >>> db = test_database()
    >>> db.connect()
    >>> scan = db.get_scan('real_plant_analyzed')
    >>> colmap_task = locate_task_filesets(scan, ['Colmap'])['Colmap']
    >>> colmap_json = scan.get_fileset(colmap_task).get_file('cameras')
    >>> colmap_cameras = json.load(colmap_json)
    >>> get_camera_kwargs_from_colmap_json(colmap_cameras)
    {'model': 'SIMPLE_RADIAL', 'f': 1166.9518889440105, 'cx': 720.0, 'cy': 540.0, 'k': -0.0013571157486977348}
    >>> db.disconnect()
    """
    # FIXME: will not work with more than one camera model!
    new_colmap_cameras = {}
    for key, value in colmap_cameras.items():
        # If loaded from JSON, camera id(s) may be str instead of int:
        if isinstance(key, str):
            new_colmap_cameras[int(key)] = value
        else:
            new_colmap_cameras[key] = value
    colmap_cameras = new_colmap_cameras.copy()
    del new_colmap_cameras
    return get_camera_kwargs_from_params_list(colmap_cameras[1]["model"], colmap_cameras[1]["params"])


def format_camera_params(colmap_cameras):
    """Format COLMAP camera parameters into a human-readable string representation.

    Creates a formatted string of camera parameters from a COLMAP camera dictionary,
    with appropriate line breaks between different parameter groups and formatted
    numerical values.

    Parameters
    ----------
    colmap_cameras : dict
        Dictionary containing COLMAP camera parameters. Expected to have camera
        model and various intrinsic parameters like focal length, principal point,
        and distortion coefficients.

    Returns
    -------
    str
        A formatted string containing camera parameters, with parameters grouped
        by prefix and formatted numbers. Small values (<0.1) are shown in
        scientific notation, others are rounded to 2 decimal places.

    Notes
    -----
    - Parameters with the same first letter are grouped on the same line
    - Values less than 0.1 are formatted in scientific notation
    - Values greater than or equal to 0.1 are rounded to 2 decimal places
    - The first parameter (typically 'model') starts the string
    - New lines are added when parameter prefixes change

    Examples
    --------
    >>> import json
    >>> from plant3dvision.camera import format_camera_params
    >>> from plantdb.commons.test_database import test_database
    >>> from plantdb.utils import locate_task_filesets
    >>> db = test_database()
    >>> db.connect()
    >>> scan = db.get_scan('real_plant_analyzed')
    >>> colmap_task = locate_task_filesets(scan, ['Colmap'])['Colmap']
    >>> colmap_json = scan.get_fileset(colmap_task).get_file('cameras')
    >>> colmap_cameras = json.load(colmap_json)
    >>> print(format_camera_params(colmap_cameras))
    model: SIMPLE_RADIAL
    f: 1166.95
    cx: 720.0, cy: 540.0
    k: -1.36e-03
    >>> db.disconnect()
    """
    camera_kwargs = get_camera_kwargs_from_colmap_json(colmap_cameras)
    return format_camera_kwargs(camera_kwargs)


def format_camera_kwargs(camera_kwargs):
    """Format COLMAP camera parameters into a human-readable string representation.

    Parameters
    ----------
    camera_kwargs : dict
        A dictionary containing named camera parameters specific to the camera model.

    Returns
    -------
    str
        A formatted string containing camera parameters, with parameters grouped
        by prefix and formatted numbers. Small values (<0.1) are shown in
        scientific notation, others are rounded to 2 decimal places.

    Examples
    --------
    >>> from plantdb.commons.test_database import test_database
    >>> from plant3dvision.camera import get_camera_kwargs_from_images_metadata
    >>> from plant3dvision.camera import format_camera_kwargs
    >>> db = test_database()
    >>> db.connect()
    >>> scan = db.get_scan('real_plant_analyzed')
    >>> image_files = scan.get_fileset('images').get_files()
    >>> img_f = image_files[0]
    >>> camera_kwargs = get_camera_kwargs_from_images_metadata(img_f)
    >>> print(format_camera_kwargs(camera_kwargs))
    model: SIMPLE_RADIAL
    f: 1166.95
    cx: 720.0, cy: 540.0
    k: -1.36e-03
    >>> db.disconnect()
    """
    prev_param = list(camera_kwargs.keys())[0]
    cam_str = f"{prev_param}: {camera_kwargs.pop(prev_param)}"  # should start by 'model' key
    for k, v in camera_kwargs.items():
        if v < 0.1:
            value = f"{v:.2e}"
        else:
            value = round(v, 2)

        if k.startswith(prev_param[0]):
            cam_str += f", {k}: {value}"
        else:
            cam_str += "\n"
            cam_str += f"{k}: {value}"
        prev_param = k
    return cam_str


def get_colmap_cameras_from_calib_scan(calibration_scan):
    # - Check an ExtrinsicCalibration task has been performed for the calibration scan:
    calib_fs = [s for s in calibration_scan.get_filesets() if "ExtrinsicCalibration" in s.id]
    if len(calib_fs) == 0:
        raise Exception(
            f"Could not find an 'ExtrinsicCalibration' fileset in calibration scan '{calibration_scan.id}'!")
    else:
        # TODO: What happens if we have more than one 'ExtrinsicCalibration' job ?!
        if len(calib_fs) > 1:
            logger.warning(
                f"More than one 'ExtrinsicCalibration' found for calibration scan '{calibration_scan.id}'!")
    # - Get the 'images' fileset from the extrinsic calibration scan
    cameras_file = calib_fs[0].get_file("cameras")
    return io.read_json(cameras_file)


def colmap_params_from_kwargs(**kwargs):
    """Convert camera parameters to COLMAP format based on camera model.

    Converts camera intrinsic parameters from different model formats (OpenCV, Radial,
    or Simple Radial) into COLMAP's parameter list format.

    Other Parameters
    ----------------
    model : str
        Camera model type ('opencv', 'radial', or 'simple_radial').
    fx, fy : float
        Focal lengths in x and y directions (OpenCV model only).
    f : float
        Focal length (Radial and Simple Radial models).
    cx, cy : float
        Principal point coordinates.
    k1, k2 : float
        Radial distortion coefficients.
    p1, p2 : float
        Tangential distortion coefficients (OpenCV model only).
    k : float
        Single radial distortion coefficient (Simple Radial model only).

    Returns
    -------
    list
        Camera parameters in COLMAP format with 8 elements:
        - For OpenCV: [fx, fy, cx, cy, k1, k2, p1, p2]
        - For Radial: [f, f, cx, cy, k1, k2, 0, 0]
        - For Simple Radial: [f, f, cx, cy, k, 0, 0, 0]

    Raises
    ------
    KeyError
        If required parameters for the specified model are missing.

    Notes
    -----
    The function assumes lowercase model names in comparison.
    Zero values are used for unused parameters in simpler models.

    Examples
    --------
    >>> # OpenCV model
    >>> params = colmap_params_from_kwargs(
    ...     model='opencv', fx=1000, fy=1000, cx=500, cy=500,
    ...     k1=0.1, k2=0.01, p1=0.001, p2=0.001)
    >>> print(params)
    [1000, 1000, 500, 500, 0.1, 0.01, 0.001, 0.001]

    >>> # Simple Radial model
    >>> params = colmap_params_from_kwargs(
    ...     model='simple_radial', f=1000, cx=500, cy=500, k=0.1)
    >>> print(params)
    [1000, 1000, 500, 500, 0.1, 0, 0, 0]
    """

    model = kwargs.get('model')
    if model.lower() == 'opencv':
        return [kwargs['fx'], kwargs['fy'], kwargs['cx'], kwargs['cy'], kwargs['k1'], kwargs['k2'], kwargs['p1'],
                kwargs['p2']]
    if model.lower() == 'radial':
        return [kwargs['f'], kwargs['f'], kwargs['cx'], kwargs['cy'], kwargs['k1'], kwargs['k2'], 0., 0.]
    if model.lower() == 'simple_radial':
        return [kwargs['f'], kwargs['f'], kwargs['cx'], kwargs['cy'], kwargs['k'], 0., 0., 0.]
