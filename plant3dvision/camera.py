#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

from plantdb import io
from romitask.log import configure_logger

logger = configure_logger(__name__)

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
    """Convert a camera model dictionary into a COLMAP string of parameters."""
    if model.lower() == 'opencv':
        return f"{kwargs['fx']},{kwargs['fy']},{kwargs['cx']},{kwargs['cy']},{kwargs['k1']},{kwargs['k2']},{kwargs['p1']},{kwargs['p2']}"
    if model.lower() == 'radial':
        return f"{kwargs['f']},{kwargs['f']},{kwargs['cx']},{kwargs['cy']},{kwargs['k1']},{kwargs['k2']},0.,0."
    if model.lower() == 'simple_radial':
        return f"{kwargs['f']},{kwargs['f']},{kwargs['cx']},{kwargs['cy']},{kwargs['k']},0.,0.,0."


def get_camera_kwargs_from_params_list(model, params):
    """Get the kwargs from the list of parameters."""
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
        if camera_kwargs['fx'] == camera_kwargs['fy'] and camera_kwargs['p1'] == camera_kwargs['p1'] == 0.:
            camera_kwargs["model"] = "RADIAL"
            camera_kwargs['f'] = camera_kwargs.pop('fx')
            camera_kwargs.pop('fy')
            camera_kwargs.pop('p1')
            camera_kwargs.pop('p2')
            # The next lines are a bit silly but useful to get correct key ordering...
            camera_kwargs['cx'] = camera_kwargs.pop('cx')
            camera_kwargs['cy'] = camera_kwargs.pop('cy')
            camera_kwargs['k1'] = camera_kwargs.pop('k1')
            camera_kwargs['k2'] = camera_kwargs.pop('k2')

    return camera_kwargs


def get_camera_kwargs_from_images_metadata(img_f):
    camera_model = img_f.get_metadata('colmap_camera')
    if camera_model is None:
        return None
    else:
        camera_model = camera_model['camera_model']
        return get_camera_kwargs_from_params_list(camera_model["model"], camera_model["params"])


def get_camera_kwargs_from_colmap_json(colmap_cameras):
    """Get a dictionary of named camera parameter depending on camera model."""
    # FIXME: will not work with more than one camera model!
    # If loaded from JSON, camera id(s) may be str instead of int:
    new_colmap_cameras = {}
    for key, value in colmap_cameras.items():
        if isinstance(key, str):
            new_colmap_cameras[int(key)] = value
        else:
            new_colmap_cameras[key] = value
    colmap_cameras = new_colmap_cameras.copy()
    del new_colmap_cameras
    return get_camera_kwargs_from_params_list(colmap_cameras[1]["model"], colmap_cameras[1]["params"])


def format_camera_params(colmap_cameras):
    """Format camera parameters from COLMAP camera dictionary."""
    camera_kwargs = get_camera_kwargs_from_colmap_json(colmap_cameras)
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
    model = kwargs.get('model')
    if model.lower() == 'opencv':
        return [kwargs['fx'], kwargs['fy'], kwargs['cx'], kwargs['cy'], kwargs['k1'], kwargs['k2'], kwargs['p1'],
                kwargs['p2']]
    if model.lower() == 'radial':
        return [kwargs['f'], kwargs['f'], kwargs['cx'], kwargs['cy'], kwargs['k1'], kwargs['k2'], 0., 0.]
    if model.lower() == 'simple_radial':
        return [kwargs['f'], kwargs['f'], kwargs['cx'], kwargs['cy'], kwargs['k'], 0., 0., 0.]
