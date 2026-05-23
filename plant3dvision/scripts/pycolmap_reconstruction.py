"""
pycolmap_reconstruction.py

Script-oriented module for running a COLMAP/pycolmap-based reconstruction pipeline, with helper
functions that prepare the runtime environment.

This file is intended to be executed as a script (e.g., inside a Docker container), while also
exposing utilities that make that execution reproducible from a host Python process.

Highlights
----------
- **Script entrypoint**: when executed directly, it expects a prepared working directory
  (typically a bind-mounted path in a container) and runs the reconstruction pipeline end-to-end.
- **Environment preparation helpers**: functions are provided to create the expected directory
  layout, stage/rename input images, write parameter/config files, and ensure auxiliary assets
  are present for the run.
- **Docker-oriented workflow**: helper functionality supports running this script *in a Docker
  container* by preparing a work directory, bind-mounting it into the container, and collecting
  results after the containerized run completes.

Typical usage patterns
----------------------
1. **Host prepares + container runs**:
   - Call the preparation utilities on the host to stage inputs in a temporary work directory.
   - Run this script within the Docker image against the bind-mounted work directory.
   - Read the produced results file from the work directory.

2. **Direct execution in a prepared environment**:
   - Ensure the expected working directory structure and parameter file exist.
   - Execute this module as a script to perform reconstruction and write outputs.

Notes
-----
- The helper functions in this module are intentionally focused on “make the environment runnable”
  and “run the script in Docker” tasks, so callers can integrate reconstruction into larger systems
  without duplicating setup logic.
"""

import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
from os.path import join
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pycolmap
import requests

from romitask.log import get_logger

logger = get_logger(__name__)

COLMAP_IMG = os.getenv("P3DV_COLMAP_IMG", "roboticsmicrofarms/colmap:3.13.0-cuda_cc89")

DATABASE_PATH = "database.sqlite"
IMAGE_PATH = "images"
PARAM_PATH = "colmap_params.json"
RESULTS_PATH = "results.json"
DOCKER_BIND_PREFIX = os.getenv("P3DV_COLMAP_WORKDIR", "/workdir")
RECONSTRUCTION_PATH = "sparse"
VOCAB_TREE_URL = "https://github.com/colmap/colmap/releases/download/3.11.1/vocab_tree_faiss_flickr100K_words32K.bin"
VOCAB_TREE_PATH = "vocab_tree_faiss_flickr100K_words32K.bin"

IMAGE_PATTERN = r"(.+)-([0-9]{5})\.(jpe?g)"
image_regex = re.compile(IMAGE_PATTERN)

PRIOR_COV = np.array([
    [20, 0, 0],
    [0, 20, 0],
    [0, 0, 20]
]) ** 2  # square of std-dev


@contextlib.contextmanager
def cd(x: str | os.PathLike[str]):
    """
    Context manager for changing the current working directory temporarily.

    This context manager allows temporarily changing the current working directory
    to a specified directory. Once the block inside the context manager is exited,
    whether normally or due to an exception, the working directory reverts to its
    original value.

    Parameters
    ----------
    x : str or os.PathLike
        The target directory to change to temporarily.
    Examples
    -------
    >>> print(os.getcwd())  # --> /original/path
    >>> with cd("/some/path"):
    >>>     print(os.getcwd())  # --> /some/path
    >>> print(os.getcwd())  # --> /original/path
    """
    d = os.getcwd()
    os.chdir(x)
    try:
        yield
    finally:
        os.chdir(d)


def _has_nvidia_gpu():
    """Returns ``True`` if an NVIDIA GPU is reachable, else ``False``."""
    try:
        out = subprocess.run('nvidia-smi', capture_output=True)
    except FileNotFoundError:
        logger.warning("nvidia-smi is not installed on your system!")
        return False
    else:
        # `nvidia-smi` utility might be installed but GPU or driver unreachable!
        if 'failed' in out.stdout.decode() or 'not found' in out.stdout.decode():
            return False
        else:
            return True


def copy_module_to_path(destination_path: str | os.PathLike[str]) -> None:
    """
    Copies the current module to a destination path.
    Works with both regular and compressed packages.
    """
    from importlib.resources import files

    module_name = __name__
    module_parts = module_name.split('.')

    # Determine package and module name
    if len(module_parts) > 1:
        package_name = '.'.join(module_parts[:-1])
        resource_name = f"{module_parts[-1]}.py"

        # Read from package using importlib.resources
        content = files(package_name).joinpath(resource_name).read_text(encoding='utf-8')
    else:
        # Top-level module
        module = sys.modules[module_name]
        with open(module.__file__, 'r', encoding='utf-8') as f:
            content = f.read()

    # Write to destination
    destination = Path(destination_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with open(destination, 'w', encoding='utf-8') as f:
        f.write(content)

    logger.debug(f"Module content copied to {destination}")


def run(config: dict, image_files: list[str], pose: dict[str, tuple[float, float, float]], colmap_iamge=COLMAP_IMG):
    """
    Runs a COLMAP reconstruction process using Docker for image-based 3D reconstruction.

    This function sets up a temporary directory to prepare the input data (images and poses),
    configures Docker to set up the necessary environment variables and volume binds, and
    executes the reconstruction pipeline. It handles compatibility with both GPU and non-GPU
    environments. The resulting output is a structured JSON file containing the reconstruction
    results.

    Parameters
    ----------
    config : dict
        Dictionary with configuration parameters for the reconstruction process.
    image_files : list[str]
        List of file paths to the images to be used in the reconstruction process.
    pose : dict[str, tuple[float, float, float]]
        Dictionary mapping image file names to their corresponding camera poses, with each
        pose represented as a tuple of 3 float values.
    colmap_iamge : str, optional
        The name of the Docker image containing the COLMAP pipeline. Defaults to `COLMAP_IMG`.

    Returns
    -------
    dict
        JSON object containing the reconstruction results.
    """
    import docker

    with TemporaryDirectory(prefix="COLMAP_") as tmpdir:
        prepare(config, tmpdir, image_files, pose)

        client = docker.from_env()
        # Defines environment variables:
        varenv = {}
        varenv.update({'PYOPENCL_CTX': os.environ.get('PYOPENCL_CTX', '0')})

        # Get the GID and UID from the workdir
        workdir_stat = os.stat(tmpdir)
        workdir_gid = str(workdir_stat.st_gid)
        workdir_uid = str(workdir_stat.st_uid)
        # Volume to bind mount
        volumes = {
            DOCKER_BIND_PREFIX: {
                'bind': tmpdir,
                'mode': 'rw',
                'uid': workdir_uid,
                'gid': workdir_gid
            }
        }
        cmd = ["python3", "pycolmap_reconstruction.py"]
        client.containers.prune()
        if _has_nvidia_gpu():
            gpu_device = docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])
            container = client.containers.run(colmap_iamge, cmd,
                                              user=workdir_uid,
                                              # group_add=["colmap_users"],
                                              environment=varenv, volumes=volumes,
                                              stdout=True, stderr=True,
                                              stream=True, detach=True,
                                              device_requests=[gpu_device], working_dir=DOCKER_BIND_PREFIX)
        else:
            container = client.containers.run(colmap_iamge, cmd,
                                              user=workdir_uid,
                                              # group_add=["colmap_users"],
                                              environment=varenv, volumes=volumes,
                                              stdout=True, stderr=True,
                                              stream=True, detach=True, working_dir=DOCKER_BIND_PREFIX)
        for line in container.logs(stream=True, follow=True):
            line = line.decode("utf-8")
            sys.stdout.write(line)
            sys.stdout.flush()
        container.wait()

        with open(join(tmpdir, RESULTS_PATH)) as f:
            results = json.load(f)

    return results


def prepare(
        config: dict, mount_path: str, image_files: list[str], pose: dict[str, tuple[float, float, float]]
) -> dict[str, str]:
    """
    Prepares the necessary directory structure, configuration, and image files for further processing.

    This function performs the following tasks:
    1. Copies a predefined script to the target directory.
    2. Downloads a specific vocab tree file and saves it in the specified mount path.
    3. Validates and creates required directories for images and reconstruction.
    4. Organizes and renames image files based on their camera source.
    5. Updates the configuration dictionary with new image names and corresponding pose priors.
    6. Saves the updated configuration to a parameter file within the directory.

    Parameters
    ----------
    config : dict
        A dictionary containing configuration data to be updated.
    mount_path : str
        The base directory where preparation will take place. It must already exist.
    image_files : list of str
        A list of paths to image files that need to be processed and organized.
    pose : dict of {str: tuple of (float, float, float)}
        A dictionary mapping original image file names to their corresponding pose priors.

    Returns
    -------
    dict of {str: str}
        A dictionary mapping original image file names to their new names based on the preparation process.
    """

    # copy script
    copy_module_to_path(join(mount_path, "script.py"))
    #
    response = requests.get(
        "https://github.com/colmap/colmap/releases/download/3.11.1/vocab_tree_faiss_flickr100K_words32K.bin")
    with open(join(mount_path, "vocab_tree_faiss_flickr100K_words32K.bin"), "wb") as f:
        f.write(response.content)

    base_dir = Path(mount_path)
    assert base_dir.exists(), "mount_path is not a path to an existing directory"

    image_dir = base_dir / Path(IMAGE_PATH)
    image_dir.mkdir(parents=True, exist_ok=True)
    camera_names = list({
        image_regex.match(os.path.basename(fname)).group(1)
        for fname in image_files
    })
    for cam_name in camera_names:
        cam_dir = image_dir / cam_name
        cam_dir.mkdir(parents=True, exist_ok=True)
    # generating names for images
    image_counters = dict.fromkeys(camera_names, 0)
    image_names = {}  # original name -> new name
    for path in sorted(image_files, key=lambda p: os.path.basename(p)):
        name = os.path.basename(path)
        match = image_regex.match(name)
        camera = match.group(1)
        extension = match.group(3)
        counter = image_counters[camera]
        if match:
            new_name = f"{camera}/image{counter:0>5}.{extension}"
            image_counters[camera] += 1
            image_names[name] = new_name
            shutil.copy(path, image_dir / new_name)
        else:
            raise ValueError(f"Image file name {name} does not match the expected pattern {IMAGE_PATTERN}")

    base_dir.joinpath(RECONSTRUCTION_PATH).mkdir(parents=True, exist_ok=True)

    new_names = list(image_names.values())
    config.update({
        "image_names": new_names,
        "pose_priors": {image_names[name]: pose[name] for name in image_names.keys()},
    })
    with open(base_dir / PARAM_PATH, "w") as f:
        json.dump(config, f, indent=4, sort_keys=True)

    return image_names


def read_params(path: os.PathLike) -> dict:
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":

    with cd(DOCKER_BIND_PREFIX):
        params = read_params(PARAM_PATH)

        image_names = params["image_names"]

        image_reader_options = pycolmap.ImageReaderOptions()
        image_reader_options.camera_model = "OPENCV"
        extraction_options = pycolmap.SiftExtractionOptions(**params["feature_extraction"])
        extraction_options.use_gpu = True
        pycolmap.extract_features(
            DATABASE_PATH,
            IMAGE_PATH,
            image_names,
            pycolmap.CameraMode.PER_FOLDER,
            camera_model=pycolmap.CameraModelId.OPENCV.name,
            reader_options=image_reader_options,
            sift_options=extraction_options,
        )

        db: pycolmap.Database = pycolmap.Database(DATABASE_PATH)
        # prior poses
        db.clear_pose_priors()
        for image_name, pose in params["pose_priors"].items():
            img: pycolmap.Image = db.read_image_with_name(image_name)
            prior = pycolmap.PosePrior(pose, PRIOR_COV, pycolmap.PosePriorCoordinateSystem.CARTESIAN)
            if db.exists_pose_prior(img.image_id):
                db.update_pose_prior(img.image_id, prior)
            else:
                db.write_pose_prior(img.image_id, prior)


        if params["matcher"] == "SequentialMatcher":
            matching_options = pycolmap.SiftMatchingOptions()
            matching_options.use_gpu = True
            matching_options.mergedict(params["matcher_options"])
            pairing_options = pycolmap.SequentialMatchingOptions()
            pairing_options.loop_detection = True
            pairing_options.vocab_tree_path = VOCAB_TREE_PATH
            pycolmap.match_sequential(DATABASE_PATH, matching_options, pairing_options)
        elif params["matcher"] == "ExhaustiveMatcher":
            matching_options = pycolmap.FeatureMatchingOptions()
            matching_options.use_gpu = True
            matching_options.mergedict(params["matcher_options"])
            pycolmap.match_exhaustive(DATABASE_PATH, matching_options)

        # reconstruction
        pipeline_options = pycolmap.IncrementalPipelineOptions()
        pipeline_options.ba_use_gpu = True
        pipeline_options.use_prior_position = True
        pipeline_options.ba_refine_principal_point = True
        pipeline_options.ba_refine_focal_length = True
        pipeline_options.ba_refine_extra_params = True
        reconstructions = pycolmap.incremental_mapping(DATABASE_PATH, IMAGE_PATH, RECONSTRUCTION_PATH, pipeline_options)

        #
        for i, recons in reconstructions.items():
            recons.write_text(join(RECONSTRUCTION_PATH, str(i)))
        reconstruction = reconstructions[0]

        i_names, points = tuple(zip(*params["pose_priors"].items()))
        print(points)
        alignment: pycolmap.Sim3d = pycolmap.align_reconstruction_to_locations(
            reconstruction,
            i_names,
            np.array(points),
            10,
            pycolmap.RANSACOptions()
        )
        if alignment:
            print(alignment.summary())
        else:
            print("Alignment FAILED !")


        results = {
            "image_names": [],
            "positions": {},
            "rotations": {},
            "viewing_direction": {},
        }
        for image in reconstruction.images.values():
            image: pycolmap.Image
            pose = image.cam_from_world()

            translation: np.ndarray = image.projection_center()
            rotation: pycolmap.Rotation3d = pose.rotation
            results["image_names"].append(image.name)
            results["positions"][image.name] = translation.tolist()
            results["rotations"][image.name] = rotation.quat.tolist()
            results["viewing_direction"][image.name] = image.viewing_direction().tolist()

        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=4, sort_keys=True)
