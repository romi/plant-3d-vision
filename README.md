# Plant 3D Vision v0.10.99 (dev)

Documentation about the "Plant Scanner" project can be found [here](https://docs.romi-project.eu/Scanner/).

# Table of Contents
- [Building and running with Docker (recommended)](#Docker)
    * [Building the image](#Building-the-image)
    * [Running the container](#Running-the-container)
    * [Performing a task with `plant-3d-vision`](#Performing-a-task-with-plant-3d-vision)
- [Install requirements for Ubuntu and Conda Environment](#Install-requirements)
- [Conda Environment](#Conda-environment)
    * [Install from sources in conda environment](#Install-from-sources-in-conda-environment)
    * [Build `plant_3d_vision` conda package](#Build-plant-3d-vision-conda-package)
    * [Install `plant_3d_vision` conda package](#Install-plant-3d-vision-conda-package)


## Docker

### Building the image
To build the Docker image of `plant-3d-vision`, you have to clone the git repository and run the script `docker/build.sh`.

```bash
git clone https://github.com/romi/plant-3d-vision.git
cd plant-3d-vision/
git submodule init  # initialize the git submodules
git submodule update  # clone the git submodules
./docker/build.sh
```
This will create a docker image named `roboticsmicrofarms/plant-3d-vision:latest`.
If you want to tag your image with a specific one, just pass the tag argument as follows
`./docker/build.sh -t mytag`

To show more options (built-in user...), just type `./docker/build.sh -h`.
Note that you must run the `build.sh` script from the root of the `plant-3d-vision` folder.

### Running the container
In the docker folder, you will find also a run script `docker/run.sh`.
You may want to mount your database folder and another folder (let's say your configs).
This can be done as follows:
```bash
./docker/run.sh \
    -v /home/${USER}/my_database:/home/${USER}/database/ \
    -v /home/${USER}/my_configs/:/home/${USER}/config/
```
Don't forget to change the paths with yours!

If you want to run a docker image with another tag, you can pass the tag name as an argument:
`./docker/run.sh -t my_tage`.

To see more running options (specif tag, command...), type `./docker/run.sh -h`

### Troubleshooting:
You must install nvidia gpu drivers, `nvidia-docker` (v2.0) and `nvidia-container-toolkit`. 
To test if everything is okay:
```bash
./docker/run.sh --gpu_test
```

This docker image has been tested successfully on:
`docker --version=19.03.6 | nvidia driver version=450.102.04 | CUDA version=11.0`

### Performing a task with plant-3d-vision
Inside the docker image there is a `romi_run_task` command which performs a task on a database according to a passed config file.

In this following example, we will use the test database and config file shipped in this repository:
 - Run the default docker image (`roboticsmicrofarms/plant-3d-vision:latest`)
 - Mount the database (`plant-3d-vision/tests/testdata/`) and configs folder (`plant-3d-vision/config/`) inside the docker container
 - Perform the task `AnglesAndInternodes` on the database with `geom_pipe_real.toml` config file

```bash
./docker/run.sh -v /path/to/plant-3d-vision/tests/testdata/:/home/$USER/database/ -v /path/to/plant-3d-vision/config/:/home/$USER/config
romi_run_task --config ~/config/geom_pipe_real.toml AnglesAndInternodes ~/database/real_plant/
```

Don't forget to replace the paths `path/to/plant-3d-vision` by the correct ones.

## Install from sources

### Install requirements
`Colmap` is required to run the reconstruction tasks, follow the official install instructions for linux [here](https://colmap.github.io/install.html#linux).

This library use `pyopencl` and thus require the following system libraries:

- `ocl-icd-libopencl1`
- `opencl-headers`

In addition, you will need:

- `git`
- `python3-pip`
- `python3-wheel`
- `miniconda`, official installation instructions [here](https://docs.conda.io/en/latest/miniconda.html)
- `poetry`, official installation instructions [here](https://python-poetry.org/docs/#installation)

As you will be using the `romicgal` library, which is a minimal wrapper for `CGAL5.0` using `pybind11` you will also need:
- `wget`
- `eigen3`
- `gmp`
- `mprf`
- `pkg-config`

On Debian and Ubuntu, you can install them (except `miniconda` and `poetry`) with:
```bash
sudo apt-get update && sudo apt-get install -y \
    git wget \
    ocl-icd-libopencl1 opencl-headers \
    python3-wheel python3-pip \
    gcc pkg-config \
    libeigen3-dev libgmp3-dev libmpfr-dev
```

If you have an NVIDIA GPU:
```bash
mkdir -p /etc/OpenCL/vendors
echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd
```

To avoid troubles during `pyopencl` install, check `/usr/lib/libOpenCL.so` exists, if not add it with:
```bash
ln -s /usr/lib/x86_64-linux-gnu/libOpenCL.so.1 /usr/lib/libOpenCL.so
```

### Install with `conda` & `poetry`

In this install instructions, we use `conda` and `poetry`.
If you do not have those tools installed, have a look at the official instructions:
 - `miniconda`: https://docs.conda.io/en/latest/miniconda.html
 - `poetry`: https://python-poetry.org/docs/#installation

1. Clone the `plant-3d-vision` sources:
    ```bash
    git clone https://github.com/romi/plant-3d-vision.git
    ```
2. Create a conda environment named `plant_3d_vision_0.10` with Python3.8 for example:
    ```bash
    conda create --name plant_3d_vision_0.10 python=3.8
    ```
3. Install sources and submodules in activated environment:
    ```bash
    conda activate plant_3d_vision_0.10
    cd plant-3d-vision/
    poetry install --no-dev
    ```
4. Test import of `plant3dvision` library:
    ```bash
    conda activate plant_3d_vision_0.10
    python3 -c 'import plant3dvision'
    ```
5. Longer tests using shipped "test dataset":
    ```bash
    cd tests/
    bash check_pipe.sh
    rm testdata/models/models/Resnet_896_896_epoch50.pt
    ```

### Troubleshooting:

- `ImportError: libGL.so.1: cannot open shared object file: No such file or directory` can be fixed with:
    ```bash
    apt-get install libgl1-mesa-glx
    ```
- `ImportError: libSM.so.6: cannot open shared object file: No such file or directory` can be fixed with:
    ```bash
    apt-get install libsm6 libxext6 libxrender-dev
    ```
