# Plant 3D Vision v0.11.99 (dev)

This Python library is part of the ROMI European project.
It provides tools to **reconstruct digital twins of plants** from a set of RGB images acquired with the "Plant Imager".
It also provides tools to **quantify some traits of the plant's aerial architecture** like angles between successive organs and inter-nodes length.

A more comprehensive documentation about the "Plant Imager" project can be found [here](https://docs.romi-project.eu/plant_imager/).

**WARNING**: this is an ongoing development and changes will arise unannounced!

**Table of Contents**
- [Building and running with Docker (recommended)](#building-and-running-with-docker-recommended)
    * [Building the image](#building-the-image)
    * [Running the container](#running-the-container)
    * [Performing a task with `romi_run_task`](#performing-a-task-with-romi_run_task)
- [Install from sources](#install-from-sources)
    * [Install requirements](#install-requirements)
    * [Install sources](#install-sources)
    * [Troubleshooting](#troubleshooting)


## Building and running with Docker (recommended)

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

### Troubleshooting
You must install nvidia gpu drivers, `nvidia-docker` (v2.0) and `nvidia-container-toolkit`. 
To test if everything is okay:
```bash
./docker/run.sh --gpu_test
```

This docker image has been tested successfully on:
`docker --version=19.03.6 | nvidia driver version=450.102.04 | CUDA version=11.0`

### Performing a task with `romi_run_task`
Inside the docker image there is a `romi_run_task` command which performs a task on a database according to a config file.

In the following example, we will use the test database and config file shipped in this repository:
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

#### Colmap
`Colmap` is required to run the reconstruction tasks.

You have two options:

1. use the `roboticsmicrofarms/colmap:3.7` available from our [docker hub](https://hub.docker.com/repository/docker/roboticsmicrofarms/colmap) [recommended]
2. follow the official install instructions to install COLMAP for linux [here](https://colmap.github.io/install.html#linux).

#### Python OpenCL
This library use `pyopencl` and thus require the following system libraries:

- `ocl-icd-libopencl1`
- `opencl-headers`

In addition, you will need:

- `git`
- `python3-pip`
- `python3-wheel`

On Debian and Ubuntu, you can install all these dependencies with:
```bash
sudo apt-get update && sudo apt-get install -y \
    ocl-icd-libopencl1 opencl-headers \
    git python3-wheel python3-pip
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

#### CGAL5.0

As you will be using the `romicgal` library, which is a minimal wrapper for `CGAL5.0` using `pybind11` you will also need:
- `wget`
- `eigen3`
- `gmp`
- `mprf`
- `pkg-config`

On Debian and Ubuntu, you can install all these dependencies with:
```bash
sudo apt-get update && sudo apt-get install -y \
    wget \
    gcc pkg-config \
    libeigen3-dev libgmp3-dev libmpfr-dev
```


### Install sources

In this install instructions, we use `conda`, have a look at the official installation instructions [here](https://docs.conda.io/en/latest/miniconda.html).

1. Clone the `plant-3d-vision` sources:
    ```bash
    git clone https://github.com/romi/plant-3d-vision.git
    ```
2. Create a conda environment named `plant3dvision` with Python3.8 for example:
    ```bash
    conda create --name plant3dvision "python=3.8"
    ```
3. Install the submodules (`plantdb`, `romitask`, `romiseg`, `romicgal` & `dtw`) and `plant3dvision` in activated environment:
    ```bash
    conda activate plant3dvision
    cd plant-3d-vision/
    git submodule init
    git submodule update
    python3 -m pip install -r ./plantdb/requirements.txt
    python3 -m pip install -e ./plantdb/.
    python3 -m pip install -r ./romitask/requirements.txt
    python3 -m pip install -e ./romitask/.
    python3 -m pip install -e ./romiseg/.
    python3 -m pip install -e ./romicgal/.
    python3 -m pip install -r ./dtw/requirements.txt
    python3 -m pip install -e ./dtw/.
    python3 -m pip install -r requirements.txt
    python3 -m pip install -e .
    ```
4. Test import of `plant3dvision` library:
    ```bash
    conda activate plant3dvision
    python3 -c 'import plant3dvision'
    ```
5. Longer tests using shipped "test dataset":
    ```bash
    cd tests/
    bash check_pipe.sh
    rm testdata/models/models/Resnet_896_896_epoch50.pt
    ```

### Troubleshooting

- `ImportError: libGL.so.1: cannot open shared object file: No such file or directory` can be fixed with:
    ```bash
    apt-get install libgl1-mesa-glx
    ```
- `ImportError: libSM.so.6: cannot open shared object file: No such file or directory` can be fixed with:
    ```bash
    apt-get install libsm6 libxext6 libxrender-dev
    ```
