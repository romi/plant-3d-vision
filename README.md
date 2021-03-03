# romiscan v0.9 (dev)

Documentation about the "Plant Scanner" project can be found [here](https://docs.romi-project.eu/Scanner/).

# Table of Contents
- [Building and running with Docker (recommended)](#Docker)
    * [Building the image](#Building-the-image)
    * [Running the container](#Running-the-container)
    * [Performing a task with `romiscan`](#Performing-a-task-with-romiscan)
- [Install requirements for Ubuntu and Conda Environment](#Install-requirements)
- [Conda Environment](#Conda-environment)
    * [Install from sources in conda environment](#Install-from-sources-in-conda-environment)
    * [Build `romiscan` conda package](#Build-romiscan-conda-package)
    * [Install `romiscan` conda package](#Install-romiscan-conda-package)

## Docker

### Building the image
To build the Docker image of romiscan, you have to clone the repo and run the script `docker/build.sh`.

```bash

git clone https://github.com/romi/romiscan.git
cd romiscan/
git submodule init
git submodule update
./docker/build.sh
```
This will create an image docker `romiscan:latest`. If you want to tag your image with a specific one, just pass the tag argument as follows
`./docker/build.sh -t mytag`

To show more options (built-in user...), just type `./docker/build.sh -h`.
Note that, you must run the script from the root of `romiscan` folder as shown previously.


### Running the container
In the docker folder, you will find also a run script `docker/run.sh`.
You may want to mount your database folder and an other folder (let's say your configs).
This can be done as follows:
```bash
./docker/run.sh \
    -v /home/${USER}/my_database:/home/${USER}/database/ \
    -v /home/${USER}/my_configs/:/home/${USER}/config/
```
Don't forget to change the paths with yours!

If you want to run a docker image with an other tag, you can pass the tag name as an argument:
`./docker/run.sh -t my_tage`.

To see more running options (specif tag, command...), type `./docker/run.sh -h`

**Troubleshooting**:

- You must install nvidia gpu drivers, nvidia-docker (v2.0) and nvidia-container-toolkit. To test if everything is okay:

```bash
./docker/run.sh --gpu_test
```

This docker image has been tested successfully on:
`docker --version=19.03.6 | nvidia driver version=450.102.04 | CUDA version=11.0`

### Performing a task with romiscan
Inside the docker image there is a `romi_run_task` command which performs a task on a database according to a passed config file.

In this following example, we will use the test database and config file shipped in this repo:
 - Run the default docker image (`romiscan:latest`)
 - Mount the database (`romiscan/tests/testdata/`) and configs folder (romiscan/config/) inside the docker container
 - Perform the task `AnglesAndInternodes` on the database with `geom_pipe_real.toml` config file

```bash
./docker/run.sh -v /path/to/romiscan/tests/testdata/:/home/$USER/database/ -v /path/to/romiscan/config/:/home/$USER/config
romi_run_task --config ~/config/geom_pipe_real.toml AnglesAndInternodes ~/database/real_plant/
```

Don't forget to replace the paths `path/to/romican` by the correct ones.

## Install requirements
Colmap is required to run the reconstruction tasks, follow the official install instructions for linux [here](https://colmap.github.io/install.html#linux).

This library use `pyopencl` and thus require the following system libraries:

- ocl-icd-libopencl1
- opencl-headers

In addition you will need:

- git
- python3-pip
- python3-wheel

As you will be using the `romicgal` library, which is a minimal wrapper for CGAL 5.0 using `pybind11` you will also need:
- wget
- eigen3
- gmp
- mprf
- pkg-config

On Debian and Ubuntu, you can install them with:
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

## Conda environment

## Install from sources in conda environment
In this install instructions, we leverage the `git submodule` functionality to clone the required ROMI libraries.

1. Clone the `romiscan` sources:
    ```bash
    git clone https://github.com/romi/romiscan.git
    ```
2. Create a conda environment named `scan_0.8` with Python3.7:
    ```bash
    conda create --name scan_0.9 python=3.7
    ```
3. Install sources and submodules in activated environment:
    ```bash
    conda activate scan_0.9
    cd romiscan
    git submodule init
    git submodule update
    python3 -m pip install -r requirements.txt
    python3 -m pip install ./romidata/
    python3 -m pip install ./romiseg/
    python3 -m pip install ./romiscanner/
    python3 -m pip install ./romicgal/
    python3 -m pip install .
    ```
4. Test import of `romiscan` library:
    ```bash
    conda activate scan_0.8
    python3 -c 'import romiscan'
    ```
5. Longer tests using shipped "test dataset":
    ```bash
    cd tests/
    bash check_pipe.sh
    rm testdata/models/models/Resnetdataset_gl_png_896_896_epoch50.pt
    rm testdata/models/models/tmp_epoch40.pt
    ```

**Troubleshooting**:

- `ImportError: libGL.so.1: cannot open shared object file: No such file or directory` can be fixed with:
    ```bash
    apt-get install libgl1-mesa-glx
    ```
- `ImportError: libSM.so.6: cannot open shared object file: No such file or directory` can be fixed with:
    ```bash
    apt-get install libsm6 libxext6 libxrender-dev
    ```

### Build romiscan conda package
From the `base` conda environment, run:
```bash
conda build conda_recipes/romiscan/ -c romi-eu -c open3d-admin -c conda-forge --user romi-eu
```


### Install romiscan conda package
```bash
conda create -n romiscan romiscan -c romi-eu -c open3d-admin --force
```
To test package install, in the activated environment import `romiscan` in python:
```bash
conda activate romiscan
python -c 'import romiscan'
```
