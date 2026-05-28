# [![ROMI_logo](docs/assets/images/ROMI_logo_green_25.svg)](https://romi-project.eu) / Plant 3D Vision v0.13.1 (dev)

[![Licence](https://img.shields.io/github/license/romi/plant-3d-vision?color=lightgray)](https://www.gnu.org/licenses/lgpl-3.0.en.html)
[![Python Version](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fromi%2Fplant-3d-vision%2Frefs%2Fheads%2Fdev%2Fpyproject.toml&logo=python&logoColor=white)]()
[![PyPI - Version](https://img.shields.io/pypi/v/plant-3d-vision?logo=pypi&logoColor=white)](https://pypi.org/project/plant-3d-vision/)
[![Conda - Version](https://img.shields.io/conda/vn/romi-eu/plant-3d-vision?logo=anaconda&logoColor=white&label=romi-eu&color=%2344A833)](https://anaconda.org/romi-eu/plant-3d-vision)
[![GitHub branch check runs](https://img.shields.io/github/check-runs/romi/plant-3d-vision/dev)](https://github.com/romi/plant-3d-vision)


## Overview
This Python library is part of the ROMI European project.
It provides tools to **reconstruct digital twins of plants** from a set of RGB images acquired with the "Plant Imager".
It also provides tools to **quantify some traits of the plant's aerial architecture** like angles between successive organs and internodes length.

More comprehensive documentation about the "Plant Imager" project can be found [here](https://docs.romi-project.eu/plant_imager/).

**WARNING**: this is an ongoing development and changes will arise unannounced!

**Table of Contents**
- [Pre-requisites](#pre-requisites)
  - [Getting Started](#getting-started)
  - [Docker Engine](#docker-engine)
  - [NVIDIA drivers](#nvidia-drivers)
  - [NVIDIA Container Toolkit](#nvidia-container-toolkit)
- [Building a Docker image (recommended)](#build-a-docker-image-recommended)
  - [Build the image](#build-the-image)
  - [Test the image](#test-the-image)
- [Install from sources](#install-from-sources)
  - [Install requirements](#requirements)
  - [Install sources](#install-sources)
- [Usage](#usage)
  - [Docker container](#docker-container)
  - [Conda environment](#conda-environment)
- [Troubleshooting](#troubleshooting)


## Pre-requisites
You will need to install:

 * the **Docker Engine** (except if you plan to install COLMAP from sources, good luck with that!)
 * the appropriate **NVIDIA driver**
 * and the **NVIDIA Container Toolkit** to benefit from GPU-accelerated algorithms inside the docker container.


### Docker Engine
To install the **Docker Engine**, you can follow the official [instructions](https://docs.docker.com/engine/install/ubuntu/) or use the convenience script:
```shell
curl https://get.docker.com | sh \
  && sudo systemctl --now enable docker
```

You also have to follow the [Post-installation steps for Linux](https://docs.docker.com/engine/install/linux-postinstall/). 

### NVIDIA drivers
On Ubuntu, you can install the latest compatible **NVIDIA drivers** with:
```shell
sudo ubuntu-drivers autoinstall
```

### NVIDIA Container Toolkit
To install the **NVIDIA Container Toolkit**, follow the official [Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) from NVIDIA.


## Build a docker image (recommended)
To avoid making a big mess while installing source code, it may be easier to build a Docker image and use it to perform reconstruction and analysis tasks.

### Build the image
To build a Docker image, you have to:

1. clone the `plant-3d-vision` git repository
2. initialize and clone the submodules (`plantdb`, `romitask`, `romiseg`, `romicgal` & `dtw`)
3. use the convenience build scripts

This can be done as follows:
```bash
# Clone the plant-3d-vision repository from GitHub
git clone https://github.com/romi/plant-3d-vision.git

# Change directory to the cloned repository
cd plant-3d-vision/

# Initialize and update git submodules (containing additional dependencies)
git submodule init
git submodule update

# Build a Docker image for COLMAP version 3.8 using a convenience script in the docker/colmap3.8 directory
./docker/colmap3.8/build.sh -t colmap3.8

# Build the main Docker image with all dependencies, specifying the previously built COLMAP image to use
./docker/build.sh --colmap 3.8
```

This will create two Docker images:
  - `roboticsmicrofarms/colmap:3.8-cuda_ccXX`.
  - `roboticsmicrofarms/plant-3d-vision:colmap3.8-cuda_ccXX`.

If you want to tag your image with a specific one, here named `mytag`, use the `docker tag` command:
```bash
docker tag roboticsmicrofarms/plant-3d-vision:colmap3.8-cuda_ccXX roboticsmicrofarms/plant-3d-vision:mytag
```
To show more options, use the help option `-h` with the convenience build scripts.

**Important**:
You must run the convenience build scripts from the root of the `plant-3d-vision` repository as it will copy the files to the filesystem of the container.

### Test the image
In the `docker` folder, you will find a convenience script named `run.sh`.
You may want to test the built image by running a container and performing some tests.

Every test assumes you are in the `plant-3d-vision` root directory of the repository.

Remember to specify your tag with the `-t` option if you changed it (_i.e._ not `latest`)

#### Test GPU access
To test if you have access to your GPU(s) can easily be done as follows:
```shell
./docker/run.sh --test-gpu
```

#### Test the geometric pipeline
To test if you can run the _geometric pipeline_:
```shell
./docker/run.sh --test-geom-pipeline
```

#### Test the machine learning pipeline
To test if you can run the _machine learning pipeline_:
```shell
./docker/run.sh --test-ml-pipeline
```

## Shared Database
### Initialise and register a local `plantdb` database
Assuming you want to put your local `plantdb` database under `/Data/ROMI/DB`.

Let's start by setting an environment variable named `$ROMI_DB` to the end of our `.bashrc` file:
```shell
cat << EOF >> /home/$USER/.bashrc
# ROMI plant-3d-vision - Set the local plantdb database location:
export ROMI_DB='/data/ROMI/DB'
EOF
```

Then create the directories and the required `romidb` marker file with:
```shell
source ~/.bashrc  # to 'activate' the $ROMI_DB environment variable
mkdir -p $ROMI_DB
touch $ROMI_DB/romidb
```

In any case, please avoid doing horrendous things like `chmod -R 777 $ROMI_DB`!

#### Create a `romi` group and give it rights over the local `plantdb` database

1. Create the `romi` group
   ```shell
   sudo addgroup romi --gid 2020
   ```
2. Add the current user to the `romi` group
   ```shell
   sudo usermod -a -G romi $USER
   ```
3. Change the group of the local DB
   ```shell
   sudo chown -R :romi $ROMI_DB
   ```
4. Check the rights with:
   ```shell
   ls -al $ROMI_DB
   ```
   This should yield something like:
   ```
   drwxrwxr-x  2 myuser romi     4096 nov.  21 12:00 .
   drwxrwxrwx 31 myuser myuser   4096 nov.  21 12:00 ..
   -rw-rw-r--  1 myuser romi        0 nov.  21 12:00 romidb
   ```
   Where `myuser` is your username.

### Enable write access to the local database with bind mount
To avoid running the container app as `root` user, we created a non-root user named `romi` with an uid of `2020`.
In turn, when you bind mount a local `plantdb` database, you will not be able to write if:
  - the owner does not have an uid of `2020`
  - the group is not set to `romi` or the

In the `./docker/run.sh` convenience script, we added a few lines to automatically get the group id of the bind-mounted host directory acting as the database location.

To be a bit cleaner and go further in sharing the database with other users, we suggest to:

1. create a group named `romi`,
2. add all potential users of the docker image to this group
3. change the group of the local `plantdb` database to the `romi` group
4. start the docker container with the `--user romi:$romi_gid` option, where `$romi_gid` is the group id (gid) of the `romi` group

This will also allow all users from the `romi` group to access the files within the database, effectively making this a shared database.

### Setgid on Directories
A directory that has ‘setgid’ on it will cause all files that are created in that directory to be owned by the group of the directories as opposed to the group of the owners.
This proves useful when sharing a database between multiple users belonging to the same group.

```shell
chmod -R g+s /data/ROMI/
```


### Systemd Service

Let's create a systemd service file that will run your Docker command at system startup.

#### 1. Create a systemd service that calls the script

``` bash
sudo nano /etc/systemd/system/romi-plant-3d.service
```

Add the following content:
``` ini
[Unit]
Description=ROMI Plant 3D Vision Service
After=docker.service network.target
Requires=docker.service

[Service]
Type=simple
ExecStartPre=-docker stop %n
ExecStartPre=-docker rm %n
ExecStart=docker run --name %n --gpus all -v /data/ROMI/test_v2/:/myapp/db -v /data/ROMI/configs/:/myapp/cfg --user romi:1003 -p 8080:8080 -d roboticsmicrofarms/plant-3d-vision:latest-cuda_cc50 "gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:8080 plant3dvision.webterm.wsgi:application"
ExecStop=docker stop %n
ExecStopPost=docker rm %n
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

#### 2. Reload and restart the service
``` bash
sudo systemctl daemon-reload
sudo systemctl restart romi-plant-3d.service
```

#### 3. Check the status
``` bash
sudo systemctl status romi-plant-3d.service
docker ps | grep romi-plant-3d
```

You can have a look at the logs with:
```bash
sudo journalctl -u romi-plant-3d.service
```

#### 4. Enable the service
``` bash
sudo systemctl enable romi-plant-3d.service
```

#### Explanation of the Service File

- `Unit` Section: Defines dependencies, this service will start after Docker service is running
- `Service` Section:
    - `ExecStartPre`: Removes any existing container with the same name
    - `ExecStart`: Runs your Docker container
    - `ExecStop`: Stop the Docker container
    - `ExecStopPost`: Removes any existing container with the same name
    - `Restart`: Automatically restarts the service if it fails
- `Install` Section: Makes the service start at boot time 

#### Managing the Service
You can manage the service using these commands:

- Check status: `sudo systemctl status romi-plant-3d.service`
- View logs: `sudo journalctl -u romi-plant-3d.service`
- Stop service: `sudo systemctl stop romi-plant-3d.service`
- Restart service: `sudo systemctl restart romi-plant-3d.service`

## Install from sources

### System Requirements

Let's first install some useful tools like `git`, `curl`, `wget` & `nano`:
```shell
sudo apt update 
sudo apt install -y git curl wget nano
```

For `romicgal` you will need several system dependencies:
```shell
sudo apt update 
sudo apt install -y build-essential gcc libcgal-dev libeigen3-dev libgmp-dev libmpfr-dev
```


### Requirements

#### Conda
In the following instructions, we use `conda`, have a look at the official installation instructions [here](https://docs.conda.io/en/latest/miniconda.html) or use this convenience script:
```shell
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
Follow the instructions and proceed to remove the convenience script:
```shell
rm Miniconda3-latest-Linux-x86_64.sh
```

#### COLMAP
`COLMAP` is required to run the reconstruction tasks.

You have two options:

1. use the `roboticsmicrofarms/colmap:3.8` available from our [docker hub](https://hub.docker.com/repository/docker/roboticsmicrofarms/colmap) [recommended]
2. follow the official installation instructions to install COLMAP for linux [here](https://colmap.github.io/install.html#linux).


#### CUDA
To use the CUDA implementation of the _backprojection_ algorithm, you will need `nvidia-cuda-toolkit`:

On Debian and Ubuntu, you can install it with:
```shell
sudo apt update 
sudo apt install -y nvidia-cuda-toolkit
```

#### Python OpenCL
To use the OpenCL implementation of the _backprojection_ algorithm, you will need `ocl-icd-libopencl1` & `opencl-headers`.

On Debian and Ubuntu, you can install these dependencies with:
```bash
sudo apt-get update && sudo apt-get install -y ocl-icd-libopencl1 opencl-headers
```

Then we add the `PYOPENCL_CTX` environment variable to the end of our `.bashrc` file:
```shell
cat << EOF >> /home/$USER/.bashrc

# ROMI plant-3d-vision - Select the first available device for PyOpenCL:
export PYOPENCL_CTX='0'
EOF
```


### Install sources

1. Clone the `plant-3d-vision` sources:
    ```bash
    git clone https://github.com/romi/plant-3d-vision.git
    cd plant-3d-vision
    git submodule init
    git submodule update
    ```
2. Create a conda environment named `plant3dvision` with Python3.8, for example:
    ```bash
    conda create --name plant3dvision "python=3.9"
    ```
3. Install the submodules (`plantdb`, `romitask`, `romiseg`, `romicgal` & `dtw`) and `plant3dvision` in activated environment:
    ```bash
    conda activate plant3dvision
    python3 -m pip install -e ./plantdb/src/commons/.
    python3 -m pip install -e ./plantdb/src/client/.
    python3 -m pip install -e ./romitask/.
    python3 -m pip install -e ./skeleton_refinement/.
    python3 -m pip install -e ./romiseg/.
    conda install -c conda-forge libstdcxx-ng  # required to use `open3d.visualization.draw`
    python3 -m pip install -e ./romicgal/.
    python3 -m pip install -r ./dtw/requirements.txt
    python3 -m pip install -e ./dtw/.
    python3 -m pip install -e .
    ```
4. Test import of `plant3dvision` library:
    ```bash
    conda activate plant3dvision
    python3 -c 'import plant3dvision'
    ```
5. Longer tests using shipped "test dataset":
    ```bash
    bash check_pipe.sh
    rm testdata/models/models/Resnet_896_896_epoch50.pt
    ```


## Usage
This package is built around `luigi` and adopt a similar **pipeline oriented** philosophy with `Tasks` and `Parameters`.

To reconstruct and analyze the RGB images acquired with the _Plant Imager_ you will have to define a pipeline as a series of tasks and set their parameters.
To simplify things, we provide two TOML configuration files defining the two main types of pipeline we use:

* the _geometric pipeline_ in `plant-3d-vision/config/geom_pipe_real.toml`
* the _machine learning pipeline_ in `plant-3d-vision/config/ml_pipe_real.toml.toml`

These configuration files should be used on "real plant" dataset, _i.e._ on RGB images acquired with the _Plant Imager_.

To start a task defined in a configuration file, use the `romi_run_task` CLI.
You will have to specify which task you want to perform, on which dataset and pass the path to the configuration file using the `--config` option.
You will find examples of this using either the Docker container or the sources installed in a conda environment.

For more details about the task and their parameters, have a look at the "Plant Imager" documentation [here](https://docs.romi-project.eu/plant_imager/specifications/tasks/reconstruction_tasks/).


### Docker container
There are now two options to use the previously built Docker image:

1. you have experience with the Docker CLI and are willing to use it
2. you prefer to use a convenience script with fewer but safer options

#### Docker CLI
You can use the Docker CLI to start a container using the previously built `roboticsmicrofarms/plant-3d-vision` image.

In the following example, we will use the `real_plant` dataset from the **test database** and the **geometric pipeline** configuration file shipped in this repository.
Assuming you are in the `plant-3d-vision` root directory of the repository:
```bash
CWD=$(pwd)  # get the absolute path to the `plant-3d-vision` directory
docker run -it --rm --gpus all \
  -v $CWD/tests/testdata/:/myapp/db \
  -v $CWD/configs/:/myapp/configs \
  roboticsmicrofarms/plant-3d-vision:latest \
  bash -c "romi_run_task AnglesAndInternodes /myapp/db/real_plant/ --config /myapp/config/geom_pipe_real.toml"
```

In details, the previous command:

* get the current working directory and assign it to `CWD` variable as docker needs **absolute path** when performing bind mount
* starts a pseudo-TTY interactive shell with `-it`
* will automatically remove the container when it exits with `--rm`
* add all available GPUs to the container
* bind mount `$CWD/tests/testdata/` from the host to `/myapp/db` in the container (created if non-existent) with the `-v` option
* bind mount `$CWD/config/` from the host to `/myapp/config` in the container (created if non-existent) with the `-v` option
* use the `roboticsmicrofarms/plant-3d-vision` image with the tag `latest` to create the container
* call a `bash` command with `bash -c "..."`
* performs the `AnglesAndInternodes` task (reconstruction and analysis) of the `real_plant` test dataset using the `geom_pipe_real.toml` configuration


#### Convenience bash script
There is a convenience bash script, named `run.sh` in the `docker/` directory, that start a docker container using the `roboticsmicrofarms/plant-3d-vision` image.
It aims at making things a bit simpler than with the Docker CLI.

In the following example, we will use the `real_plant` dataset from the **test database** and the **geometric pipeline** configuration file shipped in this repository.
Assuming you are in the `plant-3d-vision` root directory of the repository:
```bash
CWD=$(pwd)  # get the absolute path to the `plant-3d-vision` directory
./docker/run.sh \
  -db $CWD/tests/testdata/ \
  -v $CWD/configs/:/myapp/configs \
  -c "romi_run_task AnglesAndInternodes /myapp/db/real_plant/ --config /myapp/config/geom_pipe_real.toml"
```

In detail, the previous command does (the same thing as the CLI example):

* get the current working directory and assign it to `CWD` variable as docker needs **absolute path** when performing bind mount
* bind mount `$CWD/tests/testdata/` from the host to `/myapp/db` in the container (created if non-existent) with the `-db` option
* bind mount `$CWD/config/` from the host to `/myapp/config` in the container (created if non-existent) with the `-v` option
* call a `bash` command with `-c "..."`
* performs the `AnglesAndInternodes` task (reconstruction and analysis) of the `real_plant` test dataset using the `geom_pipe_real.toml` configuration

Note that:

* with the `-db` option you do not have to specify the destination in the container
* you do not have to specify the `bash` before the `-c` option


### Conda environment
In the `plant3dvision` conda environment, things are a bit simpler.

To execute the same series of tasks on the `real_plant` dataset from the **test database** and the **geometric pipeline** configuration file shipped in this repository we only have to call the `romi_run_task` CLI.
Assuming you are in the `plant-3d-vision` root directory of the repository:
```shell
cp -R tests/testdata /tmp/.  # copy the test DB to the temporary directory
conda activate plant3dvision  # activate the conda environment
romi_run_task AnglesAndInternodes /tmp/testdata/real_plant/ --config configs/test_geom_pipe_real.toml
```

This will perform the `AnglesAndInternodes` task (reconstruction and analysis) of the `real_plant` test dataset using the `geom_pipe_real.toml` configuration.


### Monitoring

#### CPU
To monitor the CPU resource usage, you can use the `htop` tool.

To install it from Ubuntu PPA repositories:
```shell
sudo apt install htop
```

To install it from the snap store:
```shell
snap install htop
```

#### GPU
To monitor the GPU resource usage, you can use the following command:
```shell
watch -n1 nvidia-smi
```
where the `-n` option is the time interval in seconds.


## Troubleshooting

### OpenCL
- `ImportError: libGL.so.1: cannot open shared object file: No such file or directory` can be fixed with:
    ```bash
    apt-get install libgl1-mesa-glx
    ```
- `ImportError: libSM.so.6: cannot open shared object file: No such file or directory` can be fixed with:
    ```bash
    apt-get install libsm6 libxext6 libxrender-dev
    ```


If you have an NVIDIA GPU, you may solve some issues with:
```bash
mkdir -p /etc/OpenCL/vendors
echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd
```

To avoid troubles during `pyopencl` install, check `/usr/lib/libOpenCL.so` exists, if not add it with:
```bash
ln -s /usr/lib/x86_64-linux-gnu/libOpenCL.so.1 /usr/lib/libOpenCL.so
```
