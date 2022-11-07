# Plant 3D Vision v0.11.99 (dev)

This Python library is part of the ROMI European project.
It provides tools to **reconstruct digital twins of plants** from a set of RGB images acquired with the "Plant Imager".
It also provides tools to **quantify some traits of the plant's aerial architecture** like angles between successive organs and inter-nodes length.

A more comprehensive documentation about the "Plant Imager" project can be found [here](https://docs.romi-project.eu/plant_imager/).

**WARNING**: this is an ongoing development and changes will arise unannounced!

**Table of Contents**
- [Pre-requisites](#pre-requisites)
  - [Getting Started](#getting-started)
  - [Docker Engine](#docker-engine)
  - [NVIDIA drivers](#nvidia-drivers)
  - [NVIDIA Container Toolkit](#nvidia-container-toolkit)
- [Building and running with Docker (recommended)](#building-and-running-with-docker-recommended)
  - [Building the image](#building-the-image)
  - [Running the container](#running-the-container)
- [Install from sources](#install-from-sources)
  - [Install requirements](#requirements)
  - [Install sources](#install-sources)
- [Usage](#usage)
  - [Docker container](#docker-container)
  - [Conda environment](#conda-environment)
- [Troubleshooting](#troubleshooting)


## Pre-requisites
You will need to install the Docker Engine, the required NVIDIA driver and NVIDIA Container Toolkit to benefit from GPU accelerated algorithms.

### Getting Started
Let's first install some useful tools like `git`, `curl` & `nano`:
```shell
sudo apt update && sudo apt install -y git curl nano
```

### Docker Engine
You can follow the official [instructions](https://docs.docker.com/engine/install/ubuntu/) or use the convenience script:
```shell
curl https://get.docker.com | sh \
  && sudo systemctl --now enable docker
```

You also have to follow the [Post-installation steps for Linux](https://docs.docker.com/engine/install/linux-postinstall/). 

### NVIDIA drivers
On Ubuntu, you can install the latest compatible NVIDIA drivers with:
```shell
sudo ubuntu-drivers autoinstall
```

### NVIDIA Container Toolkit
We strongly recommend to follow the [Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) from NVIDIA.


## Building and running with Docker (recommended)

### Building the image
To build the Docker image of `plant-3d-vision`, you have to clone the git repository and run the script `docker/build.sh`.

```bash
git clone https://github.com/romi/plant-3d-vision.git
cd plant-3d-vision/
./docker/build.sh
```
This will create a docker image named `roboticsmicrofarms/plant-3d-vision:latest`.
If you want to tag your image with a specific one, here named `mytag`, just pass the tag argument as follows
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
Do NOT forget to change the paths with yours!

If you want to run a docker image with another tag than `latest`, you can pass the tag name as an argument:
`./docker/run.sh -t mytag`.

To see more running options, type `./docker/run.sh -h`

This docker image has been tested successfully on:
`docker --version=19.03.6 | nvidia driver version=450.102.04 | CUDA version=11.0`


## Install from sources

### Requirements

#### Conda
In the following instructions, we use `conda`, have a look at the official installation instructions [here](https://docs.conda.io/en/latest/miniconda.html) or use this convenience script:
```shell
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
Follow the instructions and proceed to remove the convenience script:
```shell
rm Miniconda3-latest-Linux-x86_64.sh
```

#### COLMAP
`COLMAP` is required to run the reconstruction tasks.

You have two options:

1. use the `roboticsmicrofarms/colmap:3.7` available from our [docker hub](https://hub.docker.com/repository/docker/roboticsmicrofarms/colmap) [recommended]
2. follow the official install instructions to install COLMAP for linux [here](https://colmap.github.io/install.html#linux).

#### Python OpenCL
This library use `pyopencl` and thus require the following system libraries:

- `ocl-icd-libopencl1`
- `opencl-headers`

On Debian and Ubuntu, you can install all these dependencies with:
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
    git submodule init
    git submodule update
    ```
2. Create a conda environment named `plant3dvision` with Python3.8 for example:
    ```bash
    conda create --name plant3dvision "python=3.9"
    ```
3. Install the submodules (`plantdb`, `romitask`, `romiseg` & `dtw`) and `plant3dvision` in activated environment:
    ```bash
    conda activate plant3dvision
    cd plant-3d-vision/
    python3 -m pip install -r ./plantdb/requirements.txt
    python3 -m pip install -e ./plantdb/.
    python3 -m pip install -r ./romitask/requirements.txt
    python3 -m pip install -e ./romitask/.
    python3 -m pip install -e ./romiseg/.
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


## Usage

### Docker container
Inside the docker image there is a `romi_run_task` command which performs a task on a database according to a config file.

In the following example, we will use the test database and config file shipped in this repository:
 - Start a docker container using `roboticsmicrofarms/plant-3d-vision:latest`
 - Mount the database (`plant-3d-vision/tests/testdata/`) and configs folder (`plant-3d-vision/config/`) inside the docker container
 - Perform the task `AnglesAndInternodes` on the database with `geom_pipe_real.toml` config file

```bash
./docker/run.sh -v /path/to/plant-3d-vision/tests/testdata/:/home/$USER/database/ -v /path/to/plant-3d-vision/config/:/home/$USER/config
romi_run_task --config ~/config/geom_pipe_real.toml AnglesAndInternodes ~/database/real_plant/
```

Don't forget to replace the paths `path/to/plant-3d-vision` by the correct ones.


### Conda environment
To execute a pre-defined task and its upstream tasks, like `AnglesAndInternodes`, on the provided test scan dataset `real_plant` using the appropriate example pipeline configuration file `geom_pipe_real.toml` (from the repository root directory):
```shell
cp -R tests/testdata /tmp/.  # copy the test DB to the temporary directory
conda activate plant3dvision  # activate the conda environment
romi_run_task --config config/geom_pipe_real.toml AnglesAndInternodes /tmp/testdata/real_plant/
```


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

### Docker
To test if everything is okay:
```bash
./docker/run.sh --gpu_test
```


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
