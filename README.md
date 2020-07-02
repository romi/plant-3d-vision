# romiscan v0.7 (dev)

Documentation about the "Plant Scanner" project can be found [here](https://docs.romi-project.eu/Scanner/home/).

## Install requirements
This library use `pyopencl` and thus require the following system libraries:

- ocl-icd-libopencl1
- opencl-headers
- clinfo

On Ubuntu, you can install them with:
```bash
sudo apt-get update && apt-get install -y \
    ocl-icd-libopencl1 opencl-headers clinfo
```
You may need to make a symbolic link:
```bash
ln -s /usr/lib/x86_64-linux-gnu/libOpenCL.so.1 /usr/lib/libOpenCL.so
```


## Install from sources in conda environment:
In this install instructions, we leverage the `git submodule` functionality to clone the required ROMI libraries.

1. Clone the `romiscan` sources:
    ```bash
    git clone https://github.com/romi/romiscan.git
    ```
2. Create a conda environment named `scan_0.7` with Python3.7:
    ```bash
    conda create --name scan_0.7 python=3.7
    ```
3. Install sources and submodules in activated environment:
    ```bash
    conda activate scan_0.7
    cd romiscan
    git checkout dev
    git submodule init
    git submodule update
    python -m pip install -r requirements.txt
    python -m pip install -e ./romidata/
    python -m pip install -e ./romiseg/
    python -m pip install -e ./romiscanner/
    python -m pip install -e ./romicgal/
    python -m pip install .
    ```
4. Test `romiscan` library:
    ```bash
    conda activate scan_0.7
    python -c 'import romiscan'
    ```


## Conda packaging

### Build `romiscan` conda package:
From the `base` conda environment, run:
```bash
conda build conda_recipes/romiscan/ -c romi-eu -c open3d-admin -c conda-forge --user romi-eu
```


### Install `romiscan` conda package:
```bash
conda create -n romiscan romiscan -c romi-eu -c open3d-admin --force
```
To test package install, in the activated environment import `romiscan` in python:
```bash
conda activate romiscan
python -c 'import romiscan'
```


## Docker

### Building the image
To enable read/write access, it is required have correct user name, uid & gid.
Then we use arguments (`USER_NAME`, `USER_ID` & `GROUP_ID`) with docker image build command:
```bash
docker build -t romiscan:<tag> \
    --build-arg USER_NAME=$(id -n -u) \
    --build-arg USER_ID=$(id -u) \
    --build-arg GROUP_ID=$(id -g) .
```
Don’t forget to change the `<tag>` to a version (e.g. `v0.6`) or explicit name (e.g. `fix_colmap`) of your choosing!

Note that:

- `$(id -n -u)` will retrieve your host user name
- `$(id -u)` will retrieve your host user id
- `$(id -g)` will retrieve your host user group id


### Running the container
On the server, mount your database directory to `db_test` and optionally the config: 
```bash
docker run -it \
    -v /home/${USER}/database_<user>:/home/${USER}/db_test \
    -v /home/${USER}/configs/:/home/${USER}/config/ \
    --env PYOPENCL_CTX='0' \
    --gpus all romiscan:<tag> bash
```
Don't forget to:

- set the `<tag>` to match the one used to build!
- change the database path `database_<user>` with yours!

Also, note that:

- you are using the docker image `romiscan:0.6`
- you mount the host directory `~/database_<user>` "inside" the running container in the `~/db_test` directory
- you mount the host directory `~/configs` "inside" the running container in the `~/config` directory
- you activate all GPUs within the container with `--gpus all`
- declaring the environment variable `PYOPENCL_CTX='0'` select the first CUDA GPU capable
- `-it` & `bash` returns an interactive bash shell

You may want to name the running container (with `--name <my_name>`) if you "deamonize" it (with `-d`).
