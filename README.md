# Scan3D v0.4
This is the code for the 3D scanner, part of the ROMI project: https://romi-project.eu

It contains functions for 3D reconstruction and analysis of plants, mainly *Arabidopsis thaliana* at the moment.

## Installation
**This software is compatible with Python 3 only.**

It is possible to install `Scan3D` using various methods listed bellow.

### System-wide install from sources:

This software makes use of *Colmap* 3.6 as a Structure from Motion tool. You can download
it [here](https://colmap.github.io/). It uses OpenCL (>= 1.2) for accelerated 3D volume
processing, so you should have at least one OpenCL implementation available.

It depends on the CGAL library version 4.14. If you use Ubuntu,
the cgal in the official repositories are too old, and you have to compile
CGAL from [source](https://github.com/CGAL/cgal)

On Arch Linux, you can install cgal from the repositories.

Other dependencies should be processed automatically.

The software has only been tested for Linux systems. Other platforms might or might not work.

First clone the repository:
```
git clone https://github.com/romi/Scan3D.git
cd Scan3D
```

If you need to install Open3D, there is a helper you can run to install it:

```
./utils/install_open3d.py
```

Then, proceed to the installation using pip:
```
pip install .
```

### Source install in a conda environment:
**Note:** you will need to install `miniconda3` first!

1. Clone the repository and get all third party dependencies:
```bash
git clone https://github.com/romi/Scan3D.git
```

<!--2. Get all third party dependencies:-->
<!--```bash-->
<!--cd Scan3D-->
<!--git submodule update --init-->
<!--```-->

2. To create a conda environment and install all dependencies, in a shell run:
```bash
conda env create -f conda_recipe/scan3d_0.4-linux.yaml
```

3. Activate the conda environment:
```bash
conda activate scan3d_0.4
```

4. Then set the following **compilitation flags** for `CGAL`:
```bash
cmake romiscan/cgal/. -DCGAL_DIR="$CONDA_PREFIX/lib/cmake/CGAL" -DEIGEN3_INCLUDE_DIR="$CONDA_PREFIX/include/eigen3" -DCMAKE_BUILD_TYPE=Release
```

5. Finally run:
```bash
python setup.py install
```


## Installing `lettucethink-python`:
In order to use the `run-scan` or `run-pipeline` commands, you have to install another package developed by the ROMI team: `lettucethink-python`.

To do so, **in a shell with your conda environment active**, do:
```bash
git clone https://github.com/romi/lettucethink-python.git
cd lettucethink-python
python setup.py install
```
**TODO:** this could be done using conda after `lettucethink-python` has been packaged!


## Usage

In the following, `db` is a lettucethink `fsdb` database root location.

The following commands are then available: `run-scan` & `run-pipeline`.


### Plant scanning:
```bash
run-scan object.json scanner.json path.json db[/scan_id]
```
If you do not provide a scan ID, it will automatically be set to the current datetime.
The json files are metadata of the scan.

### Reconstruction & quantification pipeline:
To run a processing pipeline, use the command:

```bash
run-pipeline db/scan_id -c pipeline.toml -t TASK
```
where:
 * `pipeline.toml` is the file describing the process configuration. See for example
the provided `default/pipeline.toml`.
 * `TASK` can be any `RomiTask` subclass defined in `romiscan.pipeline`.

For example, if you want to compute *internode* and *angles*, on a local database for the `2019-01-29_16-56-00` scan (*eg.* located under `/data/scans/2019-01-29_16-56-00`) run:
```bash
run-pipeline /data/scans/2019-01-29_16-56-00 -c pipeline.toml -t AnglesAndInternodes
```
All necessary computation will be done automatically.


## Scans synchronization

To prepare files for vizualization and sync the files to the vizualization server, run

```
sync-scans local_db remote_db [http(s)://api_url(:api_port)]
```

This will process the `Visualization` fileset and rsync to remote_db.
api_url is used to refresh the database in the vizualizer.

For example, for ROMI people:

```
sync-scans ~/Data/scanner/processed db.romi-project.eu:/data/v0.4 https://db.romi-project.eu/
```

## Metadata structure

Here, we describe the metadata specification for v0.4 of the software

### Scanner metadata

You can find a sample `scanner.json` in the `default` folder.

Necessary key-value pairs are the following:

* `camera_firmware` : `gphoto2` or `sony_wifi`
* `camera_args` : `kwargs` for the `Camera` class (see lettucethink-python)
* `cnc_firmware` : `grbl` or `cnccontroller`
* `cnc_args` : `kwargs` for the `CNC` class (see lettucethink-python)
* `gimbal_firmware` : `dynamixel` or `blgimbal`
* `gimbal_args` : `kwargs` for the `Gimbal` class (see lettucethink-python)
* `workspace` : description of a 3D volume enclosing the scanned object.
 	It's a dictionnary, with `x`, `y` and `z` as keys and lists of extremal values as values.

### Object metadata

You can find a sample `object.json` in the `default` folder.

The only keys used in the software are `species`, `environment` and `plant_id` in the REST API.

### Path metadata

You can find a sample `path.json` in the `default` folder.
For now, `type` can only be `circular` and the corresponding arguments are found
in lettucethink-python.
