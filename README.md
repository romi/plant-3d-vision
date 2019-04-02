# Scan3D v0.4

This is the code for the 3D scanner, part of the ROMI project. It contains
various function for 3D reconstruction and analysis of plants, mainly
Arabidopsis Thaliana at the moment.

## Installation

This software is compatible with Python 3 only.

This software makes use of *Colmap* 3.6 as a Structure from Motion tool. You can download
it [here](https://colmap.github.io/). It uses OpenCL (>= 1.2) for accelerated 3D volume
processing, so you should have at least one OpenCL implementation available.

Other dependencies should be processed automatically.

The software has only been tested for Linux systems. Other platforms might or might not work.

First clone the repository and get all third party dependencies:

```
git clone https://github.com/romi/Scan3D.git
cd Scan3D
git submodule update --init
```

To install, run (I would advise to do it in a virtual environment):

```
python setup.py install
```

In the following, `db` is a lettucethink `fsdb` database root location.

The following commands are then available
```
run-scan object.json scanner.json path.json db[/scan_id]
```
If you do not provide a scan ID, it will automatically be set to the current datetime.
The json files are metadata of the scan.

To run a processing pipeline, use the command:

```
run-pipeline db/scan_id -c pipeline.toml -t TASK
```

`pipeline.toml` is the file describing the process configuration. See for example
the provided `pipeline/pipeline.toml`.
TASK can be any `RomiTask` subclass defined in `romiscan.pipeline`.

For example, if you want to compute internode and angles, run:
All necessary computation will be computed automatically.

```
run-pipeline db/scan_id -c pipeline.toml -t AnglesAndInternodes
```

## Scans synchronization

To prepare files for vizualization and sync the files to the vizualization server, run

```
sync-scans local_db remote_db [http(s)://api_url(:api_port)]
```

This will process the `Visualization` fileset and rsync to remote_db.
api_url is used to refresh the database in the vizualizer.

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

