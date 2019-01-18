# Scan3D

This repo gather the elements used to run 3D scan of individual plants by ROMI partners
More information and history record is available on the wiki

## Hardware
The following table list the different setting currently used by the different partners. The specs of each setting is detailed in the wiki.

Partner | 3D Scan hardware
--------| -------------
Lyon | Lyon.1
Paris | csl.1
Valldaurua | Vall.x
Uber | Uber.

## Acquisition

## Software

### Configuration files
* `LOCAL_APP_DATA/lettucethink/scanner/objects/` contains scanned object metadata files
* `LOCAL_APP_DATA/lettucethink/scanner/scanners/` contains scanner configuration files
* `LOCAL_APP_DATA/lettucethink/scanner/paths/` contains scanning path description files

In general, on linux systems, `LOCAL_APP_DATA` is `$HOME/.local/share`, on windows it's
`C:/Users/UserName/AppData/Local` or in the environment variable `%LOCALAPPDATA%`. On
MacOS, it should be `~/Library/Application Support`.

### Install default configuration and sample objects (on linux):

```
mkdir -p ~/.local/share/lettucethink/scanner/objects
cp default/objects/* ~/.local/share/lettucethink/scanner/objects/
mkdir -p ~/.local/share/lettucethink/scanner/scanners
cp default/scanners/* ~/.local/share/lettucethink/scanner/scanners/
mkdir -p ~/.local/share/lettucethink/scanner/paths
cp default/paths/* ~/.local/share/lettucethink/scanner/paths/
```

TODO: automate this/ assist creation of objects

### Run a scan

Run a scan using
```
run-scan OBJECT_ID SCANNER_ID PATH_ID SCAN_ID
```

To get help on any command, you can use `CMD --help`.

## Full pipeline example (with scan)

Set the ID as the current timedate (if you are using bash/zsh...)
```
ID=`date "+%Y-%m-%d_%H-%M-%S"`
```

### Run the scan

```
./run-scan GT1 lyon_1 circular_72 $ID
```

### Process the data

Compute masks
```
./compute-masks -c 0.0,1.0,0.0,0.3 -d 3 $ID
```

Compute poses using colmap
```
./compute-poses $ID
```

Compute voxel volumei with size 1mm
```
./carve-space -v 1.0 $ID
```

Process point cloud (compute pointcloud, mesh, skeleton from voxels)
```
./process-pcd $ID
```

Compute angles
```
./segment-skel $ID
```
 
### Sync data with the dedicated server

To sync data with the dedicated server:
```
rsync -av ~/.local/share/lettucethink/scanner/data/ db.romi-project.eu:/data/
```
