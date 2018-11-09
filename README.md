# Scan3D

This repo gather the elements used to run 3D scan of individual plants by ROMI partners
More information and history record is available on the wiki

## Hardware
The following table list the different setting currently used by the different partners. The specs of each setting is detailed in the wiki.

Partner | 3D Scan hardware
--------| -------------
Lyon | Lyon.1
Paris | Paris.1
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
run-scan [options] OBJECT_ID SCANNER_ID PATH_ID

Options:
    -i ID, --id=ID    scan id, default to current time formatted as
                        %Y-%m-%d_%H-%M-%S
```

For example, to use the supplied sample files:
```
run-scan GT1 lyon_1 circular_72
```

### Sync data with the dedicated server

To sync data with the dedicated server:
```
rsync -av ~/.local/share/lettucethink/scanner/data/ db.romi-project.eu:/data/
```
