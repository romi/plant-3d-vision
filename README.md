# romiscan v0.6 (dev)

## Installation instructions

You can find installation instructions in the [documentation](https://romi.github.io/html/romiscan.html#installation)

## Documentation

All documentation is [here](https://romi.github.io/html/romiscan.html)

## Install `romiscan` conda package:
```bash
conda create -n romiscan romiscan -c romi-eu -c open3d-admin --force
```
To test package install, in the activated environment import `romiscan` in python:
```bash
conda activate romiscan
python -c 'import romiscan'
```

## Install from sources in conda environment:

1. Clone the sources:
    ```bash
    git clone https://github.com/romi/conda_recipes.git
    git clone https://github.com/romi/romiscan.git
    ```
2. Create a conda environment:
    ```bash
    conda env create -n romiscan_dev -f conda_recipes/environments/romiscan_0.5.yaml
    ```
3. Install sources:
   ```bash
   conda activate romiscan_dev
   cd romiscan
   python setup.py develop
   ```
4. Test `romiscan` library:
   ```bash
   conda activate romiscan_dev
   python -c 'import romiscan'
   ```

## Build `romiscan` conda package:
From the `base` conda environment, run:
```bash
conda build conda_recipes/romiscan/ -c romi-eu -c open3d-admin -c conda-forge --user romi-eu
```