# Plant-3D-Vision base Docker Image

This Docker image provides a **ready‑to‑run environment** for the *plant‑3d‑vision* library, its sub‑modules and all required system dependencies (CUDA, COLMAP, OpenCL, Python3.10, etc.).
It is designed specifically for **automated testing** and **GitHub CI pipelines**: the image builds a non‑root user, sets up a Python virtual environment, and runs the `entrypoint.sh` script at container start‑up to install the project from a mounted source tree.

The entrypoint handles:
- mounting the repository,
- installing the packages,
- downloading optional models,
- and finally executing the command passed to the container.

## Build

Start by building the image as follows:
```shell
./docker/p3dv-devel/build.sh 
```

## Local Usage (Testing)

The helper script `run.sh` launches the container with the appropriate CUDA runtime and forwards the current repository into the container (`/workspace`).  
Replace `<TAG>` with the desired COLMAP/CUDA tag (_e.g._ `colmap3.8-cuda_cc75`).

### Unit tests

```shell
./docker/p3dv-devel/run.sh -t colmap3.8-cuda_cc75 --unittest
```

### Integration tests

```shell
./docker/p3dv-devel/run.sh -t colmap3.8-cuda_cc75 --test-integration
```

### Pipeline tests

```shell
./docker/p3dv-devel/run.sh -t colmap3.8-cuda_cc75 --test-pipelines
```

## CI / GitHub Actions

The image can be used directly in a GitHub Actions workflow to run the test suite on Linux runners with NVIDIA GPUs.

```yaml
name: CI
on:
  push
  branches
  pull_request

jobs:
  test:
    runs-on: ubuntu-latest # Require an NVIDIA‑GPU runner (self‑hosted or using the github-actions runner with GPU support)
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Build Docker image
        run: ./docker/p3dv-devel/build.sh
    
      - name: Run unit tests in Docker
        run: |
          ./docker/p3dv-devel/run.sh -t colmap3.8-cuda_cc75 --unittest
    
      - name: Run integration tests in Docker
        run: |
          ./docker/p3dv-devel/run.sh -t colmap3.8-cuda_cc75 --test-integration
    
      - name: Run pipeline tests in Docker
        run: |
          ./docker/p3dv-devel/run.sh -t colmap3.8-cuda_cc75 --test-pipelines
``` 

*Tip:* If you need to run additional commands inside the container (_e.g._, linting or custom scripts), append them after the test flags:
```shell
./docker/p3dv-devel/run.sh -t colmap3.8-cuda_cc75 --unittest && flake8 .
```

## Advanced options

- **Custom source directory** set `SOURCE_DIR` to point to a different mount point:
  ```shell
  SOURCE_DIR=/my/code ./docker/p3dv-devel/run.sh -t colmap3.8-cuda_cc75 --unittest
  ```
- **Environment variables**: the image respects the usual Docker `-e` flag; useful for tweaking OpenCL or CUDA settings (`PYCUDA_NVCC_FLAGS`, `PYOPENCL_CTX`, etc.).
