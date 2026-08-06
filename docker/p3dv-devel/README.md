# Plant-3D-Vision base Docker Image

This Docker image provides a **ready‑to‑run environment** for the *plant‑3d‑vision* library, its sub‑modules and all required system dependencies (CUDA, COLMAP, OpenCL, Python3.10, etc.).
It is designed specifically for **automated testing** and **GitHub CI pipelines**: the image builds a non‑root user, sets up a Python virtual environment, install the plant-3d-vision sources and all dependencies.
It runs the `entrypoint.sh` script at container start‑up to replace the sources and re-install the project from a mounted source tree. 
This way most of the dependencies are installed only once, and the container can be used for multiple runs. 

The entrypoint handles:
- mounting the repository,
- installing the packages,
- downloading optional models,
- and finally executing the command passed to the container.

## Build

Start by building the image as follows:
```shell
TAG=colmap$(./docker/colmap_tag.sh 3.8 | tail -n 1)
echo "Using p3dv-devel image tag: ${TAG}"
echo "Building: roboticsmicrofarms/p3dv-devel:${TAG}"
./docker/p3dv-devel/build.sh 
```

## Local Usage (Testing)

The helper script `run.sh` launches the container with the appropriate CUDA runtime and forwards the current repository into the container (`/mnt_src`).  
Replace `<TAG>` with the desired COLMAP/CUDA tag (_e.g._ `colmap3.8-cuda_cc75`).

### Unit tests

```shell
TAG=colmap$(./docker/colmap_tag.sh 3.8 | tail -n 1)
echo "Using roboticsmicrofarms/p3dv-devel:${TAG}"
./docker/p3dv-devel/run.sh -t ${TAG} --unittest
```

### Integration tests

```shell
TAG=colmap$(./docker/colmap_tag.sh 3.8 | tail -n 1)
echo "Using roboticsmicrofarms/p3dv-devel:${TAG}"
./docker/p3dv-devel/run.sh -t ${TAG} --test-integration
```

### Pipeline tests

```shell
TAG=colmap$(./docker/colmap_tag.sh 3.8 | tail -n 1)
echo "Using roboticsmicrofarms/p3dv-devel:${TAG}"
./docker/p3dv-devel/run.sh -t ${TAG} --test-pipelines
```

## CI / GitHub Actions

The image can be used directly in a GitHub Actions workflow to run the test suite on Linux runners with NVIDIA GPUs.
See the [`.github/workflows/pull_request.yml`](.github/workflows/pull_request.yml) file for an example.

## Advanced options

- **Custom source directory** use an absolute path with the `--source_dir` option to mount a specific source directory into the container:
  ```shell
  ./docker/p3dv-devel/run.sh -t ${TAG} --source_dir /path/to/source --unittest
  ```
- **Environment variables**: the image respects the usual Docker `-e` flag; useful for tweaking OpenCL or CUDA settings (`PYCUDA_NVCC_FLAGS`, `PYOPENCL_CTX`, etc.).
