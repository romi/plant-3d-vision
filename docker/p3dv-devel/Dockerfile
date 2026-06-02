# syntax = docker/dockerfile:1.12
ARG COLMAP_VERSION="3.8"
ARG CUDA_CC="75"
FROM roboticsmicrofarms/colmap:${COLMAP_VERSION}-cuda_cc${CUDA_CC}

LABEL maintainer="Jonathan LEGRAND <jonathan.legrand@ens-lyon.fr>"
LABEL corresponding_author="Peter HANAPPE <peter@romi-project.eu>"
LABEL project="Robotics for microfarms"
LABEL homepage="https://docs.romi-project.eu/documentation/"
LABEL repository="https://github.com/romi/plant-3d-vision"
LABEL license="LGPL-3.0-or-later"
LABEL description="Plant reconstruction and phenotyping pipeline."

# Set user name and id variables:
ENV USER_NAME="romi"
ENV USER_ID=2020

# Variables required by `nvidia-container-runtime`:
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
# Variables required by Python:
ENV LANG=C.UTF-8
ENV PYTHONUNBUFFERED=1
# Variables required by `pip`:
ENV PATH="${PATH}:/home/${USER_NAME}/.local/bin"

ENV APP_DIR="/myapp"
# Required by `romi_run_task`:
ENV ROMI_DB="${APP_DIR}/db"
# Required by `plant3dvision.colmap.ColmapRunner`
ENV COLMAP_EXE='colmap'

# Required by WebTerm
ENV ROMI_CFG="${APP_DIR}/cfg"
ENV ROMI_USERS="${APP_DIR}/users"
ENV WEBTERM_USERS="${ROMI_USERS}/users.csv"

ARG PYCUDA_NVCC_FLAGS=""
ENV PYCUDA_NVCC_FLAGS="${PYCUDA_NVCC_FLAGS}"

ENV PYOPENCL_CTX='0'
# Disable OpenCL automatic platform selection
ENV PYOPENCL_NO_CACHE=1

RUN apt-get update && \
    # Install the required dependencies
    apt-get install -yq --no-install-recommends \
        nano \
        wget \
        ocl-icd-libopencl1 opencl-headers clinfo \
        pocl-opencl-icd \
        nvidia-cuda-toolkit \
        python3-dev python3-pip python3-venv \
        gcc build-essential \
        libeigen3-dev libcgal-dev \
        libjpeg-dev libffi-dev && \
    # Clean-up apt cache
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    # Create a non-root user named romi' and create a "home" folder wih proper rights:
    adduser --disabled-password --gecos '' --uid ${USER_ID} ${USER_NAME} && \
    chown -R ${USER_NAME}:${USER_NAME} /home/${USER_NAME} && \
    # Create the folder used to store the app data:
    mkdir -p ${APP_DIR} && \
    mkdir -p ${APP_DIR}/plant-3d-vision && \
    mkdir -p ${ROMI_DB} && \
    mkdir -p ${ROMI_CFG} && \
    mkdir -p ${ROMI_USERS} && \
    chown -R ${USER_NAME}:${USER_NAME} ${APP_DIR} && \
    # Proper setup for OpenCL:
    mkdir -p /etc/OpenCL/vendors && \
    echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd && \
    # Give non-root user (romi) access to all COLMAP related files by making it a member of 'colmap_users' group
    usermod -a -G colmap_users ${USER_NAME} && \
    # Make sure .bashrc gets loaded in non-interactive shells by adding to .bash_profile
    echo 'if [ -f ~/.bashrc ]; then . ~/.bashrc; fi' > /home/${USER_NAME}/.bash_profile && \
    chown ${USER_NAME}:${USER_NAME} /home/${USER_NAME}/.bash_profile

# Change to non-root user:
USER ${USER_NAME}
# Change working directory:
WORKDIR /home/${USER_NAME}
# Copy .bashrc file
COPY --chown=${USER_NAME} docker/.bashrc .bashrc

# Create the virtual environment at build time (empty, ready for packages)
RUN python3 -m venv /home/${USER_NAME}/venv

# Note: Source code will be mounted at runtime, not copied during build
# The installation will happen in entrypoint.sh at container startup

COPY --chown=${USER_NAME} docker/p3dv-devel/entrypoint.sh .
# Make entrypoint executable
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]