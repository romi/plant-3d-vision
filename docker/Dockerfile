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
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
# Variables required by Python:
ENV LANG=C.UTF-8
ENV PYTHONUNBUFFERED=1
# Variables required by `pip`:
ENV PATH="${PATH}:/home/${USER_NAME}/.local/bin"

# Required by `romi_run_task`:
ENV ROMI_DB="/myapp/db"
ENV COLMAP_EXE='colmap'
ENV PYOPENCL_CTX='0'

RUN apt-get update && \
    # Install the required dependencies
    apt-get install -yq --no-install-recommends \
        nano \
        wget \
        ocl-icd-libopencl1 opencl-headers clinfo \
        python3-dev python3-pip \
        gcc build-essential \
        libeigen3-dev libcgal-dev \
        libjpeg-dev libffi-dev && \
    # Clean-up apt cache
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    # Create a non-root user named romi' and create a "home" folder wih proper rights:
    adduser --disabled-password --gecos '' --uid ${USER_ID} ${USER_NAME} && \
    chown -R ${USER_NAME}: /home/${USER_NAME} && \
    # Create the folder used to mount the database:
    mkdir -p ${ROMI_DB} && \
    chown -R ${USER_NAME}: ${ROMI_DB} && \
    # Ugly fix for OpenCL:
    ln -s /usr/lib/x86_64-linux-gnu/libOpenCL.so.1 /usr/lib/libOpenCL.so && \
    mkdir -p /etc/OpenCL/vendors && \
    echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd && \
    # Give non-root user (romi) access to all COLMAP related files by making it a member of 'colmap_users' group
    usermod -a -G colmap_users ${USER_NAME}

# Change to non-root user:
USER ${USER_NAME}
# Change working directory:
WORKDIR /home/${USER_NAME}
# Copy the source files starting from the root folder (relative to context build):
COPY --chown=${USER_NAME} ./ plant-3d-vision/

RUN cd plant-3d-vision/ && \
    # Install `plant-3d-vision` dependencies and sources:
    ./install.sh --no-env --no-cache-dir && \
    # Install dependencies be able to start the WebTerm
    python3 -m pip install -r plant3dvision/webterm/requirements.txt
    # Download the trained CNN model file:
    ./get_model.sh

COPY --chown=${USER_NAME} docker/entrypoint.sh .
ENTRYPOINT ["./entrypoint.sh"]
