FROM roboticsmicrofarms/colmap:3.8

LABEL maintainer="Jonathan LEGRAND <jonathan.legrand@ens-lyon.fr>"
LABEL corresponding_author="Peter Hanappe <peter@romi-project.eu>"
LABEL project="Robotics for microfarms"
LABEL homepage="https://docs.romi-project.eu/documentation/"
LABEL repository="https://github.com/romi/plant-3d-vision"
LABEL license="LGPL-3.0-or-later"
LABEL description="Plant reconstruction and phenotyping pipeline."

ENV LANG=C.UTF-8
ENV PYTHONUNBUFFERED=1
ENV USER_NAME=romi
ENV USER_ID=1010
# Required by `pip`:
ENV PATH=$PATH:"/home/${USER_NAME}/.local/bin"
# Required by `romi_run_task`:
ENV ROMI_DB="/myapp/db"
ENV COLMAP_EXE='colmap'
ENV PYOPENCL_CTX='0'
# Required by `nvidia-container-runtime`:
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

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
    echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

# Change to non-root user:
USER ${USER_NAME}
# Change working directory:
WORKDIR /home/${USER_NAME}
# Copy the source files starting from the root folder (relative to context build):
COPY --chown=${USER_NAME} ./ plant-3d-vision/

RUN cd plant-3d-vision/ && \
    # Install `plantdb` dependencies and sources:
    python3 -m pip install --user plantdb/ && \
    # Install `romitask` dependencies and sources:
    python3 -m pip install --user -r romitask/requirements.txt && \
    python3 -m pip install --user romitask/ && \
    # Install `romiseg` dependencies and sources:
    python3 -m pip install --user torch==1.12.1+cu102 torchvision==0.13.1+cu102 --extra-index-url https://download.pytorch.org/whl/cu102 && \
    python3 -m pip install --user romiseg/ && \
    # Install `romicgal` dependencies and sources:
    python3 -m pip install --user pybind11 && \
    python3 -m pip install --user romicgal/ && \
    # Install `dtw` dependencies and sources:
    python3 -m pip install --user -r dtw/requirements.txt && \
    python3 -m pip install --user dtw/ && \
    # Install `plant-3d-vision` dependencies and sources:
    python3 -m pip install --user -r requirements.txt && \
    python3 -m pip install --user . && \
    # Clean pip cache:
    python3 -m pip cache purge && \
    # Download the trained CNN model file:
    ./get_model.sh

CMD ["/bin/bash", "-c", "romi_run_task -h"]
