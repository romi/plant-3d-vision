FROM geki/colmap
LABEL maintainer="Jonathan LEGRAND <jonathan.legrand@ens-lyon.fr>"
# geki/colmap is based on nvidia/cuda:10.1-devel-ubuntu18.04

ENV LANG=C.UTF-8
ENV PYTHONUNBUFFERED=1
ENV USER_NAME=myuser
ENV PATH=$PATH:"/home/${USER_NAME}/.local/bin"
ENV DB_LOCATION="/myapp/db"
ENV COLMAP_EXE='colmap'
# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

USER root
# Change Shell to 'bash', default is 'sh'
SHELL [ "/bin/bash", "-c" ]
RUN apt-get update && \
    apt-get install -yq --no-install-recommends \
    python3.7 python3.7-dev python3-pip wget emacs-nox && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.7 10 && \
    apt-get install -y --no-install-recommends \
    ocl-icd-libopencl1 opencl-headers clinfo libjpeg-dev libffi-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    ln -s /usr/lib/x86_64-linux-gnu/libOpenCL.so.1 /usr/lib/libOpenCL.so && \
    # Create a non-root user and give it rights over a "myapp" folder:
    adduser --disabled-password --gecos '' ${USER_NAME} && \
    mkdir -p ${DB_LOCATION} && chown -R ${USER_NAME} /myapp && \
    # Ugly fix:
    mkdir -p /etc/OpenCL/vendors && \
    echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

# Change to non-root user:
USER ${USER_NAME}
# Change working directory:
WORKDIR /myapp
# Copy the sources:
COPY --chown=${USER_NAME} ./ plant-3d-vision/
# Download a model file:
RUN plant-3d-vision/utils/get_model.sh
# Install the required dependencies, sub-modules and plant-3d-vision:
RUN python -m pip install --upgrade pip && \
    python -m pip install setuptools wheel && \
    python -m pip install -r plant-3d-vision/plantdb/requirements.txt && \
    python -m pip install -e plant-3d-vision/plantdb/ && \
    python -m pip install -e plant-3d-vision/romitask/ && \
    python -m pip install -e plant-3d-vision/romiseg/ && \
    python -m pip install -e plant-3d-vision/romicgal/ && \
    python -m pip install -e plant-3d-vision/dtw/ && \
    python -m pip install -r plant-3d-vision/requirements.txt && \
    python -m pip install -e plant-3d-vision/ && \
    rm -rf .cache/pip

CMD ["/bin/bash", "-c", "romi_run_task -h"]
