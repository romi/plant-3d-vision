FROM geki/colmap
# geki/colmap is based on nvidia/cuda:10.1-devel-ubuntu18.04

ARG USER_NAME=scanner
ARG USER_ID=1000
ARG GROUP_NAME=scanner
ARG GROUP_ID=1000

# Change Shell to 'bash', default is 'sh'
SHELL [ "/bin/bash", "-c" ]

ENV DB_LOCATION="/home/${USER_NAME}/database"
ENV PATH=$PATH:"/home/${USER_NAME}/.local/bin"
ENV COLMAP_EXE='colmap'

USER root
RUN apt-get update && apt-get install -y \
    python3.7 python3.7-dev python3-pip wget emacs-nox && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.7 10 && \
    apt-get update && apt-get install -y --no-install-recommends \
    ocl-icd-libopencl1 opencl-headers clinfo libjpeg8-dev libffi-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    ln -s /usr/lib/x86_64-linux-gnu/libOpenCL.so.1 /usr/lib/libOpenCL.so && \
    # Create a non-root user and give it rights over its "home folder"
    addgroup --gid $GROUP_ID $GROUP_NAME && \
    adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID $USER_NAME && \
    chown -R ${USER_NAME}: /home/${USER_NAME} && \
    mkdir ${DB_LOCATION} && \
    mkdir -p /etc/OpenCL/vendors && \
    echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

# Change to non-root user:
USER ${USER_NAME}
# Change working directory:
WORKDIR /home/${USER_NAME}
# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

COPY --chown=${USER_NAME}:${GROUP_NAME} ./ plant-3d-vision/

RUN wget -nv --show-progress --progress=bar:force:noscroll https://media.romi-project.eu/data/Resnet_896_896_epoch50.pt && \
    mkdir -p plant-3d-vision/tests/testdata/models/models && \
    mv Resnet_896_896_epoch50.pt plant-3d-vision/tests/testdata/models/models/

RUN python -m pip install --upgrade pip && \
    python -m pip install -e plant-3d-vision/plantdb/ && \
    python -m pip install -e plant-3d-vision/romitask/ && \
    python -m pip install -e plant-3d-vision/romiseg/ && \
    python -m pip install -e plant-3d-vision/romicgal/ && \
    python -m pip install -e plant-3d-vision/dtw/ && \
    python -m pip install -r plant-3d-vision/requirements.txt && \
    python -m pip install -e plant-3d-vision/ && \
    rm -rf /home/jonathan/.cache/pip

CMD ["/bin/bash", "-c", "romi_run_task -h"]
