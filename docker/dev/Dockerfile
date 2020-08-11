FROM geki/colmap
# geki/colmap is based on nvidia/cuda:10.1-devel-ubuntu18.04

ARG USER_NAME=romi
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG ROMISCAN_BRANCH=dev

USER root
RUN apt-get update && apt-get install -y \
    python3.7 python3.7-dev python3-pip wget && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.7 10 && \
    apt-get update && apt-get install -y --no-install-recommends \
    ocl-icd-libopencl1 opencl-headers clinfo && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    ln -s /usr/lib/x86_64-linux-gnu/libOpenCL.so.1 /usr/lib/libOpenCL.so && \
    # Create a non-root user and give it rights over its "home folder"
    addgroup --gid $GROUP_ID $USER_NAME && \
    adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID $USER_NAME && \
    chown -R ${USER_NAME}: /home/${USER_NAME} && \
    mkdir -p /etc/OpenCL/vendors && \
    echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

USER ${USER_NAME}
WORKDIR /home/${USER_NAME}

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

RUN git clone https://github.com/romi/romiscan && \
    cd romiscan/ && \
    git checkout $ROMISCAN_BRANCH && \
    git submodule init && \
    git submodule update && \
    python -m pip install -e ./romiseg/ --no-cache-dir && \
    python -m pip install -e ./romicgal/ --no-cache-dir && \
    python -m pip install -e ./romidata/ --no-cache-dir && \
    python -m pip install -e ./romiscanner/ --no-cache-dir && \
    python -m pip install -r requirements.txt --no-cache-dir && \
    python -m pip install -e . --no-cache-dir

ENV DB_LOCATION="/home/${USER_NAME}/db_test"
ENV PATH=$PATH:"/home/${USER_NAME}/.local/bin"

CMD ["/bin/bash", "-c", "cd .."]
