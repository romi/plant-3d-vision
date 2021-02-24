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
    ocl-icd-libopencl1 opencl-headers clinfo && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    ln -s /usr/lib/x86_64-linux-gnu/libOpenCL.so.1 /usr/lib/libOpenCL.so && \
    # Create a non-root user and give it rights over its "home folder"
    addgroup --gid $GROUP_ID $GROUP_NAME && \
    adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID $USER_NAME && \
    chown -R ${USER_NAME}: /home/${USER_NAME} && \
    mkdir ${DB_LOCATION}

# Change to non-root user:
USER ${USER_NAME}
# Change working directory:
WORKDIR /home/${USER_NAME}

USER root
RUN mkdir -p /etc/OpenCL/vendors && \
    echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

USER ${USER_NAME}
# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

COPY --chown=${USER_NAME}:${GROUP_NAME} ./ romiscan/

RUN wget https://media.romi-project.eu/data/Resnet_896_896_epoch50.pt && \
    mv Resnet_896_896_epoch50.pt romiscan/tests/testdata/models/models/

RUN cd romiscan/ && \
    python -m pip install -e ./romidata/ --no-cache-dir && \
    python -m pip install -e ./romiseg/ --no-cache-dir && \
    python -m pip install -e ./romiscanner/ --no-cache-dir && \
    python -m pip install -e ./romicgal/ --no-cache-dir && \
    python -m pip install -r requirements.txt --no-cache-dir && \
    python -m pip install . --no-cache-dir

CMD ["/bin/bash", "-c", "romi_run_task -h"]
