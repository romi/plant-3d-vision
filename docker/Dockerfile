FROM roboticsmicrofarms/colmap:3.7
LABEL maintainer="Jonathan LEGRAND <jonathan.legrand@ens-lyon.fr>"
LABEL corresponding_author="Peter Hanappe <peter@romi-project.eu>"
LABEL project="Robotics for microfarms"
LABEL homepage="https://docs.romi-project.eu/documentation/"
LABEL repository="https://github.com/romi/plant-3d-vision"
LABEL license="LGPL-3.0-or-later"
LABEL description="Plant reconstruction and phenotyping pipeline."

ENV LANG=C.UTF-8
ENV PYTHONUNBUFFERED=1
ENV USER_NAME=myuser
ENV USER_ID=1000
ENV PATH=$PATH:"/home/${USER_NAME}/.local/bin"
ENV DB_LOCATION="/myapp/db"
ENV COLMAP_EXE='colmap'
# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV PYOPENCL_CTX='0'

# Change Shell to 'bash', default is 'sh'
SHELL [ "/bin/bash", "-c" ]

RUN echo "Fix CUDA GPG key..." && \
    # Install the cuda-keyring package (https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key/) \
    apt-key del 7fa2af80 && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    rm cuda-keyring_1.0-1_all.deb && \
    # Remove duplicate .list entries (https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key/)
    sed -i '/developer\.download\.nvidia\.com\/compute\/cuda\/repos/d' /etc/apt/sources.list && \
    # Remove Machine Learning repository (https://forums.developer.nvidia.com/t/updating-the-cuda-linux-gpg-repository-key/212897/8)
    sed -i '/developer\.download\.nvidia\.com\/compute\/machine-learning\/repos/d' /etc/apt/sources.list.d/* && \
    rm /etc/apt/sources.list.d/cuda*.list && \
    echo "Resume dependencies installation..." && \
    # Update package manager
    apt-get update && \
    # Install the required dependencies
    apt-get install -yq --no-install-recommends \
    emacs-nox \
    ocl-icd-libopencl1 opencl-headers clinfo \
    libjpeg-dev libffi-dev && \
    # Clean-up apt cache
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    # Create a non-root user and give it rights over a "homr" folder:
    adduser --disabled-password --gecos '' --uid ${USER_ID} ${USER_NAME} && \
    chown -R ${USER_NAME}: /home/${USER_NAME} && \
    # Create the folder used to mount the database:
    mkdir -m 777 -p ${DB_LOCATION} && \
    # Ugly fix for OpenCL:
    ln -s /usr/lib/x86_64-linux-gnu/libOpenCL.so.1 /usr/lib/libOpenCL.so && \
    mkdir -p /etc/OpenCL/vendors && \
    echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd
# Change to non-root user:
USER ${USER_NAME}
# Change working directory:
WORKDIR /home/${USER_NAME}

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Copy the source files (repository):
COPY --chown=${USER_NAME} ./ plant-3d-vision/

# Install the required dependencies, sub-modules and plant-3d-vision:
RUN cd plant-3d-vision/ && \
    python3 -m pip install -r plantdb/requirements.txt && \
    python3 -m pip install plantdb/ && \
    python3 -m pip install -r romitask/requirements.txt && \
    python3 -m pip install romitask/ && \
    python3 -m pip install romiseg/ && \
    python3 -m pip install pybind11 && \
    python3 -m pip install romicgal/ && \
    python3 -m pip install -r dtw/requirements.txt && \
    python3 -m pip install dtw/ && \
    python3 -m pip install -r requirements.txt && \
    python3 -m pip install . && \
    rm -rf .cache/pip && \
    ./get_model.sh

CMD ["/bin/bash", "-c", "romi_run_task -h"]
