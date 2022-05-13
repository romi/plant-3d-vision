FROM roboticsmicrofarms/colmap:3.7 AS base
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
ENV PATH=$PATH:"/home/${USER_NAME}/.local/bin"
ENV DB_LOCATION="/myapp/db"
ENV COLMAP_EXE='colmap'
# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

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
    # Create a non-root user and give it rights over a "myapp" folder:
    adduser --disabled-password --gecos '' ${USER_NAME} && \
    # Create the folder used to mount the database:
    mkdir -m 777 -p ${DB_LOCATION} && \
    chown -R ${USER_NAME} /myapp && \
    # Create a folder to create python venv:
    mkdir -p /venv && \
    chown -R ${USER_NAME} /venv && \
    # Ugly fix for OpenCL:
    ln -s /usr/lib/x86_64-linux-gnu/libOpenCL.so.1 /usr/lib/libOpenCL.so && \
    mkdir -p /etc/OpenCL/vendors && \
    echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd
# Change to non-root user:
USER ${USER_NAME}
# Change working directory:
WORKDIR /myapp

FROM base AS builder
LABEL stage=builder

# Upgrade pip, install poetry and create a venv:
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install poetry && \
    python3 -m venv /venv
# Copy the source files (repository):
COPY --chown=${USER_NAME} ./ ./
# In the venv: install dependencies, build a wheel package and install it:
RUN source /venv/bin/activate && \
    poetry install -n --no-dev --no-root && \
    poetry build && \
    /venv/bin/pip install dist/*.whl

FROM base AS final

# Copy the venv from the builder:
COPY --from=builder --chown=${USER_NAME} /venv /venv
COPY --chown=${USER_NAME} ./get_model.sh ./get_model.sh
COPY --chown=${USER_NAME} ./tests ./tests
COPY --chown=${USER_NAME} ./config ./config
# Download a model file:
RUN ./get_model.sh

CMD [". /venv/bin/activate && /venv/bin/romi_run_task -h"]