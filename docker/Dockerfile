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
# Update package manager & install requirements:
RUN apt-get update && \
    apt-get install -yq --no-install-recommends \
    wget emacs-nox \
    ocl-icd-libopencl1 opencl-headers clinfo libjpeg-dev libffi-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    # Ugly fix for OpenCL:
    ln -s /usr/lib/x86_64-linux-gnu/libOpenCL.so.1 /usr/lib/libOpenCL.so && \
    # Create a non-root user and give it rights over a "myapp" folder:
    adduser --disabled-password --gecos '' ${USER_NAME} && \
    mkdir -p ${DB_LOCATION} && chown -R ${USER_NAME} /myapp && \
    mkdir -p /venv && chown -R ${USER_NAME} /venv && \
    # Ugly fix for OpenCL:
    mkdir -p /etc/OpenCL/vendors && \
    echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd
# Change to non-root user:
USER ${USER_NAME}
# Change working directory:
WORKDIR /myapp

FROM base AS builder

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
# Download a model file:
RUN ./get_model.sh

CMD ["./docker-entrypoint.sh"]