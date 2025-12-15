ARG UBUNTU_VERSION=24.04
ARG NVIDIA_CUDA_VERSION=12.9.1

# -----------------------------------------------------------------------------
# Docker builder stage.
# -----------------------------------------------------------------------------
FROM nvidia/cuda:${NVIDIA_CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder

ARG COLMAP_GIT_TAG=3.13.0
ARG CUDA_ARCHITECTURES=all-major
ENV QT_XCB_GL_INTEGRATION=xcb_egl

# Prevent stop building ubuntu at time zone selection.
ENV DEBIAN_FRONTEND=noninteractive

# Prepare and empty machine for building (using `apt-mark hold` to prevent the systemd packages from being installed).
RUN apt-get update && \
    apt-mark hold systemd systemd-dev && \
    apt-get install -y \
        git \
        cmake \
        ninja-build \
        build-essential \
        libboost-program-options-dev \
        libboost-graph-dev \
        libboost-system-dev \
        libeigen3-dev \
        libfreeimage-dev \
        libmetis-dev \
        libgoogle-glog-dev \
        libgtest-dev \
        libgmock-dev \
        libsqlite3-dev \
        libglew-dev \
        qtbase5-dev \
        libqt5opengl5-dev \
        libcgal-dev \
        libceres-dev \
        libcurl4-openssl-dev \
        libmkl-full-dev \
        python3-full \
        python3-venv

# Build and install COLMAP.
ENV CMAKE_INSTALL_PREFIX=/colmap-install
RUN git clone https://github.com/colmap/colmap.git --depth 1 --shallow-submodules --no-tags --branch ${COLMAP_GIT_TAG}
RUN cd colmap && \
    git fetch --tags https://github.com/colmap/colmap.git && \
    git checkout ${COLMAP_GIT_TAG} && \
    mkdir build && \
    cd build && \
    cmake .. \
        -GNinja \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} \
        -DCMAKE_INSTALL_PREFIX=/colmap-install \
        -DBLA_VENDOR=Intel10_64lp && \
    ninja install
ENV CMAKE_PREFIX_PATH=/colmap-install
RUN apt-get install -y python3-dev
RUN python3 -m venv /home/root/colmap_env && \
    . /home/root/colmap_env/bin/activate && \
    pip install --upgrade pip && \
    cd colmap && \
    pip install .


# -----------------------------------------------------------------------------
# Docker runtime stage.
# -----------------------------------------------------------------------------
FROM nvidia/cuda:${NVIDIA_CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS runtime

# Minimal dependencies to run COLMAP binary compiled in the builder stage.
# Note: this reduces the size of the final image considerably, since all the
# build dependencies are not needed.
RUN apt-get update && \
    apt-get install -y --no-install-recommends --no-install-suggests \
        libboost-program-options1.83.0 \
        libc6 \
        libomp5 \
        libopengl0 \
        libmetis5 \
        libceres4t64 \
        libfreeimage3 \
        libgcc-s1 \
        libgl1 \
        libglew2.2 \
        libgoogle-glog0v6t64 \
        libqt5core5a \
        libqt5gui5 \
        libqt5widgets5 \
        libcurl4 \
        libmkl-locale \
        libmkl-intel-lp64 \
        libmkl-intel-thread \
        libmkl-core \
        python3-full  \
        python3-venv \
        git && \
    apt-get clean

# Add to python path to virtual env
ENV VIRTUAL_ENV=/home/root/colmap-venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
# Install Python and create virtual environment
RUN python3 -m venv $VIRTUAL_ENV

# Copy all files from /colmap-install/ in the builder stage to /usr/local/ in
# the runtime stage. This simulates installing COLMAP in the default location
# (/usr/local/), which simplifies environment variables. It also allows the user
# of this Docker image to use it as a base image for compiling against COLMAP as
# a library. For instance, CMake will be able to find COLMAP easily with the
# command: find_package(COLMAP REQUIRED).
COPY --from=builder /colmap-install/ /usr/local/
COPY --from=builder /home/root/colmap_env/lib $VIRTUAL_ENV/lib

RUN groupadd -g 1001 colmap_users && \
    # Make COLMAP files group-readable and executable
    chgrp -R colmap_users /usr/local/bin/colmap && \
    chgrp -R colmap_users /usr/local/share/colmap && \
    chmod -R g+rx /usr/local/bin/colmap && \
    chmod -R g+rx /usr/local/share/colmap

