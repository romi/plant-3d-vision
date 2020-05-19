FROM colmap/colmap:3.6-dev.3
# https://hub.docker.com/r/continuumio/miniconda3
LABEL maintainer="Jonathan LEGRAND <jonathan.legrand@ens-lyon.fr>"
# To build docker image run following command from 'docker/' folder:
# $ docker build -t interpreter romiSmartInterpreter/
# To start built docker image:
# $ docker run -it -v /data/ROMI/integration_tests:/home/romi/db_test interpreter
# Inside docker, run:
# $ ./data-storage/bin/romi_run_task AnglesAndInternodes ../db_test/arabidopsis_26/ --config Scan3D/config/original_pipe_0.toml
# To clean-up after build:
# $ docker rm $(docker ps -a -q)

# Set non-root user name:
ENV SETUSER=romi

USER root
# Change shell to 'bash', default is 'sh'
SHELL ["/bin/bash", "-c"]
# Update package manager & install requirements:
RUN apt-get update && \
    apt-get install -yq --no-install-recommends \
    git ca-certificates wget\
    python3 python3-dev python3-pip && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* && \
    useradd -m ${SETUSER} && \
    cd /home/${SETUSER} && \
    mkdir project && \
    chown -R ${SETUSER}: /home/${SETUSER}

# Change user
USER ${SETUSER}
# Change working directory:
WORKDIR /home/${SETUSER}/project

RUN pip3 install setuptools setuptools-scm pybind11 && \
    # Install "data-storage", required to have `romi_run_task` cmd:
    git clone https://github.com/romi/data-storage.git && \
    cd data-storage && \
    pip3 install . && \
    cd .. && \
    # Install "Scan3D":
    git clone https://github.com/romi/Scan3D.git && \
    cd Scan3D && \
    git checkout feature/wsgi && \
    pip3 install -r requirements.txt && \
    pip3 install . && \
    cd .. && \
    # Install "cgal_bindings_skeletonization":
    pip3 install git+https://github.com/romi/cgal_bindings_skeletonization && \
    # Initialize a test databse:
    cd ~/ && \
    mkdir db_test

ENV DB_LOCATION="/home/${SETUSER}/db_test"

CMD ["/bin/bash"]