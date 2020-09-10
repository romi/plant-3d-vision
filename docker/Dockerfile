FROM roboticsmicrofarms/romiscan_base:latest

ARG ROMISCAN_BRANCH=dev
ARG ROMIDATA_BRANCH=dev
ARG ROMISCANNER_BRANCH=master

RUN git clone --branch $ROMISCAN_BRANCH https://github.com/romi/romiscan && \
    git clone --branch $ROMIDATA_BRANCH https://github.com/romi/romidata && \
    git clone --branch $ROMISCANNER_BRANCH https://github.com/romi/romiscanner && \
    python -m pip install -e ./romidata/ --no-cache-dir && \
    python -m pip install -e ./romiscanner/ --no-cache-dir && \
    python -m pip install -r ./romiscan/requirements.txt --no-cache-dir && \
    python -m pip install -e ./romiscan/ --no-cache-dir

# Change shell to 'bash', default is 'sh'
SHELL ["/bin/bash", "-c"]
ENV DB_LOCATION="/home/${USER_NAME}/db"
ENV PATH=$PATH:"/home/${USER_NAME}/.local/bin"