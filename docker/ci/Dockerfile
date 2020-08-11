FROM roboticsmicrofarms/romiscan_base:0.7

ARG ROMISCAN_BRANCH=dev

RUN git clone --branch $ROMISCAN_BRANCH https://github.com/romi/romiscan && \
    cd romiscan/ && \
    git checkout $ROMISCAN_BRANCH && \
    git submodule init && \
    git submodule update && \
    python -m pip install -e ./romidata/ && \
    python -m pip install -e ./romiseg/ && \
    python -m pip install -e ./romiscanner/ && \
    python -m pip install -e ./romicgal/ && \
    python -m pip install -r requirements.txt && \
    python -m pip install -e . && \
    python -m pip cache purge


ENV DB_LOCATION="/home/${USER_NAME}/db_test"
ENV PATH=$PATH:"/home/${USER_NAME}/.local/bin"

CMD ["/bin/bash", "-c", "cd .."]
