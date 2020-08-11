FROM roboticsmicrofarms/romiscan_base-dev:0.7

ARG ROMISCAN_BRANCH=dev

RUN git clone --branch $ROMISCAN_BRANCH https://github.com/romi/romiscan && \
    git clone https://github.com/romi/romidata && \
    git clone https://github.com/romi/romiscanner && \
    python -m pip install -e ./romidata/ && \
    python -m pip install -e ./romiscanner/ && \
    python -m pip install -r ./romiscan/requirements.txt && \
    python -m pip install -e ./romiscan/ && \
    pip cache purge

ENV DB_LOCATION="/home/${USER_NAME}/db_test"
ENV PATH=$PATH:"/home/${USER_NAME}/.local/bin"

CMD ["/bin/bash", "-c", "cd .."]
