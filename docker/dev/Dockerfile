FROM roboticsmicrofarms/romiscan_base:latest

ARG ROMIDATA_BRANCH=dev
ARG ROMISCAN_BRANCH=dev
ARG ROMISCANNER_BRANCH=dev_lyon

RUN git clone https://github.com/romi/romidata --branch $ROMIDATA_BRANCH && \
    git clone https://github.com/romi/romiscan --branch $ROMISCAN_BRANCH && \
    git clone https://github.com/romi/romiscanner --branch $ROMISCANNER_BRANCH && \
    python -m pip install -e ./romidata/ --no-cache-dir && \
    python -m pip install -e ./romiscanner/ --no-cache-dir && \
    python -m pip install -r ./romiscan/requirements.txt --no-cache-dir && \
    python -m pip install -e ./romiscan/ --no-cache-dir

CMD ["/bin/bash", "-c", "romi_run_task -h"]