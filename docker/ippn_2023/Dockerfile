FROM roboticsmicrofarms/plant-3d-vision
LABEL maintainer="Jonathan LEGRAND <jonathan.legrand@ens-lyon.fr>"

ARG HOST_ROMI_DB
ARG HOST_NOTEBOOK

# Install the required dependencies, sub-modules and plant-3d-vision:
RUN python3 -m pip install --user notebook && \
    mkdir ${HOME}/notebooks/ && \
    rm -rf .cache/pip

# Copy the data files:
COPY --chown=${USER_NAME} ${HOST_ROMI_DB}/ ${ROMI_DB}/
# Copy the notebook files:
COPY --chown=${USER_NAME} ${HOST_NOTEBOOK}/ notebooks/

CMD ["/bin/bash", "-c", "jupyter notebook --no-browser --ip 0.0.0.0"]
