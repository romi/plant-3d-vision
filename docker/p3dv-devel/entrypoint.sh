#!/bin/bash

# Change `umask` to be able to write files and directory with 'r+w' as group rights (else just 'r'):
# - New directories will have permissions 775 (rwxrwxr-x)
# - New files will have permissions 664 (rw-rw-r--)
umask 0002

# Activate the virtual environment
source "/home/${USER_NAME}/venv/bin/activate"

# Define the source code mount point
SOURCE_DIR="${SOURCE_DIR:-/workspace}"
INSTALL_DIR="/home/${USER_NAME}/plant-3d-vision"

NO_COPY="false"
# Disable source code copy if the same directory is used
if [ "${SOURCE_DIR}" == "${INSTALL_DIR}" ]; then
  NO_COPY="true"
fi

# Check if source code is mounted
if [ -d "$SOURCE_DIR" ]; then
    echo "Source code detected at $SOURCE_DIR"

    if [ "${NO_COPY}" = "true" ]; then
        echo "Variable NO_COPY is 'true', skipping copy of source code..."
    else
        echo "Copying source code from ${SOURCE_DIR} to ${INSTALL_DIR} ..."
        mkdir -p "${INSTALL_DIR}"
        cp -R "${SOURCE_DIR}/." "${INSTALL_DIR}/."
    fi

    echo "Installing/updating plant-3d-vision from ${INSTALL_DIR}..."
    cd ${INSTALL_DIR}
    # Run installation script
    # The --no-env flag prevents creating a new venv since we're already in one
    bash install.sh --no-env --update-tools

    # Download the trained CNN model if needed
    if [ -f "./get_model.sh" ]; then
        ./get_model.sh
    fi

    echo "Installation complete!"
else
    echo "Warning: Source code not found at $SOURCE_DIR"
    echo "Please mount your repository with: -v \$(pwd):/workspace"
fi

# Move to working directory
cd /home/${USER_NAME}
# Execute whatever command was passed to the container
exec /bin/bash -c "$@"