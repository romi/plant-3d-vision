#!/bin/bash

# Change `umask` to be able to write files and directory with 'r+w' as group rights (else just 'r'):
# - New directories will have permissions 775 (rwxrwxr-x)
# - New files will have permissions 664 (rw-rw-r--)
umask 0002

# Activate the virtual environment
source "/home/${USER_NAME}/venv/bin/activate"

# Define the source code mount point
SOURCE_DIR="/mnt_src"
INSTALL_DIR="/home/${USER_NAME}/plant-3d-vision"

# Check if source code is mounted and not empty
if [ -d "$SOURCE_DIR" ] && [ "$(ls -A "$SOURCE_DIR")" ]; then
    echo "Copying source code from ${SOURCE_DIR} to ${INSTALL_DIR} ..."
    # Remove previous installation if it exists
    if [ -d "${INSTALL_DIR}" ]; then
        rm -rf "${INSTALL_DIR}"
    fi
    mkdir -p "${INSTALL_DIR}"
    cp -R "${SOURCE_DIR}/." "${INSTALL_DIR}/."

    echo "Installing plant-3d-vision sources and dependencies..."
    cd ${INSTALL_DIR}
    # Run installation script
    # The --no-env flag prevents creating a new venv since we're already in one
    bash install.sh --no-env --update-tools

    # Copy the Resnet model to the testdata directory
    cp "/home/${USER_NAME}/Resnet_896_896_epoch50.pt" "/home/${USER_NAME}/plant-3d-vision/tests/testdata/models/models/"

    echo "Installation complete!"
else
    echo "Warning: Source code not found or empty at $SOURCE_DIR"
    echo "Please mount your repository with: -v \$(pwd):/mnt_src"
fi

# Move to working directory
cd /home/${USER_NAME}
# Execute whatever command was passed to the container
exec /bin/bash -c "$@"