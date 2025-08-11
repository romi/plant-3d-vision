#!/bin/bash

# Change `umask` to be able to write files and directory with 'r+w' as group rights (else just 'r'):
umask 0002
# Activate the virtual environment
source /home/romi/venv/bin/activate
# Execute whatever command was passed to the container
/bin/bash -c "$@"