#!/bin/bash

# Change `umask` to be able to write files and directory with 'r+w' as group rights (else just 'r'):
# - New directories will have permissions 775 (rwxrwxr-x)
# - New files will have permissions 664 (rw-rw-r--)
umask 0002
# Activate the virtual environment
source /home/romi/venv/bin/activate
# Execute whatever command was passed to the container
/bin/bash -c "$@"