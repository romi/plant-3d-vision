#!/bin/bash

# Change `umask` to be able to write files and directory with 'r+w' as group rights (else just 'r'):
umask 0002
/bin/bash -c "$@"