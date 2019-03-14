import shlex
from subprocess import check_output

GIT_HEAD_REV = check_output(shlex.split('git rev-parse --short HEAD')).strip().decode()

__version__ = '0.4.dev0+' + GIT_HEAD_REV
