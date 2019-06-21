#!/usr/bin/env python
import tempfile
import os
import subprocess
from shutil import copyfile
from distutils.sysconfig import get_python_lib, get_python_inc
import site
import glob
import sysconfig
import sys

cur_dir = os.getcwd()
try:
    with tempfile.TemporaryDirectory() as tmpdir:
        python_lib_so = os.path.join(sysconfig.get_config_var('LIBDIR'), sysconfig.get_config_var('PY3LIBRARY'))
        python_include = sysconfig.get_config_var('INCLUDEDIR')
        python_exec = sys.executable
        os.chdir(tmpdir)
        subprocess.check_call(['git', 'clone', 'https://github.com/IntelVCL/Open3D'])
        os.chdir("Open3D")
        subprocess.check_call(['git', 'submodule', 'update',  '--init', '--recursive'])
        os.makedirs('build')
        os.chdir('build')
        subprocess.check_call(['cmake', '..', '-DPYTHON_EXECUTABLE=%s'%python_exec, '-DPYTHON_LIBRARY=%s'%python_lib_so, '-DPYTHON_INCLUDE=%s'%python_include])
        subprocess.check_call(['make'])
        python_lib_path = get_python_lib()
        for x in glob.glob('lib/Python/*.so'):
            print("Installing %s"%x)
            fname = os.path.basename(x)
            if hasattr(sys, 'real_prefix'):
                copyfile(x, os.path.join(python_lib_path, fname))
            else:
                copyfile(x, os.path.join(site.USER_SITE, fname))

except Exception as e:
    raise e
finally:
    os.chdir(cur_dir)


