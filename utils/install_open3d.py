#!/usr/bin/env python
import tempfile
import os
import subprocess
from shutil import copyfile
from distutils.sysconfig import get_python_lib
import site
import glob

cur_dir = os.getcwd()
try:
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        subprocess.check_call(['git', 'clone', 'https://github.com/IntelVCL/Open3D'])
        os.chdir("Open3D")
        subprocess.check_call(['git', 'submodule', 'update',  '--init', '--recursive'])
        os.makedirs('build')
        os.chdir('build')
        subprocess.check_call(['cmake', '..'])
        subprocess.check_call(['make'])
        python_lib_path = get_python_lib()
        for x in glob.glob('lib/Python/*.so'):
            print("Installing %s"%x)
            fname = os.path.basename(x)
            if site.ENABLE_USER_SITE:
                copyfile(x, os.path.join(site.getuserbase(), fname))
            else:
                copyfile(x, python_lib_path)

except Exception as e:
    raise e
finally:
    os.chdir(cur_dir)


