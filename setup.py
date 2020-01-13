import subprocess
import romiscan

from shutil import copyfile
from setuptools import setup, Extension, find_packages

s = setup(
    name='romiscan',
    packages=find_packages(),
    scripts=['bin/run-task', 'bin/scanner-rest-api'],
    author='Timothée Wintz',
    author_email='timothee@timwin.fr',
    description='A plant scanner',
    long_description='',
    zip_safe=False,
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    include_package_data=True,
)
