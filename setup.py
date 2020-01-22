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
    install_requires=[
        'romidata',
        'imageio',
        'opencv-python',
        'luigi',
        'pybind11',
        'colorlog',
        'scikit-image',
        'open3d'
    ],
    include_package_data=True,
)
