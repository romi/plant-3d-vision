#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import find_packages
from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

opts = dict(
    name='plant3dvision',
    version='0.13',
    description='3D reconstruction & quantification of single potted plants from RGB scans.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Timothée Wintz',
    author_email='timothee@timwin.fr',
    maintainer='Jonathan Legrand',
    maintainer_email='jonathan.legrand@ens-lyon.fr',
    url="https://docs.romi-project.eu/Scanner/home/",
    download_url='',
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)"
    ],
    license="LGPL-3.0",
    license_files='LICENSE',
    keywords=['ROMI', 'photogrammetry', 'COLMAP'],
    platforms=['linux'],
    scripts=[
        'bin/check_measures',
        'bin/colmap_poses',
        'bin/create_charuco_board',
        'bin/volume_viewer',
        'bin/robustness_evaluation',
        'bin/voronoi_texture_generator',
    ],
    zip_safe=False,
    install_requires=[],  # see `requirements.txt`
    python_requires='>=3.8',
    include_package_data=True,  # if `True`, include any files specified by your `MANIFEST.in`
)

if __name__ == '__main__':
    setup(**opts)
