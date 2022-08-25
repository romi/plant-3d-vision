from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

opts = dict(
    name='plant3dvision',
    packages=find_packages(),
    version='0.12.99',
    scripts=[
        'bin/colmap_poses',
        'bin/create_charuco_board',
        'bin/npz_volume_viewer',
        'bin/robustness_evaluation',
    ],
    author='Timothée Wintz',
    author_email='timothee@timwin.fr',
    description='A plant 3d vision tool',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://docs.romi-project.eu/Scanner/home/",
    zip_safe=False,
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    install_requires=[],  # see `requirements.txt`
    python_requires='>=3.8',
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)"
    ],
    # If True, include any files specified by your MANIFEST.in:
    include_package_data=True
)

if __name__ == '__main__':
    setup(**opts)