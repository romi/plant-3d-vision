from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

opts = dict(
    name='romiscan',
    packages=find_packages(),
    version='0.8.99',
    scripts=[
        'bin/romi_run_task',
        'bin/print_task_info',
        'bin/robustness_comparison',
    ],
    author='Timothée Wintz',
    author_email='timothee@timwin.fr',
    description='A plant scanner',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://docs.romi-project.eu/Scanner/home/",
    zip_safe=False,
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    install_requires=[],  # see `requirements.txt`
    python_requires='>=3.6',
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
