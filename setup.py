from setuptools import setup, find_packages

opts = dict(
    name='romiscan',
    packages=find_packages(),
    version='0.6.99',
    scripts=[
        'bin/romi_run_task',
    ],
    author='Timothée Wintz',
    author_email='timothee@timwin.fr',
    description='A plant scanner',
    long_description='',
    zip_safe=False,
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    install_requires=[],
    # If True, include any files specified by your MANIFEST.in:
    include_package_data=True
)

if __name__ == '__main__':
    setup(**opts)
