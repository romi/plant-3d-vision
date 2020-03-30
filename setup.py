from setuptools import setup, find_packages

opts = dict(
    name='romiscan',
    packages=find_packages(),
    scripts=[],
    author='Timothée Wintz',
    author_email='timothee@timwin.fr',
    description='A plant scanner',
    long_description='',
    zip_safe=False,
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    install_requires=[
        'appdirs',
        'toml',
        'tqdm',
        'romidata',
        'imageio',
        'opencv-python',
        'luigi>=2.8.11',
        'pybind11',
        'colorlog',
        'scikit-image',
        'open3d==0.9',
        'pywavefront',
        'trimesh'
    ],
    # If True, include any files specified by your MANIFEST.in:
    include_package_data=True
)

if __name__ == '__main__':
    setup(**opts)
