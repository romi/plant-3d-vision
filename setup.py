import os
import re
import sys
import platform
import subprocess
import glob
import romiscan
import site
import pathlib
from distutils.sysconfig import get_python_inc

from shutil import copyfile
from distutils.version import LooseVersion
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

dir_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
thirdparty_deps = pathlib.Path(os.path.join(dir_path, "thirdparty")).as_uri()
print(thirdparty_deps)


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=None):
        Extension.__init__(self, name, sources=[])
        if sourcedir is None:
            sourcedir = name.replace('.', '/')
        self.reldir = sourcedir
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)',
                                                   out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        print("path = ")
        print(self.get_ext_fullpath(ext.name))
        extdir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name)))
        import pybind11
        print("pybind11 = %s"%pybind11.get_include())
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable,
                      '-DPYTHON_INCLUDE_DIR=' + get_python_inc(),
                      '-DPYBIND11_INCLUDE_DIR=' + pybind11.get_include(user=site.ENABLE_USER_SITE)]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(
                cfg.upper(),
                extdir)]
            if sys.maxsize > 2 ** 32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\" -std=c++11'.format(
            env.get('CXXFLAGS', ''),
            self.distribution.get_version())
        tempdir = os.path.join(self.build_temp, ext.reldir)
        if not os.path.exists(tempdir):
            os.makedirs(tempdir)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args,
                              cwd=tempdir, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args,
                              cwd=tempdir)
        print()  # Add an empty line for cleaner output

lt_link = "https://github.com/romi/lettucethink-python/tarball/dev"
install_requires=[
        'numpy',
        'pyopencl',
        'scikit-image',
        'networkx',
        'Flask',
        'flask-restful',
        'imageio',
        'luigi',
        'pybind11',
        'requests',
        'mako', 
        'tqdm'
    ]

version_minor = sys.version_info.minor

if version_minor >= 7:
    install_requires.append('lettucethink @ %s'%lt_link)
else:
    install_requires.append('lettucethink')

s = setup(
    name='romiscan',
    version=romiscan.__version__,
    packages=find_packages(),
    scripts=['bin/run-scan', 'bin/sync-scans'],
    author='Timothée Wintz',
    author_email='timothee@timwin.fr',
    description='A plant scanner',
    long_description='',
    ext_modules=[CMakeExtension('romiscan.cgal'), CMakeExtension('romiscan.pyceres')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    install_requires=install_requires,
    include_package_data=True,
    dependency_links = ['%s#egg=lettucethink-0'%lt_link]
)
