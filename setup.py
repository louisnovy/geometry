import os
import re
import sys
import platform
import subprocess
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion

__version__ = "0.0.0"


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir="", exclude_arch=False):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)
        self.exclude_arch = exclude_arch


class CMakeBuild(build_ext):
    def run(self):
        # Check that cmake is installed
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        # If you're Windows, you need a good enough version of Cmake
        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")
        # Call to build (see below)
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        cfg = 'Release'
        build_args = ['--config', cfg]

        # Setting up call to cmake, platform-dependent
        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            if not ext.exclude_arch:
                if sys.maxsize > 2 ** 32:
                    cmake_args += ['-A', 'x64']
                else:
                    cmake_args += ['-A', 'Win32']
                build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']

        if self.distribution.verbose > 0:
            cmake_args += ['-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''), self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        if (self.distribution.verbose > 0):
            print("Running cmake configure command: " + " ".join(['cmake', ext.sourcedir] + cmake_args))
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)

        if (self.distribution.verbose > 0):
            print("Running cmake build command: " + " ".join(['cmake', '--build', '.'] + build_args))
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)


def main():
    if "--exclude-arch" in sys.argv:
        exclude_arch = True
        sys.argv.remove('--exclude-arch')
    else:
        exclude_arch = False

    setup(
        name="geometry",
        version=__version__,
        author="Louis Novy",
        author_email="novylouis@gmail.com",
        install_requires=["numpy", "scipy", "xxhash", "DracoPy"],
        extras_require={"testing": ["pytest"], "dev": ["pytest", "pre-commit"]},
        packages=find_packages("src"),
        package_dir={"": "src"},
        test_suite="test",
        python_requires=">=3.8",
        ext_modules=[CMakeExtension(".", exclude_arch=exclude_arch)],
        cmdclass=dict(build_ext=CMakeBuild),
    )


if __name__ == "__main__":
    main()
