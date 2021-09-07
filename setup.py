#!/usr/bin/env python

import codecs
import os
import pathlib
import subprocess
from glob import glob
import shutil
from shutil import copyfile
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
import numpy

HERE = pathlib.Path(__file__).parent.resolve()


def read(*parts):
    with codecs.open(pathlib.Path(HERE).joinpath(*parts), "rb", "utf-8") as f:
        return f.read()


# This custom class for building the extensions uses CMake to compile. You
# don't have to use CMake for this task, but I found it to be the easiest when
# compiling ops with GPU support since setuptools doesn't have great CUDA
# support.
class CMakeBuildExt(build_ext):
    def build_extensions(self):
        # First: configure CMake build
        import platform
        import sys
        import distutils.sysconfig

        import pybind11

        # Work out the relevant Python paths to pass to CMake, adapted from the
        # PyTorch build system
        if platform.system() == "Windows":
            cmake_python_library = "{}/libs/python{}.lib".format(
                distutils.sysconfig.get_config_var("prefix"),
                distutils.sysconfig.get_config_var("VERSION"),
            )
            if not os.path.exists(cmake_python_library):
                cmake_python_library = "{}/libs/python{}.lib".format(
                    sys.base_prefix,
                    distutils.sysconfig.get_config_var("VERSION"),
                )
        else:
            cmake_python_library = "{}/{}".format(
                distutils.sysconfig.get_config_var("LIBDIR"),
                distutils.sysconfig.get_config_var("INSTSONAME"),
            )
        cmake_python_include_dir = distutils.sysconfig.get_python_inc()
        install_dir = pathlib.Path(self.get_ext_fullpath("dummy")).parent.resolve()
        pathlib.Path(install_dir).mkdir(parents=True, exist_ok=True)
        cmake_args = [
            "-DCMAKE_INSTALL_PREFIX={}".format(install_dir),
            "-DPython_EXECUTABLE={}".format(sys.executable),
            "-DPython_LIBRARIES={}".format(cmake_python_library),
            "-DPython_INCLUDE_DIRS={}".format(cmake_python_include_dir),
            "-DCMAKE_BUILD_TYPE={}".format(
                "Debug" if self.debug else "Release"
            ),
            "-DCMAKE_PREFIX_PATH={}".format(pybind11.get_cmake_dir()),
        ]
        if os.environ.get("WITH_JAX_CUDA", "no").lower() == "yes":
            cmake_args.append("-DCUDA_SUPPORT=ON")
            cmake_args.append("-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc")

        pathlib.Path(self.build_temp).mkdir(parents=True, exist_ok=True)
        subprocess.check_call(
            ["cmake", HERE] + cmake_args, cwd=self.build_temp
        )

        # Build all the extensions
        super().build_extensions()

        # Finally run install
        subprocess.check_call(
            ["cmake", "--build", ".", "--target", "install"],
            cwd=self.build_temp,
        )

    def build_extension(self, ext):
        import distutils.sysconfig

        target_name = ext.name.split(".")[-1]
        if target_name == '_osqp':
            cmake_args_osqp = ["-DUNITTESTS=OFF"]
            cmake_args_osqp += ['-G', 'Unix Makefiles']
            cmake_args_osqp += ['-DPYTHON=ON']
            cmake_args_osqp += ['-DCUDA_SUPPORT=ON', '-DDFLOAT=ON', '-DDLONG=OFF']
            cmake_args_osqp += ['-DPYTHON_INCLUDE_DIRS=%s' % distutils.sysconfig.get_python_inc()]
            cmake_args_osqp += ['-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc']

            current_dir = os.getcwd()
            osqp_dir = os.path.join(current_dir, 'src/jbdl/experimental/cuosqp/osqp_sources')
            osqp_ext_dir = os.path.join(current_dir, 'src/jbdl/experimental/cuosqp/extension')
            osqp_build_dir = os.path.join(osqp_dir, 'build')

            if os.path.exists(osqp_build_dir):
                shutil.rmtree(osqp_build_dir)
            os.makedirs(osqp_build_dir)

            subprocess.check_call(
                ["cmake", osqp_dir] + cmake_args_osqp,
                cwd= osqp_build_dir,

            )
            subprocess.check_call(
                ["cmake", "--build", ".", "--target", 'osqpstatic'],
                cwd=osqp_build_dir,
            )

            lib_origin = os.path.join(osqp_build_dir, 'out/libosqp.a')
            lib_destination = os.path.join(osqp_ext_dir, 'src/libosqp.a')
            copyfile(lib_origin, lib_destination)


            build_ext.build_extension(self, ext)

        else:
            subprocess.check_call(
                ["cmake", "--build", ".", "--target", target_name],
                cwd=self.build_temp,
            )

        print()  # Add an empty line for cleaner output



extensions = [
    Extension(
        "jbdl.experimental.math",
        ["src/jbdl/experimental/math_bindings.cpp"],
    ),
    Extension(
        "jbdl.experimental.tools",
        ["src/jbdl/experimental/tools_bindings.cpp"],
    ),
    Extension(
        "jbdl.experimental.qpoases",
        ["src/jbdl/experimental/qpoases_bindings.cpp"],
    ),
    Extension(
        "jbdl.experimental.cpu_ops",
        ["src/jbdl/experimental/cpu_ops_bindings.cpp"],
    ),
]

if os.environ.get("WITH_JAX_CUDA", "no").lower() == "yes":
    define_macros = []
    define_macros += [('PYTHON', None)]

    libraries = ['cublas', 'cusparse', 'cudart']
    libraries += ['rt']

    # Make sure the environment variable CUDA_PATH
    # is set to the CUDA Toolkit install directory.
    CUDA_PATH = '/usr/local/cuda'
    library_dirs = [os.path.join(CUDA_PATH, 'lib64')]

    current_dir = os.getcwd()
    osqp_source_dir = os.path.join('src/jbdl/experimental/cuosqp/osqp_sources')
    osqp_extension_dir = os.path.join(current_dir, 'src/jbdl/experimental/cuosqp/extension')

    include_dirs = [
        os.path.join(osqp_source_dir, 'include'),
        os.path.join(osqp_extension_dir, 'include'),
        numpy.get_include()]

    LIB_NAME = 'libosqp.a'
    extra_objects = [os.path.join(osqp_extension_dir, 'src', LIB_NAME)]

    sources_files = glob(os.path.join(osqp_extension_dir, 'src', '*.c'))

    compile_args = ["-O3"]

    packages = find_packages("src")

    extensions.append(
        Extension('jbdl.experimental.cuosqp._osqp',
                  define_macros=define_macros,
                  libraries=libraries,
                  library_dirs=library_dirs,
                  include_dirs=include_dirs,
                  extra_objects=extra_objects,
                  sources=sources_files,
                  extra_compile_args=compile_args)
        )

    extensions.append(
        Extension(
            "jbdl.experimental.gpu_ops",
            ["src/jbdl/experimental/lcp_gpu/gpu_ops.cc",
             "src/jbdl/experimental/lcp_gpu/kernels.cc.cu", ], )
        )


setup(
    name="jbdl",
    version="0.0.89",
    author="mikechzhou",
    author_email="mikechzhou@tencent.com",
    url="https://git.woa.com/AgentLearningRobotX/jbdl",
    license="MIT",
    description=(
        "A Jax Physical Body Dynamic Library with GPU Support"
        "CUDA ops"
    ),
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    python_requires='>=3.6, <4',
    packages=find_packages("src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=["numpy", "wheel", "jax", "jaxlib", "matplotlib",
                      "chex", "cvxopt", "gym", "pybullet", "meshcat", "urdf_parser_py"],
    extras_require={"test": "pytest"},
    ext_modules=extensions,
    cmdclass={"build_ext": CMakeBuildExt},
)
