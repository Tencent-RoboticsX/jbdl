#!/usr/bin/env python

import codecs
import os
import pathlib
import subprocess

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

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
        pathlib.Path(install_dir).mkdir(exist_ok=True)
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
        if os.environ.get("KEPLER_JAX_CUDA", "no").lower() == "yes":
            cmake_args.append("-DKEPLER_JAX_CUDA=yes")

        pathlib.Path(self.build_temp).mkdir(exist_ok=True)
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
        target_name = ext.name.split(".")[-1]
        subprocess.check_call(
            ["cmake", "--build", ".", "--target", target_name],
            cwd=self.build_temp,
        )
        print() # Add an empty line for cleaner output


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

if os.environ.get("KEPLER_JAX_CUDA", "no").lower() == "yes":
    extensions.append(
        Extension(
            "kepler_jax.gpu_ops",
            [
                "src/kepler_jax/src/gpu_ops.cc",
                "src/kepler_jax/src/cuda_kernels.cc.cu",
            ],
        )
    )


setup(
    name="jbdl",
    version="0.1.0",
    author="mikechzhou",
    author_email="mikechzhou@tencent.com",
    url="https://git.woa.com/AgentLearningRobotX/jbdl",
    license="MIT",
    description=(
        "A simple demonstration of how you can extend JAX with custom C++ and "
        "CUDA ops"
    ),
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    python_requires='>=3.6, <4',
    packages=find_packages("src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=['numpy', 'wheel',"jax", "jaxlib", "matplotlib"],
    extras_require={"test": "pytest"},
    ext_modules=extensions,
    cmdclass={"build_ext": CMakeBuildExt},
)
