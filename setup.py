#!/usr/bin/env python3

import io
import os
import re
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import List
from jinja2 import Template

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


def use_cxx11_abi():
    try:
        import torch

        return torch._C._GLIBCXX_USE_CXX11_ABI
    except ImportError:
        return False


def get_torch_root():
    try:
        import torch

        return str(Path(torch.__file__).parent)
    except ImportError:
        return None


def get_nccl_root():
    try:
        from nvidia import nccl

        return str(Path(nccl.__file__).parent)
    except ImportError:
        return None


def get_base_dir():
    return os.path.abspath(os.path.dirname(__file__))


def join_path(*paths):
    return os.path.join(get_base_dir(), *paths)


def extract_version():
    # first read from environment variable
    version = os.getenv("SCALELLM_VERSION")
    if not version:
        # then read from version file
        with open("version.txt", "r") as f:
            version = f.read().strip()

    # strip the leading 'v' if present
    if version and version.startswith("v"):
        version = version[1:]

    if not version:
        raise RuntimeError("Version is not set")
    return version


def get_scalellm_version():
    version = extract_version()
    version_suffix = os.getenv("SCALELLM_VERSION_SUFFIX")
    if version_suffix:
        version += version_suffix
    return version


def gen_version_file(version):
    # read the template file
    with open("scalellm/version.py.jinja", "r") as fin:
        template_str = fin.read()
    # render the template
    rendered = Template(template_str).render(
        {
            "VERSION": version,
        }
    )
    # write the rendered content to version.py
    with open("scalellm/version.py", "w") as fout:
        fout.write(rendered)


def read_readme() -> str:
    p = join_path("README.md")
    if os.path.isfile(p):
        return io.open(p, "r", encoding="utf-8").read()
    else:
        return ""


def read_requirements() -> List[str]:
    file = join_path("requirements.txt")
    with open(file) as f:
        return f.read().splitlines()


# ---- cmake extension ----
def get_cmake_dir():
    plat_name = sysconfig.get_platform()
    python_version = sysconfig.get_python_version().replace(".", "")
    dir_name = f"cmake.{plat_name}-{sys.implementation.name}-{python_version}"
    cmake_dir = Path(get_base_dir()) / "build" / dir_name
    cmake_dir.mkdir(parents=True, exist_ok=True)
    return cmake_dir


# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
# If you need multiple extensions, see scikit-build.
class CMakeExtension(Extension):
    def __init__(self, name: str, path: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())
        self.path = path


class CMakeBuild(build_ext):
    user_options = build_ext.user_options + [
        ("base-dir=", None, "base directory of ScaleLLM project"),
    ]

    def initialize_options(self):
        build_ext.initialize_options(self)
        self.base_dir = get_base_dir()

    def finalize_options(self):
        build_ext.finalize_options(self)

    def run(self):
        # check if cmake is installed
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        match = re.search(
            r"version\s*(?P<major>\d+)\.(?P<minor>\d+)([\d.]+)?", out.decode()
        )
        cmake_major, cmake_minor = int(match.group("major")), int(match.group("minor"))
        if (cmake_major, cmake_minor) < (3, 18):
            raise RuntimeError("CMake >= 3.18.0 is required")

        # build extensions
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext: CMakeExtension):
        ninja_dir = shutil.which("ninja")
        # the output dir for the extension
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.path)))

        # create build directory
        os.makedirs(self.build_temp, exist_ok=True)

        # Using this requires trailing slash for auto-detection & inclusion of
        # auxiliary "native" libs

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        build_type = "Debug" if debug else "Release"

        # python directories
        cuda_architectures = "80;89;90"
        cmake_args = [
            "-G",
            "Ninja",  # Ninja is much faster than make
            f"-DCMAKE_MAKE_PROGRAM={ninja_dir}",  # pass in the ninja build path
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            "-DUSE_CCACHE=ON",  # use ccache if available
            "-DUSE_MANYLINUX:BOOL=ON",  # use manylinux settings
            f"-DPython_EXECUTABLE:FILEPATH={sys.executable}",
            f"-DCMAKE_CUDA_ARCHITECTURES={cuda_architectures}",
            f"-DCMAKE_BUILD_TYPE={build_type}",  # not used on MSVC, but no harm
        ]

        # Adding CMake arguments set as environment variable
        # (needed e.g. to build for ARM OSx on conda-forge)
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        # check if torch binary is built with cxx11 abi
        if use_cxx11_abi():
            cmake_args += ["-DUSE_CXX11_ABI=ON"]
        else:
            cmake_args += ["-DUSE_CXX11_ABI=OFF"]

        build_args = ["--config", build_type]
        max_jobs = os.getenv("MAX_JOBS", str(os.cpu_count()))
        build_args += ["-j" + max_jobs]

        env = os.environ.copy()
        LIBTORCH_ROOT = get_torch_root()
        if LIBTORCH_ROOT is None:
            raise RuntimeError(
                "Please install requirements first, pip install -r requirements.txt"
            )
        env["LIBTORCH_ROOT"] = LIBTORCH_ROOT

        NCCL_ROOT = get_nccl_root()
        if NCCL_ROOT is not None:
            env["NCCL_ROOT"] = NCCL_ROOT
            env["NCCL_VERSION"] = "2"

        # print cmake args
        print("CMake Args: ", cmake_args)
        print("Env: ", env)

        cmake_dir = get_cmake_dir()
        subprocess.check_call(
            ["cmake", self.base_dir] + cmake_args, cwd=cmake_dir, env=env
        )

        # add build target to speed up the build process
        build_args += ["--target", ext.name]
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=cmake_dir)


# The information here can also be placed in setup.cfg - better separation of
# logic and declaration, and simpler if you include description/version in a file.
scalellm_package_data = []

if __name__ == "__main__":
    version = get_scalellm_version()
    # generate version file
    gen_version_file(version)

    setup(
        name="scalellm",
        version=version,
        license="Apache 2.0",
        author="ScaleLLM Team",
        description="A high-performance inference system for large language models.",
        long_description=read_readme(),
        long_description_content_type="text/markdown",
        url="https://github.com/vectorch-ai/ScaleLLM",
        project_url={
            "Homepage": "https://github.com/vectorch-ai/ScaleLLM",
        },
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Intended Audience :: Developers",
            "Operating System :: POSIX",
            "License :: OSI Approved :: Apache Software License",
        ],
        packages=["scalellm", "scalellm/serve", "scalellm/_C", "examples"],
        ext_modules=[CMakeExtension("_C", "scalellm/")],
        cmdclass={"build_ext": CMakeBuild},
        zip_safe=False,
        package_data={
            "scalellm": scalellm_package_data,
        },
        python_requires=">=3.8",
        install_requires=read_requirements(),
    )
