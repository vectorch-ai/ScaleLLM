# How to Contribute

Thanks for your interest in contributing to this project! We welcome contributions from anyone, and are grateful for even the smallest of fixes! There are several ways to help out:

* **Report bugs**: If you notice any bugs, please [file an issue]
* **Fix bugs**: If you can fix any of the open issues, please [submit a pull request]
* **Add features**: If you have a cool idea for a new feature, please [file an issue]
* **Write tests**: If you can write tests, please [submit a pull request]
* **Write documentation**: If you can improve the documentation, please [submit a pull request]

## Getting Started

Let's assume you have a Linux machine with Nvidia GPU. You can follow the steps below to get started.

### Install cuda toolkit 11.8

Since this project depends on libtorch 2.0.0, which is built with cuda 11.7 and 11.8. You can download the cuda toolkit from nvidia website <https://developer.nvidia.com/cuda-downloads>. You can leverage the script to install it as well.

``` bash
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run
```

### Install build tools

Before you start building the project, you need to install the build tools. You can install them by running the following command. Please note it is not a complete list. You may need to install more depending on your system.
[to be updated]

``` bash
sudo apt-get install build-essential ninja-build cmake python3-dev
```

### Install rust

You can install rust by running the following command.

``` bash
curl https://sh.rustup.rs -sSf | sh
source "$HOME/.cargo/env"
```

### Enlist code

You can enlist the project by running the following command.

``` bash
git clone --recursive https://github.com/vectorch-ai/LLMInfer.git
```

### Install libtorch 2.0.0

You can download specific version of libtorch from pytorch website. Here is the link: <https://pytorch.org/get-started/locally/>. We are using libtorch 2.0.0 for this project and you can leverage the script to install it.

``` bash
./scripts/install_libtorch.sh
```

### Install vcpkg

Vcpkg helps you manage C and C++ libraries on Windows, Linux and MacOS. This tool and ecosystem are constantly evolving; and we are using it for the development of this project. You can install it by running the following command.

``` bash
./scripts/install_vcpkg.sh
```

### Build

Config and build the project in the build directory.

``` bash
cmake -G Ninja -S . -B build
cmake --build build
```

### Run tests

You can run all tests in build directory by running the following command.

``` bash
cd build && ctest --output-on-failure
``````

## Compiler

You may want to chose different compiler for the project. You can do it by setting the environment variable `CXX` and `CC` before running cmake. For example, if you want to use gcc-12 and g++-12, you can run the following command.

``` bash
CXX=/usr/bin/g++-12 CC=/usr/bin/gcc-12 cmake -G Ninja -S . -B build
```

Following compilers are tested and supported.

* gcc-12
* clang-14

## IDE

Your productivity can be improved by using IDE. We recommend you to use [Visual Studio Code](https://code.visualstudio.com/) with [CMake Tools](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cmake-tools) extension. You can also use [CLion](https://www.jetbrains.com/clion/) if you have a license.

* [Cmake Tools](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cmake-tools) allows you to configure and build your cmake project. It also provides you with the ability to debug your project without adding any extra configuration. All those functionalities are provided by the extension and available in the command palette and bottom bar. (Recommended)
