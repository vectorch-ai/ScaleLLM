# How to Contribute

Thanks for your interest in contributing to this project! We welcome contributions from anyone, and are grateful for even the smallest of fixes! There are several ways to help out:

* **Report bugs**: If you notice any bugs, please [file an issue]
* **Fix bugs**: If you can fix any of the open issues, please [submit a pull request]
* **Add features**: If you have a cool idea for a new feature, please [file an issue]
* **Write tests**: If you can write tests, please [submit a pull request]
* **Write documentation**: If you can improve the documentation, please [submit a pull request]

## Getting Started

Let's assume you have a Linux machine with Nvidia GPU. You can follow the steps below to get started.

### Install build tools

Before you start building the project, you need to install the build tools. You can install them by running the following command. Please note it is not a complete list. You may need to install more depending on your system.
[to be updated]

``` bash
sudo apt-get install build-essential ninja-build cmake python3-dev zip pkg-config libssl-dev libboost-all-dev wget curl
```
### Install cuda toolkit and nccl

This project depends on libtorch 2.1, which is compatible with both cuda 12.1 and cuda 11.8. Consequently, it's essential to install either CUDA Toolkit 12.1 or CUDA Toolkit 11.8.

To download the necessary CUDA Toolkit, please visit NVIDIA's [official website](https://developer.nvidia.com/cuda-downloads). Additionally, you have the option to use the following script for the installation process.

#### install cuda 12.1 with nccl 2.17
``` bash
# install cuda 12.1
wget -q https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run --no-drm --no-man-page --override --toolkit --silent
# install nccl 2.17
wget -q https://developer.download.nvidia.com/compute/redist/nccl/v2.17.1/nccl_2.17.1-1+cuda12.1_x86_64.txz
tar xf nccl_2.17.1-1+cuda12.1_x86_64.txz
sudo cp -a nccl_2.17.1-1+cuda12.1_x86_64/include/* /usr/local/cuda-12.1/include/
sudo cp -a nccl_2.17.1-1+cuda12.1_x86_64/lib/* /usr/local/cuda-12.1/lib64/
sudo ldconfig
```

#### install cuda 11.8 with nccl 2.15
``` bash
# install cuda 11.8
wget -q https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run --no-drm --no-man-page --override --toolkit --silent
# install nccl 2.15
wget -q https://developer.download.nvidia.com/compute/redist/nccl/v2.15.5/nccl_2.15.5-1+cuda11.8_x86_64.txz
tar xf nccl_2.15.5-1+cuda11.8_x86_64.txz
sudo cp -a nccl_2.15.5-1+cuda11.8_x86_64/include/* /usr/local/cuda-11.8/include/
sudo cp -a nccl_2.15.5-1+cuda11.8_x86_64/lib/* /usr/local/cuda-11.8/lib64/
sudo ldconfig
```

### Install rust

ScaleLLM relies on two open-source projects: [safetensors](https://github.com/huggingface/safetensors/) and [tokenizers](https://github.com/huggingface/tokenizers), both of which are developed in Rust. To successfully build ScaleLLM, installing Rust is a prerequisite. You can install Rust by executing the following command:

``` bash
curl https://sh.rustup.rs -sSf | sh
source "$HOME/.cargo/env"
```

### Enlist code

You can enlist the project by running the following command.

``` bash
git clone --recursive https://github.com/vectorch-ai/ScaleLLM.git
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

## Devel Image

You can use development image which have installed cuda 12.1, nccl 2.17 and list of tools for build and debug.

```
vectorchai/scalellm:devel
```

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
