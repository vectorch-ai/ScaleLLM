#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd $PROJECT_ROOT

# set max cache size to 25GiB
command -v ccache >/dev/null && ccache -M 25Gi

# zero out ccache if ccache is installed
command -v ccache >/dev/null && ccache -z

# build
cmake -G Ninja -S . -B build -DCMAKE_CUDA_ARCHITECTURES="80;89;90a;100a;120a"
cmake --build build --target scalellm --config Release -j$(nproc)

# show ccache statistics if ccache is installed
command -v ccache >/dev/null && ccache -vs

# install
cmake --install build --prefix ./app
