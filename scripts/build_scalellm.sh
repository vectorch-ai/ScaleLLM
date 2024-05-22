#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd $PROJECT_ROOT

# zero out ccache if ccache is installed
command -v ccache >/dev/null && ccache -z

# build
cmake -G Ninja -S . -B build -DCMAKE_CUDA_ARCHITECTURES="80;89;90"
cmake --build build --target scalellm --config Release -j$(nproc)

# show ccache statistics if ccache is installed
command -v ccache >/dev/null && ccache -vs

# install
cmake --install build --prefix ./app