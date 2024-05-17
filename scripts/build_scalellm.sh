#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd $PROJECT_ROOT

# build
cmake -G Ninja -S . -B build -DCMAKE_CUDA_ARCHITECTURES="80;89;90"
cmake --build build --target scalellm --config Release -j$(nproc)

# install
cmake --install build --prefix ./app