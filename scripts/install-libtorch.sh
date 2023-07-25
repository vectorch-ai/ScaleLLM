#!/bin/bash

# download torchlib and put into libtorch folder
if lspci | grep -i -e nvidia > /dev/null; then
    echo "GPU found, downloading libtorch 2.0 with cuda 11.8"
    LIBTORCH_URL="https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcu118.zip"
else
    echo "No GPU found, downloading libtorch 2.0 with cpu"
    LIBTORCH_URL="https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.7.1%2Bcpu.zip"
fi

if [ -d "libtorch" ]; then
    echo "libtorch folder already exist"
else 
    LIBTORCH=libtorch.zip
    wget -O $LIBTORCH $LIBTORCH_URL
    unzip $LIBTORCH && rm $LIBTORCH
fi
