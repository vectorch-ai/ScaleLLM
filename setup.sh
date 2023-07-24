#!/bin/bash

# Install Vcpkg into .vcpkg folder
./scripts/install-vcpkg.sh

# download torchlib and put into libtorch folder
LIBTORCH_URL="https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcu118.zip"
if [ -d "libtorch" ]; then
    echo "libtorch folder already exist"
else 
    LIBTORCH=libtorch.zip
    wget -O $LIBTORCH $LIBTORCH_URL
    unzip $LIBTORCH && rm $LIBTORCH
fi
