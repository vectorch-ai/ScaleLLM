#!/bin/bash
set -e

# Determine script's parent directory
PARENT_DIR=$(dirname "$0")/..

# Check if libtorch directory already exists
if [ -d "${PARENT_DIR}/libtorch" ]; then
    echo "libtorch folder already exists. Exiting to prevent overwrite."
    exit 1
fi

# Determine libtorch version to download
if lspci | grep -i -e nvidia > /dev/null; then
    echo "GPU found, downloading libtorch 2.0 with cuda 11.8"
    LIBTORCH_URL="https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcu118.zip"
else
    echo "No GPU found, downloading libtorch 2.0 with CPU support"
    LIBTORCH_URL="https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.7.1%2Bcpu.zip"
fi

# Download, unzip, and remove the zip file
LIBTORCH=libtorch.zip
wget -O $LIBTORCH $LIBTORCH_URL
unzip $LIBTORCH && rm $LIBTORCH
