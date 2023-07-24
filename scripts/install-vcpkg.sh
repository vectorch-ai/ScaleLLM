#!/bin/bash

# Install Vcpkg into .vcpkg folder

set -e
cd $(dirname "$0")/..
VCPKG_DIR="$(pwd)/.vcpkg"
echo "VCPKG_DIR: $VCPKG_DIR"

if [[ -d "$VCPKG_DIR" ]]; then
    pushd "$VCPKG_DIR"
    ./bootstrap-vcpkg.sh --disableMetrics
    popd
else
    # install Vcpkg
    pushd "$(pwd)"
    git clone -b 2023.06.20 https://github.com/Microsoft/vcpkg.git .vcpkg
    cd .vcpkg
    ./bootstrap-vcpkg.sh --disableMetrics
    popd
fi

# Create binary cache folder
VCPKG_CACHE_DIR=${VCPKG_DEFAULT_BINARY_CACHE:-$VCPKG_DIR/bincache}
mkdir -p "$VCPKG_CACHE_DIR"