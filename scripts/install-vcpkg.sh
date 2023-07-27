#!/bin/bash
set -e

# Install Vcpkg into $(PARENT_DIR)/.vcpkg folder

# Determine script's parent directory
PARENT_DIR=$(dirname "$0")/..

VCPKG_DIR="${PARENT_DIR}/.vcpkg"
echo "VCPKG_DIR: $VCPKG_DIR"

if [[ -d "$VCPKG_DIR" ]]; then
    echo ".vcpkg folder already exists."
else
    echo ".vcpkg folder does not exist. Cloning Vcpkg into $VCPKG_DIR"
    # install Vcpkg
    git clone -b 2023.06.20 https://github.com/Microsoft/vcpkg.git ${VCPKG_DIR}
fi

# bootstrap Vcpkg
${VCPKG_DIR}/bootstrap-vcpkg.sh --disableMetrics

# Create binary cache folder if it does not exist
VCPKG_CACHE_DIR=${VCPKG_DEFAULT_BINARY_CACHE:-$VCPKG_DIR/bincache}
mkdir -p "$VCPKG_CACHE_DIR"
