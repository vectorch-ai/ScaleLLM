#!/bin/bash

set -ex

[ -n "$CMAKE_VERSION" ]

# Uninstall cmake package if it exists
command -v pip3 >/dev/null && pip3 uninstall -y cmake

# Remove existing CMake installation
rm -f /usr/local/bin/cmake

path="v${CMAKE_VERSION}"
ARCH=$(uname -m)
file="cmake-${CMAKE_VERSION}-linux-${ARCH}.tar.gz"

# Download and install specific CMake version in /usr/local
pushd /tmp
wget -q "https://github.com/Kitware/CMake/releases/download/${path}/${file}"
tar -C /usr/local --strip-components 1 --no-same-owner -zxf ${file}
rm -f cmake-*.tar.gz
popd