#!/bin/bash

set -ex

[ -n "$CMAKE_VERSION" ]

# Remove system cmake install so it won't get used instead
ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')
case "$ID" in
  ubuntu)
    apt-get remove cmake -y
    ;;
  centos)
    rm -f /usr/local/bin/cmake
    ;;
  almalinux)
    rm -f /usr/local/bin/cmake
    ;;
  *)
    echo "Unable to determine OS..."
    exit 1
    ;;
esac

path="v${CMAKE_VERSION}"
file="cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz"

# Download and install specific CMake version in /usr/local
pushd /tmp
wget -q "https://github.com/Kitware/CMake/releases/download/${path}/${file}"
tar -C /usr/local --strip-components 1 --no-same-owner -zxf ${file}
rm -f cmake-*.tar.gz
popd