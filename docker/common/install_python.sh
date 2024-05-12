#!/bin/bash

set -ex

PYTHON_VERSION="$1"
shift

NO_RC_PYTHON_VERSION="${PYTHON_VERSION%rc*}"

url="https://www.python.org/ftp/python/${NO_RC_PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz"

pushd /tmp
wget "$url"
tar xvzf "Python-${PYTHON_VERSION}.tgz"
cd "Python-${PYTHON_VERSION}"
./configure \
  --with-lto \
  --enable-optimizations \
  --enable-shared \
  --prefix=/opt/python/${PYTHON_VERSION} \
  LDFLAGS=-Wl,-rpath=/opt/python/${PYTHON_VERSION}/lib

make -j$(nproc) install
cp /opt/python/${PYTHON_VERSION}/bin/pip3 /opt/python/${PYTHON_VERSION}/bin/pip
ln -s /opt/python/${PYTHON_VERSION}/bin/python3 /opt/python/${PYTHON_VERSION}/bin/python

rm -rf "Python-${PYTHON_VERSION}"
popd

