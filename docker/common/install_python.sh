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

# Extract major and minor version number
MAJOR=$(echo "${PYTHON_VERSION}" | cut -d . -f 1)
MINOR=$(echo "${PYTHON_VERSION}" | cut -d . -f 2)

INSTALL_FOLDER="/opt/python/cp${MAJOR}${MINOR}-cp${MAJOR}${MINOR}"

./configure \
  --enable-shared \
  --enable-ipv6 \
  --prefix=${INSTALL_FOLDER} \
  LDFLAGS=-Wl,-rpath=${INSTALL_FOLDER}/lib,--disable-new-dtags

make -j$(nproc) install
# upgrade pip, setuptools and wheel
${INSTALL_FOLDER}/bin/python3 -m pip install --upgrade pip setuptools wheel
# create symlinks
cp ${INSTALL_FOLDER}/bin/pip3 ${INSTALL_FOLDER}/bin/pip
ln -s ${INSTALL_FOLDER}/bin/python3 ${INSTALL_FOLDER}/bin/python

rm -rf "Python-${PYTHON_VERSION}"
popd

