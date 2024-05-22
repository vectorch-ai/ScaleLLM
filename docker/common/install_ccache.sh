#!/bin/bash

set -ex

[ -n "$CCACHE_VERSION" ]

ARCH=$(uname -m)
url=https://github.com/ccache/ccache/releases/download/v${CCACHE_VERSION}/ccache-${CCACHE_VERSION}-linux-${ARCH}.tar.xz

pushd /tmp
curl -L "$url" | xz -d | tar -x
cp ./ccache-${CCACHE_VERSION}-linux-x86_64/ccache /usr/bin/ccache
popd

# set max cache size to 25GiB
/usr/bin/ccache -M 25Gi