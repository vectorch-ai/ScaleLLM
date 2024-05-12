#!/bin/bash

set -ex

install_ubuntu() {  
  # Install common dependencies
  apt-get update
  apt-get install -y --no-install-recommends \
    build-essential \
    ninja-build \
    cmake \
    ccache \
    python3-dev \
    python3-pip \
    zip \
    pkg-config \
    libssl-dev \
    libboost-all-dev \
    software-properties-common \
    curl \
    git \
    wget \
    sudo \
    vim \
    jq \
    libtool \
    unzip \
    gdb

  # Cleanup package manager
  apt-get autoclean && apt-get clean
  rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
}

ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')
case "$ID" in
  ubuntu)
    install_ubuntu
    ;;
  *)
    echo "Unable to determine OS..."
    exit 1
    ;;
esac