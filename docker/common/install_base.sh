#!/bin/bash

set -ex

install_ubuntu() {
  deploy_deps="libffi-dev libbz2-dev libreadline-dev libncurses5-dev libncursesw5-dev libgdbm-dev libsqlite3-dev uuid-dev tk-dev"
  # Install common dependencies
  apt-get update
  apt-get install -y --no-install-recommends \
    ${deploy_deps} \
    build-essential \
    zip \
    pkg-config \
    libssl-dev \
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

install_almalinux() {
  yum -y update
  yum -y install \
    zip \
    wget \
    curl \
    perl \
    sudo \
    vim \
    jq \
    libtool \
    unzip
  
  # Cleanup
  yum clean all
}

ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')
case "$ID" in
  ubuntu)
    install_ubuntu
    ;;
  almalinux)
    install_almalinux
    ;;
  *)
    echo "Unable to determine OS..."
    exit 1
    ;;
esac