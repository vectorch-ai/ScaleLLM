#!/bin/bash

# Adapted from https://github.com/pytorch/pytorch/blob/main/.ci/docker/common/install_cuda.sh

set -ex

arch_path=''
targetarch=${TARGETARCH:-$(uname -m)}
if [ ${targetarch} = 'amd64' ] || [ "${targetarch}" = 'x86_64' ]; then
  arch_path='x86_64'
else
  arch_path='sbsa'
fi

NVSHMEM_VERSION=3.3.24

function install_cuda {
  version=$1
  runfile=$2
  major_minor=${version%.*}
  rm -rf /usr/local/cuda-${major_minor} /usr/local/cuda
  if [[ ${arch_path} == 'sbsa' ]]; then
      runfile="${runfile}_sbsa"
  fi
  runfile="${runfile}.run"
  wget -q https://developer.download.nvidia.com/compute/cuda/${version}/local_installers/${runfile} -O ${runfile}
  chmod +x ${runfile}
  ./${runfile} --toolkit --silent
  rm -f ${runfile}
  rm -f /usr/local/cuda && ln -s /usr/local/cuda-${major_minor} /usr/local/cuda
}

function install_cudnn {
  cuda_major_version=$1
  cudnn_version=$2
  mkdir tmp_cudnn && cd tmp_cudnn
  # cuDNN license: https://developer.nvidia.com/cudnn/license_agreement
  filepath="cudnn-linux-${arch_path}-${cudnn_version}_cuda${cuda_major_version}-archive"
  wget -q https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-${arch_path}/${filepath}.tar.xz
  tar xf ${filepath}.tar.xz
  cp -a ${filepath}/include/* /usr/local/cuda/include/
  cp -a ${filepath}/lib/* /usr/local/cuda/lib64/
  cd ..
  rm -rf tmp_cudnn
}

function install_nccl {
  nccl_version=$1
  # NCCL license: https://docs.nvidia.com/deeplearning/nccl/#licenses
  # Follow build: https://github.com/NVIDIA/nccl/tree/master?tab=readme-ov-file#build
  git clone -b ${nccl_version} --depth 1 https://github.com/NVIDIA/nccl.git
  cd nccl
  make -j src.build
  cp -a build/include/* /usr/local/cuda/include/
  cp -a build/lib/* /usr/local/cuda/lib64/
  cd ..
  rm -rf nccl
}

function install_cusparselt {
  cusparselt_version=$1
  # cuSPARSELt license: https://docs.nvidia.com/cuda/cusparselt/license.html
  mkdir tmp_cusparselt && cd tmp_cusparselt
  cusparselt_name="libcusparse_lt-linux-${arch_path}-${cusparselt_version}-archive"
  curl --retry 3 -OLs https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-${arch_path}/${cusparselt_name}.tar.xz

  tar xf ${cusparselt_name}.tar.xz
  cp -a ${cusparselt_name}/include/* /usr/local/cuda/include/
  cp -a ${cusparselt_name}/lib/* /usr/local/cuda/lib64/
  cd ..
  rm -rf tmp_cusparselt
}

function install_nvshmem {
  cuda_major_version=$1      # e.g. "12"
  nvshmem_version=$2         # e.g. "3.3.9"

  case "${arch_path}" in
    sbsa)
      dl_arch="aarch64"
      ;;
    x86_64)
      dl_arch="x64"
      ;;
    *)
      dl_arch="${arch}"
      ;;
  esac

  tmpdir="tmp_nvshmem"
  mkdir -p "${tmpdir}" && cd "${tmpdir}"

  # nvSHMEM license: https://docs.nvidia.com/nvshmem/api/sla.html
  # This pattern is a lie as it is not consistent across versions, for 3.3.9 it was cuda_ver-arch-nvshhem-ver
  filename="libnvshmem-linux-${arch_path}-${nvshmem_version}_cuda${cuda_major_version}-archive"
  suffix=".tar.xz"
  url="https://developer.download.nvidia.com/compute/nvshmem/redist/libnvshmem/linux-${arch_path}/${filename}${suffix}"

  # download, unpack, install
  wget -q "${url}"
  tar xf "${filename}${suffix}"
  cp -a "${filename}/include/"* /usr/local/cuda/include/
  cp -a "${filename}/lib/"*     /usr/local/cuda/lib64/

  # cleanup
  cd ..
  rm -rf "${tmpdir}"

  echo "nvSHMEM ${nvshmem_version} for CUDA ${cuda_major_version} (${arch_path}) installed."
}

function install_126 {
  CUDNN_VERSION=9.10.2.21
  NCCL_VERSION=v2.27.5-1
  CUSPARSELT_VERSION=0.7.1.0

  echo "Installing CUDA 12.6.3, cuDNN ${CUDNN_VERSION}, NCCL ${NCCL_VERSION}, NVSHMEM ${NVSHMEM_VERSION} and cuSparseLt ${CUSPARSELT_VERSION}"

  install_cuda 12.6.3 cuda_12.6.3_560.35.05_linux

  install_cudnn 12 $CUDNN_VERSION

  install_nvshmem 12 $NVSHMEM_VERSION

  install_nccl $NCCL_VERSION

  install_cusparselt $CUSPARSELT_VERSION

  ldconfig
}

function install_129 {
  CUDNN_VERSION=9.10.2.21
  NCCL_VERSION=v2.27.5-1
  CUSPARSELT_VERSION=0.7.1.0

  echo "Installing CUDA 12.9.1, cuDNN ${CUDNN_VERSION}, NCCL ${NCCL_VERSION}, NVSHMEM ${NVSHMEM_VERSION} and cuSparseLt ${CUSPARSELT_VERSION}"

  # install CUDA 12.9.1 in the same container
  install_cuda 12.9.1 cuda_12.9.1_575.57.08_linux

  # cuDNN license: https://developer.nvidia.com/cudnn/license_agreement
  install_cudnn 12 $CUDNN_VERSION

  install_nvshmem 12 $NVSHMEM_VERSION

  install_nccl $NCCL_VERSION

  install_cusparselt $CUSPARSELT_VERSION

  ldconfig
}

function install_128 {
  CUDNN_VERSION=9.8.0.87
  NCCL_VERSION=v2.27.5-1
  CUSPARSELT_VERSION=0.7.1.0

  echo "Installing CUDA 12.8.1, cuDNN ${CUDNN_VERSION}, NCCL ${NCCL_VERSION}, NVSHMEM ${NVSHMEM_VERSION} and cuSparseLt ${CUSPARSELT_VERSION}"

  # install CUDA 12.8.1 in the same container
  install_cuda 12.8.1 cuda_12.8.1_570.124.06_linux

  # cuDNN license: https://developer.nvidia.com/cudnn/license_agreement
  install_cudnn 12 $CUDNN_VERSION

  install_nvshmem 12 $NVSHMEM_VERSION

  install_nccl $NCCL_VERSION

  install_cusparselt $CUSPARSELT_VERSION

  ldconfig
}

function install_130 {
  CUDNN_VERSION=9.13.0.50
  NCCL_VERSION=v2.27.7-1
  CUSPARSELT_VERSION=0.8.0.4_cuda13

  echo "Installing CUDA 12.8.1, cuDNN ${CUDNN_VERSION}, NCCL ${NCCL_VERSION}, NVSHMEM ${NVSHMEM_VERSION} and cuSparseLt ${CUSPARSELT_VERSION}"

  # install CUDA 13.0 in the same container
  install_cuda 13.0.0 cuda_13.0.0_580.65.06_linux

  # cuDNN license: https://developer.nvidia.com/cudnn/license_agreement
  install_cudnn 13 $CUDNN_VERSION

  install_nvshmem 13 $NVSHMEM_VERSION

  install_nccl $NCCL_VERSION

  install_cusparselt $CUSPARSELT_VERSION

  ldconfig
}

# idiomatic parameter and option handling in sh
while test $# -gt 0
do
    case "$1" in
    12.6|12.6.*) install_126;
        ;;
    12.8|12.8.*) install_128;
        ;;
    12.9|12.9.*) install_129;
        ;;
    13.0|13.0.*) install_130;
        ;;
    *) echo "bad argument $1"; exit 1
        ;;
    esac
    shift
done
