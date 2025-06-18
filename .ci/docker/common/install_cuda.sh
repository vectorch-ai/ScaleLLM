#!/bin/bash

# adapted from https://github.com/pytorch/pytorch/blob/main/.ci/docker/common/install_cuda.sh

set -ex

arch_path=''
targetarch=${TARGETARCH:-$(uname -m)}
if [ ${targetarch} = 'amd64' ] || [ "${targetarch}" = 'x86_64' ]; then
  arch_path='x86_64'
else
  arch_path='sbsa'
fi

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

function install_126 {
  CUDNN_VERSION=9.10.2.21
  echo "Installing CUDA 12.6.3 and cuDNN ${CUDNN_VERSION} and NCCL and cuSparseLt-0.7.1"
  install_cuda 12.6.3 cuda_12.6.3_560.35.05_linux

  install_cudnn 12 $CUDNN_VERSION

  CUDA_VERSION=12.6 bash install_nccl.sh

  CUDA_VERSION=12.6 bash install_cusparselt.sh

  ldconfig
}

function install_129 {
  CUDNN_VERSION=9.10.2.21
  echo "Installing CUDA 12.9.1 and cuDNN ${CUDNN_VERSION} and NCCL and cuSparseLt-0.7.1"
  # install CUDA 12.9.1 in the same container
  install_cuda 12.9.1 cuda_12.9.1_575.57.08_linux

  # cuDNN license: https://developer.nvidia.com/cudnn/license_agreement
  install_cudnn 12 $CUDNN_VERSION

  CUDA_VERSION=12.9 bash install_nccl.sh

  CUDA_VERSION=12.9 bash install_cusparselt.sh

  ldconfig
}

function prune_126 {
  echo "Pruning CUDA 12.6"
  #####################################################################################
  # CUDA 12.6 prune static libs
  #####################################################################################
  export NVPRUNE="/usr/local/cuda-12.6/bin/nvprune"
  export CUDA_LIB_DIR="/usr/local/cuda-12.6/lib64"

  export GENCODE="-gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90"
  export GENCODE_CUDNN="-gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90"

  if [[ -n "$OVERRIDE_GENCODE" ]]; then
      export GENCODE=$OVERRIDE_GENCODE
  fi
  if [[ -n "$OVERRIDE_GENCODE_CUDNN" ]]; then
      export GENCODE_CUDNN=$OVERRIDE_GENCODE_CUDNN
  fi

  # all CUDA libs except CuDNN and CuBLAS
  ls $CUDA_LIB_DIR/ | grep "\.a" | grep -v "culibos" | grep -v "cudart" | grep -v "cudnn" | grep -v "cublas" | grep -v "metis"  \
      | xargs -I {} bash -c \
                "echo {} && $NVPRUNE $GENCODE $CUDA_LIB_DIR/{} -o $CUDA_LIB_DIR/{}"

  # prune CuDNN and CuBLAS
  $NVPRUNE $GENCODE_CUDNN $CUDA_LIB_DIR/libcublas_static.a -o $CUDA_LIB_DIR/libcublas_static.a
  $NVPRUNE $GENCODE_CUDNN $CUDA_LIB_DIR/libcublasLt_static.a -o $CUDA_LIB_DIR/libcublasLt_static.a

  #####################################################################################
  # CUDA 12.6 prune visual tools
  #####################################################################################
  export CUDA_BASE="/usr/local/cuda-12.6/"
  rm -rf $CUDA_BASE/libnvvp $CUDA_BASE/nsightee_plugins $CUDA_BASE/nsight-compute-2024.3.2 $CUDA_BASE/nsight-systems-2024.5.1/
}

function install_128 {
  CUDNN_VERSION=9.8.0.87
  echo "Installing CUDA 12.8.1 and cuDNN ${CUDNN_VERSION} and NCCL and cuSparseLt-0.7.1"
  # install CUDA 12.8.1 in the same container
  install_cuda 12.8.1 cuda_12.8.1_570.124.06_linux

  # cuDNN license: https://developer.nvidia.com/cudnn/license_agreement
  install_cudnn 12 $CUDNN_VERSION

  CUDA_VERSION=12.8 bash install_nccl.sh

  CUDA_VERSION=12.8 bash install_cusparselt.sh

  ldconfig
}

# idiomatic parameter and option handling in sh
while test $# -gt 0
do
    case "$1" in
    12.6|12.6.*) install_126; prune_126
        ;;
    12.8|12.8.*) install_128;
        ;;
    12.9|12.9.*) install_129;
        ;;
    *) echo "bad argument $1"; exit 1
        ;;
    esac
    shift
done