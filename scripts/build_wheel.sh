#!/bin/bash
set -e

ensure_env() {
    local var_name="$1"
    if [ -z "${!var_name}" ]; then
        echo "Error: Environment variable '$var_name' is not set."
        exit 1
    fi
}

ensure_env PYTHON_VERSION
ensure_env TORCH_VERSION
ensure_env CUDA_VERSION
# ensure_env BUILD_VERSION

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

export HOME=/tmp/home
mkdir -p $HOME
export PATH="$HOME/.local/bin:$PATH"
CUDA_MAJOR="${CUDA_VERSION%.*}"
CUDA_MINOR="${CUDA_VERSION#*.}"
TORCH_MAJOR="${TORCH_VERSION%.*}"
TORCH_MINOR="${TORCH_VERSION#*.}"

# choose the right python version
PYVER="${PYTHON_VERSION//./}"
export PATH="/opt/python/cp${PYVER}-cp${PYVER}/bin:$PATH"


echo "::group::Install PyTorch"
pip install torch==$TORCH_VERSION --index-url "https://download.pytorch.org/whl/cu${CUDA_MAJOR}${CUDA_MINOR}"
echo "::endgroup::"


echo "::group::Build wheel for ScaleLLM"
cd "$PROJECT_ROOT/python"
python setup.py bdist_wheel
echo "::endgroup::"