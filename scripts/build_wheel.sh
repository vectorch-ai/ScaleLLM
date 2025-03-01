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

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

export HOME=/tmp/home
mkdir -p $HOME
export PATH="$HOME/.local/bin:$PATH"

# choose the right python version
PYVER="${PYTHON_VERSION//./}"
export PATH="/opt/python/cp${PYVER}-cp${PYVER}/bin:$PATH"

# update pip
python -m pip install --upgrade pip

# install PyTorch
pip install torch==$TORCH_VERSION --index-url "https://download.pytorch.org/whl/cu${CUDA_VERSION//./}"

# install other dependencies
pip install numpy jinja2
pip install --upgrade setuptools wheel

# set max cache size to 25GiB
command -v ccache >/dev/null && ccache -M 25Gi

# zero out ccache if ccache is installed
command -v ccache >/dev/null && ccache -z

cd "$PROJECT_ROOT"
python setup.py bdist_wheel

# show ccache statistics
command -v ccache >/dev/null && ccache -vs

# bundle external shared libraries into wheel
# pip install auditwheel
# cd "$PROJECT_ROOT"
# for whl in dist/*.whl; do
#     auditwheel repair "$whl" --plat manylinux1_x86_64 -w dist/
# done