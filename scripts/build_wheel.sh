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

RENAME_WHL=${RENAME_WHL:-false}

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


# install PyTorch
pip install torch==$TORCH_VERSION --index-url "https://download.pytorch.org/whl/cu${CUDA_MAJOR}${CUDA_MINOR}"

# install other dependencies
pip install numpy
pip install --upgrade setuptools wheel

# zero out ccache if ccache is installed
command -v ccache >/dev/null && ccache -z

cd "$PROJECT_ROOT/python"
python setup.py bdist_wheel

# show ccache statistics
command -v ccache >/dev/null && ccache -vs

# rename wheel to include torch and cuda versions
if [ "$RENAME_WHL" = "true" ]; then
    cd "$PROJECT_ROOT/python"
    for whl in dist/*.whl; do
        python rename_whl.py "$whl"
    done
fi

# bundle external shared libraries into wheel
# pip install auditwheel
# cd "$PROJECT_ROOT/python"
# for whl in dist/*.whl; do
#     auditwheel repair "$whl" --plat manylinux1_x86_64 -w dist/
# done