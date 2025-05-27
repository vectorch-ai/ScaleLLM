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
cd "$PROJECT_ROOT"

export HOME=/tmp/home
mkdir -p $HOME
export PATH="$HOME/.local/bin:$PATH"

# choose python version
PYVER="${PYTHON_VERSION//./}"
export PATH="/opt/python/cp${PYVER}-cp${PYVER}/bin:$PATH"

# update pip
python -m pip install --upgrade pip

# install PyTorch
pip install torch==$TORCH_VERSION -i "https://download.pytorch.org/whl/cu${CUDA_VERSION//./}"

# install dependencies
pip install -r requirements-test.txt

# install scalellm wheel
pip install dist/*.whl

# run pytest within the tests directory
cd tests
pytest
