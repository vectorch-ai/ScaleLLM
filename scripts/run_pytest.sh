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

# choose python version
PYVER="${PYTHON_VERSION//./}"
export PATH="/opt/python/cp${PYVER}-cp${PYVER}/bin:$PATH"

# install PyTorch
pip install torch==$TORCH_VERSION -i "https://download.pytorch.org/whl/cu${CUDA_VERSION//./}"

cd "$PROJECT_ROOT"

# install dependencies
pip install numpy
pip install -r requirements-test.txt

# install scalellm wheel
pip install dist/*.whl

# run pytest
printf "\n\nRunning pytest\n\n"
# cd tests
python3 -m pytest