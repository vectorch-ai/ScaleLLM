#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Construct the arguments to pass to the 'scalellm' command
ARGS=""

# Check if HF_MODEL_ID is defined; if so, download the model from the Hugging Face hub
if [ -n "$HF_MODEL_ID" ]; then
    echo "Downloading model from the Hugging Face hub for model id: "$HF_MODEL_ID" and revision: "$HF_MODEL_REVISION""

    MODEL_PATH=$(python3 ${SCRIPT_DIR}/download_hf_models.py --repo_id "$HF_MODEL_ID" --revision "$HF_MODEL_REVISION" --cache_dir "$HF_MODEL_CACHE_DIR" --allow_patterns "$HF_MODEL_ALLOW_PATTERN")
    # return if error
    if [ $? -ne 0 ]; then
        echo "Error downloading model from the Hugging Face hub for model id: "$HF_MODEL_ID" and revision: "$HF_MODEL_REVISION""
        exit 1
    fi
    ARGS+=" --model_path "$MODEL_PATH" --model_id "$HF_MODEL_ID""
fi

# Run the 'scalellm' with the specified arguments
$SCRIPT_DIR/../build/src/server/scalellm $ARGS "$@"
