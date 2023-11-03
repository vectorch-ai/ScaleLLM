#!/bin/bash

# Set default values for HF_MODEL_REVISION and HF_MODEL_ALLOW_PATTERN
HF_MODEL_REVISION=${HF_MODEL_REVISION:-main}
# Define allowed file patterns for config, tokenizer, and model weights
HF_MODEL_ALLOW_PATTERN=${HF_MODEL_ALLOW_PATTERN:-"*.json,*.safetensors,*.model"}
# HF_MODEL_CACHE_DIR=${HF_MODEL_CACHE_DIR:-$HOME/.cache/huggingface_hub}
HF_MODEL_CACHE_DIR=${HF_MODEL_CACHE_DIR:-/models}

ARGS=""

# Check if HF_MODEL_ID is defined; if so, download the model from the Hugging Face hub
if [ -n "$HF_MODEL_ID" ]; then
    echo "Downloading model from the Hugging Face hub for model id: "$HF_MODEL_ID" and revision: "$HF_MODEL_REVISION""

    MODEL_PATH=$(python -c 'from huggingface_hub import snapshot_download; path = snapshot_download("'"$HF_MODEL_ID"'", revision="'"$HF_MODEL_REVISION"'", cache_dir="'"$HF_MODEL_CACHE_DIR"'", allow_patterns="'"$HF_MODEL_ALLOW_PATTERN"'".split(",")); print(path)')
    # return if error
    if [ $? -ne 0 ]; then
        echo "Error downloading model from the Hugging Face hub for model id: "$HF_MODEL_ID" and revision: "$HF_MODEL_REVISION""
        exit 1
    fi
    ARGS+=" --model_path "$MODEL_PATH" --model_id "$HF_MODEL_ID""
elif [ -n "$HF_MODEL_PATH" ]; then
    echo "Using model from the specified path "$HF_MODEL_PATH""

    ARGS+=" --model_path "$HF_MODEL_PATH""
fi

# Run the 'scalellm' command with the specified arguments
scalellm $ARGS "$@"
