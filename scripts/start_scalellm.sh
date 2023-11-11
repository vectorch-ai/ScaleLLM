#!/bin/bash
DEVICE=${DEVICE:-"auto"}
# Set default values for HF_MODEL_REVISION and HF_MODEL_ALLOW_PATTERN
HF_MODEL_REVISION=${HF_MODEL_REVISION:-main}
# Define allowed file patterns for config, tokenizer, and model weights
HF_MODEL_ALLOW_PATTERN=${HF_MODEL_ALLOW_PATTERN:-"*.json,*.safetensors,*.model"}

ARGS=""

# Check if HF_MODEL_ID is defined; if so, download the model from the Hugging Face hub
if [ -n "$HF_MODEL_ID" ]; then
    echo "Downloading model from the Hugging Face hub for model id: "$HF_MODEL_ID" and revision: "$HF_MODEL_REVISION""

    MODEL_PATH=$(python3 -c 'from huggingface_hub import snapshot_download; path = snapshot_download("'"$HF_MODEL_ID"'", revision="'"$HF_MODEL_REVISION"'", allow_patterns="'"$HF_MODEL_ALLOW_PATTERN"'".split(",")); print(path)')
    # return if error
    if [ $? -ne 0 ]; then
        echo "Error downloading model from the Hugging Face hub for model id: "$HF_MODEL_ID" and revision: "$HF_MODEL_REVISION""
        exit 1
    fi
    ARGS+=" --model_path "$MODEL_PATH" --model_id "$HF_MODEL_ID""
fi

ARGS+=" --device "$DEVICE""

# Run the 'scalellm' command with the specified arguments
$HOME/code/ScaleLLM/build/src/server/scalellm $ARGS "$@"
