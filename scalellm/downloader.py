import os


def convert_pickle_to_safetensors(path):
    import torch
    from safetensors.torch import save_file
                
    for filename in os.listdir(path):
        if filename.endswith(".bin") or filename.endswith(".pth"):
            file_path = os.path.join(path, filename)
            # Replace the extension with .safetensors
            st_file = os.path.splitext(filename)[0] + ".safetensors"
            st_file_path = os.path.join(path, st_file)
            if os.path.exists(st_file_path):
                # already converted
                continue
            
            # load the model
            model = torch.load(file_path, map_location="cpu")
            if hasattr(model, "state_dict"):
                state_dict = model.state_dict()
            else:
                state_dict = model
            
            if not isinstance(state_dict, dict):
                continue
            
            is_pickle_with_tensors = all(isinstance(state_dict[key], torch.Tensor) for key in state_dict)
            if is_pickle_with_tensors:
                # Clone the tensors to avoid shared tensors
                state_dict = {key: tensor.clone() for key, tensor in state_dict.items()}
                # Replace the extension with .safetensors
                st_file = os.path.splitext(filename)[0] + ".safetensors"
                st_file_path = os.path.join(path, st_file)
                save_file(state_dict, st_file_path)
                print(f"Converted {filename} to {st_file}")
            else:
                print(f"Ignore non-tensor pickle file: {filename}")
    

def download_hf_model(repo_id, revision=None, allow_patterns=None, cache_dir=None, convert_to_safetensors=False):
    from huggingface_hub import HfApi, snapshot_download

    # check if >safetensors are present in the model repo
    def check_safetensors_present(model_id, revision):
        # Authenticate with HF token
        api = HfApi()
        files = api.list_repo_files(repo_id=model_id, revision=revision)
        for file in files:
            _, extension = os.path.splitext(file)
            if extension == ".safetensors":
                return True
        return False

    repo_id = repo_id
    assert repo_id, "Please provide a repo_id"

    revision = revision if revision else "main"
    cache_dir = cache_dir if cache_dir else None

    has_safetensors = check_safetensors_present(repo_id, revision)
    if not allow_patterns:
        # Define allowed file patterns for config, tokenizer, and model weights
        # download tokenizer and json configs
        allow_patterns = "*.json,*.tiktoken,*.model"
        # download safetensors if present, otherwise download pickle files
        allow_patterns += ",*.safetensors" if has_safetensors else ",*.bin,*.pth"

    path = snapshot_download(
        repo_id,
        revision=revision,
        cache_dir=cache_dir,
        allow_patterns=allow_patterns.split(","),
    )
    
    # convert to safetensors from pickle
    if convert_to_safetensors and not has_safetensors:
        convert_pickle_to_safetensors(path)
    
    # print download path
    return path
