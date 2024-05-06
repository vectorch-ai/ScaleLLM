
def download_hf_model(repo_id, revision=None, allow_patterns=None, cache_dir=None):
    import os

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

    if not allow_patterns:
        # Define allowed file patterns for config, tokenizer, and model weights
        has_safetensors = check_safetensors_present(repo_id, revision)
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
    # print download path
    return path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, default=None)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--allow_patterns", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    args = parser.parse_args()

    path = download_hf_model(
        args.repo_id,
        revision=args.revision,
        allow_patterns=args.allow_patterns,
        cache_dir=args.cache_dir,
    )
    # print download path
    print(path)
