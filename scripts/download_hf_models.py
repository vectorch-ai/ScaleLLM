#!/usr/bin/env python3

import argparse
import os
from huggingface_hub import snapshot_download

# check if >safetensors are present in the model repo
def check_safetensors_present(model_id, revision):
  from huggingface_hub import HfApi
  # Authenticate with HF token
  api = HfApi()
  files = api.list_repo_files(repo_id=model_id, 
                              revision=revision)
  for file in files:
    _, extension = os.path.splitext(file)
    if extension == '.safetensors':
      return True
  return False

if __name__ == '__main__':  
  parser = argparse.ArgumentParser()
  parser.add_argument('--repo_id', type=str, default=None)
  parser.add_argument('--revision', type=str, default=None)
  parser.add_argument('--allow_patterns', type=str, default=None)
  parser.add_argument('--cache_dir', type=str, default=None)
  args = parser.parse_args()
  
  repo_id = args.repo_id
  assert args.repo_id, "Please provide a repo_id"

  revision = args.revision if args.revision else "main"
  cache_dir = args.cache_dir if args.cache_dir else None
  allow_patterns = args.allow_patterns
    
  if not allow_patterns:
    # Define allowed file patterns for config, tokenizer, and model weights
    has_safetensors = check_safetensors_present(repo_id, revision)
    # download tokenizer and json configs
    allow_patterns = "*.json,*.tiktoken,*.model"
    # download safetensors if present, otherwise download pickle files
    allow_patterns += ",*.safetensors" if has_safetensors else ",*.bin,*.pth"
    
  path = snapshot_download(args.repo_id, 
                           revision=revision, 
                           cache_dir=cache_dir,
                           allow_patterns=allow_patterns.split(","))
  # print download path
  print(path)
  
  

