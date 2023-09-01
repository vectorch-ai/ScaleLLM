#pragma once

#include <torch/torch.h>

#include <optional>

namespace llm {

class ModelArgs {
 public:
  bool load_from_file(const std::string& file_path);

  TORCH_ARG(std::string, model_name);

  TORCH_ARG(int64_t, dim) = 4096;

  TORCH_ARG(int64_t, n_layers) = 32;

  TORCH_ARG(int64_t, n_heads) = 32;

  TORCH_ARG(std::optional<int64_t>, n_kv_heads);

  // defined later by tokenizer
  TORCH_ARG(int64_t, vocab_size) = -1;

  // make SwiGLU hidden layer size multiple of large power of 2
  TORCH_ARG(int64_t, multiple_of) = 256;

  TORCH_ARG(std::optional<float>, ffn_dim_multiplier);

  TORCH_ARG(float, norm_eps) = 1e-5;

  // TODO: following two should not be part of model args
  TORCH_ARG(int64_t, max_batch_size) = 32;

  TORCH_ARG(int64_t, max_seq_len) = 2048;
};

}  // namespace llm
