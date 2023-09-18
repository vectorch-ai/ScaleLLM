#pragma once

#include <optional>

#include "common/arg.h"

namespace llm {

class ModelArgs {
 public:
  DEFINE_ARG(std::vector<std::string>, architectures);

  DEFINE_ARG(int64_t, dim) = 4096;

  DEFINE_ARG(int64_t, n_layers) = 32;

  DEFINE_ARG(int64_t, n_heads) = 32;

  DEFINE_ARG(std::optional<int64_t>, n_kv_heads);

  // defined later by tokenizer
  DEFINE_ARG(int64_t, vocab_size) = -1;

  // make SwiGLU hidden layer size multiple of large power of 2
  DEFINE_ARG(int64_t, multiple_of) = 256;

  DEFINE_ARG(std::optional<float>, ffn_dim_multiplier);

  DEFINE_ARG(float, norm_eps) = 1e-5;

  // TODO: following two should not be part of model args
  DEFINE_ARG(int64_t, max_batch_size) = 32;

  DEFINE_ARG(int64_t, max_seq_len) = 2048;
};

}  // namespace llm
