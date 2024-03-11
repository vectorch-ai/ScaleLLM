#pragma once

#include <gflags/gflags.h>
#include <torch/torch.h>

#include "layers/attention/handler.h"
#include "memory/kv_cache.h"
#include "models/input_parameters.h"

DECLARE_bool(disable_custom_kernels);

namespace llm {

class AttentionImpl : public torch::nn::Module {
 public:
  AttentionImpl(int64_t n_heads,
                int64_t n_kv_heads,
                int64_t head_dim,
                AttentionHandler* handler);

  // query: [n_tokens, n_heads, head_dim]
  // key/value: [n_tokens, n_kv_heads, head_dim]
  // positions: [n_tokens]
  // return: [n_tokens, n_heads, head_dim]
  torch::Tensor forward(const torch::Tensor& query,
                        const torch::Tensor& key,
                        const torch::Tensor& value,
                        const torch::Tensor& positions,
                        KVCache& kv_cache,
                        const InputParameters& input_params);

 private:
  int64_t n_heads_ = 0;
  int64_t n_kv_heads_ = 0;
  int64_t head_dim_ = 0;

  // handler for attention operations
  AttentionHandler* handler_ = nullptr;
};
TORCH_MODULE(Attention);

}  // namespace llm
