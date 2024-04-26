#pragma once
#include <torch/torch.h>

namespace llm::kernel {

void set_kv_cache(
    const torch::Tensor& slot_ids,  // [n_tokens]
    const torch::Tensor& keys,      // [n_tokens, n_kv_heads, head_dim]
    const torch::Tensor& values,    // [n_tokens, n_kv_heads, head_dim]
    torch::Tensor& key_cache,       // [n_blocks, block_size, n_heads, head_dim]
    torch::Tensor& value_cache);

}  // namespace llm::kernel
