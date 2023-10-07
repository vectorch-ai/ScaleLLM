#pragma once
#include <torch/torch.h>

namespace llm::kernel {

// apply rotary embedding to query and key inplace
void apply_rotary_pos_emb(
    torch::Tensor& query,            // [n_tokens, n_heads, head_dim]
    torch::Tensor& key,              // [n_tokens, n_kv_heads, head_dim]
    const torch::Tensor& positions,  // [n_tokens]
    const torch::Tensor& cos_sin,    // [max_positions, 2, rotary_dim/2]
    int rotary_dim,
    bool interleaved);

}  // namespace llm::kernel
