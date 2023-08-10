#pragma once

#include <c10/core/TensorImpl.h>
#include <glog/logging.h>
#include <torch/torch.h>

namespace llm::attention {

torch::Tensor varlen_masked_self_attention(
    torch::Tensor query,                     // [num_tokens, n_heads, head_dim]
    torch::Tensor key,                       // [num_tokens, n_heads, head_dim]
    torch::Tensor value,                     // [num_tokens, n_heads, head_dim]
    const std::vector<int64_t>& cu_seq_lens  // cumulative sequence lengths
);

}  // namespace llm::attention
