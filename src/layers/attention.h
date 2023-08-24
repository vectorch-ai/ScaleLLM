#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

#include "memory/kv_cache.h"

namespace llm::attention {

// self attention with variable length sequence
// used in prefill stage
void varlen_masked_self_attention(
    const torch::Tensor& query,        // [num_tokens, n_heads, head_dim]
    const torch::Tensor& key,          // [num_tokens, n_heads, head_dim]
    const torch::Tensor& value,        // [num_tokens, n_heads, head_dim]
    const torch::Tensor& cu_seq_lens,  // [num_seq + 1]
    int32_t max_seq_len,               // maximum sequence length
    torch::Tensor output               // [num_tokens, n_heads, head_dim]
);

// self attention with single token as query
// used in generate stage
void single_token_masked_self_attention(
    const KVCache& kv_cache,     // where to get key and value
    torch::Tensor query,         // [num_tokens/num_seq, n_heads, head_dim]
    torch::Tensor block_tables,  // [num_tokens, num_blocks]
    torch::Tensor context_lens,  // [num_tokens] the length of each sequence
    int32_t max_context_len,     // maximum context length
    torch::Tensor& output        // [num_tokens, n_heads, head_dim]
);

}  // namespace llm::attention
