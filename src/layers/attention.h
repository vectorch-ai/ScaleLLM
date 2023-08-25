#pragma once

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include "memory/kv_cache.h"

DECLARE_string(varlen_masked_self_attention);
DECLARE_string(single_token_masked_self_attention);

namespace llm::attention {

// self attention with variable length sequence
// used in prefill stage
void varlen_masked_self_attention(
    const torch::Tensor& query,        // [num_tokens, n_heads, head_dim]
    const torch::Tensor& key,          // [num_tokens, n_kv_heads, head_dim]
    const torch::Tensor& value,        // [num_tokens, n_kv_heads, head_dim]
    const torch::Tensor& cu_seq_lens,  // [num_seq + 1]
    int32_t max_seq_len,               // maximum sequence length
    const torch::Tensor&);

void varlen_masked_self_attention_slow(
    const torch::Tensor& query,        // [num_tokens, n_heads, head_dim]
    const torch::Tensor& key,          // [num_tokens, n_kv_heads, head_dim]
    const torch::Tensor& value,        // [num_tokens, n_kv_heads, head_dim]
    const torch::Tensor& cu_seq_lens,  // [num_seq + 1]
    int32_t max_seq_len,               // maximum sequence length
    torch::Tensor output);

void varlen_masked_self_attention_cuda(
    const torch::Tensor& query,        // [num_tokens, n_heads, head_dim]
    const torch::Tensor& key,          // [num_tokens, n_kv_heads, head_dim]
    const torch::Tensor& value,        // [num_tokens, n_kv_heads, head_dim]
    const torch::Tensor& cu_seq_lens,  // [num_seq + 1]
    int32_t max_seq_len,               // maximum sequence length
    torch::Tensor output);

// self attention with single token as query
// used in decode stage
void single_token_masked_self_attention(
    const KVCache& kv_cache,     // kv cache
    const torch::Tensor& query,  // [num_tokens/num_seq, n_heads, head_dim]
    const torch::Tensor& block_tables,  // [num_tokens, max_num_blocks]
    const torch::Tensor& context_lens,  // [num_tokens]
    int32_t max_context_len,            // maximum context length
    const torch::Tensor& output);

void single_token_masked_self_attention_slow(
    const KVCache& kv_cache,     // kv cache
    const torch::Tensor& query,  // [num_tokens/num_seq, n_heads, head_dim]
    const torch::Tensor& block_tables,  // [num_tokens, max_num_blocks]
    const torch::Tensor& context_lens,  // [num_tokens]
    int32_t max_context_len,            // maximum context length
    torch::Tensor output);

void single_token_masked_self_attention_cuda(
    const KVCache& kv_cache,     // kv cache
    const torch::Tensor& query,  // [num_tokens/num_seq, n_heads, head_dim]
    const torch::Tensor& block_tables,  // [num_tokens, max_num_blocks]
    const torch::Tensor& context_lens,  // [num_tokens]
    int32_t max_context_len,            // maximum context length
    torch::Tensor output);

}  // namespace llm::attention
