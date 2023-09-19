#pragma once

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include "memory/kv_cache.h"
#include "models/input_parameters.h"

DECLARE_string(varlen_masked_self_attention);
DECLARE_string(single_token_masked_self_attention);

namespace llm {

class AttentionImpl : public torch::nn::Module {
 public:
  AttentionImpl(int64_t n_heads,
                int64_t n_kv_heads,
                float scale,
                const torch::ScalarType& dtype,
                const torch::Device& device);

  // query: [num_tokens, n_heads, head_dim]
  // key/value: [num_tokens, n_kv_heads, head_dim]
  // return: [num_tokens, n_heads, head_dim]
  torch::Tensor forward(const torch::Tensor& query,
                        const torch::Tensor& key,
                        const torch::Tensor& value,
                        const KVCache& kv_cache,
                        const InputParameters& input_params);

 private:
  // scale factor
  float scale_;

  // head mapping used for single_token_masked_self_attention
  // [num_heads]
  torch::Tensor kv_head_mapping_;
};
TORCH_MODULE(Attention);

// expose the attention functions for testing
// self attention with variable length sequence
// used in prefill stage
void varlen_masked_self_attention(
    const torch::Tensor& query,        // [num_tokens, n_heads, head_dim]
    const torch::Tensor& key,          // [num_tokens, n_kv_heads, head_dim]
    const torch::Tensor& value,        // [num_tokens, n_kv_heads, head_dim]
    const torch::Tensor& cu_seq_lens,  // [num_seq + 1]
    int32_t max_seq_len,               // maximum sequence length
    float scale,
    const torch::Tensor&);

void varlen_masked_self_attention_slow(
    const torch::Tensor& query,        // [num_tokens, n_heads, head_dim]
    const torch::Tensor& key,          // [num_tokens, n_kv_heads, head_dim]
    const torch::Tensor& value,        // [num_tokens, n_kv_heads, head_dim]
    const torch::Tensor& cu_seq_lens,  // [num_seq + 1]
    int32_t max_seq_len,               // maximum sequence length
    float scale,
    torch::Tensor output);

void varlen_masked_self_attention_cuda(
    const torch::Tensor& query,        // [num_tokens, n_heads, head_dim]
    const torch::Tensor& key,          // [num_tokens, n_kv_heads, head_dim]
    const torch::Tensor& value,        // [num_tokens, n_kv_heads, head_dim]
    const torch::Tensor& cu_seq_lens,  // [num_seq + 1]
    int32_t max_seq_len,               // maximum sequence length
    float scale,
    torch::Tensor output);

// self attention with single token as query
// used in decode stage
void single_token_masked_self_attention(
    const KVCache& kv_cache,               // kv cache
    const torch::Tensor& kv_head_mapping,  // [num_heads]
    const torch::Tensor& query,  // [num_tokens/num_seq, n_heads, head_dim]
    const torch::Tensor& block_tables,  // [num_tokens, max_num_blocks]
    const torch::Tensor& context_lens,  // [num_tokens]
    int32_t max_context_len,            // maximum context length
    float scale,
    const torch::Tensor& output);

void single_token_masked_self_attention_slow(
    const KVCache& kv_cache,     // kv cache
    const torch::Tensor& query,  // [num_tokens/num_seq, n_heads, head_dim]
    const torch::Tensor& block_tables,  // [num_tokens, max_num_blocks]
    const torch::Tensor& context_lens,  // [num_tokens]
    int32_t max_context_len,            // maximum context length
    float scale,
    torch::Tensor output);

void single_token_masked_self_attention_cuda(
    const KVCache& kv_cache,               // kv cache
    const torch::Tensor& kv_head_mapping,  // [num_heads]
    const torch::Tensor& query,  // [num_tokens/num_seq, n_heads, head_dim]
    const torch::Tensor& block_tables,  // [num_tokens, max_num_blocks]
    const torch::Tensor& context_lens,  // [num_tokens]
    int32_t max_context_len,            // maximum context length
    float scale,
    torch::Tensor output);

}  // namespace llm
