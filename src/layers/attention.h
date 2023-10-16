#pragma once

#include <gflags/gflags.h>
#include <torch/torch.h>

#include "layers/pos_embedding.h"
#include "memory/kv_cache.h"
#include "models/input_parameters.h"

DECLARE_string(varlen_masked_self_attention);
DECLARE_string(single_query_masked_self_attention);

namespace llm {

class AttentionImpl : public torch::nn::Module {
 public:
  AttentionImpl(int64_t n_heads,
                int64_t n_kv_heads,
                int64_t head_dim,
                float scale,
                torch::ScalarType dtype,
                const torch::Device& device);

  // query: [num_tokens, n_heads, head_dim]
  // key/value: [num_tokens, n_kv_heads, head_dim]
  // return: [num_tokens, n_heads, head_dim]
  torch::Tensor forward(const torch::Tensor& query,
                        const torch::Tensor& key,
                        const torch::Tensor& value,
                        KVCache& kv_cache,
                        const InputParameters& input_params);

 private:
  int64_t n_heads_ = 0;
  int64_t n_kv_heads_ = 0;
  int64_t head_dim_ = 0;

  // scale factor
  float scale_ = 0.0;

  // head mapping used for single_query_masked_self_attention
  // [num_heads]
  torch::Tensor kv_head_mapping_;
};
TORCH_MODULE(Attention);

namespace detail {
// expose the attention functions for testing
// self attention with variable length sequence
// used in prefill stage
void varlen_masked_self_attention(
    const torch::Tensor& query,        // [num_tokens, n_heads, head_dim]
    const torch::Tensor& key,          // [num_tokens, n_kv_heads, head_dim]
    const torch::Tensor& value,        // [num_tokens, n_kv_heads, head_dim]
    const torch::Tensor& alibi_slope,  // [n_heads]
    const torch::Tensor& cu_seq_lens,  // [num_seq + 1]
    int32_t max_seq_len,               // maximum sequence length
    float scale,
    const torch::Tensor&);

// self attention with single token as query
// used in decode stage
void single_query_masked_self_attention(
    const KVCache& kv_cache,               // kv cache
    const torch::Tensor& kv_head_mapping,  // [num_heads]
    const torch::Tensor& query,  // [num_tokens/num_seq, n_heads, head_dim]
    const torch::Tensor& block_tables,  // [num_tokens, max_num_blocks]
    const torch::Tensor& context_lens,  // [num_tokens]
    int32_t max_context_len,            // maximum context length
    float scale,
    const torch::Tensor& output);

// different implementations start here
// slow version of varlen_masked_self_attention
void varlen_masked_self_attention_generic(
    const torch::Tensor& query,        // [num_tokens, n_heads, head_dim]
    const torch::Tensor& key,          // [num_tokens, n_kv_heads, head_dim]
    const torch::Tensor& value,        // [num_tokens, n_kv_heads, head_dim]
    const torch::Tensor& alibi_slope,  // [n_heads]
    const torch::Tensor& cu_seq_lens,  // [num_seq + 1]
    int32_t max_seq_len,               // maximum sequence length
    float scale,
    torch::Tensor output);

// fast version of varlen_masked_self_attention with CUDA kernel
void varlen_masked_self_attention_cuda(
    const torch::Tensor& query,        // [num_tokens, n_heads, head_dim]
    const torch::Tensor& key,          // [num_tokens, n_kv_heads, head_dim]
    const torch::Tensor& value,        // [num_tokens, n_kv_heads, head_dim]
    const torch::Tensor& alibi_slope,  // [n_heads]
    const torch::Tensor& cu_seq_lens,  // [num_seq + 1]
    int32_t max_seq_len,               // maximum sequence length
    float scale,
    torch::Tensor output);

// slow version of single_query_masked_self_attention
// mainly used for testing
void single_query_masked_self_attention_generic(
    const KVCache& kv_cache,     // kv cache
    const torch::Tensor& query,  // [num_tokens/num_seq, n_heads, head_dim]
    const torch::Tensor& block_tables,  // [num_tokens, max_num_blocks]
    const torch::Tensor& context_lens,  // [num_tokens]
    int32_t max_context_len,            // maximum context length
    float scale,
    torch::Tensor output);

// fast version of single_query_masked_self_attention with CUDA kernel
void single_query_masked_self_attention_cuda(
    const KVCache& kv_cache,               // kv cache
    const torch::Tensor& kv_head_mapping,  // [num_heads]
    const torch::Tensor& query,  // [num_tokens/num_seq, n_heads, head_dim]
    const torch::Tensor& block_tables,  // [num_tokens, max_num_blocks]
    const torch::Tensor& context_lens,  // [num_tokens]
    int32_t max_context_len,            // maximum context length
    float scale,
    torch::Tensor output);

}  // namespace detail

}  // namespace llm
