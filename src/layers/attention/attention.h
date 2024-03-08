#pragma once

#include <gflags/gflags.h>
#include <torch/torch.h>

#include "memory/kv_cache.h"
#include "models/input_parameters.h"

DECLARE_bool(disable_custom_kernels);

namespace llm {

class AttentionImpl : public torch::nn::Module {
 public:
  AttentionImpl(int64_t n_heads,
                int64_t n_kv_heads,
                int64_t head_dim,
                float scale,
                torch::ScalarType dtype,
                const torch::Device& device);

  // query: [n_tokens, n_heads, head_dim]
  // key/value: [n_tokens, n_kv_heads, head_dim]
  // return: [n_tokens, n_heads, head_dim]
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
};
TORCH_MODULE(Attention);

namespace detail {

// expose the attention functions for testing
// self attention with variable length sequence
// used in prefill stage
void varlen_masked_self_attention(
    const torch::Tensor& query,        // [n_tokens, n_heads, head_dim]
    const torch::Tensor& key,          // [n_tokens, n_kv_heads, head_dim]
    const torch::Tensor& value,        // [n_tokens, n_kv_heads, head_dim]
    const torch::Tensor& cu_seq_lens,  // [n_seq + 1]
    torch::optional<torch::Tensor> alibi_slopes,  // [n_heads]
    int32_t max_seq_len,                          // maximum sequence length
    float scale,
    torch::Tensor& output);

// self attention with multiple tokens as query
// used in decode stage
void multiple_query_masked_self_attention(
    const torch::Tensor& query,           // [n_q_tokens, n_heads, head_dim]
    const KVCache& kv_cache,              // where to get key and value
    const torch::Tensor& q_cu_seq_lens,   // [n_seqs + 1]
    const torch::Tensor& kv_cu_seq_lens,  // [n_seqs + 1]
    torch::optional<torch::Tensor> block_tables,  // [n_seqs, max_n_blocks]
    torch::optional<torch::Tensor> alibi_slopes,  // [n_heads]
    int32_t q_max_seq_len,  // maximum sequence length for Q
    int32_t k_max_seq_len,  // maximum sequence length for K/V
    float scale,
    torch::Tensor& output);

// fast version of varlen_masked_self_attention with CUDA kernel
void varlen_masked_self_attention_cuda(
    const torch::Tensor& query,        // [n_tokens, n_heads, head_dim]
    const torch::Tensor& key,          // [n_tokens, n_kv_heads, head_dim]
    const torch::Tensor& value,        // [n_tokens, n_kv_heads, head_dim]
    const torch::Tensor& cu_seq_lens,  // [n_seq + 1]
    torch::optional<torch::Tensor> alibi_slopes,  // [n_heads]
    int32_t max_seq_len,                          // maximum sequence length
    float scale,
    torch::Tensor& output);

// key/value support two type of inputs:
// * packed continuous memory : [n_kv_tokens, n_kv_heads, head_dim]
// * block-wise memory: [n_blocks, block_size, n_kv_heads, head_dim]
void multiple_query_masked_self_attention_cuda(
    const torch::Tensor& query,          // [n_q_tokens, n_heads, head_dim]
    const torch::Tensor& key,            // [..., n_kv_heads, head_dim]
    const torch::Tensor& value,          // [..., n_kv_heads, head_dim]
    const torch::Tensor& q_cu_seq_lens,  // [n_seqs + 1]
    const torch::Tensor& k_cu_seq_lens,  // [n_seqs + 1]
    torch::optional<torch::Tensor> block_tables,  // [n_seqs, max_n_blocks]
    torch::optional<torch::Tensor> alibi_slopes,  // [n_heads]
    int32_t q_max_seq_len,  // maximum sequence length for Q
    int32_t k_max_seq_len,  // maximum sequence length for K/V
    float scale,
    torch::Tensor& output,
    int32_t num_splits);

}  // namespace detail

}  // namespace llm
