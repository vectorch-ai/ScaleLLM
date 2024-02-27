#pragma once

#include <gflags/gflags.h>
#include <torch/torch.h>

#include "layers/pos_embedding.h"
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

// returns IntTensor [n_heads], mapping from query head to kv head
torch::Tensor prepare_kv_head_mapping(int64_t n_heads,
                                      int64_t n_kv_heads,
                                      const torch::Device& device);

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

// self attention with single token as query
// used in decode stage
void single_query_masked_self_attention(
    const KVCache& kv_cache,            // kv cache
    int32_t n_kv_heads,                 // number of kv heads
    const torch::Tensor& query,         // [n_tokens/n_seq, n_heads, head_dim]
    const torch::Tensor& block_tables,  // [n_tokens, max_num_blocks]
    const torch::Tensor& context_lens,  // [n_tokens]
    torch::optional<torch::Tensor> alibi_slopes,  // [n_heads]
    int32_t max_context_len,                      // maximum context length
    float scale,
    torch::Tensor& output);

// different implementations start here
// slow version of varlen_masked_self_attention
void varlen_masked_self_attention_generic(
    const torch::Tensor& query,        // [n_tokens, n_heads, head_dim]
    const torch::Tensor& key,          // [n_tokens, n_kv_heads, head_dim]
    const torch::Tensor& value,        // [n_tokens, n_kv_heads, head_dim]
    const torch::Tensor& cu_seq_lens,  // [n_seq + 1]
    torch::optional<torch::Tensor> alibi_slopes,  // [n_heads]
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

// slow version of single_query_masked_self_attention
// mainly used for testing
void single_query_masked_self_attention_generic(
    const KVCache& kv_cache,            // kv cache
    const torch::Tensor& query,         // [n_tokens/n_seq, n_heads, head_dim]
    const torch::Tensor& block_tables,  // [n_tokens, max_num_blocks]
    const torch::Tensor& context_lens,  // [n_tokens]
    torch::optional<torch::Tensor> alibi_slopes,  // [n_heads]
    int32_t max_context_len,                      // maximum context length
    float scale,
    torch::Tensor& output);

// fast version of single_query_masked_self_attention with CUDA kernel
void single_query_masked_self_attention_cuda(
    const KVCache& kv_cache,            // kv cache
    int32_t n_kv_heads,                 // number of kv heads
    const torch::Tensor& query,         // [n_tokens/n_seq, n_heads, head_dim]
    const torch::Tensor& block_tables,  // [n_seq, max_n_blocks]
    const torch::Tensor& context_lens,  // [n_tokens]
    torch::optional<torch::Tensor> alibi_slopes,  // [n_heads]
    int32_t max_context_len,                      // maximum context length
    float scale,
    torch::Tensor& output);

// key/value support two type of inputs:
// * packed continuous memory : [n_kv_tokens, n_kv_heads, head_dim]
// * block-wise memory: [n_blocks, block_size, n_kv_heads, head_dim]
void masked_self_attention_cuda(
    const torch::Tensor& query,          // [n_q_tokens, n_heads, head_dim]
    const torch::Tensor& key,            // [..., n_kv_heads, head_dim]
    const torch::Tensor& value,          // [..., n_kv_heads, head_dim]
    const torch::Tensor& q_cu_seq_lens,  // [n_seq + 1]
    const torch::Tensor& k_cu_seq_lens,  // [n_seq + 1]
    torch::optional<torch::Tensor> block_tables,  // [n_seq, max_n_blocks]
    torch::optional<torch::Tensor> alibi_slopes,  // [n_heads]
    int32_t q_max_seq_len,  // maximum sequence length for Q
    int32_t k_max_seq_len,  // maximum sequence length for K/V
    float scale,
    torch::Tensor& output);

void masked_self_attention_with_newk_cuda(
    const torch::Tensor& query,           // [n_q_tokens, n_heads, head_dim]
    const torch::Tensor& kcache,          // [n_blocks, block_size, n_kv_heads, head_dim]
    const torch::Tensor& vcache,          // [n_blocks, block_size, n_kv_heads, head_dim]
    torch::optional<torch::Tensor> knew,  // [new_tokens, n_kv_heads, head_dim]
    torch::optional<torch::Tensor> vnew,  // [new_tokens, n_kv_heads, head_dim]
    const torch::Tensor& q_cu_seq_lens,   // [n_seq + 1]
    const torch::Tensor& k_cu_seq_lens,   // [n_seq + 1]
    torch::optional<torch::Tensor> knew_cu_seq_lens,  // [n_seq + 1]
    torch::optional<torch::Tensor> block_tables,      // [n_seq, max_n_blocks]
    torch::optional<torch::Tensor> alibi_slopes,      // [n_heads]
    int32_t q_max_seq_len,  // maximum sequence length for Q
    int32_t k_max_seq_len,  // maximum sequence length for K/V
    float scale,
    torch::Tensor& output);

}  // namespace detail

}  // namespace llm
