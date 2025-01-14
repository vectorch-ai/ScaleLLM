#pragma once

#include <cute/config.hpp>
#include <cute/tensor.hpp>

#include "cute/layout.hpp"
namespace llm {

// common params for attention kernels
struct AttentionParamsCommon {
  const void* __restrict__ q_ptr = nullptr;
  const void* __restrict__ k_ptr = nullptr;
  const void* __restrict__ v_ptr = nullptr;
  void* __restrict__ o_ptr = nullptr;

  // input shapes
  int batch_size = 0;
  int n_heads = 0;
  int n_kv_heads = 0;
  int head_dim = 0;

  // mask
  int sliding_window = -1;

  // softcap
  float logits_soft_cap = 0.0;

  // softmax scaling
  float sm_scale = 1.0;

  // alibi
  const float* __restrict__ alibi_slopes_ptr = nullptr;  // [n_heads]
};

struct AttentionParams : public AttentionParamsCommon {
  // (batch, head, len, 1): last dimension is contiguous
  using Stride = cute::Stride<int64_t, int64_t, int64_t /*,_1*/>;

  Stride q_stride;
  Stride k_stride;
  Stride v_stride;
  Stride o_stride;

  // input shapes
  int q_len = 0;
  int kv_len = 0;
};

// variable length sequence
struct VarLenAttentionParams : public AttentionParamsCommon {
  // (head, seq, 1): last dimension is contiguous
  using Stride = cute::Stride<int64_t, int64_t /*,_1*/>;

  Stride q_stride;
  Stride k_stride;
  Stride v_stride;
  Stride o_stride;

  // input shapes
  // array of length batch_size + 1 holding starting offset of each sequence.
  const int* __restrict__ cu_seqlens_q = nullptr;
  const int* __restrict__ cu_seqlens_kv = nullptr;
};

// paged KV cache
struct PagedKVAttentionParams : public VarLenAttentionParams {
  // Paged KV cache
  const int* __restrict__ block_table = nullptr;
  const int* __restrict__ cu_block_lens = nullptr;
  int block_size = 0;
};

}  // namespace llm