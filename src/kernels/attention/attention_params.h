#pragma once

#include <cute/config.hpp>
#include <cute/tensor.hpp>

#include "cute/layout.hpp"
namespace llm {

struct AttentionParams {
  // (batch, head, len, 1): last dimension is contiguous
  using Stride = cute::Stride<int64_t, int64_t, int64_t /*,_1*/>;

  const void* __restrict__ q_ptr = nullptr;
  Stride q_stride;

  const void* __restrict__ k_ptr = nullptr;
  Stride k_stride;

  const void* __restrict__ v_ptr = nullptr;
  Stride v_stride;

  void* __restrict__ o_ptr = nullptr;
  Stride o_stride;

  const float* __restrict__ alibi_slopes_ptr = nullptr;  // [n_heads]

  // input shapes
  int batch_size = 0;
  int n_heads = 0;
  int n_kv_heads = 0;
  int head_dim = 0;
  int q_len = 0;
  int kv_len = 0;

  // mask
  int sliding_window = -1;

  // softcap
  float logits_soft_cap = 0.0;

  // softmax scaling
  float sm_scale = 1.0;
};

// TODO: variable length sequence
struct VarLenAttentionParams {};

// TODO: paged KV cache
struct PagedKVAttentionParams {};

}  // namespace llm