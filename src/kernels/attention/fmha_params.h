#pragma once

#include <cstdint>

namespace llm {

// Params for fused multi-head attention (FMHA) kernels
struct FmhaParams {
  ////////////////////////////////////////////////
  // Parameters for input/output tensors
  ////////////////////////////////////////////////
  const void* __restrict__ q_ptr = nullptr;
  const void* __restrict__ k_ptr = nullptr;
  const void* __restrict__ v_ptr = nullptr;
  void* __restrict__ o_ptr = nullptr;

  // Parameters for input shapes
  int batch_size = 0;
  int n_heads = 0;
  int n_kv_heads = 0;
  int head_dim = 0;

  // strides for query, key, value, and output tensors
  // Tensor shape: (batch, seq, head, dim): last dimension is contiguous
  // N.B. for variable length sequence, the q/k/v/o_batch_stride is not used.
  int64_t q_batch_stride;
  int64_t q_seq_stride;
  int64_t q_head_stride;

  int64_t k_batch_stride;
  int64_t k_seq_stride;
  int64_t k_head_stride;

  int64_t v_batch_stride;
  int64_t v_seq_stride;
  int64_t v_head_stride;

  int64_t o_batch_stride;
  int64_t o_seq_stride;
  int64_t o_head_stride;

  ////////////////////////////////////////////////
  // Parameters for sequence length
  ////////////////////////////////////////////////
  // Only used for fix length sequence
  int q_len = 0;
  int kv_len = 0;

  // // Only used for variable length sequence
  // // array of length batch_size + 1 holding starting offset of each sequence.
  // const int* __restrict__ q_cu_lens = nullptr;
  // const int* __restrict__ kv_cu_lens = nullptr;

  // ////////////////////////////////////////////////
  // // Parameters for paged KV cache
  // ////////////////////////////////////////////////
  // // size for each cache block
  // int block_size = 1;
  // // the first slot id of each block
  // const int* __restrict__ block_table = nullptr;
  // // array of length batch_size + 1 holding starting offset of each sequence.
  // const int* __restrict__ block_cu_lens = nullptr;

  ////////////////////////////////////////////////
  // Parameters for local attention
  ////////////////////////////////////////////////
  // left sliding window size
  int sliding_window = -1;

  ////////////////////////////////////////////////
  // Parameters for logits soft cap
  ////////////////////////////////////////////////
  float logits_soft_cap = 0.0;

  ////////////////////////////////////////////////
  // Parameters for softmax
  ////////////////////////////////////////////////
  // softmax scaling
  float sm_scale = 1.0;

  ////////////////////////////////////////////////
  // Parameters for alibi positional encoding
  ////////////////////////////////////////////////
  const float* __restrict__ alibi_slopes_ptr = nullptr;  // [n_heads]

  ////////////////////////////////////////////////
  // Parameters for scheduling
  ////////////////////////////////////////////////
  // TODO: remove it after persistent kernel
  // int max_q_len = 0;
};

}  // namespace llm
