#pragma once

#include <cute/config.hpp>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>

#include "fast_math.h"
namespace llm {

// common params for attention kernels
struct MLAParamsCommon {
  const void* __restrict__ q_ptr = nullptr;
  const void* __restrict__ kv_ptr = nullptr;

  const void* __restrict__ q_rope_ptr = nullptr;
  const void* __restrict__ k_rope_ptr = nullptr;

  void* __restrict__ o_ptr = nullptr;

  // input shapes
  int batch_size = 0;

  int n_heads = 0;
  int head_dim = 0;
  int rope_head_dim = 0;

  // softmax scaling
  float sm_scale = 1.0;

  // used for scheduling
  // TODO: remove it after persistent kernel
  int max_q_len = 0;

  // block size, only used for paged KV cache
  int block_size = 1;

  // private:
  // used for performance optimization, don't change it
  bool normalized = false;
  float sm_scale_log2 = 0.0;
  int32_t block_shift_right = 0;
  int32_t block_mask = 0;
  FastDivmod group_size;

  // used to initialize the params that used for performance optimization
  void normalize() {
    if (normalized) {
      // already normalized
      return;
    }
    sm_scale_log2 = static_cast<float>(sm_scale * M_LOG2E);

    // block size must be power of 2
    assert(block_size > 0 && is_pow2(block_size));
    block_shift_right = log2(block_size);
    block_mask = block_size - 1;

    // for MLA/MQA, group_size = n_heads
    group_size = n_heads;

    normalized = true;
  }
};

struct MLAParams : public MLAParamsCommon {
  // Q/O: (batch, seq, head, dim): last dimension is contiguous
  using Stride = cute::Stride<int64_t, int64_t, int64_t, cute::_1>;
  // KV: (batch, seq, dim): last dimension is contiguous
  using KV_Stride = cute::Stride<int64_t, int64_t, cute::_1>;

  Stride q_stride;
  Stride q_rope_stride;

  KV_Stride kv_stride;
  KV_Stride k_rope_stride;

  Stride o_stride;

  // input shapes
  int q_len = 0;
  int kv_len = 0;
};

// paged KV cache + variable length sequence
struct MLAPagedKVParams : public MLAParamsCommon {
  // Q/O: (seq, head, dim): last dimension is contiguous
  using Stride = cute::Stride<int64_t, int64_t, cute::_1>;
  // KV: (seq, dim): last dimension is contiguous
  using KV_Stride = cute::Stride<int64_t, cute::_1>;

  Stride q_stride;
  Stride q_rope_stride;

  KV_Stride kv_stride;
  KV_Stride k_rope_stride;

  Stride o_stride;

  // input shapes
  // array of length batch_size + 1 holding starting offset of each sequence.
  const int* __restrict__ q_cu_lens = nullptr;
  const int* __restrict__ kv_cu_lens = nullptr;

  // Paged KV cache
  // the first slot id of each block
  const int* __restrict__ block_table = nullptr;
  // array of length batch_size + 1 holding starting offset of each sequence.
  const int* __restrict__ block_cu_lens = nullptr;
};

}  // namespace llm
