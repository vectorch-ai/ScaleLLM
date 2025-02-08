#pragma once

#include <cute/config.hpp>
#include <cute/tensor.hpp>

#include "cute/layout.hpp"
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

  // private:
  // used for performance optimization, don't change it
  bool normalized = false;
  float sm_scale_log2 = 0.0;

  // used to initialize the params that used for performance optimization
  void normalize() {
    if (normalized) {
      // already normalized
      return;
    }
    sm_scale_log2 = static_cast<float>(sm_scale * M_LOG2E);
    
    normalized = true;
  }
};

struct MLAParams : public MLAParamsCommon {
  // Q/O: (batch, seq, head, dim): last dimension is contiguous
  using Stride = cute::Stride<int64_t, int64_t, int64_t /*,_1*/>;
  // KV: (batch, seq, dim): last dimension is contiguous
  using KV_Stride = cute::Stride<int64_t, int64_t /*,_1*/>;

  Stride q_stride;
  Stride q_rope_stride;

  KV_Stride kv_stride;
  KV_Stride k_rope_stride;

  Stride o_stride;

  // input shapes
  int q_len = 0;
  int kv_len = 0;
};

}  // namespace llm