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

  // used for scheduling
  // TODO: remove it after persistent kernel
  int max_q_len = 0;

  // mask
  int sliding_window = -1;

  // softcap
  float logits_soft_cap = 0.0;

  // softmax scaling
  float sm_scale = 1.0;

  // alibi
  const float* __restrict__ alibi_slopes_ptr = nullptr;  // [n_heads]

  // block size, only used for paged KV cache
  int block_size = 0;

  // private:
  // used for performance optimization, don't change it
  bool normalized = false;
  float sm_scale_log2 = 0.0;
  int32_t block_shift_right = 0;
  int32_t block_mask = 0;
  int group_size = 0;

  // used to initialize the params that used for performance optimization
  void normalize() {
    if (normalized) {
      // already normalized
      return;
    }

    if (logits_soft_cap > 0.0) {
      //    Softmax(x * sm_scale) + apply_logits_soft_cap
      // => Softmax(Tanh(x * sm_scale / soft_cap) * soft_cap)
      // => Softmax(S' * sm_scale') where
      //    S'        = Tanh(x * sm_scale / soft_cap)
      //              = Tanh(x * soft_cap')
      //    soft_cap' = sm_scale / soft_cap
      //    sm_scale' = soft_cap
      const auto sm_scale_hat = logits_soft_cap;
      logits_soft_cap = sm_scale / logits_soft_cap;
      sm_scale = sm_scale_hat;
    }
    sm_scale_log2 = static_cast<float>(sm_scale * M_LOG2E);

    // block size must be power of 2
    assert(block_size > 0 && (block_size & (block_size - 1)) == 0);
    auto int_log2 = [](int x) {
      int n = 0;
      while (x >>= 1) {
        ++n;
      }
      return n;
    };
    block_shift_right = int_log2(block_size);
    block_mask = block_size - 1;

    assert(n_heads % n_kv_heads == 0);
    group_size = n_heads / n_kv_heads;

    normalized = true;
  }
};

struct AttentionParams : public AttentionParamsCommon {
  // (batch, seq, head, dim): last dimension is contiguous
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
  // (seq, head, dim): last dimension is contiguous
  using Stride = cute::Stride<int64_t, int64_t /*,_1*/>;

  Stride q_stride;
  Stride k_stride;
  Stride v_stride;
  Stride o_stride;

  // input shapes
  // array of length batch_size + 1 holding starting offset of each sequence.
  const int* __restrict__ q_cu_lens = nullptr;
  const int* __restrict__ kv_cu_lens = nullptr;
};

// paged KV cache
struct PagedKVAttentionParams : public VarLenAttentionParams {
  // Paged KV cache
  // the first slot id of each block
  const int* __restrict__ block_table = nullptr;
  // array of length batch_size + 1 holding starting offset of each sequence.
  const int* __restrict__ block_cu_lens = nullptr;
};

}  // namespace llm