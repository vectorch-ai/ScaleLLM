#pragma once
#include <ATen/core/interned_strings.h>

#include <cute/config.hpp>
#include <cute/tensor.hpp>

#include "attention_params.h"
#include "gather_tensor.hpp"

namespace llm {
using namespace cute;

template <typename Params>
struct AttentionTile {
  static_assert(cute::dependent_false<Params>, "not implemented");
};

// AttentionTile specialization for AttentionParams
template <>
struct AttentionTile<AttentionParams> {
  // NOLINTNEXTLINE
  const AttentionParams& params_;

  CUTE_HOST_DEVICE AttentionTile(const AttentionParams& params)
      : params_(params) {}

  // return the query tile: (q_len, head_dim)
  template <typename Element>
  CUTE_HOST_DEVICE auto get_q_tile(int batch_idx, int head_idx) const {
    // (batch, head, len, d)
    const auto offset = batch_idx * get<0>(params_.q_stride) +
                        head_idx * get<1>(params_.q_stride);
    // q[batch_idx, head_idx, :, :]
    return make_tensor(make_gmem_ptr((const Element*)params_.q_ptr + offset),
                       make_shape(params_.q_len, params_.head_dim),
                       make_stride(get<2>(params_.q_stride), _1{}));
  }

  // return the output tile: (q_len, head_dim)
  template <typename Element>
  CUTE_HOST_DEVICE auto get_o_tile(int batch_idx, int head_idx) const {
    // (batch, head, len, d)
    const auto offset = batch_idx * get<0>(params_.o_stride) +
                        head_idx * get<1>(params_.o_stride);
    // o[batch_idx, head_idx, :, :]
    return make_tensor(make_gmem_ptr((Element*)params_.o_ptr + offset),
                       make_shape(params_.q_len, params_.head_dim),
                       make_stride(get<2>(params_.o_stride), _1{}));
  }

  // return the key tile: (kv_len, head_dim)
  template <typename Element>
  CUTE_HOST_DEVICE auto get_k_tile(int batch_idx, int kv_head_idx) const {
    // (batch, head, len, d)
    const auto offset = batch_idx * get<0>(params_.k_stride) +
                        kv_head_idx * get<1>(params_.k_stride);
    // k[batch_idx, kv_head_idx, :, :]
    return make_tensor(make_gmem_ptr((const Element*)params_.k_ptr + offset),
                       make_shape(params_.kv_len, params_.head_dim),
                       make_stride(get<2>(params_.k_stride), _1{}));
  }

  // return the value tile: (kv_len, head_dim)
  template <typename Element>
  CUTE_HOST_DEVICE auto get_v_tile(int batch_idx, int kv_head_idx) const {
    // (batch, head, len, d)
    const auto offset = batch_idx * get<0>(params_.v_stride) +
                        kv_head_idx * get<1>(params_.v_stride);
    // v[batch_idx, kv_head_idx, :, :]
    return make_tensor(make_gmem_ptr((const Element*)params_.v_ptr + offset),
                       make_shape(params_.kv_len, params_.head_dim),
                       make_stride(get<2>(params_.v_stride), _1{}));
  }
};

// variable length sequence
template <>
struct AttentionTile<VarLenAttentionParams> {
  // NOLINTNEXTLINE
  const VarLenAttentionParams& params_;

  CUTE_HOST_DEVICE AttentionTile(const VarLenAttentionParams& params)
      : params_(params) {}

  // return the query tile: (q_len, head_dim)
  template <typename Element>
  CUTE_HOST_DEVICE auto get_q_tile(int batch_idx, int head_idx) const {
    const auto begin = params_.cu_seqlens_q[batch_idx];
    const auto q_len = params_.cu_seqlens_q[batch_idx + 1] - begin;
    // (head, len, d)
    const auto offset =
        head_idx * get<0>(params_.q_stride) + begin * get<1>(params_.q_stride);
    // q[head_idx, begin:begin + q_len, :]
    return make_tensor(make_gmem_ptr((const Element*)params_.q_ptr + offset),
                       make_shape(q_len, params_.head_dim),
                       make_stride(get<1>(params_.q_stride), _1{}));
  }

  // return the output tile: (q_len, head_dim)
  template <typename Element>
  CUTE_HOST_DEVICE auto get_o_tile(int batch_idx, int head_idx) const {
    const auto begin = params_.cu_seqlens_q[batch_idx];
    const auto o_len = params_.cu_seqlens_q[batch_idx + 1] - begin;
    // (seq, head, len, d)
    const auto offset =
        head_idx * get<0>(params_.o_stride) + begin * get<1>(params_.o_stride);
    // o[head_idx, begin:begin + o_len, :]
    return make_tensor(make_gmem_ptr((Element*)params_.o_ptr + offset),
                       make_shape(o_len, params_.head_dim),
                       make_stride(get<1>(params_.o_stride), _1{}));
  }

  // return the key tile: (kv_len, head_dim)
  template <typename Element>
  CUTE_HOST_DEVICE auto get_k_tile(int batch_idx, int kv_head_idx) const {
    const auto begin = params_.cu_seqlens_kv[batch_idx];
    const auto kv_len = params_.cu_seqlens_kv[batch_idx + 1] - begin;
    // (seq, head, len, d)
    const auto offset = kv_head_idx * get<0>(params_.k_stride) +
                        begin * get<1>(params_.k_stride);
    // k[kv_head_idx, begin:begin + kv_len, :]
    return make_tensor(make_gmem_ptr((const Element*)params_.k_ptr + offset),
                       make_shape(kv_len, params_.head_dim),
                       make_stride(get<1>(params_.k_stride), _1{}));
  }

  // return the value tile: (kv_len, head_dim)
  template <typename Element>
  CUTE_HOST_DEVICE auto get_v_tile(int batch_idx, int kv_head_idx) const {
    const auto begin = params_.cu_seqlens_kv[batch_idx];
    const auto kv_len = params_.cu_seqlens_kv[batch_idx + 1] - begin;
    // (seq, head, len, d)
    const auto offset = kv_head_idx * get<0>(params_.v_stride) +
                        begin * get<1>(params_.v_stride);
    // v[kv_head_idx, begin:begin + kv_len, :]
    return make_tensor(make_gmem_ptr((const Element*)params_.v_ptr + offset),
                       make_shape(kv_len, params_.head_dim),
                       make_stride(get<1>(params_.v_stride), _1{}));
  }
};

// paged KV cache
template <>
struct AttentionTile<PagedKVAttentionParams> {
  // NOLINTNEXTLINE
  const PagedKVAttentionParams& params_;

  CUTE_HOST_DEVICE AttentionTile(const PagedKVAttentionParams& params)
      : params_(params) {}

  // return the query tile: (q_len, head_dim)
  template <typename Element>
  CUTE_HOST_DEVICE auto get_q_tile(int batch_idx, int head_idx) const {
    const auto begin = params_.cu_seqlens_q[batch_idx];
    const auto q_len = params_.cu_seqlens_q[batch_idx + 1] - begin;
    // (seq, head, len, d)
    const auto offset =
        head_idx * get<0>(params_.q_stride) + begin * get<1>(params_.q_stride);
    // q[head_idx, begin:begin + q_len, :]
    return make_tensor(make_gmem_ptr((const Element*)params_.q_ptr + offset),
                       make_shape(q_len, params_.head_dim),
                       make_stride(get<1>(params_.q_stride), _1{}));
  }

  // return the output tile: (q_len, head_dim)
  template <typename Element>
  CUTE_HOST_DEVICE auto get_o_tile(int batch_idx, int head_idx) const {
    const auto begin = params_.cu_seqlens_q[batch_idx];
    const auto o_len = params_.cu_seqlens_q[batch_idx + 1] - begin;
    // (seq, head, len, d)
    const auto offset =
        head_idx * get<0>(params_.o_stride) + begin * get<1>(params_.o_stride);
    // o[head_idx, begin:begin + o_len, :]
    return make_tensor(make_gmem_ptr((Element*)params_.o_ptr + offset),
                       make_shape(o_len, params_.head_dim),
                       make_stride(get<1>(params_.o_stride), _1{}));
  }

  // return the key tile: (kv_len, head_dim)
  template <typename Element>
  CUTE_HOST_DEVICE auto get_k_tile(int batch_idx, int kv_head_idx) const {
    const auto kv_len =
        params_.cu_seqlens_kv[batch_idx + 1] - params_.cu_seqlens_kv[batch_idx];

    // map seq_idx to slot_idx
    const int* block_table =
        params_.block_table + params_.cu_block_lens[batch_idx];
    const int block_size = params_.block_size;
    auto idx_to_slot = [block_table, block_size](int idx) {
      return block_table[idx / block_size] * block_size + idx % block_size;
    };

    // v[kv_head_idx, :, :]
    const auto offset = kv_head_idx * get<0>(params_.k_stride);
    return make_gather_tensor(
        make_gmem_ptr((const Element*)params_.k_ptr + offset),
        make_shape(kv_len, params_.head_dim),
        make_stride(get<1>(params_.k_stride), _1{}),
        idx_to_slot);
  }

  // return the value tile: (kv_len, head_dim)
  template <typename Element>
  CUTE_HOST_DEVICE auto get_v_tile(int batch_idx, int kv_head_idx) const {
    const auto kv_len =
        params_.cu_seqlens_kv[batch_idx + 1] - params_.cu_seqlens_kv[batch_idx];

    // map seq_idx to slot_idx
    const int* block_table =
        params_.block_table + params_.cu_block_lens[batch_idx];
    const int block_size = params_.block_size;
    auto idx_to_slot = [block_table, block_size](int idx) {
      return block_table[idx / block_size] * block_size + idx % block_size;
    };

    // v[kv_head_idx, :, :]
    const auto offset = kv_head_idx * get<0>(params_.v_stride);
    return make_gather_tensor(
        make_gmem_ptr((const Element*)params_.v_ptr + offset),
        make_shape(kv_len, params_.head_dim),
        make_stride(get<1>(params_.v_stride), _1{}),
        idx_to_slot);
  }
};

}  // namespace llm