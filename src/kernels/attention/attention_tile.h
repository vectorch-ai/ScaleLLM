#pragma once
#include <cute/config.hpp>
#include <cute/tensor.hpp>

#include "attention_params.h"

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

  // return the query tile for a given block: (q_len, head_dim)
  template <typename Element>
  CUTE_HOST_DEVICE auto get_query_tile(int batch_idx, int head_idx) const {
    // (batch, head, len, d)
    const auto offset = batch_idx * get<0>(params_.q_stride) +
                        head_idx * get<1>(params_.q_stride);
    return make_tensor(make_gmem_ptr((const Element*)params_.q_ptr + offset),
                       make_shape(params_.q_len, params_.head_dim),
                       make_stride(get<2>(params_.q_stride), _1{}));
  }

  // return the output tile for a given block: (q_len, head_dim)
  template <typename Element>
  CUTE_HOST_DEVICE auto get_output_tile(int batch_idx, int head_idx) const {
    // (batch, head, len, d)
    const auto offset = batch_idx * get<0>(params_.o_stride) +
                        head_idx * get<1>(params_.o_stride);
    return make_tensor(make_gmem_ptr((Element*)params_.o_ptr + offset),
                       make_shape(params_.q_len, params_.head_dim),
                       make_stride(get<2>(params_.o_stride), _1{}));
  }

  // return the key tile for a given block: (kv_len, head_dim)
  template <typename Element>
  CUTE_HOST_DEVICE auto get_key_tile(int batch_idx, int kv_head_idx) const {
    // (batch, head, len, d)
    const auto offset = batch_idx * get<0>(params_.k_stride) +
                        kv_head_idx * get<1>(params_.k_stride);
    return make_tensor(make_gmem_ptr((Element*)params_.k_ptr + offset),
                       make_shape(params_.kv_len, params_.head_dim),
                       make_stride(get<2>(params_.k_stride), _1{}));
  }

  // return the value tile for a given block: (kv_len, head_dim)
  template <typename Element>
  CUTE_HOST_DEVICE auto get_value_tile(int batch_idx, int kv_head_idx) const {
    // (batch, head, len, d)
    const auto offset = batch_idx * get<0>(params_.v_stride) +
                        kv_head_idx * get<1>(params_.v_stride);
    return make_tensor(make_gmem_ptr((Element*)params_.v_ptr + offset),
                       make_shape(params_.kv_len, params_.head_dim),
                       make_stride(get<2>(params_.v_stride), _1{}));
  }

  CUTE_HOST_DEVICE int get_q_len() const { return params_.q_len; }
  CUTE_HOST_DEVICE int get_kv_len() const { return params_.kv_len; }
};

// TODO: variable length sequence
template <>
struct AttentionTile<VarLenAttentionParams> {};

// TODO: paged KV cache
template <>
struct AttentionTile<PagedKVAttentionParams> {};

}  // namespace llm