#pragma once
#include <cute/config.hpp>
#include <cute/tensor.hpp>

#include "gather_tensor.hpp"
#include "mla_params.h"

namespace llm {
using namespace cute;

template <typename Params>
struct MLATile {
  static_assert(cute::dependent_false<Params>, "not implemented");
};

// AttentionTile specialization for AttentionParams
template <>
struct MLATile<MLAParams> {
  // NOLINTNEXTLINE
  const MLAParams& params_;

  CUTE_HOST_DEVICE MLATile(const MLAParams& params) : params_(params) {}

  // return the query/output tile: (q_packed_len, head_dim)
  // return q_rope tile: (q_packed_len, qk_rope_head_dim)
  template <typename Element>
  CUTE_HOST_DEVICE auto get_qo_tile(int batch_idx) const {
    // (batch, seq, head, dim)
    const auto q_packed_len = params_.q_len * params_.n_heads;
    const auto q_offset = batch_idx * get<0>(params_.q_stride);
    auto q =
        make_tensor(make_gmem_ptr((const Element*)params_.q_ptr + q_offset),
                    make_shape(q_packed_len, params_.head_dim),
                    make_stride(get<2>(params_.q_stride), _1{}));
    
    // (batch, seq, head, rope_head_dim)
    const auto q_rope_offset = batch_idx * get<0>(params_.q_rope_stride);
    auto q_rope = make_tensor(
        make_gmem_ptr((const Element*)params_.q_rope_ptr + q_rope_offset),
        make_shape(q_packed_len, params_.rope_head_dim),
        make_stride(get<2>(params_.q_rope_stride), _1{}));

    // (batch, seq, head, dim)
    const auto o_offset = batch_idx * get<0>(params_.o_stride);
    auto o = make_tensor(make_gmem_ptr((Element*)params_.o_ptr + o_offset),
                         make_shape(q_packed_len, params_.head_dim),
                         make_stride(get<2>(params_.o_stride), _1{}));
    return make_tuple(q, q_rope, o);
  }

  // return the key/value tile: (kv_len, head_dim)
  template <typename Element>
  CUTE_HOST_DEVICE auto get_kv_tile(int batch_idx) const {
    // (batch, seq, dim)
    const auto kv_offset = batch_idx * get<0>(params_.kv_stride);
    // k[batch_idx, :, kv_head_idx, :]
    auto kv =
        make_tensor(make_gmem_ptr((const Element*)params_.kv_ptr + kv_offset),
                    make_shape(params_.kv_len, params_.head_dim),
                    make_stride(get<1>(params_.kv_stride), _1{}));
    
    // (batch, seq, rope_head_dim)
    const auto k_rope_offset = batch_idx * get<0>(params_.k_rope_stride);
    auto k_rope = make_tensor(
        make_gmem_ptr((const Element*)params_.k_rope_ptr + k_rope_offset),
        make_shape(params_.kv_len, params_.rope_head_dim),
        make_stride(get<1>(params_.k_rope_stride), _1{}));
    return make_tuple(kv, k_rope);
  }
};

}  // namespace llm