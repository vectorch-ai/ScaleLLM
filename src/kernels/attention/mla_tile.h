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

  // return the query/output tile: (q_len, head_dim)
  template <typename Element>
  CUTE_HOST_DEVICE auto get_qo_tile(int batch_idx, int kv_head_idx) const {
    // (batch, seq, head, dim)

    // packed all q/o in the same kv head group together
    // q/o [batch, n_tokens, n_heads, dim]
    //   => q/o [*batch_idx, n_tokens, n_heads, dim]
    //   => q/o [n_tokens, group_size, n_kv_heads, dim]
    //   => q/o [n_tokens, group_size, *kv_head_idx, dim]
    //   => q/o [(group_size, n_tokens), dim]
    //   => q/o [packed_len, dim]
    const auto group_size = params_.n_heads;
    const auto head_base = kv_head_idx * group_size;
    auto packed_idx_to_coord = [group_size, head_base](int packed_idx) {
      const int idx = packed_idx / group_size;
      const int offset = packed_idx % group_size;
      // (group_size, n_tokens)
      return make_coord(head_base + offset, idx);
    };

    const auto packed_len = params_.q_len * group_size;
    const auto q_offset = batch_idx * get<0>(params_.q_stride);
    auto q = make_gather_tensor(
        make_gmem_ptr((const Element*)params_.q_ptr + q_offset),
        make_shape(packed_len,
                   params_.qk_nope_head_dim + params_.qk_rope_head_dim),
        make_stride(
            make_stride(get<2>(params_.q_stride), get<1>(params_.q_stride)),
            _1{}),
        packed_idx_to_coord);

    const auto o_offset = batch_idx * get<0>(params_.o_stride);
    auto o = make_gather_tensor(
        make_gmem_ptr((Element*)params_.o_ptr + o_offset),
        make_shape(packed_len, params_.qk_nope_head_dim),
        make_stride(
            make_stride(get<2>(params_.o_stride), get<1>(params_.o_stride)),
            _1{}),
        packed_idx_to_coord);
    return make_tuple(q, o);
  }

  // return the key/value tile: (kv_len, head_dim)
  template <typename Element>
  CUTE_HOST_DEVICE auto get_kv_tile(int batch_idx, int kv_head_idx) const {
    // (batch, seq, kv_head, dim)
    const auto kv_offset = batch_idx * get<0>(params_.kv_stride) +
                           kv_head_idx * get<2>(params_.kv_stride);
    // k[batch_idx, :, kv_head_idx, :]
    auto kv =
        make_tensor(make_gmem_ptr((const Element*)params_.k_ptr + kv_offset),
                    make_shape(params_.kv_len, params_.v_head_dim),
                    make_stride(get<1>(params_.kv_stride), _1{}));
    return kv;
  }
};

}  // namespace llm