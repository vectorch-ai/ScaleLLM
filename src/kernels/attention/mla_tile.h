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
  const MLAParams& params_;
  const int batch_idx_;

  CUTE_HOST_DEVICE MLATile(const MLAParams& params, int batch_idx)
      : params_(params), batch_idx_(batch_idx) {}

  // return the query/output tile: (q_packed_len, head_dim)
  // return q_rope tile: (q_packed_len, rope_head_dim)
  template <typename Element>
  CUTE_HOST_DEVICE auto get_qo_tile() const {
    // (batch, seq, head, dim)
    const auto q_packed_len = params_.q_len * params_.group_size;
    const auto q_offset = batch_idx_ * get<0>(params_.q_stride);
    auto q =
        make_tensor(make_gmem_ptr((const Element*)params_.q_ptr + q_offset),
                    make_shape(q_packed_len, params_.head_dim),
                    make_stride(get<2>(params_.q_stride), _1{}));

    // (batch, seq, head, rope_head_dim)
    const auto q_rope_offset = batch_idx_ * get<0>(params_.q_rope_stride);
    auto q_rope = make_tensor(
        make_gmem_ptr((const Element*)params_.q_rope_ptr + q_rope_offset),
        make_shape(q_packed_len, params_.rope_head_dim),
        make_stride(get<2>(params_.q_rope_stride), _1{}));

    // (batch, seq, head, dim)
    const auto o_offset = batch_idx_ * get<0>(params_.o_stride);
    auto o = make_tensor(make_gmem_ptr((Element*)params_.o_ptr + o_offset),
                         make_shape(q_packed_len, params_.head_dim),
                         make_stride(get<2>(params_.o_stride), _1{}));
    return make_tuple(q, q_rope, o);
  }

  // return the kv: (kv_len, head_dim)
  // return k_rope: (kv_len, rope_head_dim)
  template <typename Element>
  CUTE_HOST_DEVICE auto get_kv_tile() const {
    // (batch, seq, dim)
    const auto kv_offset = batch_idx_ * get<0>(params_.kv_stride);
    // k[batch_idx, :, kv_head_idx, :]
    auto kv =
        make_tensor(make_gmem_ptr((const Element*)params_.kv_ptr + kv_offset),
                    make_shape(params_.kv_len, params_.head_dim),
                    make_stride(get<1>(params_.kv_stride), _1{}));

    // (batch, seq, rope_head_dim)
    const auto k_rope_offset = batch_idx_ * get<0>(params_.k_rope_stride);
    auto k_rope = make_tensor(
        make_gmem_ptr((const Element*)params_.k_rope_ptr + k_rope_offset),
        make_shape(params_.kv_len, params_.rope_head_dim),
        make_stride(get<1>(params_.k_rope_stride), _1{}));
    return make_tuple(kv, k_rope);
  }
};

// paged KV cache + variable length sequence
template <>
struct MLATile<MLAPagedKVParams> {
  // NOLINTNEXTLINE
  const MLAPagedKVParams& params_;
  const int batch_idx_;

  CUTE_HOST_DEVICE MLATile(const MLAPagedKVParams& params, int batch_idx)
      : params_(params), batch_idx_(batch_idx) {}

  // return the query/output tile: (q_packed_len, head_dim)
  // return q_rope tile: (q_packed_len, rope_head_dim)
  template <typename Element>
  CUTE_HOST_DEVICE auto get_qo_tile() const {
    const auto begin = params_.q_cu_lens[batch_idx_];
    const auto qo_len = params_.q_cu_lens[batch_idx_ + 1] - begin;

    // (seq, head, dim)
    const auto q_packed_len = qo_len * params_.group_size;
    const auto q_offset = begin * get<0>(params_.q_stride);

    auto q =
        make_tensor(make_gmem_ptr((const Element*)params_.q_ptr + q_offset),
                    make_shape(q_packed_len, params_.head_dim),
                    make_stride(get<1>(params_.q_stride), _1{}));

    // (seq, head, rope_head_dim)
    const auto q_rope_offset = begin * get<0>(params_.q_rope_stride);
    auto q_rope = make_tensor(
        make_gmem_ptr((const Element*)params_.q_rope_ptr + q_rope_offset),
        make_shape(q_packed_len, params_.rope_head_dim),
        make_stride(get<1>(params_.q_rope_stride), _1{}));

    // (seq, head, dim)
    const auto o_offset = begin * get<0>(params_.o_stride);
    auto o = make_tensor(make_gmem_ptr((Element*)params_.o_ptr + o_offset),
                         make_shape(q_packed_len, params_.head_dim),
                         make_stride(get<1>(params_.o_stride), _1{}));
    return make_tuple(q, q_rope, o);
  }

  // return the kv: (kv_len, head_dim)
  // return k_rope: (kv_len, rope_head_dim)
  template <typename Element>
  CUTE_HOST_DEVICE auto get_kv_tile() const {
    const auto kv_len =
        params_.kv_cu_lens[batch_idx_ + 1] - params_.kv_cu_lens[batch_idx_];

    // map seq_idx to slot_idx
    const int* block_table =
        params_.block_table + params_.block_cu_lens[batch_idx_];
    auto idx_to_slot = [block_table,
                        shr = params_.block_shift_right,
                        mask = params_.block_mask](int idx) {
      return block_table[idx >> shr] + (idx & mask);
    };

    // kv: (seq, dim)
    auto kv = make_gather_tensor(make_gmem_ptr((const Element*)params_.kv_ptr),
                                 make_shape(kv_len, params_.head_dim),
                                 make_stride(get<0>(params_.kv_stride), _1{}),
                                 idx_to_slot);

    // k_rope: (seq, rope_head_dim)
    auto k_rope =
        make_gather_tensor(make_gmem_ptr((const Element*)params_.k_rope_ptr),
                           make_shape(kv_len, params_.rope_head_dim),
                           make_stride(get<0>(params_.k_rope_stride), _1{}),
                           idx_to_slot);
    return make_tuple(kv, k_rope);
  }
};

}  // namespace llm