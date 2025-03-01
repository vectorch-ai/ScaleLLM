#pragma once
#include <cute/config.hpp>
#include <cute/tensor.hpp>

#include "fast_math.h"
#include "gather_tensor.hpp"
#include "mha_params.h"

namespace llm {
using namespace cute;

template <typename Params>
struct MHATile {
  static_assert(cute::dependent_false<Params>, "not implemented");
};

// AttentionTile specialization for AttentionParams
template <>
struct MHATile<MHAParams> {
  const MHAParams& params_;
  const int batch_idx_;
  const int kv_head_idx_;

  CUTE_HOST_DEVICE MHATile(const MHAParams& params,
                           int batch_idx,
                           int kv_head_idx)
      : params_(params), batch_idx_(batch_idx), kv_head_idx_(kv_head_idx) {}

  // return the query/output tile: (q_len, head_dim)
  template <typename Element>
  CUTE_HOST_DEVICE auto get_qo_tile() const {
    // (batch, seq, head, dim)

    // packed all q/o in the same kv head group together
    const auto head_base = kv_head_idx_ * params_.group_size;
    auto packed_idx_to_coord = [this, head_base](int packed_idx) {
      int idx, offset;
      params_.group_size.divmod(packed_idx, idx, offset);
      return make_coord(idx, head_base + offset);
    };

    const auto packed_len = params_.q_len * params_.group_size;
    const auto q_offset = batch_idx_ * get<0>(params_.q_stride);
    auto q = make_gather_tensor(
        make_gmem_ptr((const Element*)params_.q_ptr + q_offset),
        make_shape(packed_len, params_.head_dim),
        make_stride(
            make_stride(get<1>(params_.q_stride), get<2>(params_.q_stride)),
            _1{}),
        packed_idx_to_coord);

    const auto o_offset = batch_idx_ * get<0>(params_.o_stride);
    auto o = make_gather_tensor(
        make_gmem_ptr((Element*)params_.o_ptr + o_offset),
        make_shape(packed_len, params_.head_dim),
        make_stride(
            make_stride(get<1>(params_.o_stride), get<2>(params_.o_stride)),
            _1{}),
        packed_idx_to_coord);
    return make_tuple(q, o);
  }

  // return the key/value tile: (kv_len, head_dim)
  template <typename Element>
  CUTE_HOST_DEVICE auto get_kv_tile() const {
    // (batch, seq, kv_head, dim)
    const auto k_offset = batch_idx_ * get<0>(params_.k_stride) +
                          kv_head_idx_ * get<2>(params_.k_stride);
    const auto v_offset = batch_idx_ * get<0>(params_.v_stride) +
                          kv_head_idx_ * get<2>(params_.v_stride);
    // k[batch_idx, :, kv_head_idx, :]
    auto k =
        make_tensor(make_gmem_ptr((const Element*)params_.k_ptr + k_offset),
                    make_shape(params_.kv_len, params_.head_dim),
                    make_stride(get<1>(params_.k_stride), _1{}));
    // v[batch_idx, :, kv_head_idx, :]
    auto v =
        make_tensor(make_gmem_ptr((const Element*)params_.v_ptr + v_offset),
                    make_shape(params_.kv_len, params_.head_dim),
                    make_stride(get<1>(params_.v_stride), _1{}));
    return make_tuple(k, v);
  }
};

// paged KV cache + variable length sequence
template <>
struct MHATile<MHAPagedKVParams> {
  // NOLINTNEXTLINE
  const MHAPagedKVParams& params_;
  const int batch_idx_;
  const int kv_head_idx_;

  CUTE_HOST_DEVICE MHATile(const MHAPagedKVParams& params,
                           int batch_idx,
                           int kv_head_idx)
      : params_(params), batch_idx_(batch_idx), kv_head_idx_(kv_head_idx) {}

  // return the query/output tile: (q_len, head_dim)
  template <typename Element>
  CUTE_HOST_DEVICE auto get_qo_tile() const {
    const auto begin = params_.q_cu_lens[batch_idx_];
    const auto qo_len = params_.q_cu_lens[batch_idx_ + 1] - begin;
    const auto head_base = kv_head_idx_ * params_.group_size;
    auto packed_idx_to_coord = [this, head_base](int packed_idx) {
      int idx, offset;
      params_.group_size.divmod(packed_idx, idx, offset);
      return make_coord(idx, head_base + offset);
    };

    const auto packed_len = qo_len * params_.group_size;
    const auto q_offset = begin * get<0>(params_.q_stride);
    auto q = make_gather_tensor(
        make_gmem_ptr((const Element*)params_.q_ptr + q_offset),
        make_shape(packed_len, params_.head_dim),
        make_stride(
            make_stride(get<0>(params_.q_stride), get<1>(params_.q_stride)),
            _1{}),
        packed_idx_to_coord);

    const auto o_offset = begin * get<0>(params_.o_stride);
    auto o = make_gather_tensor(
        make_gmem_ptr((Element*)params_.o_ptr + o_offset),
        make_shape(packed_len, params_.head_dim),
        make_stride(
            make_stride(get<0>(params_.o_stride), get<1>(params_.o_stride)),
            _1{}),
        packed_idx_to_coord);
    return make_tuple(q, o);
  }

  // return the key/value tile: (kv_len, head_dim)
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

    // v[:, kv_head_idx, :]
    const auto k_offset = kv_head_idx_ * get<1>(params_.k_stride);
    auto k = make_gather_tensor(
        make_gmem_ptr((const Element*)params_.k_ptr + k_offset),
        make_shape(kv_len, params_.head_dim),
        make_stride(get<0>(params_.k_stride), _1{}),
        idx_to_slot);

    // v[:, kv_head_idx, :]
    const auto v_offset = kv_head_idx_ * get<1>(params_.v_stride);
    auto v = make_gather_tensor(
        make_gmem_ptr((const Element*)params_.v_ptr + v_offset),
        make_shape(kv_len, params_.head_dim),
        make_stride(get<0>(params_.v_stride), _1{}),
        idx_to_slot);
    return make_tuple(k, v);
  }
};

}  // namespace llm