#pragma once
#include <cute/config.hpp>
#include <cute/tensor.hpp>

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
  // NOLINTNEXTLINE
  const MHAParams& params_;

  CUTE_HOST_DEVICE MHATile(const MHAParams& params) : params_(params) {}

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
    const auto group_size = params_.group_size;
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
        make_shape(packed_len, params_.head_dim),
        make_stride(
            make_stride(get<2>(params_.q_stride), get<1>(params_.q_stride)),
            _1{}),
        packed_idx_to_coord);

    const auto o_offset = batch_idx * get<0>(params_.o_stride);
    auto o = make_gather_tensor(
        make_gmem_ptr((Element*)params_.o_ptr + o_offset),
        make_shape(packed_len, params_.head_dim),
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
    const auto k_offset = batch_idx * get<0>(params_.k_stride) +
                          kv_head_idx * get<2>(params_.k_stride);
    const auto v_offset = batch_idx * get<0>(params_.v_stride) +
                          kv_head_idx * get<2>(params_.v_stride);
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

// variable length sequence
template <>
struct MHATile<MHAVarLenParams> {
  // NOLINTNEXTLINE
  const MHAVarLenParams& params_;

  CUTE_HOST_DEVICE MHATile(const MHAVarLenParams& params) : params_(params) {}

  // return the query tile: (q_len, head_dim)
  template <typename Element>
  CUTE_HOST_DEVICE auto get_qo_tile(int batch_idx, int kv_head_idx) const {
    const auto begin = params_.q_cu_lens[batch_idx];
    const auto qo_len = params_.q_cu_lens[batch_idx + 1] - begin;

    const auto group_size = params_.group_size;
    const auto head_base = kv_head_idx * group_size;
    auto packed_idx_to_coord = [group_size, head_base](int packed_idx) {
      const int idx = packed_idx / group_size;
      const int offset = packed_idx % group_size;
      // (group_size, n_tokens)
      return make_coord(head_base + offset, idx);
    };

    const auto packed_len = qo_len * group_size;
    const auto q_offset = begin * get<0>(params_.q_stride);
    auto q = make_gather_tensor(
        make_gmem_ptr((const Element*)params_.q_ptr + q_offset),
        make_shape(packed_len, params_.head_dim),
        make_stride(
            make_stride(get<1>(params_.q_stride), get<0>(params_.q_stride)),
            _1{}),
        packed_idx_to_coord);

    const auto o_offset = begin * get<0>(params_.o_stride);
    auto o = make_gather_tensor(
        make_gmem_ptr((Element*)params_.o_ptr + o_offset),
        make_shape(packed_len, params_.head_dim),
        make_stride(
            make_stride(get<1>(params_.o_stride), get<0>(params_.o_stride)),
            _1{}),
        packed_idx_to_coord);
    return make_tuple(q, o);
  }

  // return the key/value tile: (kv_len, head_dim)
  template <typename Element>
  CUTE_HOST_DEVICE auto get_kv_tile(int batch_idx, int kv_head_idx) const {
    const auto begin = params_.kv_cu_lens[batch_idx];
    const auto kv_len = params_.kv_cu_lens[batch_idx + 1] - begin;
    // (seq, head, dim)
    const auto k_offset = begin * get<0>(params_.k_stride) +
                          kv_head_idx * get<1>(params_.k_stride);
    const auto v_offset = begin * get<0>(params_.v_stride) +
                          kv_head_idx * get<1>(params_.v_stride);

    // k[begin:begin + kv_len, kv_head_idx, :]
    auto k =
        make_tensor(make_gmem_ptr((const Element*)params_.k_ptr + k_offset),
                    make_shape(kv_len, params_.head_dim),
                    make_stride(get<0>(params_.k_stride), _1{}));
    // v[begin:begin + kv_len, kv_head_idx, :]
    auto v =
        make_tensor(make_gmem_ptr((const Element*)params_.v_ptr + v_offset),
                    make_shape(kv_len, params_.head_dim),
                    make_stride(get<0>(params_.v_stride), _1{}));
    return make_tuple(k, v);
  }
};

// paged KV cache
template <>
struct MHATile<MHAPagedKVParams> {
  // NOLINTNEXTLINE
  const MHAPagedKVParams& params_;

  CUTE_HOST_DEVICE MHATile(const MHAPagedKVParams& params) : params_(params) {}

  // return the query/output tile: (q_len, head_dim)
  template <typename Element>
  CUTE_HOST_DEVICE auto get_qo_tile(int batch_idx, int kv_head_idx) const {
    const auto begin = params_.q_cu_lens[batch_idx];
    const auto qo_len = params_.q_cu_lens[batch_idx + 1] - begin;
    const auto group_size = params_.group_size;
    const auto head_base = kv_head_idx * group_size;
    auto packed_idx_to_coord = [group_size, head_base](int packed_idx) {
      const int idx = packed_idx / group_size;
      const int offset = packed_idx % group_size;
      // (group_size, n_tokens)
      return make_coord(head_base + offset, idx);
    };

    const auto packed_len = qo_len * group_size;
    const auto q_offset = begin * get<0>(params_.q_stride);
    auto q = make_gather_tensor(
        make_gmem_ptr((const Element*)params_.q_ptr + q_offset),
        make_shape(packed_len, params_.head_dim),
        make_stride(
            make_stride(get<1>(params_.q_stride), get<0>(params_.q_stride)),
            _1{}),
        packed_idx_to_coord);

    const auto o_offset = begin * get<0>(params_.o_stride);
    auto o = make_gather_tensor(
        make_gmem_ptr((Element*)params_.o_ptr + o_offset),
        make_shape(packed_len, params_.head_dim),
        make_stride(
            make_stride(get<1>(params_.o_stride), get<0>(params_.o_stride)),
            _1{}),
        packed_idx_to_coord);
    return make_tuple(q, o);
  }

  // return the key/value tile: (kv_len, head_dim)
  template <typename Element>
  CUTE_HOST_DEVICE auto get_kv_tile(int batch_idx, int kv_head_idx) const {
    const auto kv_len =
        params_.kv_cu_lens[batch_idx + 1] - params_.kv_cu_lens[batch_idx];

    // map seq_idx to slot_idx
    const int* block_table =
        params_.block_table + params_.block_cu_lens[batch_idx];
    auto idx_to_slot = [block_table,
                        right_shift = params_.block_shift_right,
                        mask = params_.block_mask](int idx) {
      // idx / block_size;
      const int block_idx = idx >> right_shift;
      // idx % block_size;
      const int block_offset = idx & mask;
      return block_table[block_idx] + block_offset;
    };

    // v[:, kv_head_idx, :]
    const auto k_offset = kv_head_idx * get<1>(params_.k_stride);
    auto k = make_gather_tensor(
        make_gmem_ptr((const Element*)params_.k_ptr + k_offset),
        make_shape(kv_len, params_.head_dim),
        make_stride(get<0>(params_.k_stride), _1{}),
        idx_to_slot);

    // v[:, kv_head_idx, :]
    const auto v_offset = kv_head_idx * get<1>(params_.v_stride);
    auto v = make_gather_tensor(
        make_gmem_ptr((const Element*)params_.v_ptr + v_offset),
        make_shape(kv_len, params_.head_dim),
        make_stride(get<0>(params_.v_stride), _1{}),
        idx_to_slot);
    return make_tuple(k, v);
  }
};

}  // namespace llm