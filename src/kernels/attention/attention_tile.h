#pragma once
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

  // return the query/output tile: (q_len, head_dim)
  template <typename Element>
  CUTE_HOST_DEVICE auto get_qo_tile(int batch_idx, int head_idx) const {
    // (batch, seq, head, dim)
    const auto q_offset = batch_idx * get<0>(params_.q_stride) +
                          head_idx * get<2>(params_.q_stride);
    const auto o_offset = batch_idx * get<0>(params_.o_stride) +
                          head_idx * get<2>(params_.o_stride);

    // q[batch_idx, :, head_idx, :]
    auto q =
        make_tensor(make_gmem_ptr((const Element*)params_.q_ptr + q_offset),
                    make_shape(params_.q_len, params_.head_dim),
                    make_stride(get<1>(params_.q_stride), _1{}));
    auto o = make_tensor(make_gmem_ptr((Element*)params_.o_ptr + o_offset),
                         make_shape(params_.q_len, params_.head_dim),
                         make_stride(get<1>(params_.o_stride), _1{}));
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
struct AttentionTile<VarLenAttentionParams> {
  // NOLINTNEXTLINE
  const VarLenAttentionParams& params_;

  CUTE_HOST_DEVICE AttentionTile(const VarLenAttentionParams& params)
      : params_(params) {}

  // return the query tile: (q_len, head_dim)
  template <typename Element>
  CUTE_HOST_DEVICE auto get_qo_tile(int batch_idx, int head_idx) const {
    const auto begin = params_.q_cu_lens[batch_idx];
    const auto qo_len = params_.q_cu_lens[batch_idx + 1] - begin;
    // (seq, head, dim)
    const auto q_offset =
        begin * get<0>(params_.q_stride) + head_idx * get<1>(params_.q_stride);
    const auto o_offset =
        begin * get<0>(params_.o_stride) + head_idx * get<1>(params_.o_stride);

    // q[begin:begin + q_len, head_idx, :]
    auto q =
        make_tensor(make_gmem_ptr((const Element*)params_.q_ptr + q_offset),
                    make_shape(qo_len, params_.head_dim),
                    make_stride(get<0>(params_.q_stride), _1{}));
    // o[begin:begin + o_len, head_idx, :]
    auto o = make_tensor(make_gmem_ptr((Element*)params_.o_ptr + o_offset),
                         make_shape(qo_len, params_.head_dim),
                         make_stride(get<0>(params_.o_stride), _1{}));
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
struct AttentionTile<PagedKVAttentionParams> {
  // NOLINTNEXTLINE
  const PagedKVAttentionParams& params_;

  CUTE_HOST_DEVICE AttentionTile(const PagedKVAttentionParams& params)
      : params_(params) {}

  // return the query/output tile: (q_len, head_dim)
  template <typename Element>
  CUTE_HOST_DEVICE auto get_qo_tile(int batch_idx, int head_idx) const {
    const auto begin = params_.q_cu_lens[batch_idx];
    const auto qo_len = params_.q_cu_lens[batch_idx + 1] - begin;
    // (seq, head, dim)
    const auto q_offset =
        begin * get<0>(params_.q_stride) + head_idx * get<1>(params_.q_stride);
    const auto o_offset =
        begin * get<0>(params_.o_stride) + head_idx * get<1>(params_.o_stride);

    // q[begin:begin + q_len, head_idx, :]
    auto q =
        make_tensor(make_gmem_ptr((const Element*)params_.q_ptr + q_offset),
                    make_shape(qo_len, params_.head_dim),
                    make_stride(get<0>(params_.q_stride), _1{}));
    // o[begin:begin + o_len, head_idx, :]
    auto o = make_tensor(make_gmem_ptr((Element*)params_.o_ptr + o_offset),
                         make_shape(qo_len, params_.head_dim),
                         make_stride(get<0>(params_.o_stride), _1{}));
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
    const int block_size = params_.block_size;
    auto idx_to_slot = [block_table,
                        block_size = cutlass::FastDivmod(block_size)](int idx) {
      int block_idx;     // idx / block_size;
      int block_offset;  // idx % block_size
      block_size.fast_divmod(block_idx, block_offset, idx);
      return block_table[block_idx] * block_size + block_offset;
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