#pragma once

#include <cute/config.hpp>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>

#include "common/fast_math.h"
#include "gather_tensor.h"

namespace llm {

using namespace cute;

// AttentionTile specialization for AttentionParams
template <typename TileShape,  // (BLK_M, BLK_N, BLK_K)
          typename Element,    // Element type
          typename StrideQ,    // (B, Q, H, D)
          typename StrideK,    // (B, Q, H, D)
          typename StrideV,    // (B, Q, KH, D)
          typename StrideO,    // (B, Q, KH, D)
          bool kLocal>
struct FmhaBlock {
  // Host side parameters
  struct Arguments {
    const void* __restrict__ q_ptr;
    const void* __restrict__ k_ptr;
    const void* __restrict__ v_ptr;
    void* __restrict__ o_ptr;

    StrideQ q_stride;
    StrideK k_stride;
    StrideV v_stride;
    StrideO o_stride;

    int sliding_window = -1;  // -1 means no sliding window
  };

  // Device side parameters
  struct Params {
    const void* __restrict__ q_ptr;
    const void* __restrict__ k_ptr;
    const void* __restrict__ v_ptr;
    void* __restrict__ o_ptr;

    StrideQ q_stride;
    StrideK k_stride;
    StrideV v_stride;
    StrideO o_stride;

    int sliding_window;

    // Parameters from problem shape
    int q_len;
    int kv_len;
    int head_dim;
    FastDivmod group_size;
  };

  template <class ProblemShape>
  static Params to_underlying_arguments(const ProblemShape& problem_shape,
                                        const Arguments& args,
                                        void* workspace = nullptr) {
    // ProblemShape: (Q, K, D, ((KH, G), B))
    const int q_len = size<0>(problem_shape);
    const int kv_len = size<1>(problem_shape);
    const int head_dim = size<2>(problem_shape);
    const int group_size = size<3, 0, 1>(problem_shape);

    // TODO: construct tma_load for k/v tensors
    return {
        .q_ptr = args.q_ptr,
        .k_ptr = args.k_ptr,
        .v_ptr = args.v_ptr,
        .o_ptr = args.o_ptr,
        .q_stride = args.q_stride,
        .k_stride = args.k_stride,
        .v_stride = args.v_stride,
        .o_stride = args.o_stride,
        .sliding_window = args.sliding_window,
        .q_len = q_len,
        .kv_len = kv_len,
        .head_dim = head_dim,
        .group_size = FastDivmod(group_size),
    };
  }

  static constexpr int kBlockM = get<0>(TileShape{});
  static constexpr int kBlockN = get<1>(TileShape{});
  static constexpr int kBlockK = get<2>(TileShape{});

  using BLK_M = Int<kBlockM>;
  using BLK_N = Int<kBlockN>;
  using BLK_K = Int<kBlockK>;

  // hold a reference to the parameters and block coordination
  const Params& params_;
  // TODO: (m_block_idx, (kv_head_idx, batch_idx))
  // (batch_idx, m_block_idx, kv_head_idx)
  const tuple<int, int, int>& blk_coord_;

  // derived parameters to avoid recomputation
  int m_block_base_;
  int packed_len_;

  // Constructor
  CUTE_HOST_DEVICE FmhaBlock(const Params& params,
                             const tuple<int, int, int>& blk_coord)
      : params_(params), blk_coord_(blk_coord) {
    // derive parameters
    m_block_base_ = get<1>(blk_coord) * get<0>(TileShape{});
    packed_len_ = params_.q_len * params_.group_size;
  }

  // check if the m_block is valid
  CUTE_HOST_DEVICE bool is_valid() const {
    // check if m_block_idx is out of range
    return packed_len_ > m_block_base_;
  }

  // returns packed_len
  CUTE_HOST_DEVICE int get_packed_len() const { return packed_len_; }

  // returns actual query length
  CUTE_HOST_DEVICE int get_q_len() const { return params_.q_len; }

  // returns actual kv length
  CUTE_HOST_DEVICE int get_kv_len() const { return params_.kv_len; }

  // returns head_dim
  CUTE_HOST_DEVICE int get_head_dim() const { return params_.head_dim; }

  // returns group size
  CUTE_HOST_DEVICE const FastDivmod& get_group_size() const {
    return params_.group_size;
  }

  // returns redidue mnk
  CUTE_HOST_DEVICE auto get_residue_mnk() const {
    return make_tuple(packed_len_, params_.kv_len, params_.head_dim);
  }

  // returns (m_block_idx, (kv_head_idx, batch_idx))
  // return (batch_idx, m_block_idx, kv_head_idx)
  CUTE_HOST_DEVICE const auto& get_block_coord() const { return blk_coord_; }

  // returns kv block range: (n_block_min, n_block_max]
  CUTE_HOST_DEVICE auto get_kv_blocks() const {
    static constexpr int kBlockM = get<0>(TileShape{});
    static constexpr int kBlockN = get<1>(TileShape{});

    const int q_len = params_.q_len;
    const int kv_len = params_.kv_len;
    const int q_idx = m_block_base_ / params_.group_size;
    // take care of causal mask
    const int diagonal = q_idx + kv_len - q_len;
    const int kv_idx_max = std::min(kv_len, diagonal + kBlockM);
    const int n_block_max = cute::ceil_div(kv_idx_max, kBlockN);

    if constexpr (kLocal) {
      const int kv_idx_min = std::max(0, diagonal - params_.sliding_window);
      const int n_block_min = kv_idx_min / kBlockN;
      return make_tuple(n_block_min, n_block_max);
    } else {
      return make_tuple(0, n_block_max);
    }
  }

  // return the query tile: (BLK_M, BLK_K) => (M, K)
  CUTE_HOST_DEVICE auto get_q_tile() const {
    const auto& [batch_idx, m_block_idx, kv_head_idx] = blk_coord_;

    // packing all q in the same kv head group together
    auto packed_idx_to_coord = [this](int packed_idx) {
      // packed_idx => (seq, kv_heads):(group_size, 1)
      int idx, offset;
      params_.group_size.divmod(packed_idx, idx, offset);
      return make_coord(idx, offset);
    };

    // (batch, seq, head, dim)
    // => (batch, seq, (kv_heads, group), dim)
    // => (seq, group, dim)
    const auto offset =
        batch_idx * get<0>(params_.q_stride) +
        kv_head_idx * params_.group_size * get<2>(params_.q_stride);
    // gmem tensor: (packed_len, dim) => ((seq, group), dim)
    auto Q = make_gather_tensor(
        make_gmem_ptr((const Element*)params_.q_ptr + offset),
        make_shape(packed_len_, params_.head_dim),
        make_stride(select<1, 2>(params_.q_stride), get<3>(params_.q_stride)),
        packed_idx_to_coord);

    // (BLK_M, BLK_K)
    Tensor gQ =
        local_tile(Q, Shape<BLK_M, BLK_K>{}, make_coord(m_block_idx, _0{}));
    // (BLK_M, BLK_K) => (M, K)
    Tensor cQ = local_tile(make_identity_tensor(shape(Q)),
                           Shape<BLK_M, BLK_K>{},
                           make_coord(m_block_idx, _0{}));
    return make_tuple(gQ, cQ);
  }

  // return the output tile: (BLK_M, BLK_K) => (M, K)
  CUTE_HOST_DEVICE auto get_o_tile() const {
    const auto& [batch_idx, m_block_idx, kv_head_idx] = blk_coord_;

    // packing all q in the same kv head group together
    auto packed_idx_to_coord = [this](int packed_idx) {
      // packed_idx => (seq, kv_heads):(group_size, 1)
      int idx, offset;
      params_.group_size.divmod(packed_idx, idx, offset);
      return make_coord(idx, offset);
    };

    // (batch, seq, head, dim)
    // => (batch, seq, (kv_heads, group), dim)
    // => (seq, group, dim)
    const auto offset =
        batch_idx * get<0>(params_.o_stride) +
        kv_head_idx * params_.group_size * get<2>(params_.o_stride);
    // gmem tensor: (packed_len, dim) => ((seq, group), dim)
    auto O = make_gather_tensor(
        make_gmem_ptr((Element*)params_.o_ptr + offset),
        make_shape(packed_len_, params_.head_dim),
        make_stride(select<1, 2>(params_.o_stride), get<3>(params_.o_stride)),
        packed_idx_to_coord);

    // (BLK_M, BLK_K)
    Tensor gO =
        local_tile(O, Shape<BLK_M, BLK_K>{}, make_coord(m_block_idx, _0{}));
    // (BLK_M, BLK_K) => (M, K)
    Tensor cQ = local_tile(make_identity_tensor(shape(O)),
                           Shape<BLK_M, BLK_K>{},
                           make_coord(m_block_idx, _0{}));
    return make_tuple(gO, cQ);
  }

  // return the key/value tile: (BLK_N, BLK_K, n) => (N, K)
  CUTE_HOST_DEVICE auto get_kv_tile() const {
    const auto& [batch_idx, m_block_idx, kv_head_idx] = blk_coord_;

    const auto k_offset = (batch_idx * get<0>(params_.k_stride)) +
                          (kv_head_idx * get<2>(params_.k_stride));
    const auto v_offset = (batch_idx * get<0>(params_.v_stride)) +
                          (kv_head_idx * get<2>(params_.v_stride));

    // (batch, seq, kv_head, dim) => (seq, dim)
    auto K =
        make_tensor(make_gmem_ptr((const Element*)params_.k_ptr + k_offset),
                    make_shape(params_.kv_len, params_.head_dim),
                    select<1, 3>(params_.k_stride));
    auto V =
        make_tensor(make_gmem_ptr((const Element*)params_.v_ptr + v_offset),
                    make_shape(params_.kv_len, params_.head_dim),
                    select<1, 3>(params_.v_stride));

    // (BLK_N, BLK_K, n)
    Tensor gK = local_tile(K, Shape<BLK_N, BLK_K>{}, make_coord(_, _0{}));
    Tensor gV = local_tile(V, Shape<BLK_N, BLK_K>{}, make_coord(_, _0{}));
    // (BLK_N, BLK_K, n) => (N, K)
    Tensor cKV = local_tile(make_identity_tensor(shape(K)),
                            Shape<BLK_N, BLK_K>{},
                            make_coord(_, _0{}));
    return make_tuple(gK, gV, cKV);
  }

  // functions for tma load
  // returns kv tma tile: (BLK_N, BLK_K, n) => (1@0, 1@1, 1@2)
  template <class TMA_K, class TMA_V>
  CUTE_HOST_DEVICE auto get_kv_tma_tile(TMA_K tma_k, TMA_V tma_v) const {
    // 1: make_gather_tma_tensor()
    // tma_tensor = (seq, dim, kv_head) => (1@0, 1@1, 1@2)
    // 2: partition into tiles
    // tma_tile = (BLK_N, BLK_K, n) => (1@0, 1@1, 1@2)

    // (Q, D, (B, H))
    // Tensor mQ_qdl_p = tma_k.get_tma_tensor(select<0,2,3>(problem_shape));
    // Tensor mQ_qdl = domain_offset(make_coord(q_offs_0, _0{}, make_coord(_0{},
    // q_offs_2_1)), mQ_qdl_p); (BLK_N, BLK_K, m, k, (b)) Tensor gQ_qdl =
    // local_tile(mQ_qdl, TileShapeQK{}, make_coord(_, _, _), Step<_1, X,
    // _1>{});

    // outside in caller part
    // (BLK_N, BLK_K, n) => (TMA,TMA_M,TMA_N, n)
    // auto cta_tma = tma.get_slice(Int<0>{});  // CTA slice
    // (TMA,TMA_M,TMA_N,REST_M,REST_N)
    // Tensor tAgA_x = cta_tma.partition_S(gA);
    // (TMA,TMA_M,TMA_N)
    // Tensor tAsA_x = cta_tma.partition_D(sA);

    return;
  }
};

}  // namespace llm
