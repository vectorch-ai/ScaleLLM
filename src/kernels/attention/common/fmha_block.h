#pragma once

#include <cute/config.hpp>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>

#include "common/fast_math.h"
#include "gather_tensor.h"

namespace llm {

using namespace cute;

// AttentionTile specialization for AttentionParams
template <typename ProblemShape,  // (Q, K, D, ((KH, G), B))
          typename TileShape,     // (BLK_M, BLK_N, BLK_K)
          typename BlocKCoord,  // (m_block_idx, ((kv_head_idx, _0), batch_idx))
          typename Element,     // Element type
          typename StrideQ,     // (Q, D, ((KH, G), B))
          typename StrideK,     // (K, D, ((KH, _0), B))
          typename StrideV,     // (V, D, ((KH, _0), B))
          typename StrideO      // (Q, Q, ((KH, G), B))
          >
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
  };

  // Device side parameters
  struct Params {
    const Element* __restrict__ q_ptr;
    const Element* __restrict__ k_ptr;
    const Element* __restrict__ v_ptr;
    Element* __restrict__ o_ptr;

    StrideQ q_stride;
    StrideK k_stride;
    StrideV v_stride;
    StrideO o_stride;

    ProblemShape problem_shape;
    // for fast divmod
    FastDivmod group_size;
  };

  static Params to_underlying_arguments(const ProblemShape& problem_shape,
                                        const Arguments& args,
                                        void* /*workspace*/) {
    // ProblemShape: (Q, K, D, ((KH, G), B))
    const int group_size = get<3, 0, 1>(problem_shape);

    // TODO: construct tma_load for k/v tensors
    return {
        .q_ptr = (const Element*)args.q_ptr,
        .k_ptr = (const Element*)args.k_ptr,
        .v_ptr = (const Element*)args.v_ptr,
        .o_ptr = (Element*)args.o_ptr,
        .q_stride = args.q_stride,
        .k_stride = args.k_stride,
        .v_stride = args.v_stride,
        .o_stride = args.o_stride,
        .problem_shape = problem_shape,
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
  const BlocKCoord& coord_;

  // derived parameters to avoid recomputation
  int m_block_base_;
  int packed_len_;

  // Constructor
  CUTE_HOST_DEVICE FmhaBlock(const Params& params, const BlocKCoord& coord)
      : params_(params), coord_(coord) {
    // derived parameters
    m_block_base_ = get<0>(coord) * get<0>(TileShape{});
    packed_len_ = get<0>(params_.problem_shape) * params_.group_size;
  }

  // check if the m_block is valid
  CUTE_HOST_DEVICE bool is_valid() const {
    // check if m_block_idx is out of range
    return packed_len_ > m_block_base_;
  }

  // returns packed_len
  CUTE_HOST_DEVICE int get_packed_len() const { return packed_len_; }

  // returns problem shape: (Q, K, D, ((KH, G), B))
  CUTE_HOST_DEVICE const auto& get_problem_shape() const {
    return params_.problem_shape;
  }

  // returns (m_block_idx, ((kv_head_idx, _0), batch_idx))
  CUTE_HOST_DEVICE const auto& get_coord() const { return coord_; }

  // returns group size fast divmod
  CUTE_HOST_DEVICE const FastDivmod& get_group_size() const {
    return params_.group_size;
  }

  // returns redidue mnk
  CUTE_HOST_DEVICE auto get_residue_mnk() const {
    auto residue_mnk = select<0, 1, 2>(params_.problem_shape);
    get<0>(residue_mnk) = packed_len_;
    return residue_mnk;
  }

  // returns kv block range: (n_block_min, n_block_max]
  template <bool kLocal>
  CUTE_HOST_DEVICE auto get_kv_blocks(int sliding_window) const {
    static constexpr int kBlockM = get<0>(TileShape{});
    static constexpr int kBlockN = get<1>(TileShape{});

    const auto [q_len, kv_len] = select<0, 1>(params_.problem_shape);
    const int q_idx = m_block_base_ / params_.group_size;
    // take care of causal mask
    const int diagonal = q_idx + kv_len - q_len;
    const int kv_idx_max = std::min(kv_len, diagonal + kBlockM);
    const int n_block_max = cute::ceil_div(kv_idx_max, kBlockN);

    if constexpr (kLocal) {
      const int kv_idx_min = std::max(0, diagonal - sliding_window);
      const int n_block_min = kv_idx_min / kBlockN;
      return make_tuple(n_block_min, n_block_max);
    } else {
      return make_tuple(0, n_block_max);
    }
  }

  // return the query tile: (BLK_M, BLK_K) => (M, K)
  CUTE_HOST_DEVICE auto get_q_tile() const {
    // (Q, D, ((KH, G), B))
    auto q_shape = select<0, 2, 3>(params_.problem_shape);
    auto mQ =
        make_tensor(make_gmem_ptr(params_.q_ptr), q_shape, params_.q_stride);
    // (Q, D, G*)
    auto Q = mQ(_, _, get<1>(coord_));

    // packing all q in the same kv head group together
    auto packed_idx_to_coord = [this](int packed_idx) {
      // packed_idx => (Q, G):(G, 1)
      int idx, offset;
      params_.group_size.divmod(packed_idx, idx, offset);
      return make_coord(idx, offset);
    };

    // ((Q, G), D)
    auto q_stride = make_stride(
        make_stride(get<0>(params_.q_stride), get<2, 0, 1>(params_.q_stride)),
        get<1>(params_.q_stride));

    // packed tensor: (pQ, D) => ((Q, G), D)
    const int head_dim = get<2>(params_.problem_shape);
    auto pQ = make_gather_tensor(Q.data(),
                                 make_shape(packed_len_, head_dim),
                                 q_stride,
                                 packed_idx_to_coord);

    const auto m_block_idx = get<0>(coord_);
    // (BLK_M, BLK_K)
    Tensor gQ =
        local_tile(pQ, Shape<BLK_M, BLK_K>{}, make_coord(m_block_idx, _0{}));
    // (BLK_M, BLK_K) => (M, K)
    Tensor cQ = local_tile(make_identity_tensor(shape(pQ)),
                           Shape<BLK_M, BLK_K>{},
                           make_coord(m_block_idx, _0{}));
    return make_tuple(gQ, cQ);
  }

  // return the output tile: (BLK_M, BLK_K) => (M, K)
  CUTE_HOST_DEVICE auto get_o_tile() const {
    // (Q, D, ((KH, G), B))
    auto o_shape = select<0, 2, 3>(params_.problem_shape);
    auto mO =
        make_tensor(make_gmem_ptr(params_.o_ptr), o_shape, params_.o_stride);
    // (Q, D, G*)
    auto O = mO(_, _, get<1>(coord_));

    // packing all q in the same kv head group together
    auto packed_idx_to_coord = [this](int packed_idx) {
      // packed_idx => (Q, G):(G, 1)
      int idx, offset;
      params_.group_size.divmod(packed_idx, idx, offset);
      return make_coord(idx, offset);
    };
    auto o_stride = make_stride(
        make_stride(get<0>(params_.o_stride), get<2, 0, 1>(params_.o_stride)),
        get<1>(params_.o_stride));
    const int head_dim = get<2>(params_.problem_shape);
    // packed tensor: (pO, D) => ((O, G), D)
    auto pO = make_gather_tensor(O.data(),
                                 make_shape(packed_len_, head_dim),
                                 o_stride,
                                 packed_idx_to_coord);

    const auto m_block_idx = get<0>(coord_);
    // (BLK_M, BLK_K)
    Tensor gO =
        local_tile(pO, Shape<BLK_M, BLK_K>{}, make_coord(m_block_idx, _0{}));
    // (BLK_M, BLK_K) => (M, K)
    Tensor cO = local_tile(make_identity_tensor(shape(pO)),
                           Shape<BLK_M, BLK_K>{},
                           make_coord(m_block_idx, _0{}));
    return make_tuple(gO, cO);
  }

  // return the key/value tile: (BLK_N, BLK_K, n) => (N, K)
  CUTE_HOST_DEVICE auto get_kv_tile() const {
    // (KV, D, ((KH, G), B))
    auto kv_shape = select<1, 2, 3>(params_.problem_shape);
    auto mK =
        make_tensor(make_gmem_ptr(params_.k_ptr), kv_shape, params_.k_stride);
    auto mV =
        make_tensor(make_gmem_ptr(params_.v_ptr), kv_shape, params_.v_stride);

    // (K/V, D)
    auto K = mK(_, _, get<1>(coord_));
    auto V = mV(_, _, get<1>(coord_));

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

  // using StrideK = ...;

  // using TMA_K = decltype(make_tma_copy(
  //       GmemTiledCopy{}, // TMA_COPY
  //       make_tensor(static_cast<InternalElementA const*>(nullptr),
  //       repeat_like(StrideK{}, int32_t(0)), StrideK{}),
  //       SmemLayoutK{}(_,_,_0{})));

  // Tensor tensor_k = make_tensor(ptr_k, make_layout(make_shape(M,K,L),
  // args.stride_k)); auto tma_load_k = make_tma_copy(SM90_TMA_LOAD{},
  // gtensor_k, SmemLayoutK{}(_,_,_0{}));

  // returns kv tma tile: (BLK_N, BLK_K, n) => (1@0, 1@1, 1@2)
  // template <class TMA_K, class TMA_V>
  // CUTE_HOST_DEVICE auto get_kv_tma_tile(TMA_K tma_k, TMA_V tma_v) const {
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

  //   return;
  // }
};

}  // namespace llm
