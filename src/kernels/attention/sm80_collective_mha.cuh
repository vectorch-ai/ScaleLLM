#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cute/config.hpp>
#include <cute/container/array_aligned.hpp>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>

#include "cute_extensions.cuh"
#include "fast_cast.cuh"
#include "layout_convertor.h"
#include "mask.h"

namespace llm {

using namespace cute;

template <class TileShape_,
          class Element_,
          int HeadDim_,
          bool EVEN_K,
          bool ALIBI,
          bool SOFT_CAP,
          bool LOCAL>
struct Sm80CollectiveMha {
  // TODO: multiple stages
  using TileShape = TileShape_;
  using Element = Element_;
  using ElementAccum = float;

  static constexpr int kHeadDim = HeadDim_;
  static constexpr int kBlockM = get<0>(TileShape{});
  static constexpr int kBlockN = get<1>(TileShape{});
  static constexpr int kBlockK = get<2>(TileShape{});

  static_assert(kBlockK == 32 || kBlockK == 64);
  static_assert(kHeadDim % kBlockK == 0);

  using BLK_M = Int<kBlockM>;
  using BLK_N = Int<kBlockN>;
  using BLK_K = Int<kBlockK>;
  using HEAD_DIM = Int<kHeadDim>;

  // TiledMMA (64x16x16) for gemm-I and gemm-II
  using MMA_Atom_ =
      std::conditional_t<std::is_same_v<Element, cute::half_t>,
                         MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
                         MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>>;
  using TiledMma = TiledMMA<MMA_Atom_,
                            Layout<Shape<_4, _1, _1>>,  // warp layout 4x1x1
                            Tile<_64, _16, _16>>;       // Tile Shape 64x16x16

  static constexpr int kRowsPerMMA = 2;
  static constexpr int kMmaThreads = size(TiledMma{});

  // Atom layout: (8, BLK_K):(BLK_K, 1) k-major
  using SmemLayoutAtom_ =
      decltype(composition(Swizzle<3, 3, 3>{},
                           Layout<Shape<_8, BLK_K>, Stride<BLK_K, _1>>{}));

  // Q smem: (BLK_M, HEAD_DIM)
  using SmemLayoutQ =
      decltype(tile_to_shape(SmemLayoutAtom_{}, Shape<BLK_M, HEAD_DIM>{}));

  // KV smem: (BLK_N, HEAD_DIM)
  using SmemLayoutK =
      decltype(tile_to_shape(SmemLayoutAtom_{}, Shape<BLK_N, HEAD_DIM>{}));
  using SmemLayoutV =
      decltype(tile_to_shape(SmemLayoutAtom_{}, Shape<BLK_N, HEAD_DIM>{}));

  // V^T smem: (HEAD_DIM, BLK_N)
  using SmemLayoutVt = decltype(select<1, 0>(SmemLayoutV{}));

  // Thr layout for gmem copy
  using GmemCopyThrLayout_ =
      std::conditional_t<kBlockK == 32,
                         Layout<Shape<_32, _4>, Stride<_4, _1>>,
                         Layout<Shape<_16, _8>, Stride<_8, _1>>>;

  // g2s tiled copy for q
  using GmemTiledCopyQ = decltype(make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, Element>{},
      GmemCopyThrLayout_{},    // Thr layout: (_16,_8)/(_32, _4)
      Layout<Shape<_1, _8>>{}  // Val layout: 8 vals per read
      ));

  // g2s tiled copy for kv
  using GmemTiledCopyKV = GmemTiledCopyQ;

  // s2r tiled copy for gemm-I
  using SmemTiledCopyQ =
      decltype(make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, Element>{},
                                 TiledMma{}));
  using SmemTiledCopyK =
      decltype(make_tiled_copy_B(Copy_Atom<SM75_U32x4_LDSM_N, Element>{},
                                 TiledMma{}));

  // s2r tiled copy for gemm-II
  using SmemTiledCopyVt =
      decltype(make_tiled_copy_B(Copy_Atom<SM75_U16x8_LDSM_T, Element>{},
                                 TiledMma{}));

  struct SharedStorage : cute::aligned_struct<128> {
    union {
      cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> q_smem;
      struct {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>> k_smem;
        union {
          cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>> v_smem;
          cute::array_aligned<Element, cute::cosize_v<SmemLayoutVt>> vt_smem;
        };
      };
    };
  };

  // Host side arguments
  struct Arguments {
    // mask
    int sliding_window = -1;

    // softcap
    float logits_soft_cap = 0.0;

    // softmax scaling
    float sm_scale = 1.0;
    float sm_scale_log2 = 0.0;

    // alibi: (n_heads)
    const float* __restrict__ alibi_slopes_ptr = nullptr;

    FastDivmod group_size;
  };

  // Device side params
  using Params = Arguments;

  // Convert host side arguments to device side params
  static Params to_underlying_arguments(Arguments const& args) {
    // no convertion needed.
    return args;
  }

  // returns false if the block has been skipped
  template <class TensorQ,
            class TensorK,
            class TensorV,
            class FrgTensor,
            class Softmax,
            class ResiduelMNK>
  CUTLASS_DEVICE bool mha(const Params& params,
                          const TensorQ& gQ,  // (BLK_M, HEAD_DIM)
                          const TensorK& gK,  // (BLK_N, HEAD_DIM, n)
                          const TensorV& gV,  // (BLK_N, HEAD_DIM, n)
                          FrgTensor& tOrO,    // (MMA, MMA_M, MMA_N)
                          Softmax& softmax,
                          int tidx,
                          const dim3& block_coord,
                          ResiduelMNK residue_mnk,
                          char* smem) {
    // TODO: port logics from mha_kernel_sm80.cuh
    static_assert(is_rmem<FrgTensor>::value,
                  "Accum tensor must be rmem resident.");
    static_assert(is_gmem<TensorQ>::value, "Q tensor must be gmem resident.");
    static_assert(is_gmem<TensorK>::value, "K tensor must be gmem resident.");
    static_assert(is_gmem<TensorV>::value, "V tensor must be gmem resident.");

    const auto [m_block_idx, batch_idx, kv_head_idx] = block_coord;
    const auto [q_packed_len, kv_len, head_dim] = residue_mnk;

    if (m_block_idx * kBlockM >= q_packed_len) {
      // m out of bound, return
      return false;
    }

    const int sliding_window = LOCAL ? params.sliding_window : kv_len;
    const float logits_soft_cap = params.logits_soft_cap;
    const float sm_scale = params.sm_scale;
    const float sm_scale_log2 = params.sm_scale_log2;
    const auto& group_size = params.group_size;
    const int q_len = q_packed_len / group_size;

    // Construct shared memory tiles
    auto& ss = *reinterpret_cast<SharedStorage*>(smem);

    // (BLK_M, HEAD_DIM), k-major
    Tensor sQ = make_tensor(make_smem_ptr(ss.q_smem.data()), SmemLayoutQ{});
    // (BLK_N, HEAD_DIM), k-major
    Tensor sK = make_tensor(make_smem_ptr(ss.k_smem.data()), SmemLayoutK{});
    Tensor sV = make_tensor(make_smem_ptr(ss.v_smem.data()), SmemLayoutV{});

    // Tensor for V^t; used in GEMM-II.
    // (HEAD_DIM, BLK_N), k-major
    Tensor sVt = make_tensor(make_smem_ptr(ss.vt_smem.data()), SmemLayoutVt{});

    // g2s tiled copy for qkv
    GmemTiledCopyQ gmem_tiled_copy_Q;
    GmemTiledCopyKV gmem_tiled_copy_KV;
    auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tidx);
    auto gmem_thr_copy_KV = gmem_tiled_copy_KV.get_thread_slice(tidx);

    // coordinate tensor for oob handling
    // (BLK_M, HEAD_DIM) -> (blk_m, head_dim)
    Tensor cQ = make_identity_tensor(Shape<BLK_M, HEAD_DIM>{});
    Tensor tQcQ = gmem_thr_copy_Q.partition_S(cQ);

    auto produce_query = [&]() {
      auto tQgQ = gmem_thr_copy_Q.partition_S(gQ);
      auto tQsQ = gmem_thr_copy_Q.partition_D(sQ);
      auto max_coord =
          make_coord(q_packed_len - m_block_idx * kBlockM, head_dim);
      safe_copy</*EVEN_M=*/false, EVEN_K, /*ZFILL_M=*/true, /*ZFILL_K=*/true>(
          gmem_tiled_copy_Q, tQgQ, tQsQ, tQcQ, max_coord);
    };

    // (BLK_N, HEAD_DIM) -> (blk_n, head_dim)
    Tensor cKV = make_identity_tensor(Shape<BLK_N, HEAD_DIM>{});
    Tensor tKVcKV = gmem_thr_copy_KV.partition_S(cKV);

    Tensor tKsK = gmem_thr_copy_KV.partition_D(sK);
    auto produce_key = [&](int ni) {
      auto tKgK = gmem_thr_copy_KV.partition_S(gK(_, _, ni));
      auto max_coord = make_coord(kv_len - ni * kBlockN, head_dim);
      // skip ZFILL_MN for key since Mask will mask out oob with -inf
      safe_copy</*EVEN_N=*/false, EVEN_K, /*ZFILL_N=*/false, /*ZFILL_K=*/true>(
          gmem_tiled_copy_KV, tKgK, tKsK, tKVcKV, max_coord);
    };

    // produce key without oob handling
    auto produce_key_no_oob = [&](int ni) {
      auto tKgK = gmem_thr_copy_KV.partition_S(gK(_, _, ni));
      auto max_coord = make_coord(kv_len - ni * kBlockN, head_dim);
      safe_copy</*EVEN_N=*/true, EVEN_K, /*ZFILL_N=*/false, /*ZFILL_K=*/false>(
          gmem_tiled_copy_KV, tKgK, tKsK, tKVcKV, max_coord);
    };

    Tensor tVsV = gmem_thr_copy_KV.partition_D(sV);
    auto produce_value = [&](int ni) {
      auto tVgV = gmem_thr_copy_KV.partition_S(gV(_, _, ni));
      auto max_coord = make_coord(kv_len - ni * kBlockN, head_dim);
      // skipping ZFILL_MN for v may cause nan issue
      safe_copy</*EVEN_N=*/false, EVEN_K, /*ZFILL_N=*/true, /*ZFILL_K=*/true>(
          gmem_tiled_copy_KV, tVgV, tVsV, tKVcKV, max_coord);
    };

    // produce value without oob handling
    auto produce_value_no_oob = [&](int ni) {
      auto tVgV = gmem_thr_copy_KV.partition_S(gV(_, _, ni));
      auto max_coord = make_coord(kv_len - ni * kBlockN, head_dim);
      safe_copy</*EVEN_N=*/true, EVEN_K, /*ZFILL_N=*/false, /*ZFILL_K=*/false>(
          gmem_tiled_copy_KV, tVgV, tVsV, tKVcKV, max_coord);
    };

    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_slice(tidx);
    // GEMM-I: S = Q@K.T
    auto tSrQ = thr_mma.partition_fragment_A(sQ);  // (MMA,MMA_M,MMA_K)
    auto tSrK = thr_mma.partition_fragment_B(sK);  // (MMA,MMA_N,MMA_K)

    // s2r tiled copy for qkv
    // copy query to rmem
    SmemTiledCopyQ smem_tiled_copy_Q;
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    auto tSsQ = smem_thr_copy_Q.partition_S(sQ);
    auto tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);

    SmemTiledCopyK smem_tiled_copy_K;
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    auto tSsK = smem_thr_copy_K.partition_S(sK);
    auto tSrK_copy_view = smem_thr_copy_K.retile_D(tSrK);

    // S = Q@K.T
    // tSrAccS: (MMA,MMA_M,MMA_N)
    auto compute_qk = [&](auto& tSrAccS) {
      // prefetch key
      cute::copy(
          smem_tiled_copy_K, tSsK(_, _, _0{}), tSrK_copy_view(_, _, _0{}));

      CUTE_UNROLL
      for (int ki = 0; ki < size<2>(tSrQ); ++ki) {
        // prefetch next key
        if (ki != size<2>(tSrQ) - 1) {
          const auto next_ki = ki + 1;
          cute::copy(smem_tiled_copy_K,
                     tSsK(_, _, next_ki),
                     tSrK_copy_view(_, _, next_ki));
        }
        cute::gemm(tiled_mma, tSrQ(_, _, ki), tSrK(_, _, ki), tSrAccS);
      }
    };

    // GEMM-II: O = softmax(S)@V
    auto tOrVt = thr_mma.partition_fragment_B(sVt);  // (MMA,MMA_K,MMA_N)

    SmemTiledCopyVt smem_tiled_copy_Vt;
    auto smem_thr_copy_Vt = smem_tiled_copy_Vt.get_thread_slice(tidx);
    auto tOsVt = smem_thr_copy_Vt.partition_S(sVt);
    auto tOrVt_copy_view = smem_thr_copy_Vt.retile_D(tOrVt);

    // O = softmax(S)*V
    // tSrAccS: (MMA,MMA_M,MMA_N)
    // tOrAccO: (MMA,MMA_M,MMA_K)
    auto compute_sv = [&](const auto& tSrAccS, auto& tOrAccO) {
      // cast scores from Accumulator to Element
      auto tSrS = make_tensor_like<Element>(tSrAccS);
      fast_cast(tSrAccS, tSrS);

      // convert layout from gemm-I C to gemm-II A
      auto tOrS =
          make_tensor(tSrS.data(), LayoutConvertor::to_mma_a(tSrS.layout()));

      // prefetch V^t
      cute::copy(
          smem_tiled_copy_Vt, tOsVt(_, _, _0{}), tOrVt_copy_view(_, _, _0{}));
      CUTE_UNROLL
      for (int ki = 0; ki < size<2>(tOrS); ++ki) {
        // prefetch next V^t
        if (ki != size<2>(tOrS) - 1) {
          const auto next_ki = ki + 1;
          cute::copy(smem_tiled_copy_Vt,
                     tOsVt(_, _, next_ki),
                     tOrVt_copy_view(_, _, next_ki));
        }
        cute::gemm(tiled_mma, tOrS(_, _, ki), tOrVt(_, _, ki), tOrAccO);
      }
    };

    // output accumulator, (MMA,MMA_M,MMA_K)
    // auto tOrO = partition_fragment_C(tiled_mma, Shape<_BLK_M, _HEAD_DIM>{});
    // clear(tOrO);
    auto tOrO_mn =
        make_tensor(tOrO.data(), LayoutConvertor::to_mn(tOrO.layout()));

    const int diagonal = (m_block_idx * kBlockM) / group_size + kv_len - q_len;
    // process kv in range: [kv_idx_min, kv_idx_max)
    const int kv_idx_min = std::max(0, diagonal - sliding_window);
    const int kv_idx_max = std::min(kv_len, diagonal + kBlockM);
    const int n_block_min = LOCAL ? kv_idx_min / kBlockN : 0;
    const int n_block_max = cute::ceil_div(kv_idx_max, kBlockN);

    if (n_block_min >= n_block_max) {
      return true;
    }

    auto apply_logits_soft_cap = [&](auto& tSrAccS) {
      if constexpr (SOFT_CAP) {
        CUTE_UNROLL
        for (int i = 0; i < size(tSrAccS); ++i) {
          tSrAccS(i) = tanh(tSrAccS(i) * logits_soft_cap);
        }
      }
    };

    // ###############  Prologue  ###############
    // produce query: [] => [q]
    produce_query();
    cp_async_fence();

    // wait g2s copy done for query
    cp_async_wait<0>();
    __syncthreads();

    // copy query from smem to rmem
    cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
    // wait s2r copy done for query
    __syncthreads();

    // produce key: [q] => [q, k]
    produce_key(n_block_max - 1);
    cp_async_fence();

    // ###############  Mainloop  ###############
    constexpr int n_oob_mask = cute::ceil_div(kBlockM, kBlockN) + 1;
    const int n_blocks = n_block_max - n_block_min;

    // attention score accumulator, (MMA,MMA_M,MMA_N)
    auto tSrS = partition_fragment_C(tiled_mma, Shape<BLK_M, BLK_N>{});
    auto tSrS_mn =
        make_tensor(tSrS.data(), LayoutConvertor::to_mn(tSrS.layout()));

    // identity tensor for score accumulator
    auto tScS =
        thr_mma.partition_C(make_identity_tensor(Shape<BLK_M, BLK_N>{}));
    auto tScS_mn =
        make_tensor(tScS.data(), LayoutConvertor::to_mn(tScS.layout()));

    constexpr int kRowsPerThr = kRowsPerMMA * size<1>(tSrS);
    using Mask = Mask<kRowsPerThr, ALIBI, LOCAL>;
    Mask mask(q_len, kv_len, group_size, sliding_window);
    if constexpr (ALIBI) {
      mask.init_alibi(tScS_mn,
                      m_block_idx * kBlockM,
                      kv_head_idx,
                      sm_scale,
                      params.alibi_slopes_ptr);
    }

    CUTE_NO_UNROLL
    for (int i = 0; i < n_blocks; ++i) {
      const int n_block_idx = n_block_max - 1 - i;
      clear(tSrS);

      // wait key, queue: [q, k] => []
      cp_async_wait<0>();
      __syncthreads();

      // produce value, [] => [v]
      if (i == 0) {
        produce_value(n_block_idx);
      } else {
        produce_value_no_oob(n_block_idx);
      }
      cp_async_fence();

      // 1> S = Q@K.T
      compute_qk(tSrS);

      // wait value, [v] => []
      cp_async_wait<0>();
      __syncthreads();

      if constexpr (SOFT_CAP) {
        apply_logits_soft_cap(tSrS);
      }

      if (i < n_oob_mask) {
        mask.apply(
            tSrS_mn, tScS_mn, m_block_idx * kBlockM, n_block_idx * kBlockN);
      } else {
        mask.apply</*OOB_MASK=*/false>(
            tSrS_mn, tScS_mn, m_block_idx * kBlockM, n_block_idx * kBlockN);
      }
      softmax.rescale(tSrS_mn, tOrO_mn);

      // produce next key: [] => [k]
      if (n_block_idx > n_block_min) {
        produce_key_no_oob(n_block_idx - 1);
      }
      cp_async_fence();

      // 2> O = softmax(S)*V
      compute_sv(tSrS, tOrO);
    }

    // normalize output: o /= rowsum
    softmax.finalize(tOrO_mn);
    return true;
  }
};

}  // namespace llm
