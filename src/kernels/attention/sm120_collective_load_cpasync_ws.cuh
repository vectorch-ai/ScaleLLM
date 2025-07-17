#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cutlass/arch/barrier.h>

#include <cute/config.hpp>
#include <cute/container/array_aligned.hpp>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>

#include "safe_copy.h"

namespace llm {

using namespace cute;

template <class TileShape_,
          class Element_,
          int HeadDim_,
          bool EVEN_K,
          bool LOCAL>
struct Sm120CollectiveLoadCpAsyncWs {
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
  // using SmemLayoutVt = decltype(select<1, 0>(SmemLayoutV{}));

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

  struct SharedStorage : cute::aligned_struct<128> {
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> smem_q;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>> smem_k;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>> smem_v;
    // union {
    //   cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> smem_q;
    //   struct {
    //     cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>> smem_k;
    //     union {
    //       cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>> smem_v;
    //       cute::array_aligned<Element, cute::cosize_v<SmemLayoutVt>> smem_vt;
    //     };
    //   };
    // };
  };

  // Host side arguments
  struct Arguments {};

  // Device side params
  using Params = Arguments;

  // Convert host side arguments to device side params
  static Params to_underlying_arguments(Arguments const& args) {
    // no convertion needed.
    return args;
  }

  // load Q/K/V from gmem to smem using cp_async
  template <class TensorQ,
            class TensorCQ,
            class TensorK,
            class TensorV,
            class TensorCKV,
            class ResidueMNK,
            class PipelineQ,
            class PipelineKV>
  CUTE_DEVICE void operator()(
      const Params& /*params*/,
      const TensorQ& gQ,     // (BLK_M, HEAD_DIM)
      const TensorCQ& cQ,    // (BLK_M, HEAD_DIM) => (M, K)
      const TensorK& gK,     // (BLK_N, HEAD_DIM, n)
      const TensorV& gV,     // (BLK_N, HEAD_DIM, n)
      const TensorCKV& cKV,  // (BLK_N, HEAD_DIM, n) => (N, K)
      int tidx,
      const ResidueMNK& residue_mnk,  // (M, N, K)
      PipelineQ& q_pipeline,
      typename PipelineQ::PipelineState& q_state,
      PipelineKV& kv_pipeline,
      typename PipelineKV::PipelineState& kv_state,
      int n_block_min,
      int n_block_max,
      char* smem) {
    static_assert(is_gmem<TensorQ>::value, "Q tensor must be gmem resident.");
    static_assert(is_gmem<TensorK>::value, "K tensor must be gmem resident.");
    static_assert(is_gmem<TensorV>::value, "V tensor must be gmem resident.");

    static constexpr int kBlockM = get<0>(TileShape{});
    static constexpr int kBlockN = get<1>(TileShape{});

    // Construct shared memory tiles
    auto& ss = *reinterpret_cast<SharedStorage*>(smem);

    // (BLK_M, HEAD_DIM), k-major
    Tensor sQ = make_tensor(make_smem_ptr(ss.smem_q.data()), SmemLayoutQ{});
    // (BLK_N, HEAD_DIM), k-major
    Tensor sK = make_tensor(make_smem_ptr(ss.smem_k.data()), SmemLayoutK{});
    Tensor sV = make_tensor(make_smem_ptr(ss.smem_v.data()), SmemLayoutV{});

    // g2s tiled copy for qkv
    GmemTiledCopyQ gmem_tiled_copy_Q;
    GmemTiledCopyKV gmem_tiled_copy_KV;
    auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tidx);
    auto gmem_thr_copy_KV = gmem_tiled_copy_KV.get_thread_slice(tidx);

    // (CPY, CPY_N, CPY_K, n) => (N, K)
    Tensor tGcKV = gmem_thr_copy_KV.partition_S(cKV);
    // (CPY, CPY_N, CPY_K, n)
    Tensor tGgK = gmem_thr_copy_KV.partition_S(gK);
    Tensor tGgV = gmem_thr_copy_KV.partition_S(gV);

    // (CPY, CPY_N, CPY_K)
    Tensor tGsK = gmem_thr_copy_KV.partition_D(sK);
    Tensor tGsV = gmem_thr_copy_KV.partition_D(sV);

    const auto residue_mk = select<0, 2>(residue_mnk);
    const auto residue_nk = select<1, 2>(residue_mnk);

    auto produce_query = [&]() {
      auto tGcQ = gmem_thr_copy_Q.partition_S(cQ);
      auto tGgQ = gmem_thr_copy_Q.partition_S(gQ);
      auto tGsQ = gmem_thr_copy_Q.partition_D(sQ);
      safe_copy</*EVEN_MN=*/false, EVEN_K, /*ZFILL_MN=*/true, /*ZFILL_K=*/true>(
          gmem_tiled_copy_Q, tGgQ, tGsQ, tGcQ, residue_mk);
    };

    auto produce_key = [&](int ni) {
      // skip ZFILL_MN for key since Mask will mask out oob with -inf
      safe_copy</*EVEN_N=*/false, EVEN_K, /*ZFILL_N=*/false, /*ZFILL_K=*/true>(
          gmem_tiled_copy_KV,
          tGgK(_, _, _, ni),
          tGsK,
          tGcKV(_, _, _, ni),
          residue_nk);
    };

    // produce key without oob handling
    auto produce_key_no_oob = [&](int ni) {
      safe_copy</*EVEN_N=*/true, EVEN_K, /*ZFILL_N=*/false, /*ZFILL_K=*/false>(
          gmem_tiled_copy_KV,
          tGgK(_, _, _, ni),
          tGsK,
          tGcKV(_, _, _, ni),
          residue_nk);
    };

    auto produce_value = [&](int ni) {
      // skipping ZFILL_MN for v may cause nan issue
      safe_copy</*EVEN_N=*/false, EVEN_K, /*ZFILL_N=*/true, /*ZFILL_K=*/true>(
          gmem_tiled_copy_KV,
          tGgV(_, _, _, ni),
          tGsV,
          tGcKV(_, _, _, ni),
          residue_nk);
    };

    // produce value without oob handling
    auto produce_value_no_oob = [&](int ni) {
      safe_copy</*EVEN_N=*/true, EVEN_K, /*ZFILL_N=*/false, /*ZFILL_K=*/false>(
          gmem_tiled_copy_KV,
          tGgV(_, _, _, ni),
          tGsV,
          tGcKV(_, _, _, ni),
          residue_nk);
    };

    // Q1, K(n-1), V(n-1), ..., K2, V2, K1, V1

    // ###############  Prologue  ###############
    // produce query: [] => [q]
    q_pipeline.produce_acquire(q_state);
    produce_query();
    q_pipeline.producer_commit(q_state, cutlass::arch::cpasync_barrier_arrive);
    ++q_state;

    // produce key: [q] => [q, k]
    kv_pipeline.produce_acquire(kv_state);
    produce_key(n_block_max - 1);
    kv_pipeline.producer_commit(kv_state,
                                cutlass::arch::cpasync_barrier_arrive);
    ++kv_state;

    // ###############  Mainloop  ###############
    const int n_blocks = n_block_max - n_block_min;
    CUTE_NO_UNROLL
    for (int i = 0; i < n_blocks; ++i) {
      const int ni = n_block_max - 1 - i;

      // produce value, [] => [v]
      kv_pipeline.produce_acquire(kv_state);
      if (i == 0) {
        produce_value(ni);
      } else {
        produce_value_no_oob(ni);
      }
      kv_pipeline.producer_commit(kv_state,
                                  cutlass::arch::cpasync_barrier_arrive);
      ++kv_state;

      // produce next key: [] => [k]
      kv_pipeline.produce_acquire(kv_state);
      if (ni > n_block_min) {
        produce_key_no_oob(ni - 1);
      }
      kv_pipeline.producer_commit(kv_state,
                                  cutlass::arch::cpasync_barrier_arrive);
      ++kv_state;
    }
  }
};

}  // namespace llm
