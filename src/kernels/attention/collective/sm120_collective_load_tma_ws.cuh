#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cutlass/arch/barrier.h>

#include <cute/config.hpp>
#include <cute/container/array_aligned.hpp>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>

#include "common/safe_copy.h"

namespace llm {

using namespace cute;

template <class TileShape,
          class Element,
          class TensorStorage,
          class SmemLayoutQ,
          class SmemLayoutK,
          class SmemLayoutV,
          class PipelineQ,
          class PipelineKV,
          bool EVEN_K>
struct Sm120CollectiveLoadTmaWs {
  // load K/V from gmem to smem using tma
  template <class TensorQ,
            class TensorCQ,
            class ResidueMNK>
  CUTE_DEVICE void operator()(
      const TensorQ& gQ,   // (BLK_M, HEAD_DIM)
      const TensorCQ& cQ,  // (BLK_M, HEAD_DIM) => (M, K)
      int tidx,
      const ResidueMNK& residue_mnk,  // (M, N, K)
      PipelineQ& q_pipeline,
      typename PipelineQ::PipelineState& q_state,
      PipelineKV& kv_pipeline,
      typename PipelineKV::PipelineState& kv_state,
      int n_block_min,
      int n_block_max,
      TensorStorage& ss) {
    static_assert(is_gmem<TensorQ>::value, "Q tensor must be gmem resident.");

    static constexpr int kBlockM = get<0>(TileShape{});
    static constexpr int kBlockN = get<1>(TileShape{});
    static constexpr int kBlockK = get<2>(TileShape{});

    // Construct smem tensors
    // (BLK_M, HEAD_DIM), k-major
    Tensor sQ = make_tensor(make_smem_ptr(ss.smem_q.data()), SmemLayoutQ{});
    // (BLK_N, HEAD_DIM), k-major
    Tensor sK = make_tensor(make_smem_ptr(ss.smem_k.data()), SmemLayoutK{});
    Tensor sV = make_tensor(make_smem_ptr(ss.smem_v.data()), SmemLayoutV{});

    // Thr thread layout for gmem copy (4 warps = 128 threads)
    using GmemCopyThrLayout_ =
        std::conditional_t<kBlockK == 32,
                           Layout<Shape<_32, _4>, Stride<_4, _1>>,
                           Layout<Shape<_16, _8>, Stride<_8, _1>>>;

    // g2s tiled copy for q/kv
    auto gmem_tiled_copy = make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, Element>{},
        GmemCopyThrLayout_{},    // Thr layout: (_16,_8)/(_32, _4)
        Layout<Shape<_1, _8>>{}  // Val layout: 8 vals per read
    );
    auto gmem_thr_copy = gmem_tiled_copy.get_thread_slice(tidx);

    // TODO: copy k/v using TMA

    const auto residue_mk = select<0, 2>(residue_mnk);
    auto produce_query = [&](auto& state) {
      q_pipeline.producer_acquire(state);

      auto tGcQ = gmem_thr_copy.partition_S(cQ);
      auto tGgQ = gmem_thr_copy.partition_S(gQ);
      auto tGsQ = gmem_thr_copy.partition_D(sQ);
      safe_copy</*EVEN_MN=*/false, EVEN_K, /*ZFILL_MN=*/true, /*ZFILL_K=*/true>(
          gmem_tiled_copy, tGgQ, tGsQ, tGcQ, residue_mk);

      q_pipeline.producer_commit(state, cutlass::arch::cpasync_barrier_arrive);
      ++state;
    };

    auto produce_key = [&](int ni, auto& state) {
      kv_pipeline.producer_acquire(state);
      // TMA copy

      kv_pipeline.producer_commit(state);  // no op for tma
      ++state;
    };

    auto produce_value = [&](int ni, auto& state) {
      kv_pipeline.producer_acquire(state);
      // TMA copy

      kv_pipeline.producer_commit(state);  // no op for tma
      ++state;
    };

    // async copy gmem to smem in following order:
    //    Q1, Kn-1, Vn-1, ..., K2, V2, K1, V1

    // produce Q1
    produce_query(q_state);

    // produce Kn-1, Vn-1
    CUTE_NO_UNROLL
    for (int ni = n_block_max - 1; ni >= n_block_min; --ni) {
      // produce Ki
      produce_key(ni, kv_state);
      // produce Vi
      produce_value(ni, kv_state);
    }
  }
};

}  // namespace llm
