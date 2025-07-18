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
  // load Q using cp_async and K/V using tma
  template <class Block>
  CUTE_DEVICE void operator()(const Block& block,
                              int tidx,
                              PipelineQ& q_pipeline,
                              typename PipelineQ::PipelineState& q_state,
                              PipelineKV& kv_pipeline,
                              typename PipelineKV::PipelineState& kv_state,
                              TensorStorage& ss) {
    static constexpr int kBlockK = get<2>(TileShape{});

    if (!block.is_valid()) {
      // skip invalid block
      return;
    }

    const auto [n_block_min, n_block_max] = block.get_kv_blocks();
    if (n_block_min >= n_block_max) {
      return;  // no kv blocks to process
    }

    // (M, N, K)
    const auto residue_mnk = block.get_residue_mnk();

    // (BLK_M, HEAD_DIM) => (M, K)
    auto [gQ, cQ] = block.get_q_tile();

    // Construct smem tensors
    // (BLK_M, HEAD_DIM), k-major
    Tensor sQ = make_tensor(make_smem_ptr(ss.smem_q.data()), SmemLayoutQ{});
    // (BLK_N, HEAD_DIM, KVStages), k-major
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
    auto gmem_thr_copy = gmem_tiled_copy.get_slice(tidx);

    // (CPY, CPY_N, CPY_K) => (M, K)
    Tensor tGcQ = gmem_thr_copy.partition_S(cQ);
    // (CPY, CPY_N, CPY_K)
    Tensor tGgQ = gmem_thr_copy.partition_S(gQ);
    Tensor tGsQ = gmem_thr_copy.partition_D(sQ);

    // TODO: copy k/v using TMA

    const auto residue_mk = select<0, 2>(residue_mnk);
    auto load_query = [&](auto& state) {
      q_pipeline.producer_acquire(state);
      safe_copy</*EVEN_MN=*/false, EVEN_K, /*ZFILL_MN=*/true, /*ZFILL_K=*/true>(
          gmem_tiled_copy, tGgQ, tGsQ, tGcQ, residue_mk);
      q_pipeline.producer_commit(state, cutlass::arch::cpasync_barrier_arrive);
      ++state;
    };

    auto load_key = [&](int ni, auto& state) {
      kv_pipeline.producer_acquire(state);
      // TMA copy

      kv_pipeline.producer_commit(state);  // no op for tma
      ++state;
    };

    auto load_value = [&](int ni, auto& state) {
      kv_pipeline.producer_acquire(state);
      // TMA copy

      kv_pipeline.producer_commit(state);  // no op for tma
      ++state;
    };

    // async copy gmem to smem in following order:
    //    Q0, Kn-1, Vn-1, ..., K1, V1, K0, V0

    // load Q1
    load_query(q_state);

    // load Kn-1, Vn-1
    CUTE_NO_UNROLL
    for (int ni = n_block_max - 1; ni >= n_block_min; --ni) {
      // load Ki
      load_key(ni, kv_state);
      // load Vi
      load_value(ni, kv_state);
    }
  }
};

}  // namespace llm
