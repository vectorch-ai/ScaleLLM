#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cutlass/arch/barrier.h>

#include <cute/config.hpp>
#include <cute/container/array_aligned.hpp>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>

#include "common/safe_copy.h"
#include "common/selector.h"
#include "common/static_dispatch.h"

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
struct Sm120CollectiveLoadCpAsyncWs {
  static constexpr int kThreads = 128;
  static constexpr int kBlockK = get<2>(TileShape{});
  // g2s tiled copy for Q/K/V
  using GmemTiledCopy =
      decltype(gmem_tiled_copy_selector<Element, kThreads, kBlockK>(
          Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<cute::uint128_t>,
                    Element>{}));

  // load Q/K/V tiles from gmem to smem using cp_async
  template <class Block>
  CUTE_DEVICE void operator()(const Block& block,
                              int tidx,
                              int n_block_min,
                              int n_block_max,
                              PipelineQ& q_pipeline,
                              typename PipelineQ::PipelineState& q_state,
                              PipelineKV& kv_pipeline,
                              typename PipelineKV::PipelineState& kv_state,
                              TensorStorage& ss) {
    static constexpr int kStages = size<2>(SmemLayoutK{});

    // (M, N, K)
    const auto residue_mnk = block.get_residue_mnk();

    // (BLK_M, BLK_K) => (M, K)
    auto [gQ, cQ] = block.get_q_tile();
    // (BLK_N, BLK_K, n) => (N, K)
    auto [gK, gV, cKV] = block.get_kv_tile();

    // Construct smem tensors
    // (BLK_M, BLK_K), k-major
    Tensor sQ = make_tensor(make_smem_ptr(ss.smem_q.data()), SmemLayoutQ{});
    // (BLK_N, BLK_K, KVStages), k-major
    Tensor sK = make_tensor(make_smem_ptr(ss.smem_k.data()), SmemLayoutK{});
    Tensor sV = make_tensor(make_smem_ptr(ss.smem_v.data()), SmemLayoutV{});

    // g2s tiled copy for q/kv
    GmemTiledCopy gmem_tiled_copy;
    auto gmem_thr_copy = gmem_tiled_copy.get_slice(tidx);

    // (CPY, CPY_N, CPY_K) => (M, K)
    Tensor tGcQ = gmem_thr_copy.partition_S(cQ);
    // (CPY, CPY_N, CPY_K)
    Tensor tGgQ = gmem_thr_copy.partition_S(gQ);
    Tensor tGsQ = gmem_thr_copy.partition_D(sQ);

    // (CPY, CPY_N, CPY_K, n) => (N, K)
    Tensor tGcKV = gmem_thr_copy.partition_S(cKV);
    // (CPY, CPY_N, CPY_K, n)
    Tensor tGgK = gmem_thr_copy.partition_S(gK);
    Tensor tGgV = gmem_thr_copy.partition_S(gV);

    // (CPY, CPY_N, CPY_K, KVStages)
    Tensor tGsK = gmem_thr_copy.partition_D(sK);
    Tensor tGsV = gmem_thr_copy.partition_D(sV);

    const auto residue_mk = select<0, 2>(residue_mnk);
    const auto residue_nk = select<1, 2>(residue_mnk);

    auto load_query = [&]() {
      q_pipeline.producer_acquire(q_state);
      safe_copy</*EVEN_M=*/false, EVEN_K, /*ZFILL_M=*/true, /*ZFILL_K=*/true>(
          gmem_tiled_copy, tGgQ, tGsQ, tGcQ, residue_mk);
      q_pipeline.producer_commit(q_state,
                                 cutlass::arch::cpasync_barrier_arrive);
      ++q_state;
    };

    auto load_key = [&](int ni) {
      kv_pipeline.producer_acquire(kv_state);
      // skip ZFILL_MN for key since Mask will mask out oob with -inf
      safe_copy</*EVEN_N=*/false, EVEN_K, /*ZFILL_N=*/false, /*ZFILL_K=*/true>(
          gmem_tiled_copy,
          tGgK(_, _, _, ni),
          tGsK(_, _, _, kv_state.index()),
          tGcKV(_, _, _, ni),
          residue_nk);
      kv_pipeline.producer_commit(kv_state,
                                  cutlass::arch::cpasync_barrier_arrive);
      ++kv_state;
    };

    // load key without oob handling
    auto load_key_no_oob = [&](int ni) {
      kv_pipeline.producer_acquire(kv_state);
      if constexpr (EVEN_K) {
        safe_copy</*EVEN_N=*/true,
                  EVEN_K,
                  /*ZFILL_N=*/false,
                  /*ZFILL_K=*/false>(gmem_tiled_copy,
                                     tGgK(_, _, _, ni),
                                     tGsK(_, _, _, kv_state.index()),
                                     tGcKV(_, _, _, ni),
                                     residue_nk);
      } else {
        DISPATCH_BOOL(kv_state.count() < kStages, ZFILL_K, [&] {
          safe_copy</*EVEN_N=*/true, EVEN_K, /*ZFILL_N=*/false, ZFILL_K>(
              gmem_tiled_copy,
              tGgK(_, _, _, ni),
              tGsK(_, _, _, kv_state.index()),
              tGcKV(_, _, _, ni),
              residue_nk);
        });
      }
      kv_pipeline.producer_commit(kv_state,
                                  cutlass::arch::cpasync_barrier_arrive);
      ++kv_state;
    };

    auto load_value = [&](int ni) {
      kv_pipeline.producer_acquire(kv_state);
      // skipping ZFILL_MN for v may cause nan issue
      safe_copy</*EVEN_N=*/false, EVEN_K, /*ZFILL_N=*/true, /*ZFILL_K=*/true>(
          gmem_tiled_copy,
          tGgV(_, _, _, ni),
          tGsV(_, _, _, kv_state.index()),
          tGcKV(_, _, _, ni),
          residue_nk);
      kv_pipeline.producer_commit(kv_state,
                                  cutlass::arch::cpasync_barrier_arrive);
      ++kv_state;
    };

    // load value without oob handling
    auto load_value_no_oob = [&](int ni) {
      kv_pipeline.producer_acquire(kv_state);
      if constexpr (EVEN_K) {
        safe_copy</*EVEN_N=*/true,
                  EVEN_K,
                  /*ZFILL_N=*/false,
                  /*ZFILL_K=*/false>(gmem_tiled_copy,
                                     tGgV(_, _, _, ni),
                                     tGsV(_, _, _, kv_state.index()),
                                     tGcKV(_, _, _, ni),
                                     residue_nk);
      } else {
        DISPATCH_BOOL(kv_state.count() < kStages, ZFILL_K, [&] {
          safe_copy</*EVEN_N=*/true, EVEN_K, /*ZFILL_N=*/false, ZFILL_K>(
              gmem_tiled_copy,
              tGgV(_, _, _, ni),
              tGsV(_, _, _, kv_state.index()),
              tGcKV(_, _, _, ni),
              residue_nk);
        });
      }
      kv_pipeline.producer_commit(kv_state,
                                  cutlass::arch::cpasync_barrier_arrive);
      ++kv_state;
    };

    // async copy gmem to smem in following order:
    //    Q0, Kn-1, Vn-1, ..., K1, V1, K0, V0

    // load Q1
    load_query();

    // load Kn-1, Vn-1 with oob handling
    int ni = n_block_max - 1;
    load_key(ni);
    load_value(ni);
    --ni;

    CUTE_NO_UNROLL
    while (ni >= n_block_min) {
      // load Ki
      load_key_no_oob(ni);
      // load Vi
      load_value_no_oob(ni);
      // advance to next kv block
      --ni;
    }
  }
};

}  // namespace llm
