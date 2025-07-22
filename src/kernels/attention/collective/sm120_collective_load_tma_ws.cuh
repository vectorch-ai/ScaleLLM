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

namespace llm {

using namespace cute;

template <class TileShape,
          // class ClusterShape,
          class Element,
          class TensorStorage,
          class SmemLayoutQ,
          class SmemLayoutK,
          class SmemLayoutV,
          class PipelineQ,
          class PipelineKV,
          bool EVEN_K>
struct Sm120CollectiveLoadTmaWs {
  static constexpr int kThreads = 128;
  static constexpr int kBlockK = get<2>(TileShape{});
  // g2s tiled copy for Q
  using GmemTiledCopyQ =
      decltype(gmem_tiled_copy_selector<Element, kThreads, kBlockK>(
          Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<cute::uint128_t>,
                    Element>{}));

  // using StrideK = ...;

  // using TMA_K = decltype(make_tma_copy(
  //       GmemTiledCopy{}, // TMA_COPY
  //       make_tensor(static_cast<InternalElementA const*>(nullptr),
  //       repeat_like(StrideK{}, int32_t(0)), StrideK{}),
  //       SmemLayoutK{}(_,_,_0{})));

  // Tensor tensor_k = make_tensor(ptr_k, make_layout(make_shape(M,K,L),
  // args.stride_k)); auto tma_load_k = make_tma_copy(SM90_TMA_LOAD{},
  // gtensor_k, SmemLayoutK{}(_,_,_0{}));

  // load Q using cp_async and K/V using tma
  template <class Block>
  CUTE_DEVICE void operator()(const Block& block,
                              int tidx,
                              PipelineQ& q_pipeline,
                              typename PipelineQ::PipelineState& q_state,
                              PipelineKV& kv_pipeline,
                              typename PipelineKV::PipelineState& kv_state,
                              TensorStorage& ss) {
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

    // (BLK_M, BLK_K) => (M, K)
    auto [gQ, cQ] = block.get_q_tile();

    // Construct smem tensors
    // (BLK_M, BLK_K), k-major
    Tensor sQ = make_tensor(make_smem_ptr(ss.smem_q.data()), SmemLayoutQ{});
    // (BLK_N, BLK_K, KVStages), k-major
    Tensor sK = make_tensor(make_smem_ptr(ss.smem_k.data()), SmemLayoutK{});
    Tensor sV = make_tensor(make_smem_ptr(ss.smem_v.data()), SmemLayoutV{});

    // g2s tiled copy for q
    GmemTiledCopyQ gmem_tiled_copy_q;
    auto gmem_thr_copy_q = gmem_tiled_copy_q.get_slice(tidx);

    // (CPY, CPY_N, CPY_K) => (M, K)
    Tensor tGcQ = gmem_thr_copy_q.partition_S(cQ);
    // (CPY, CPY_N, CPY_K)
    Tensor tGgQ = gmem_thr_copy_q.partition_S(gQ);
    Tensor tGsQ = gmem_thr_copy_q.partition_D(sQ);

    // TODO: copy k/v using TMA
    // ??? where to define TMA copy?
    // 1> block, need smem layout (pass)
    // 2> tma_load, need gmem tensor (pass)
    // 3> mainloop, has smem layout and gtensor (x)

    // where to keep tma_load_kv?
    // as args in load_tma_ws? or pass in as parameters?

    const auto residue_mk = select<0, 2>(residue_mnk);
    auto load_query = [&]() {
      q_pipeline.producer_acquire(q_state);
      safe_copy</*EVEN_MN=*/false, EVEN_K, /*ZFILL_MN=*/true, /*ZFILL_K=*/true>(
          gmem_tiled_copy_q, tGgQ, tGsQ, tGcQ, residue_mk);
      q_pipeline.producer_commit(q_state,
                                 cutlass::arch::cpasync_barrier_arrive);
      ++q_state;
    };

    auto load_key = [&](int ni) {
      kv_pipeline.producer_acquire(kv_state);
      // TMA copy

      // kv_pipeline.producer_commit(state);
      ++kv_state;
    };

    auto load_value = [&](int ni) {
      kv_pipeline.producer_acquire(kv_state);
      // TMA copy

      // kv_pipeline.producer_commit(state);
      ++kv_state;
    };

    // async copy gmem to smem in following order:
    //    Q0, Kn-1, Vn-1, ..., K1, V1, K0, V0

    // load Q1
    load_query();

    // load Kn-1, Vn-1
    CUTE_NO_UNROLL
    for (int ni = n_block_max - 1; ni >= n_block_min; --ni) {
      // load Ki
      load_key(ni);
      // load Vi
      load_value(ni);
    }
  }
};

}  // namespace llm
