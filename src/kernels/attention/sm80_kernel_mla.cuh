#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cute/layout.hpp>
#include <cute/tensor.hpp>

#include "mla_tile.h"
#include "online_softmax.cuh"

namespace llm {

using namespace cute;

template <class CollectiveMainloop_,
          class CollectiveEpilogue_,
          class TileScheduler_>
class Sm80KernelMla {
 public:
  using CollectiveMainloop = CollectiveMainloop_;
  using CollectiveEpilogue = CollectiveEpilogue_;
  using TileScheduler = TileScheduler_;

  using TiledMma_PV = typename CollectiveMainloop::TiledMma_PV;

  using Element = typename CollectiveMainloop::Element;
  using BLK_M = typename CollectiveMainloop::BLK_M;
  using BLK_N = typename CollectiveMainloop::BLK_N;
  using BLK_K = typename CollectiveMainloop::BLK_K;
  using HEAD_DIM = typename CollectiveMainloop::HEAD_DIM;
  using ROPE_HEAD_DIM = typename CollectiveMainloop::ROPE_HEAD_DIM;
  using STEPS = typename CollectiveMainloop::STEPS;

  static constexpr int kBlockM = CollectiveMainloop::kBlockM;

  static constexpr int kRowsPerMMA = CollectiveMainloop::kRowsPerMMA;

  static constexpr int kSharedStorageSize =
      cute::max(sizeof(typename CollectiveMainloop::SharedStorage),
                sizeof(typename CollectiveEpilogue::SharedStorage));

  static constexpr int kMmaThreads = CollectiveMainloop::kMmaThreads;

  // Kernel params
  using MainloopParams = typename CollectiveMainloop::Params;
  using EpilogueParams = typename CollectiveEpilogue::Params;
  using TileSchedulerParams = typename TileScheduler::Params;

  // returns grid and block shape for kernel launch
  using TileSchedulerArgs = typename TileScheduler::Arguments;
  static dim3 get_grid_shape(TileSchedulerArgs const& args) {
    return TileScheduler::get_grid_shape(args);
  }
  static dim3 get_block_shape() { return kMmaThreads; }

  template <class Params>
  CUTE_DEVICE void operator()(const Params& params,
                              const TileSchedulerParams& scheduler_params,
                              char* smem) {
    CollectiveMainloop mha;
    CollectiveEpilogue epilogue;
    TileScheduler scheduler(scheduler_params);

    // construct params
    MainloopParams mainloop_params{params.group_size};
    EpilogueParams epilogue_params;

    // process each block
    const auto& group_size = params.group_size;

    for (const auto block_coord : scheduler) {
      // block coord: (batch_idx, m_block_idx, kv_head_idx)
      const auto [batch_idx, m_block_idx, kv_head_idx] = block_coord;
      const auto tidx = threadIdx.x;

      // Q/O: (q_packed_len, HEAD_DIM)
      // Q_ROPE: (q_packed_len, ROPE_HEAD_DIM)
      MLATile<Params> tile(params, batch_idx);
      auto [Q, Q_ROPE, O] = tile.template get_qo_tile<Element>();
      // KV: (kv_len, HEAD_DIM)
      // K_ROPE: (kv_len, ROPE_HEAD_DIM)
      auto [KV, K_ROPE] = tile.template get_kv_tile<Element>();

      // problem shape
      const int q_packed_len = size<0>(Q);
      const int q_len = q_packed_len / group_size;
      const int kv_len = size<0>(KV);

      if (m_block_idx * kBlockM >= size<0>(Q)) {
        // m out of bound, return
        return;
      }

      const auto head_dim = params.head_dim;
      auto problem_shape_mnk = make_shape(q_packed_len, kv_len, head_dim);

      // (BLK_M, BLK_K, k)
      Tensor gQ =
          local_tile(Q, Shape<BLK_M, BLK_K>{}, make_coord(m_block_idx, _));
      Tensor gO =
          local_tile(O, Shape<BLK_M, BLK_K>{}, make_coord(m_block_idx, _));
      // (BLK_N, BLK_K, n, k)
      Tensor gKV = local_tile(KV, Shape<BLK_N, BLK_K>{}, make_coord(_, _));

      // (BLK_M, ROPE_HEAD_DIM)
      Tensor gQ_rope = local_tile(
          Q_ROPE, Shape<BLK_M, ROPE_HEAD_DIM>{}, make_coord(m_block_idx, _0{}));
      // (BLK_N, ROPE_HEAD_DIM, n)
      Tensor gK_rope = local_tile(
          K_ROPE, Shape<BLK_N, ROPE_HEAD_DIM>{}, make_coord(_, _0{}));

      TiledMma_PV tiled_mma_pv;
      // accumulator: MMA,MMA_M,MMA_K, k)
      auto tOrAccO =
          partition_fragment_C(tiled_mma_pv, Shape<BLK_M, BLK_K, STEPS>{});
      clear(tOrAccO);

      constexpr int kRowsPerThr = kRowsPerMMA * size<1>(tOrAccO);
      OnlineSoftmax<kRowsPerThr> softmax(params.sm_scale_log2);

      // mainloop
      mha(mainloop_params,
          gQ,
          gKV,
          gQ_rope,
          gK_rope,
          tOrAccO,
          softmax,
          tidx,
          block_coord,
          problem_shape_mnk,
          smem);

      // epilogue
      epilogue(epilogue_params,
               tOrAccO,
               tiled_mma_pv,
               gO,
               tidx,
               block_coord,
               problem_shape_mnk,
               smem);
    }
  }
};

}  // namespace llm
