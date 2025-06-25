#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cute/layout.hpp>
#include <cute/tensor.hpp>

#include "mha_tile.h"
#include "online_softmax.cuh"

namespace llm {

using namespace cute;

template <class CollectiveMainloop_,
          class CollectiveEpilogue_,
          class TileScheduler_>
class Sm80KernelMha {
 public:
  using CollectiveMainloop = CollectiveMainloop_;
  using CollectiveEpilogue = CollectiveEpilogue_;
  using TileScheduler = TileScheduler_;

  using TiledMma = typename CollectiveMainloop::TiledMma;

  using Element = typename CollectiveMainloop::Element;
  using BLK_M = typename CollectiveMainloop::BLK_M;
  using BLK_N = typename CollectiveMainloop::BLK_N;
  using HEAD_DIM = typename CollectiveMainloop::HEAD_DIM;

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
    MainloopParams mainloop_params{params.sliding_window,
                                   params.logits_soft_cap,
                                   params.sm_scale,
                                   params.sm_scale_log2,
                                   params.alibi_slopes_ptr,
                                   params.group_size};
    EpilogueParams epilogue_params;

    // process each block
    for (const auto block_coord : scheduler) {
      // block coord: (batch_idx, m_block_idx, kv_head_idx)
      const auto [batch_idx, m_block_idx, kv_head_idx] = block_coord;
      const auto tidx = threadIdx.x;

      // (q_packed_len, HEAD_DIM)
      MHATile<Params> tile(params, batch_idx, kv_head_idx);
      auto [Q, O] = tile.template get_qo_tile<Element>();
      // (kv_len, HEAD_DIM)
      auto [K, V] = tile.template get_kv_tile<Element>();

      // problem shape
      const int q_packed_len = size<0>(Q);
      const int kv_len = size<0>(K);
      const int head_dim = params.head_dim;
      if (m_block_idx * kBlockM >= q_packed_len) {
        // m out of bound, skip this block
        continue;
      }
      const auto residue_mnk = make_tuple(q_packed_len, kv_len, head_dim);

      // (BLK_M, HEAD_DIM)
      Tensor gQ = local_tile(
          Q, Shape<BLK_M, HEAD_DIM>{}, make_coord(m_block_idx, _0{}));
      Tensor gO = local_tile(
          O, Shape<BLK_M, HEAD_DIM>{}, make_coord(m_block_idx, _0{}));
      // (BLK_M, HEAD_DIM) => (M, K)
      Tensor cQ = local_tile(make_identity_tensor(Q.shape()),
                             Shape<BLK_M, HEAD_DIM>{},
                             make_coord(m_block_idx, _0{}));

      // (BLK_N, HEAD_DIM, n)
      Tensor gK = local_tile(K, Shape<BLK_N, HEAD_DIM>{}, make_coord(_, _0{}));
      Tensor gV = local_tile(V, Shape<BLK_N, HEAD_DIM>{}, make_coord(_, _0{}));
      // (BLK_N, HEAD_DIM, n) => (N, K)
      Tensor cKV = local_tile(make_identity_tensor(K.shape()),
                              Shape<BLK_N, HEAD_DIM>{},
                              make_coord(_, _0{}));

      TiledMma tiled_mma;
      // accumulator: MMA,MMA_M,MMA_K)
      auto tOrAccO = partition_fragment_C(tiled_mma, Shape<BLK_M, HEAD_DIM>{});
      clear(tOrAccO);

      constexpr int kRowsPerThr = kRowsPerMMA * size<1>(tOrAccO);
      OnlineSoftmax<kRowsPerThr> softmax(params.sm_scale_log2);

      // mainloop
      mha(mainloop_params,
          gQ,
          cQ,
          gK,
          gV,
          cKV,
          tOrAccO,
          softmax,
          tidx,
          block_coord,
          residue_mnk,
          smem);

      // epilogue
      epilogue(
          epilogue_params, tOrAccO, tiled_mma, gO, cQ, tidx, residue_mnk, smem);
    }
  }
};

}  // namespace llm
