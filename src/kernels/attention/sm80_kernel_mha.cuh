#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cute/layout.hpp>
#include <cute/tensor.hpp>

#include "mha_tile.h"
#include "online_softmax.cuh"

namespace llm {

using namespace cute;

template <class CollectiveMainloop_, class CollectiveEpilogue_>
class Sm80KernelMha {
 public:
  using CollectiveMainloop = CollectiveMainloop_;
  using CollectiveEpilogue = CollectiveEpilogue_;

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

  template <class Params>
  CUTE_DEVICE void operator()(const Params& params, char* smem) {
    CollectiveMainloop mha;
    CollectiveEpilogue epilogue;

    const auto tidx = threadIdx.x;

    // block coord
    const int m_block_idx = blockIdx.x;
    const int batch_idx = blockIdx.y;
    const int kv_head_idx = blockIdx.z;
    auto block_coord_mnk = make_coord(m_block_idx, batch_idx, kv_head_idx);

    // (q_packed_len, HEAD_DIM)
    MHATile<Params> tile(params, batch_idx, kv_head_idx);
    auto [Q, O] = tile.template get_qo_tile<Element>();
    // (kv_len, HEAD_DIM)
    auto [K, V] = tile.template get_kv_tile<Element>();

    // problem shape
    const int q_packed_len = size<0>(Q);
    const int kv_len = size<0>(K);
    const int head_dim = params.head_dim;
    auto problem_shape_mnk = make_shape(q_packed_len, kv_len, head_dim);

    if (m_block_idx * kBlockM >= q_packed_len) {
      // m out of bound, return
      return;
    }

    // (BLK_M, HEAD_DIM)
    Tensor gQ =
        local_tile(Q, Shape<BLK_M, HEAD_DIM>{}, make_coord(m_block_idx, _0{}));
    Tensor gO =
        local_tile(O, Shape<BLK_M, HEAD_DIM>{}, make_coord(m_block_idx, _0{}));
    // (BLK_N, HEAD_DIM, n)
    Tensor gK = local_tile(K, Shape<BLK_N, HEAD_DIM>{}, make_coord(_, _0{}));
    Tensor gV = local_tile(V, Shape<BLK_N, HEAD_DIM>{}, make_coord(_, _0{}));

    // construct params
    MainloopParams mainloop_params{params.sliding_window,
                                   params.logits_soft_cap,
                                   params.sm_scale,
                                   params.sm_scale_log2,
                                   params.alibi_slopes_ptr,
                                   params.group_size};
    EpilogueParams epilogue_params;

    TiledMma tiled_mma;
    // accumulator: MMA,MMA_M,MMA_K)
    auto tOrAccO = partition_fragment_C(tiled_mma, Shape<BLK_M, HEAD_DIM>{});
    clear(tOrAccO);

    constexpr int kRowsPerThr = kRowsPerMMA * size<1>(tOrAccO);
    OnlineSoftmax<kRowsPerThr> softmax(params.sm_scale_log2);

    // mainloop
    mha(mainloop_params,
        gQ,
        gK,
        gV,
        tOrAccO,
        softmax,
        tidx,
        block_coord_mnk,
        problem_shape_mnk,
        smem);

    // epilogue
    epilogue(epilogue_params,
             tOrAccO,
             tiled_mma,
             gO,
             tidx,
             block_coord_mnk,
             problem_shape_mnk,
             smem);
  }
};

}  // namespace llm
