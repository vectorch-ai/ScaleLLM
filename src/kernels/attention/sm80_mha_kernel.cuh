#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cute/layout.hpp>
#include <cute/tensor.hpp>

#include "cute/atom/mma_atom.hpp"
#include "cute/config.hpp"
#include "cute/container/array_aligned.hpp"
#include "cute_extensions.cuh"
#include "fast_cast.cuh"
#include "layout_convertor.h"
#include "mask.h"
#include "mha_tile.h"
#include "online_softmax.cuh"

namespace llm {

using namespace cute;

template <class CollectiveMainloop_, class CollectiveEpilogue_, class Params_>
class Sm80MhaKernel {
 public:
  using CollectiveMainloop = CollectiveMainloop_;
  using CollectiveEpilogue = CollectiveEpilogue_;
  using Params = Params_;

  // Mainloop derived types
  using TileShape_MNK = typename CollectiveMainloop::TileShape_MNK;
  using TiledMma = typename CollectiveMainloop::TiledMma;
  using Element = typename CollectiveMainloop::Element;
  using BLK_M = typename CollectiveMainloop::BLK_M;
  using BLK_N = typename CollectiveMainloop::BLK_N;
  using HEAD_DIM = typename CollectiveMainloop::HEAD_DIM;

  using MainloopParams = typename CollectiveMainloop::Params;

  static constexpr int kRowsPerMMA = CollectiveMainloop::kRowsPerMMA;
  static constexpr int kMmaThreads = CollectiveMainloop::kMmaThreads;

  // Epilogue derived types
  using EpilogueParams = typename CollectiveEpilogue::Params;

  CUTLASS_DEVICE
  void operator()(const Params& params, char* smem) {
    CollectiveMainloop mainloop;
    CollectiveEpilogue epilogue;

    const auto tidx = threadIdx.x;
    const int m_block_idx = blockIdx.x;
    const int batch_idx = blockIdx.y;
    const int kv_head_idx = blockIdx.z;

    // ProblemShape
    // (q_packed_len, HEAD_DIM)
    MHATile<Params> tile(params, batch_idx, kv_head_idx);
    auto [Q, O] = tile.template get_qo_tile<Element>();
    // (kv_len, HEAD_DIM)
    auto [K, V] = tile.template get_kv_tile<Element>();

    const int q_packed_len = size<0>(Q);
    const int kv_len = size<0>(K);
    const int head_dim = params.head_dim;

    // (BLK_M, HEAD_DIM)
    Tensor gQ =
        local_tile(Q, Shape<BLK_M, HEAD_DIM>{}, make_coord(m_block_idx, _0{}));
    Tensor gO =
        local_tile(O, Shape<BLK_M, HEAD_DIM>{}, make_coord(m_block_idx, _0{}));
    // (BLK_N, HEAD_DIM, n)
    Tensor gK = local_tile(K, Shape<BLK_N, HEAD_DIM>{}, make_coord(_, _0{}));
    Tensor gV = local_tile(V, Shape<BLK_N, HEAD_DIM>{}, make_coord(_, _0{}));

    MainloopParams mainloop_params{params.sliding_window,
                                   params.logits_soft_cap,
                                   params.sm_scale,
                                   params.sm_scale_log2,
                                   params.alibi_slopes_ptr,
                                   params.group_size};

    EpilogueParams epilogue_params;

    const auto block_coord = make_tuple(m_block_idx, batch_idx, kv_head_idx);
    const auto residue_mnk = make_tuple(q_packed_len, kv_len, head_dim);

    TiledMma tiled_mma;
    // accumulator: MMA,MMA_M,MMA_K)
    auto tOrAccO = partition_fragment_C(tiled_mma, Shape<BLK_M, HEAD_DIM>{});
    clear(tOrAccO);

    constexpr int kRowsPerThr = kRowsPerMMA * size<1>(tOrAccO);
    using Softmax = OnlineSoftmax<kRowsPerThr>;
    Softmax softmax(params.sm_scale_log2);

    // mainloop
    mainloop.mha(mainloop_params,
                 gQ,
                 gK,
                 gV,
                 tOrAccO,
                 softmax,
                 tidx,
                 block_coord,
                 residue_mnk,
                 smem);
    // epilogue
    epilogue.store(
        epilogue_params, tOrAccO, gO, tidx, block_coord, residue_mnk, smem);
  }
};

}  // namespace llm
