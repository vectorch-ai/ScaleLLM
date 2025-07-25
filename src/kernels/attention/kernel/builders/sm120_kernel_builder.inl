#pragma once

#include <cutlass/arch/arch.h>

#include <cute/tensor.hpp>

#include "collective/sm120_collective_epilogue.cuh"
#include "collective/sm120_collective_fmha_mainloop_ws.cuh"
#include "common/fmha_block.h"
#include "common/tile_scheduler.cuh"
#include "kernel/sm120_kernel_fmha_ws.cuh"
#include "kernel_builder_decl.h"

namespace llm {

template <class ProblemShape,
          class TileShape,
          class Element,
          class StrideQ,
          class StrideK,
          class StrideV,
          class StrideO,
          bool EVEN_K,
          bool ALIBI,
          bool SOFT_CAP,
          bool LOCAL,
          bool KV_USE_TMA>
struct KernelBuilder<cutlass::arch::Sm120,
                     ProblemShape,
                     TileShape,
                     Element,
                     StrideQ,
                     StrideK,
                     StrideV,
                     StrideO,
                     EVEN_K,
                     ALIBI,
                     SOFT_CAP,
                     LOCAL,
                     KV_USE_TMA,
                     cute::enable_if_t<not cute::is_tuple_v<Element>>> {
  // TODO: support persistent kernels
  using TileScheduler = SingleTileScheduler;
  using BlocKCoord = TileScheduler::BlocKCoord;
  using Block = FmhaBlock<ProblemShape,
                          TileShape,
                          BlocKCoord,
                          Element,
                          StrideQ,
                          StrideK,
                          StrideV,
                          StrideO>;

  using CollectiveMainloop = Sm120CollectiveFMhaWs<TileShape,
                                                   Element,
                                                   EVEN_K,
                                                   ALIBI,
                                                   SOFT_CAP,
                                                   LOCAL,
                                                   KV_USE_TMA>;
  // using SmemLayoutK = typename CollectiveMainloop::SmemLayoutK;
  // using SmemLayoutV = typename CollectiveMainloop::SmemLayoutV;
  // TODO: pass in SmemLayout to Block for TMA definitions
  using CollectiveEpilogue =
      Sm120CollectiveEpilogue<TileShape, Element, EVEN_K>;

  using Kernel = Sm120KernelFmhaWs<ProblemShape,
                                   Block,
                                   CollectiveMainloop,
                                   CollectiveEpilogue,
                                   TileScheduler>;
};

}  // namespace llm
