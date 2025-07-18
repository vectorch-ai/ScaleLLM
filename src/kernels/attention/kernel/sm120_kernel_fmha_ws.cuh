#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cutlass/pipeline/pipeline.hpp>

#include "common/fmha_block.h"

namespace llm {

using namespace cute;

namespace detail {

struct Sm120WarpSpecializedScheduler {
  enum class WarpRole : uint8_t {
    Load,  // load Q/K/V from gmem to smem
    FMHA,  // collective FMHA mainloop
    Empty  // no work, used to donate regs
  };

  // one warpgroup for loading Q, K/V from gmem to smem
  static constexpr int kNumWarpsLoad = 4;
  // one warpgroup for FMHA mainloop
  static constexpr int kNumWarpsFMHA = 4;
  // 0 warps for empty workgroup to donate registers
  static constexpr int kNumWarpsEmpty = 0;
  // total number of warps in the kernel
  static constexpr int kNumWarps =
      kNumWarpsLoad + kNumWarpsFMHA + kNumWarpsEmpty;

  // TODO: Tune the number of registers for each role
  // valid value expected to be multiple of 8
  // 96*128 + 248 * 128 = 44032 < 65536 (64 KB per SM)
  static constexpr int kNumRegLoad = 96;
  static constexpr int kNumRegFMHA = 248;
  static constexpr int kNumRegEmpty = 24;

  static constexpr WarpRole warp_idx_to_role(int warp_idx) {
    const auto wg_idx = warp_idx / 4;  // 4 warps per workgroup
    // warp 0 is for loading Q, warp 1 is for loading K/V
    switch (wg_idx) {
      case 0:
        return WarpRole::Load;
      case 1:
        return WarpRole::FMHA;
      default:
        return WarpRole::Empty;
    }
    return WarpRole::Empty;  // unreachable
  }
};

template <uint32_t RegCount>
CUTE_DEVICE void warpgroup_reg_set() {
  if constexpr (RegCount < 128) {
    cutlass::arch::warpgroup_reg_dealloc<RegCount>();
  } else {
    cutlass::arch::warpgroup_reg_alloc<RegCount>();
  }
}

}  // namespace detail

template <class CollectiveMainloop,
          class CollectiveEpilogue,
          class TileScheduler,
          class WarpScheduler = detail::Sm120WarpSpecializedScheduler>
class Sm120KernelFmhaWs {
 public:
  using TileShape = typename CollectiveMainloop::TileShape;
  using Element = typename CollectiveMainloop::Element;
  using ClusterShape = typename CollectiveMainloop::ClusterShape;

  static const int kThreadsPerBlock =
      WarpScheduler::kNumWarps * cutlass::NumThreadsPerWarp;

  using PipelineQ = typename CollectiveMainloop::PipelineQ;
  using PipelineKV = typename CollectiveMainloop::PipelineKV;

  using PipelineParamsQ = typename PipelineQ::Params;
  using PipelineParamsKV = typename PipelineKV::Params;

  using PipelineStateQ = typename PipelineQ::PipelineState;
  using PipelineStateKV = typename PipelineKV::PipelineState;

  struct SharedStorage {
    typename CollectiveMainloop::TensorStorage mainloop;
    typename CollectiveEpilogue::TensorStorage epilogue;

    struct PipelineStorage {
      alignas(16) typename PipelineQ::SharedStorage load_q;
      alignas(16) typename PipelineKV::SharedStorage load_kv;
    } pipelines;
  };

  static constexpr int kSharedStorageSize = sizeof(SharedStorage);

  // Kernel params
  using MainloopParams = typename CollectiveMainloop::Params;
  using EpilogueParams = typename CollectiveEpilogue::Params;
  using TileSchedulerParams = typename TileScheduler::Params;

  // returns grid and block shape for kernel launch
  using TileSchedulerArgs = typename TileScheduler::Arguments;
  static dim3 get_grid_shape(TileSchedulerArgs const& args) {
    return TileScheduler::get_grid_shape(args);
  }
  static dim3 get_block_shape() { return kThreadsPerBlock; }

  template <class Params>
  CUTE_DEVICE void load_loop(const Params& params,
                             const TileSchedulerParams& scheduler_params,
                             PipelineQ& q_pipeline,
                             PipelineKV& kv_pipeline,
                             SharedStorage& ss) {
    static constexpr int kHeadDim = CollectiveMainloop::kHeadDim;
    static constexpr bool kLocal = CollectiveMainloop::kLocal;
    using Block = FmhaBlock<Params, TileShape, Element, kHeadDim, kLocal>;

    auto q_state = cutlass::make_producer_start_state<PipelineQ>();
    auto kv_state = cutlass::make_producer_start_state<PipelineKV>();

    CollectiveMainloop mainloop;
    TileScheduler scheduler(scheduler_params);

    // thread idx within warp group (4 warps = 128 threads)
    const auto tidx = threadIdx.x % cutlass::NumThreadsPerWarpGroup;

    // process each block
    for (const auto blk_coord : scheduler) {
      // block coord: (batch_idx, m_block_idx, kv_head_idx)
      const Block block(params, blk_coord);
      mainloop.load(
          block, tidx, q_pipeline, q_state, kv_pipeline, kv_state, ss.mainloop);
    }

    // prevent early exit of producer blocks in cluster
    q_pipeline.producer_tail(q_state);
    kv_pipeline.producer_tail(kv_state);
  }  // end of load_loop

  template <class Params>
  CUTE_DEVICE void fmha_loop(const Params& params,
                             const TileSchedulerParams& scheduler_params,
                             PipelineQ& q_pipeline,
                             PipelineKV& kv_pipeline,
                             SharedStorage& ss) {
    static constexpr int kHeadDim = CollectiveMainloop::kHeadDim;
    static constexpr bool kLocal = CollectiveMainloop::kLocal;
    using TiledMma = typename CollectiveMainloop::TiledMma;
    using BLK_M = typename CollectiveMainloop::BLK_M;
    using HEAD_DIM = typename CollectiveMainloop::HEAD_DIM;

    using Block = FmhaBlock<Params, TileShape, Element, kHeadDim, kLocal>;

    PipelineStateQ q_state;
    PipelineStateKV kv_state;

    CollectiveMainloop mainloop;
    CollectiveEpilogue epilogue;

    // construct params
    MainloopParams mainloop_params{params.sliding_window,
                                   params.logits_soft_cap,
                                   params.sm_scale,
                                   params.sm_scale_log2,
                                   params.group_size,
                                   params.alibi_slopes_ptr};

    EpilogueParams epilogue_params;

    TileScheduler scheduler(scheduler_params);

    // thread idx within warp group (4 warps = 128 threads)
    const auto tidx = threadIdx.x % cutlass::NumThreadsPerWarpGroup;

    // process each block
    const auto& group_size = params.group_size;
    for (const auto blk_coord : scheduler) {
      // block coord: (batch_idx, m_block_idx, kv_head_idx)
      const Block block(params, blk_coord);

      TiledMma tiled_mma;
      // accumulator: (MMA,MMA_M,MMA_K)
      auto tOrAccO = partition_fragment_C(tiled_mma, Shape<BLK_M, HEAD_DIM>{});
      clear(tOrAccO);

      mainloop.fmha(mainloop_params,
                    block,
                    tOrAccO,
                    tidx,
                    q_pipeline,
                    q_state,
                    kv_pipeline,
                    kv_state,
                    ss.mainloop);

      epilogue(epilogue_params, block, tOrAccO, tiled_mma, tidx, ss.epilogue);
    }
  }  // end of fmha_loop

  template <class Params>
  CUTE_DEVICE void operator()(const Params& params,
                              const TileSchedulerParams& scheduler_params,
                              char* smem) {
    static constexpr bool kKVUseTma = CollectiveMainloop::kKVUseTma;
    static constexpr int kNumThreadsLoad =
        WarpScheduler::kNumWarpsLoad * cutlass::NumThreadsPerWarp;
    static constexpr int kNumThreadsFMHA =
        WarpScheduler::kNumWarpsFMHA * cutlass::NumThreadsPerWarp;

    using WarpRole = typename WarpScheduler::WarpRole;

    const int warp_idx = cutlass::canonical_warp_idx_sync();
    const auto role = WarpScheduler::warp_idx_to_role(warp_idx);
    const uint32_t lane_predicate = cute::elect_one_sync();

    // thread idx within warp group (4 warps = 128 threads)
    const auto tidx_in_wg = threadIdx.x % cutlass::NumThreadsPerWarpGroup;

    auto& ss = *reinterpret_cast<SharedStorage*>(smem);

    // define pipelines for loading Q, KV
    PipelineQ q_pipeline = [&] {
      PipelineParamsQ pipeline_params;
      if (role == WarpRole::Load) {
        pipeline_params.role = PipelineQ::ThreadCategory::Producer;
      } else if (role == WarpRole::FMHA) {
        pipeline_params.role = PipelineQ::ThreadCategory::Consumer;
      }
      pipeline_params.producer_arv_count = kNumThreadsLoad;
      pipeline_params.consumer_arv_count = kNumThreadsFMHA;
      return PipelineQ(ss.pipelines.load_q, pipeline_params);
    }();

    PipelineKV kv_pipeline = [&] {
      PipelineParamsKV pipeline_params;
      if (role == WarpRole::Load) {
        pipeline_params.role = PipelineKV::ThreadCategory::Producer;
      } else if (role == WarpRole::FMHA) {
        pipeline_params.role = PipelineKV::ThreadCategory::Consumer;
      }
      if constexpr (kKVUseTma) {
        pipeline_params.transaction_bytes =
            CollectiveMainloop::kTmaTransactionBytes;
        pipeline_params.is_leader = (tidx_in_wg == 0);
        pipeline_params.num_producers = 1;  // only one thread issuing tma
        pipeline_params.num_consumers = kNumThreadsFMHA;
        return PipelineKV(
            ss.pipelines.load_kv, pipeline_params, ClusterShape{});
      } else {
        pipeline_params.producer_arv_count = kNumThreadsLoad;
        pipeline_params.consumer_arv_count = kNumThreadsFMHA;
        return PipelineKV(ss.pipelines.load_kv, pipeline_params);
      }
    }();

    // ensure the pipeline init is visible to all blocks in the cluster
    __syncthreads();

    if (role == WarpRole::Load) {
      detail::warpgroup_reg_set<WarpScheduler::kNumRegLoad>();
      // load Q, K, V from gmem to smem
      load_loop(params, scheduler_params, q_pipeline, kv_pipeline, ss);
    } else if (role == WarpRole::FMHA) {
      detail::warpgroup_reg_set<WarpScheduler::kNumRegFMHA>();
      // FMHA mainloop
      fmha_loop(params, scheduler_params, q_pipeline, kv_pipeline, ss);
    } else if (role == WarpRole::Empty) {
      // Empty warp, do nothing except donating registers
      detail::warpgroup_reg_set<WarpScheduler::kNumRegEmpty>();
    }
  }
};

}  // namespace llm
