#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cute/layout.hpp>
#include <cute/tensor.hpp>

#include "common/gather_tensor.h"

namespace llm {

using namespace cute;

template <class CollectiveMainloop_,
          class CollectiveEpilogue_,
          class TileScheduler_>
class Sm80KernelGroupedGEMM {
 public:
  using CollectiveMainloop = CollectiveMainloop_;
  using CollectiveEpilogue = CollectiveEpilogue_;
  using TileScheduler = TileScheduler_;

  using TiledMma = typename CollectiveMainloop::TiledMma;

  using Element = typename CollectiveMainloop::Element;
  using BLK_M = typename CollectiveMainloop::BLK_M;
  using BLK_N = typename CollectiveMainloop::BLK_N;
  using BLK_K = typename CollectiveMainloop::BLK_K;

  static constexpr int kBlockM = CollectiveMainloop::kBlockM;
  static constexpr int kBlockN = CollectiveMainloop::kBlockN;

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
    CollectiveMainloop gemm;
    CollectiveEpilogue epilogue;
    TileScheduler scheduler(scheduler_params);

    // ProblemShape
    const auto M = kBlockM * params.m_blocks;
    const auto N = params.n;
    const auto K = params.k;
    const auto topk = params.topk;
    const auto n_experts = params.n_experts;
    const auto n_flatten_tokens = params.m * topk;

    const auto residue_mnk = make_tuple(M, N, K);

    const int* sorted_token_idxes = params.sorted_token_idxes_ptr;
    auto idx_to_t_idx = [sorted_token_idxes, topk](int idx) {
      return sorted_token_idxes[idx] / topk;
    };
    // A: (M, K), k-major
    auto A = make_gather_tensor(make_gmem_ptr((const Element*)params.a_ptr),
                                make_shape(M, K),
                                params.a_stride,
                                idx_to_t_idx);
    // (M, K) => (BLK_M, BLK_K, m, k)
    Tensor gA_t = local_tile(A, Shape<BLK_M, BLK_K>{}, make_coord(_, _));
    // (BLK_M, BLK_K, m, k) => (M, K)
    Tensor cA_t = local_tile(make_identity_tensor(make_shape(M, K)),
                             Shape<BLK_M, BLK_K>{},
                             make_coord(_, _));

    // B: (E, N, K), k-major
    auto B = make_tensor(make_gmem_ptr((const Element*)params.b_ptr),
                         make_shape(n_experts, N, K),
                         params.b_stride);
    // (E, N, K) => (_1, BLK_N, BLK_K, e, n, k)
    Tensor gB_t = local_tile(B, Shape<_1, BLK_N, BLK_K>{}, make_coord(_, _, _));
    // (BLK_N, BLK_K, n, k) => (N, K)
    Tensor cB_t = local_tile(make_identity_tensor(make_shape(N, K)),
                             Shape<BLK_N, BLK_K>{},
                             make_coord(_, _));

    // C: (M, N), n-major
    auto idx_to_f_idx = [sorted_token_idxes](int idx) {
      return sorted_token_idxes[idx];
    };
    auto C = make_gather_tensor(make_gmem_ptr((Element*)params.c_ptr),
                                make_shape(M, N),
                                params.c_stride,
                                idx_to_f_idx);
    // (M, N) => (BLK_M, BLK_N, m, n)
    Tensor gC_t = local_tile(C, Shape<BLK_M, BLK_N>{}, make_coord(_, _));
    // (BLK_M, BLK_N, m, n) => (M, N)
    Tensor cC_t = local_tile(make_identity_tensor(make_shape(M, N)),
                             Shape<BLK_M, BLK_N>{},
                             make_coord(_, _));

    // construct params
    MainloopParams mainloop_params{params.sorted_token_idxes_ptr,
                                   n_flatten_tokens};
    EpilogueParams epilogue_params{params.sorted_token_idxes_ptr,
                                   n_flatten_tokens};

    // process each block
    for (const auto block_coord : scheduler) {
      // block coord: (batch_idx, m_block_idx, kv_head_idx)
      const auto [m_block_idx, n_block_idx] = block_coord;
      const auto tidx = threadIdx.x;
      const int expert_id = params.expert_ids_ptr[m_block_idx];

      // (BLK_M, BLK_K, m, k) => (BLK_M, BLK_K, k)
      auto gA = gA_t(_, _, m_block_idx, _);
      auto cA = cA_t(_, _, m_block_idx, _);
      // (_1, BLK_N, BLK_K, e, n, k) => (BLK_N, BLK_K, k)
      auto gB = gB_t(_0{}, _, _, expert_id, n_block_idx, _);
      // (BLK_N, BLK_K, n, k) => (BLK_N, BLK_K, k)
      auto cB = cB_t(_, _, n_block_idx, _);
      // (BLK_M, BLK_N, m, n) => (BLK_M, BLK_N)
      auto gC = gC_t(_, _, m_block_idx, n_block_idx);
      auto cC = cC_t(_, _, m_block_idx, n_block_idx);

      TiledMma tiled_mma;
      // (BLK_M, BLK_N) => (MMA, MMA_M, MMA_N)
      auto tCrAccC = partition_fragment_C(tiled_mma, Shape<BLK_M, BLK_N>{});
      cute::clear(tCrAccC);  // Clear the accumulator

      // mainloop
      gemm(mainloop_params, gA, cA, gB, cB, tCrAccC, tidx, residue_mnk, smem);

      // epilogue
      epilogue(
          epilogue_params, tCrAccC, tiled_mma, gC, cC, tidx, residue_mnk, smem);
    }
  }
};

}  // namespace llm
