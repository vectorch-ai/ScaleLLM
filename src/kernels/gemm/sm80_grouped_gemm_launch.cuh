#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cute/layout.hpp>
#include <cute/tensor.hpp>

#include "sm80_collective_epilogue.cuh"
#include "sm80_collective_grouped_gemm.cuh"
// #include "sm80_kernel_mha.cuh"
#include "tile_scheduler.cuh"

namespace llm {

struct GEMMParams {
  using AStride = Stride<int64_t, _1>;
  using BStride = Stride<int64_t, int64_t, _1>;
  using CStride = Stride<int64_t, _1>;

  // A: (m, k)
  const void* __restrict__ a_ptr = nullptr;
  AStride a_stride;

  // B: (e, n, k)
  const void* __restrict__ b_ptr = nullptr;
  BStride b_stride;

  // C: ((m, topk), n)
  void* __restrict__ c_ptr = nullptr;
  CStride c_stride;

  // (m_blocks*BLK_M)
  const int* __restrict__ sorted_token_idxes_ptr = nullptr;
  // (m_blocks)
  const int* __restrict__ expert_ids_ptr = nullptr;

  const int* __restrict__ n_tokens_padded = nullptr;

  int m = 0;
  int n = 0;
  int k = 0;
  int topk = 0;

  int m_blocks = 0;
  int n_blocks = 0;
};

namespace detail {
/// Generic kernel template.
template <typename Operator, typename Params>
__global__ __launch_bounds__(Operator::kMmaThreads) void device_kernel(
    __grid_constant__ const Params params,
    __grid_constant__ const typename Operator::TileSchedulerParams
        scheduler_params) {
  extern __shared__ char smem[];
  Operator op;
  op(params, scheduler_params, smem);
}
}  // namespace detail

// template <bool EVEN_N, bool EVEN_K, typename Traits, typename Params>
// void launch_grouped_gemm_kernel_sm80(const Params& params,
//                                      cudaStream_t stream) {
//   const auto smem_size = sizeof(GEMMSharedStorageSM80<Traits>);
//   // std::cout << "SMEM size: " << smem_size << " bytes\n";

//   auto gemm_kernel = grouped_gemm_kernel_sm80<EVEN_N, EVEN_K, Traits,
//   Params>; cudaFuncSetAttribute(
//       gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
//   // TODO: support persistent kernels
//   dim3 grid(params.m_blocks, params.n_blocks);
//   dim3 block = Traits::kThreadNum;
//   gemm_kernel<<<grid, block, smem_size, stream>>>(params);
// }

template <typename Dtype, bool EVEN_N, bool EVEN_K, typename Params>
void sm80_launch_grouped_gemm_kernel(const Params& params,
                                     cudaStream_t stream) {
  // TODO: tune block shape MNK based on the head dim and smem size
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications-technical-specifications-per-compute-capability
  // SM           | 7.0 | 7.2 | 7.5 | 8.0 | 8.6 | 8.7 | 8.9 | 9.0 | 10.x | 12.0|
  // Max SMEM (KB)|     96    |  64 | 164 | 100 | 164 | 100 |     228    | 100 |
  // valid dynamic shared memory sizes for different compute capabilities:
  // * 7.0 | 7.2 : 0, 8, 16, 32, 64, 96
  // * 7.5       : 0, 32, 64
  // * 8.0 | 8.7 : 0, 8, 16, 32, 64, 100, 132, 164
  // * 8.6 | 8.9 : 0, 8, 16, 32, 64, 100
  // * 9.0 | 10.x: 0, 8, 16, 32, 64, 100, 132, 164, 196, 228
  // * 12.0      : 0, 8, 16, 32, 64, 100
  constexpr int BLK_M = 64;
  constexpr int BLK_N = 64;
  constexpr int BLK_K = 64;
  constexpr int Stages = 2;

  using TileShape = Shape<Int<BLK_M>, Int<BLK_N>, Int<BLK_K>>;
  using CollectiveMainloop =
      Sm80CollectiveGroupedGEMM<Stages, TileShape, Dtype, EVEN_N, EVEN_K>;

  using CollectiveEpilogue = Sm80CollectiveEpilogue<TileShape, Dtype, EVEN_N>;

  // TODO: support persistent kernels
  using TileScheduler = SingleTileScheduler;

  //   const auto m_blocks = cute::ceil_div(max_q_packed_len, BLK_M);
  //   typename TileScheduler::Arguments scheduler_args{
  //       batch_size, m_blocks, n_kv_heads};
  //   auto scheduler_params =
  //       TileScheduler::to_underlying_arguments(scheduler_args);

  //   using AttnKernel =
  //       Sm80KernelMha<CollectiveMainloop, CollectiveEpilogue, TileScheduler>;

  //   auto mha_kernel = detail::device_kernel<AttnKernel, Params>;

  //   const auto smem_size = AttnKernel::kSharedStorageSize;
  //   if (smem_size >= 48 * 1024) {
  //     cudaFuncSetAttribute(
  //         mha_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
  //         smem_size);
  //   }

  //   const dim3 grid = AttnKernel::get_grid_shape(scheduler_args);
  //   const dim3 block = AttnKernel::get_block_shape();

  //   mha_kernel<<<grid, block, smem_size, stream>>>(params, scheduler_params);
  // TODO: check launch status
}

}  // namespace llm
