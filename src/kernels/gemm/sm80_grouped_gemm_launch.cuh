#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cute/layout.hpp>
#include <cute/tensor.hpp>

#include "sm80_collective_epilogue.cuh"
#include "sm80_collective_grouped_gemm.cuh"
#include "sm80_kernel_grouped_gemm.cuh"
#include "tile_scheduler.cuh"

namespace llm {

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

template <int Stages,
          typename TileShape,
          typename Dtype,
          bool EVEN_N,
          bool EVEN_K,
          typename Params>
void sm80_launch_grouped_gemm_kernel(const Params& params,
                                     cudaStream_t stream) {
  using CollectiveMainloop =
      Sm80CollectiveGroupedGEMM<Stages, TileShape, Dtype, EVEN_N, EVEN_K>;

  using CollectiveEpilogue = Sm80CollectiveEpilogue<TileShape, Dtype, EVEN_N>;

  // TODO: support persistent kernels
  using TileScheduler = SingleTileScheduler;

  typename TileScheduler::Arguments scheduler_args{params.m_blocks,
                                                   params.n_blocks};
  auto scheduler_params =
      TileScheduler::to_underlying_arguments(scheduler_args);

  using GEMMKernel = Sm80KernelGroupedGEMM<CollectiveMainloop,
                                           CollectiveEpilogue,
                                           TileScheduler>;

  auto gemm_kernel = detail::device_kernel<GEMMKernel, Params>;

  const auto smem_size = GEMMKernel::kSharedStorageSize;
  if (smem_size >= 48 * 1024) {
    cudaFuncSetAttribute(
        gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  }

  const dim3 grid = GEMMKernel::get_grid_shape(scheduler_args);
  const dim3 block = GEMMKernel::get_block_shape();

  gemm_kernel<<<grid, block, smem_size, stream>>>(params, scheduler_params);
  // TODO: check launch status
}

}  // namespace llm
