#pragma once

#include "cutlass/cluster_launch.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include "cutlass/kernel_launch.h"

////////////////////////////////////////////////////////////////////////////////

namespace llm {
using namespace cute;

template <class Kernel>
class Fmha {
 public:
  using Arguments = typename Kernel::Arguments;
  using Params = typename Kernel::Params;
  using ClusterShape = typename Kernel::ClusterShape;

  bool initialize(Arguments const& args, void* workspace = nullptr) {
    params_ = Kernel::to_underlying_arguments(args, workspace);
    if (is_initialized_) {
      return true;
    }

    const int smem_size = Kernel::kSharedStorageSize;
    if (smem_size >= (48 << 10)) {
      cudaError_t result =
          cudaFuncSetAttribute(cutlass::device_kernel<Kernel>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               smem_size);
      if (cudaSuccess != result) {
        result = cudaGetLastError();  // to clear the error bit
        return false;
      }
    }
    is_initialized_ = true;
    return true;
  }

  bool run(cudaStream_t stream = nullptr) {
    const dim3 block = Kernel::get_block_shape();
    const dim3 grid = Kernel::get_grid_shape(params_);
    constexpr int smem_size = Kernel::kSharedStorageSize;

    cutlass::Status status;
    if constexpr (Kernel::ArchTag::kMinComputeCapability >= 90) {
      dim3 cluster(size<0>(ClusterShape{}),
                   size<1>(ClusterShape{}),
                   size<2>(ClusterShape{}));

      cutlass::ClusterLaunchParams launch_params{
          .grid_dims = grid,
          .block_dims = block,
          .cluster_dims = cluster,
          .smem_size_in_bytes = smem_size,
          .cuda_stream = stream,
      };
      void const* kernel = (void const*)cutlass::device_kernel<Kernel>;
      status =
          cutlass::launch_kernel_on_cluster(launch_params, kernel, params_);
    } else {
      status = cutlass::kernel_launch<Kernel>(
          grid, block, smem_size, stream, params_, /*launch_with_pdl=*/false);
    }
    // check_launch_status
    return cutlass::Status::kSuccess == status;
  }

 private:
  Params params_;
  bool is_initialized_ = false;
};

}  // namespace llm
