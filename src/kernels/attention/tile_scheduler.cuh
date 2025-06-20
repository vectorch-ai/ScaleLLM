#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cute/layout.hpp>
#include <cute/tensor.hpp>

namespace llm {

class SingleTileScheduler {
 public:
  // Host side kernel arguments
  struct Arguments {
    int m_blocks = 0;
    int batch_size = 0;
    int n_kv_heads = 0;
  };

  // Device side kernel params
  using Params = Arguments;

  static Params to_underlying_arguments(const Arguments& args) { return args; }

  static dim3 get_grid_dim(Arguments const& args) {
    return {(uint32_t)args.m_blocks,
            (uint32_t)args.batch_size,
            (uint32_t)args.n_kv_heads};
  }

  struct WorkTileInfo {
    int m_block_idx = 0;
    int batch_idx = 0;
    int kv_head_idx = 0;
    bool is_valid = false;

    CUTE_DEVICE
    bool valid() const { return is_valid; }

    CUTE_DEVICE
    auto get_block_coord() const {
      return cute::tuple{m_block_idx, batch_idx, kv_head_idx};
    }
  };

  CUTE_DEVICE
  SingleTileScheduler(const Params& params) {}

  CUTE_DEVICE
  WorkTileInfo get_initial_work() const {
    return {(int)blockIdx.x,
            (int)blockIdx.y,
            (int)blockIdx.z,
            /*is_valid_tile*/ true};
  }

  CUTE_DEVICE WorkTileInfo
  get_next_work(const WorkTileInfo& /*current_work*/) const {
    // no more works
    return {0, 0, 0, /*is_valid_tile*/ false};
  }
};

}  // namespace llm
