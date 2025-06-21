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
    int n_blocks = 0;
  };
  static dim3 get_grid_shape(Arguments const& args) {
    return {(uint32_t)args.m_blocks, (uint32_t)args.n_blocks};
  }

  // Device side kernel params
  using Params = Arguments;
  static Params to_underlying_arguments(const Arguments& args) { return args; }

  // End Iterator tag
  class EndIterator {};
  class Iterator {
   public:
    CUTE_DEVICE
    Iterator() = default;

    CUTE_DEVICE
    cute::tuple<int, int> operator*() const { return {blockIdx.x, blockIdx.y}; }

    CUTE_DEVICE
    Iterator& operator++() {
      valid_ = false;
      return *this;
    }

    // compare against end iterator
    CUTE_DEVICE
    bool operator!=(const EndIterator&) const { return valid_; }

   private:
    bool valid_ = true;
  };

  CUTE_DEVICE
  SingleTileScheduler(const Params& params) {}

  CUTE_DEVICE
  Iterator begin() const { return {}; }

  CUTE_DEVICE
  EndIterator end() const { return {}; }
};

}  // namespace llm
