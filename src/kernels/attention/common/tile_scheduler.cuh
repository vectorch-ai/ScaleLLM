#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cute/layout.hpp>
#include <cute/tensor.hpp>

namespace llm {
using namespace cute;
class SingleTileScheduler {
 public:
  // (m_block_idx, ((kv_head_idx, _0), batch_idx))
  using BlocKCoord = Coord<int, Coord<Coord<int, _0>, int>>;

  // Device side kernel arguments
  struct Params {
    int batch_size = 0;
    int m_blocks = 0;
    int n_kv_heads = 0;
  };

  static dim3 get_grid_shape(Params const& params) {
    return {(uint32_t)params.m_blocks,
            (uint32_t)params.n_kv_heads,
            (uint32_t)params.batch_size};
  }

  template <class ProblemShape, class TileShape>
  static Params to_underlying_arguments(const ProblemShape& problem_shape,
                                        const TileShape& tile_shape) {
    // problem_shape: (Q, K, D, ((KH, G), B))
    const int max_q_len = size<0>(problem_shape);
    const int n_kv_heads = size<3, 0, 0>(problem_shape);
    const int group_size = size<3, 0, 1>(problem_shape);
    const int batch_size = size<3, 1>(problem_shape);

    const int max_q_packed_len = max_q_len * group_size;
    const int m_blocks = ceil_div(max_q_packed_len, size<0>(tile_shape));

    return {batch_size, m_blocks, n_kv_heads};
  }

  // End Iterator tag
  class EndIterator {};
  class Iterator {
   public:
    CUTE_DEVICE
    Iterator() = default;

    CUTE_DEVICE
    BlocKCoord operator*() const {
      return make_coord(
          (int)blockIdx.x,
          make_coord(make_coord((int)blockIdx.y, _0{}), (int)blockIdx.z));
    }

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
