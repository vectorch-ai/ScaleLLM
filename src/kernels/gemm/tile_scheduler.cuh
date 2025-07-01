#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cute/config.hpp>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>

#include "fast_math.h"

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

enum class RasterOrder { AlongM, AlongN };

class StaticPersistentTileScheduler {
 public:
  // Host side kernel arguments
  struct Arguments {
    FastDivmod cluster_shape_m = 0;
    FastDivmod cluster_shape_n = 0;
    int grid_shape_m = 0;
    int grid_shape_n = 0;
    FastDivmod swizzle = 0;
    RasterOrder raster_order = RasterOrder::AlongM;
  };

  static dim3 get_grid_shape(Arguments const& args, int n_sms) {
    return {(uint32_t)n_sms};
  }

  // Device side kernel params
  using Params = Arguments;
  static Params to_underlying_arguments(const Arguments& args) { return args; }

  class Iterator {
   public:
    CUTE_DEVICE
    Iterator(int start, int step, const Params& params)
        : linear_idx_(start), step_(step), params_(params) {}

    CUTE_DEVICE
    cute::tuple<int, int> operator*() const {
      return swizzle_and_rasterize(linear_idx_, params_);
    }

    CUTE_DEVICE
    Iterator& operator++() {
      linear_idx_ += step_;
      return *this;
    }

    CUTE_DEVICE
    bool operator!=(const Iterator& e) const {
      return linear_idx_ < e.linear_idx_;
    }

   private:
    int linear_idx_;
    int step_;
    const Params& params_;
  };

  CUTE_DEVICE StaticPersistentTileScheduler(const Params& params)
      : params_(params) {
    linear_idx_ = blockIdx.x;
    grid_size_ = gridDim.x * gridDim.y * gridDim.z;
  }

  CUTE_DEVICE
  Iterator begin() const { return {linear_idx_, grid_size_, params_}; }

  CUTE_DEVICE
  Iterator end() const {
    const int problem_tiles = params_.grid_shape_m * params_.grid_shape_n;
    return {problem_tiles, 0, params_};
  }

  // For example: ClusterShape: (2, 1), GridShape(4, 4) and Swizzle=2. Along M
  //
  //                             Reduce Cluster           Reduce Swizzle
  //    <---- N ---->
  //  ^ +--+--+--+--+  ^
  //  | |00|04|08|12|  |          <---- N ---->              <- N ->
  //  C +--+--+--+--+  |          <- S ->
  //  | |01|05|09|13|  |          +--+--+--+--+  ^           +--+--+  ^
  //  v +--+--+--+--+  M   --->   |00|02|04|06|  |    --->   |00|02|  |
  //    |02|06|10|14|  |          +--+--+--+--+  M           +--+--+  M
  //    +--+--+--+--+  |          |01|03|05|07|  |           |01|03|  |
  //    |03|07|11|15|  |          +--+--+--+--+  v           +--+--+  v
  //    +--+--+--+--+  v
  //                                                           |
  //                                                           v
  //                            ^ +--+--+--+--+
  //                            | |00|02|12|14|            <---- N ---->
  //                            C +--+--+--+--+            <- S ->
  //                            | |01|03|13|15|            +--+--+--+--+  ^
  //                            v +--+--+--+--+     <---   |00|01|04|05|  |
  //                              |04|06|08|10|            +--+--+--+--+  M
  //                              +--+--+--+--+            |02|03|06|07|  |
  //                              |05|07|09|11|            +--+--+--+--+  v
  //                              +--+--+--+--+
  //
  //                              Expand  Cluster          Expand Swizzle

  // compute tile coord from linear idx
  CUTE_HOST_DEVICE static cute::tuple<int, int> swizzle_and_rasterize(
      int linear_idx,
      const Params& params) {
    // number of ctas per cluster
    const int cluster_size = params.cluster_shape_m * params.cluster_shape_n;
    // number of cluster along major
    const int major_clusters =
        params.raster_order == RasterOrder::AlongM
            ? params.grid_shape_m / params.cluster_shape_m
            : params.grid_shape_n / params.cluster_shape_n;

    // Convert linear CTA idx/coord into cluster coord.
    // Layout: (cluster_size, clusters):(1, cluster_size)
    const int cluster_idx = linear_idx / cluster_size;
    const int cluster_offset = linear_idx % cluster_size;

    // Convert linear idx within cluster into cta coord
    // Layout: (cluster_shape_m, cluster_shape_n):(1, cluster_shape_m)
    int cluster_offset_m, cluster_offset_n;
    params.cluster_shape_m.divmod(
        cluster_offset, cluster_offset_n, cluster_offset_m);

    int major_idx, minor_idx, panel_idx;
    if (params.swizzle > 1) {
      // Convert cluster linear idx into swizzled coord
      // Layout: (swizzle, panels): (1, swizzle)
      int swizzle_idx, swizzle_offset;
      params.swizzle.divmod(cluster_idx, swizzle_idx, swizzle_offset);

      major_idx = swizzle_idx % major_clusters;
      panel_idx = swizzle_idx / major_clusters;
      minor_idx = panel_idx * params.swizzle + swizzle_offset;
    } else {
      // no swizzle, panel size = 1
      major_idx = cluster_idx % major_clusters;
      panel_idx = cluster_idx / major_clusters;
      minor_idx = panel_idx;
    }

    if ((panel_idx & 1) != 0) {
      // odd idx within panel, reverse major index
      major_idx = (major_clusters - 1 - major_idx);
    }

    if (params.raster_order == RasterOrder::AlongM) {
      // Map the swizzled cluster tile back to a CTA tile
      major_idx = major_idx * params.cluster_shape_m + cluster_offset_m;
      minor_idx = minor_idx * params.cluster_shape_n + cluster_offset_n;
      return {major_idx, minor_idx};
    }

    // raster_order == AlongN
    // Map the swizzled cluster tile back to a CTA tile
    minor_idx = minor_idx * params.cluster_shape_m + cluster_offset_m;
    major_idx = major_idx * params.cluster_shape_n + cluster_offset_n;
    return {minor_idx, major_idx};
  }

 private:
  int linear_idx_ = 0;
  int grid_size_ = 0;
  Params params_;
};

}  // namespace llm
