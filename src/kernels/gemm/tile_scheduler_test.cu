#include <gtest/gtest.h>

#include <cute/tensor.hpp>

#include "tile_scheduler.cuh"

namespace llm {

class TileSchedulerTest
    : public ::testing::TestWithParam<std::tuple<int32_t /*cluster_m*/,
                                                 int32_t /*cluster_n*/,
                                                 int32_t /*grid_m*/,
                                                 int32_t /*grid_n*/,
                                                 int32_t /*swizzle*/,
                                                 RasterOrder /*order*/>> {};

// StaticPersistentTileScheduler
TEST_P(TileSchedulerTest, StaticPersistent) {
  using TileScheduler = StaticPersistentTileScheduler;
  using namespace cute;

  const auto [cluster_m, cluster_n, grid_m, grid_n, swizzle, order] =
      GetParam();

  TileScheduler::Params params{
      cluster_m, cluster_n, grid_m, grid_n, swizzle, order};

  const int problem_tiles = params.grid_shape_m * params.grid_shape_n;
  // std::vector<int> mapping_data(problem_tiles);
  // auto mapping =
  //     make_tensor(mapping_data.data(),
  //                 make_shape(params.grid_shape_m, params.grid_shape_n));
  int pre_tile_m = 0, pre_tile_n = 0;
  const int max_dist = order == RasterOrder::AlongM
                           ? (swizzle * cluster_n) + cluster_m
                           : (swizzle * cluster_m) + cluster_n;
  int32_t valid = 0;
  for (int linear_idx = 0; linear_idx < problem_tiles; ++linear_idx) {
    const auto [tile_m, tile_n] =
        TileScheduler::swizzle_and_rasterize(linear_idx, params);

    const int dist =
        std::abs(tile_m - pre_tile_m) + std::abs(tile_n - pre_tile_n);
    pre_tile_m = tile_m;
    pre_tile_n = tile_n;
    EXPECT_LE(dist, max_dist);
    // mapping(tile_m, tile_n) = linear_idx;

    // (grid_m, grid_n):(1, grid_m)
    const int idx = tile_m + (tile_n * grid_m);
    valid ^= idx;
    valid ^= linear_idx;
  }
  EXPECT_EQ(valid, 0);

  // print_tensor(mapping);
}

INSTANTIATE_TEST_SUITE_P(
    TileScheduler,
    TileSchedulerTest,
    ::testing::Combine(::testing::Values(1, 2),     // cluster_m
                       ::testing::Values(1, 2),     // cluster_n
                       ::testing::Values(8, 16),    // grid_m
                       ::testing::Values(8, 16),    // grid_n
                       ::testing::Values(1, 2, 4),  // swizzle
                       ::testing::Values(RasterOrder::AlongM,
                                         RasterOrder::AlongN)  // order
                       ));

}  // namespace llm
