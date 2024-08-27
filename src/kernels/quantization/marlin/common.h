#pragma once

namespace marlin {

// Marlin params

// 8 warps are a good choice since every SM has 4 schedulers and having more
// than 1 warp per schedule allows some more latency hiding. At the same time,
// we want relatively few warps to have many registers per warp and small tiles.
static constexpr int default_threads = 256;

// 4 pipeline stages fit into shared memory
static constexpr int pipe_stages = 4;

static constexpr int min_thread_n = 64;
static constexpr int min_thread_k = 64;

// 16x16 tiles
static constexpr int tile_size = 16;

// max number of parallelism
static constexpr int max_par = 16;

// Repack params
// number of stages in the repack kernel
static constexpr int repack_stages = 8;

// number of threads in the repack kernel
static constexpr int repack_threads = 256;

// 16x64 tiles
static constexpr int tile_k_size = tile_size;
static constexpr int tile_n_size = tile_k_size * 4;

constexpr int div_ceil(int a, int b) { return (a + b - 1) / b; }

}  // namespace marlin