#pragma once

#include <cute/int_tuple.hpp>
#include <cute/layout.hpp>

#include "huggingface/safetensors.h"
#include "sm80_grouped_gemm_launch.cuh"
#include "static_dispatch.h"

namespace llm {
using namespace cute;

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

// forward declaration
// template <int Stages,
//           typename TileShape,
//           typename Dtype,
//           bool EVEN_N,
//           bool EVEN_K,
//           typename Params>
// void sm80_launch_grouped_gemm_kernel(const Params& params, cudaStream_t
// stream);

// user-facing function to run the attention kernel
template <typename Dtype, typename Params>
void sm80_run_grouped_gemm(Params& params, cudaStream_t stream = nullptr) {
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

  // dispatch to proper kernel instantiation based on params
  DISPATCH_BOOL((params.n % BLK_N) == 0, EVEN_N, [&] {
    DISPATCH_BOOL((params.k % BLK_K) == 0, EVEN_K, [&] {
      sm80_launch_grouped_gemm_kernel<Stages,
                                      TileShape,
                                      Dtype,
                                      EVEN_N,
                                      EVEN_K,
                                      Params>(params, stream);
    });
  });
}

}  // namespace llm
