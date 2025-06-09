#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include <cute/config.hpp>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>

#include "gather_tensor.hpp"

namespace llm {
using namespace cute;

template <typename DTYPE, int DIM, int BLK_M, int BLK_N, int BLK_K, int STAGES>
struct GEMMTraitsSM80 {
  static constexpr int kDim = DIM;
  static constexpr int kBlockM = BLK_M;
  static constexpr int kBlockN = BLK_N;
  static constexpr int kBlockK = BLK_K;
  static constexpr int kStages = STAGES;

  static_assert(kBlockM % 64 == 0);
  static_assert(kBlockN % 32 == 0);
  static_assert(kBlockK % 16 == 0);

  // helpful aliases
  using DType = DTYPE;
  using _BLK_M = Int<kBlockM>;
  using _BLK_N = Int<kBlockN>;
  using _BLK_K = Int<kBlockK>;
  using _STAGES = Int<kStages>;
  using _DIM = Int<kDim>;

  // TiledMMA: (64x32x16)
  using MMA_Atom_ =
      std::conditional_t<std::is_same_v<DType, cute::half_t>,
                         MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
                         MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>>;
  using TiledMma =
      TiledMMA<MMA_Atom_, Layout<Shape<_4, _2, _1>>, Tile<_64, _32, _16>>;

  //   // Shared memory LayoutAtom (8x64)
  //   using SmemLayoutAtom_8x64 =
  //       decltype(composition(Swizzle<3, 3, 3>{},
  //                            Layout<Shape<_8, _64>, Stride<_64, _1>>{}));
  //   using SmemLayoutAtom_8x32 =
  //       decltype(composition(Swizzle<2, 3, 3>{},
  //                            Layout<Shape<_8, _32>, Stride<_32, _1>>{}));

  //   using SmemLayoutAtomK = std::conditional_t<kBlockK % 64 == 0,
  //                                              SmemLayoutAtom_8x64,
  //                                              SmemLayoutAtom_8x32>;
  //   // SMEM Layout for A: (BLK_M, BLK_K, STAGES)
  //   using SmemLayoutA =
  //       decltype(tile_to_shape(SmemLayoutAtomK{}, Shape<_BLK_M, _BLK_K>{}));
  //   // SMEM Layout for B: (BLK_N, BLK_K, STAGES)
  //   using SmemLayoutB =
  //       decltype(tile_to_shape(SmemLayoutAtomK{}, Shape<_BLK_N, _BLK_K>{}));

  //   // Gmem tiled copy: copy A/B from global memory to shared memory (32x64)
  //   using GmemTiledCopy = decltype(make_tiled_copy(
  //       Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, DType>{},
  //       Layout<Shape<_32, _8>, Stride<_8, _1>>{},  // Thr layout: (_32, _8)
  //       Layout<Shape<_1, _8>>{}                    // Val layout: 8 vals per
  //       read
  //       ));

  // constexpr values for kernel launch
  static constexpr size_t kThreadNum = size(TiledMma{});
};

template <typename Traits>
struct GEMMSharedStorageSM80 {};

struct GEMMParams {
  using AStride = Stride<int64_t /*,_1*/>;
  using BStride = Stride<int64_t, int64_t /*,_1*/>;
  using CStride = Stride<int64_t /*,_1*/>;

  // A: (m, k)
  const void* __restrict__ a_ptr = nullptr;
  AStride a_stride;

  // B: (e, n, k)
  const void* __restrict__ b_ptr = nullptr;
  BStride b_stride;

  // C: ((m, topk), n)
  void* __restrict__ c_ptr = nullptr;
  CStride c_stride;

  // (m_blocks, BLK_M)
  const int* __restrict__ sorted_token_idxes_ptr = nullptr;
  // (m_blocks)
  const int* __restrict__ expert_ids_ptr = nullptr;

  int m = 0;
  int n = 0;
  int k = 0;
  int topk = 0;
  int n_tokens_padded = 0;
};

template <typename Traits, typename Params>
__global__ __launch_bounds__(Traits::kThreadNum) void grouped_gemm_kernel_sm80(
    __grid_constant__ const Params params) {
  // Traits
  constexpr int kBlockM = Traits::kBlockM;
  constexpr int kBlockN = Traits::kBlockN;
  constexpr int kBlockK = Traits::kBlockK;

  using _BLK_M = Int<kBlockM>;
  using _BLK_N = Int<kBlockN>;
  using _BLK_K = Int<kBlockK>;

  using DTYPE = typename Traits::DType;

  const auto topk = params.topk;
  // ProblemShape
  // each thread block takes care of one block: (BLK_M, BLK_N)
  const auto m_block_idx = blockIdx.x;
  const auto n_block_idx = blockIdx.y;
  // const auto expert_id = params.expert_ids_ptr[m_block_idx];
  const auto expert_id = 0;

  // 1: load A to smem: (BLK_M, BLK_K, STAGES)
  //  load sorted_token_idxes from gmem, (m, topk) => (BLK_M)
  const int* sorted_token_idxes =
      params.sorted_token_idxes_ptr + m_block_idx * kBlockM;
  auto idx_to_t_idx = [sorted_token_idxes, topk](int idx) {
    // Convert to token index
    return sorted_token_idxes[idx] / topk;
  };
  // A: (BLK_M, K)
  auto A = make_gather_tensor(make_gmem_ptr((const DTYPE*)params.a_ptr),
                              make_shape(kBlockM, params.k),
                              make_stride(get<0>(params.a_stride), _1{}),
                              idx_to_t_idx);
  if (thread0()) {
    print("A: ");
    print(A);
    print("\n");
  }

  // 2: load B to smem: (BLK_N, BLK_K, STAGES)
  //  load expert_id for current block from gmem, (1)
  // B: (BLK_N, K)
  // (e, n, k) => (BLK_N, k)
  const auto b_offset = expert_id * get<0>(params.b_stride) +
                        n_block_idx * get<1>(params.b_stride);
  auto B = make_tensor(make_gmem_ptr((const DTYPE*)params.b_ptr + b_offset),
                       make_shape(kBlockN, params.k),
                       make_stride(get<1>(params.b_stride), _1{}));
  if (thread0()) {
    print("B: ");
    print(B);
    print("\n");
  }

  // Accumulator: (BLK_M, BLK_N)
  // 3: iterate over k
  // 4:     partition A to tCsA, tCrA
  // 5:     partition B to tCsB, tCrB
  //        load a, b to registers
  // 6:     compute tCrA * tCrB with gemm

  // C: (BLK_M, BLK_N)
  // 7:  write tCrC to global memory using sorted_token_idxes (m, topk)
  auto idx_to_f_idx = [sorted_token_idxes](int idx) {
    // Convert to token index
    return sorted_token_idxes[idx];
  };
  auto C = make_gather_tensor(make_gmem_ptr((const DTYPE*)params.c_ptr),
                              make_shape(kBlockM, kBlockN),
                              make_stride(get<0>(params.c_stride), _1{}),
                              idx_to_f_idx);
  if (thread0()) {
    print("C: ");
    print(C);
    print("\n");
  }
}

template <typename Traits, typename Params>
void launch_grouped_gemm_kernel_sm80(const Params& params,
                                     cudaStream_t stream) {
  //   const auto batch_size = params.batch_size;
  //   const auto max_q_packed_len = params.max_q_len * params.n_heads;

  const auto smem_size = sizeof(GEMMSharedStorageSM80<Traits>);

  auto gemm_kernel = grouped_gemm_kernel_sm80<Traits, Params>;
  cudaFuncSetAttribute(
      gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  // TODO: support persistent kernels
  // dim3 grid(cute::ceil_div(max_q_packed_len, Traits::kBlockM), batch_size,
  // 1);
  dim3 grid(1, 1, 1);  // Placeholder for grid dimensions, adjust as needed
  dim3 block = Traits::kThreadNum;
  gemm_kernel<<<grid, block, smem_size, stream>>>(params);
}

}  // namespace llm
