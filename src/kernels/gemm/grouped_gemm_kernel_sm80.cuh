#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include <cute/config.hpp>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>

#include "gather_tensor.hpp"

namespace llm {
using namespace cute;

template <typename DTYPE, int BLK_M, int BLK_N, int BLK_K, int PIPE>
struct GEMMTraitsSM80 {
  static constexpr int kBlockM = BLK_M;
  static constexpr int kBlockN = BLK_N;
  static constexpr int kBlockK = BLK_K;
  static constexpr int kPipe = PIPE;

  static_assert(kBlockM % 64 == 0);
  static_assert(kBlockN % 32 == 0);
  static_assert(kBlockK % 16 == 0);

  // helpful aliases
  using DType = DTYPE;
  using _BLK_M = Int<kBlockM>;
  using _BLK_N = Int<kBlockN>;
  using _BLK_K = Int<kBlockK>;
  using _PIPE = Int<kPipe>;

  // MMA Atom: (16x8x16) for F32F16F16F32 or F32BF16BF16F32
  using MMA_Atom_ =
      std::conditional_t<std::is_same_v<DType, cute::half_t>,
                         MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
                         MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>>;

  // TiledMMA: (64x16x16)
  using TiledMma = TiledMMA<MMA_Atom_,
                            Layout<Shape<_4, _1, _1>>,  // warp layout: (4x1x1)
                            Tile<_64, _16, _16>>;  // tile layout: (64x16x16)

  // Shared memory LayoutAtom (8x64)
  using SmemLayoutAtom_8x64 =
      decltype(composition(Swizzle<3, 3, 3>{},
                           Layout<Shape<_8, _64>, Stride<_64, _1>>{}));
  using SmemLayoutAtom_8x32 =
      decltype(composition(Swizzle<2, 3, 3>{},
                           Layout<Shape<_8, _32>, Stride<_32, _1>>{}));

  using SmemLayoutAtom = std::conditional_t<kBlockK % 64 == 0,
                                            SmemLayoutAtom_8x64,
                                            SmemLayoutAtom_8x32>;
  // SMEM Layout for A: (BLK_M, BLK_K, PIPE)
  using SmemLayoutA =
      decltype(tile_to_shape(SmemLayoutAtom{}, Shape<_BLK_M, _BLK_K, _PIPE>{}));
  // SMEM Layout for B: (BLK_N, BLK_K, PIPE)
  using SmemLayoutB =
      decltype(tile_to_shape(SmemLayoutAtom{}, Shape<_BLK_N, _BLK_K, _PIPE>{}));

  // Thread layout for gmem copy: (_16,_8)/(_32, _4)
  using GmemCopyThrLayout =
      std::conditional_t<BLK_K == 32,
                         Layout<Shape<_32, _4>, Stride<_4, _1>>,
                         Layout<Shape<_16, _8>, Stride<_8, _1>>>;
  // g2s tiled copy: copy A/B from global memory to shared memory
  using GmemTiledCopyAB = decltype(make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, DType>{},
      GmemCopyThrLayout{},     // Thr layout: (_16,_8)/(_32, _4)
      Layout<Shape<_1, _8>>{}  // Val layout: 8 vals per read
      ));

  // s2r tiled copy for A and B
  using SmemTiledCopyA =
      decltype(make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, DType>{},
                                 TiledMma{}));
  using SmemTiledCopyB =
      decltype(make_tiled_copy_B(Copy_Atom<SM75_U32x4_LDSM_N, DType>{},
                                 TiledMma{}));

  // ******* Epilogue *******

  using SmemLayoutAtomC = std::conditional_t<kBlockN % 64 == 0,
                                             SmemLayoutAtom_8x64,
                                             SmemLayoutAtom_8x32>;
  using SmemLayoutC =
      decltype(tile_to_shape(SmemLayoutAtomC{}, Shape<_BLK_M, _BLK_N>{}));

  // use 128-bit vectorizing copy
  using VectorizingCopy = AutoVectorizingCopyWithAssumedAlignment<128>;
  // r2s tiled copy for C
  using SmemTiledCopyC =
      decltype(make_tiled_copy_C(Copy_Atom<VectorizingCopy, DType>{},
                                 TiledMma{}));

  // s2g tiled copy for O
  using GmemTiledCopyC = decltype(make_tiled_copy(
      Copy_Atom<VectorizingCopy, DType>{},
      GmemCopyThrLayout{},     // Thr layout: (_16,_8)/(_32, _4)
      Layout<Shape<_1, _8>>{}  // Val layout: 8 vals per read
      ));

  // constexpr values for kernel launch
  static constexpr size_t kThreadNum = size(TiledMma{});
};

template <typename Traits>
struct GEMMSharedStorageSM80 {
  using DType = typename Traits::DType;
  using SmemLayoutA = typename Traits::SmemLayoutA;
  using SmemLayoutB = typename Traits::SmemLayoutB;
  using SmemLayoutC = typename Traits::SmemLayoutC;

  union {
    struct {
      // Shared memory for A: (BLK_M, BLK_K, PIPE)
      cute::array_aligned<DType, cute::cosize_v<SmemLayoutA>> a_smem;
      // Shared memory for B: (BLK_N, BLK_K, PIPE)
      cute::array_aligned<DType, cute::cosize_v<SmemLayoutB>> b_smem;
    };
    // Shared memory for C: (BLK_M, BLK_N)
    cute::array_aligned<DType, cute::cosize_v<SmemLayoutC>> c_smem;
  };
};

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

  // (m_blocks*BLK_M)
  const int* __restrict__ sorted_token_idxes_ptr = nullptr;
  // (m_blocks)
  const int* __restrict__ expert_ids_ptr = nullptr;

  const int* __restrict__ n_tokens_padded = nullptr;

  int m = 0;
  int n = 0;
  int k = 0;
  int topk = 0;
  int n_experts = 0;

  int m_blocks = 0;
  int n_blocks = 0;
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
  using TiledMma = typename Traits::TiledMma;

  using SmemLayoutA = typename Traits::SmemLayoutA;
  using SmemLayoutB = typename Traits::SmemLayoutB;
  using SmemLayoutC = typename Traits::SmemLayoutC;

  using GmemTiledCopyAB = typename Traits::GmemTiledCopyAB;
  using SmemTiledCopyA = typename Traits::SmemTiledCopyA;
  using SmemTiledCopyB = typename Traits::SmemTiledCopyB;
  using SmemTiledCopyC = typename Traits::SmemTiledCopyC;
  using GmemTiledCopyC = typename Traits::GmemTiledCopyC;

  using SharedStorage = GEMMSharedStorageSM80<Traits>;

  const auto M = kBlockM * gridDim.x;
  const auto N = params.n;
  const auto K = params.k;

  const auto topk = params.topk;
  const auto n_experts = params.n_experts;

  // each thread block takes care of one block: (BLK_M, BLK_N)
  const auto m_block_idx = blockIdx.x;
  const auto n_block_idx = blockIdx.y;
  const auto tidx = threadIdx.x;

  const int expert_id = params.expert_ids_ptr[m_block_idx];

  // ProblemShape
  const int* sorted_token_idxes = params.sorted_token_idxes_ptr;
  auto idx_to_t_idx = [sorted_token_idxes, topk](int idx) {
    return sorted_token_idxes[idx] / topk;
  };
  // A: (M, K), k-major
  auto A = make_gather_tensor(make_gmem_ptr((const DTYPE*)params.a_ptr),
                              make_shape(M, K),
                              make_stride(get<0>(params.a_stride), _1{}),
                              idx_to_t_idx);

  // B: (N, K), k-major
  const auto b_offset = expert_id * get<0>(params.b_stride);
  auto B = make_tensor(make_gmem_ptr((const DTYPE*)params.b_ptr + b_offset),
                       make_shape(N, K),
                       make_stride(get<1>(params.b_stride), _1{}));

  // C: (M, N), n-major
  auto idx_to_f_idx = [sorted_token_idxes](int idx) {
    return sorted_token_idxes[idx];
  };
  auto C = make_gather_tensor(make_gmem_ptr((DTYPE*)params.c_ptr),
                              make_shape(M, N),
                              make_stride(get<0>(params.c_stride), _1{}),
                              idx_to_f_idx);

  // (M, K) => (BLK_M, BLK_K, k)
  Tensor gA =
      local_tile(A, Shape<_BLK_M, _BLK_K>{}, make_coord(m_block_idx, _));
  // (N, K) => (BLK_N, BLK_K, k)
  Tensor gB =
      local_tile(B, Shape<_BLK_N, _BLK_K>{}, make_coord(n_block_idx, _));
  // (M, N) => (BLK_M, BLK_N)
  Tensor gC = local_tile(
      C, Shape<_BLK_M, _BLK_N>{}, make_coord(m_block_idx, n_block_idx));

  // Smem
  extern __shared__ char smem[];
  auto& ss = *reinterpret_cast<SharedStorage*>(smem);

  // (BLK_M, BLK_K, PIPE)
  Tensor sA = make_tensor(make_smem_ptr(ss.a_smem.data()), SmemLayoutA{});
  // (BLK_N, BLK_K, PIPE)
  Tensor sB = make_tensor(make_smem_ptr(ss.b_smem.data()), SmemLayoutB{});
  // (BLK_M, BLK_N)
  // Tensor sC = make_tensor(make_smem_ptr(ss.c_smem.data()), SmemLayoutC{});

  // Tiled Copy
  GmemTiledCopyAB gmem_tiled_copy_ab;
  auto gmem_thr_copy_ab = gmem_tiled_copy_ab.get_thread_slice(tidx);

  // (BLK_M, BLK_K, k) => (COPY, CP_M, CP_K, k)
  auto tAgA = gmem_thr_copy_ab.partition_S(gA);
  // (BLK_M, BLK_K, PIPE) => (COPY, CP_M, CP_K, PIPE)
  auto tAsA = gmem_thr_copy_ab.partition_D(sA);

  // (BLK_N, BLK_K, k) => (COPY, CP_N, CP_K, k)
  auto tBgB = gmem_thr_copy_ab.partition_S(gB);
  // (BLK_N, BLK_K, PIPE) => (COPY, CP_N, CP_K, PIPE)
  auto tBsB = gmem_thr_copy_ab.partition_D(sB);

  // GEMM: C = A@B.T
  TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(tidx);
  // rA: (BLK_M, BLK_K) => (MMA,MMA_M,MMA_K)
  auto tCrA = thr_mma.partition_fragment_A(sA(_, _, _0{}));
  // rB: (BLK_N, BLK_K) => (MMA,MMA_N,MMA_K)
  auto tCrB = thr_mma.partition_fragment_B(sB(_, _, _0{}));

  // s2r tiled copy for A and B
  auto smem_tiled_copy_a = SmemTiledCopyA{};
  auto smem_thr_copy_a = smem_tiled_copy_a.get_thread_slice(tidx);
  // (BLK_M, BLK_K, PIPE) => (COPY, COPY_M, COPY_K, PIPE)
  auto tCsA = smem_thr_copy_a.partition_S(sA);
  // (COPY, COPY_M, COPY_K)
  auto tCrA_cpv = smem_thr_copy_a.retile_D(tCrA);

  auto smem_tiled_copy_b = SmemTiledCopyB{};
  auto smem_thr_copy_b = smem_tiled_copy_b.get_thread_slice(tidx);
  // (BLK_N, BLK_K, PIPE) => (COPY, COPY_N, COPY_K, PIPE)
  auto tCsB = smem_thr_copy_b.partition_S(sB);
  // (COPY, COPY_N, COPY_K)
  auto tCrB_cpv = smem_thr_copy_b.retile_D(tCrB);

  // ###############  Prologue  ###############
  // remaining k-tile count
  int k_tile_remaining = size<3>(tAgA);
  // next tile index in gmem to read from
  int k_tile_next = 0;

  // async loads for all pipes except the last one
  auto kPipe = size<3>(tAsA);
  CUTE_UNROLL
  for (int k_pipe = 0; k_pipe < kPipe - 1; ++k_pipe) {
    copy(gmem_tiled_copy_ab, tAgA(_, _, _, k_tile_next), tAsA(_, _, _, k_pipe));
    copy(gmem_tiled_copy_ab, tBgB(_, _, _, k_tile_next), tBsB(_, _, _, k_pipe));
    cp_async_fence();

    // advance to next k-tile
    if (--k_tile_remaining > 0) {
      ++k_tile_next;
    }
  }

  // ###############  Mainloop  ###############
  // (BLK_M, BLK_N) => (MMA, MMA_M, MMA_N)
  auto tCrC = partition_fragment_C(tiled_mma, Shape<_BLK_M, _BLK_N>{});
  cute::clear(tCrC);  // Clear the accumulator

  // pipe index in smem to read from
  int pipe_read = 0;
  // pipe index in smem to write to
  int pipe_write = kPipe - 1;

  // pipe to read from: (COPY, COPY_N, COPY_K)
  Tensor tCsA_p = tCsA(_, _, _, pipe_read);
  Tensor tCsB_p = tCsB(_, _, _, pipe_read);

  // Size of the register pipeline
  auto kBlocks = size<2>(tCrA);

  // prefetch register pipeline
  if (kBlocks > 1) {
    // wait until our first prefetched tile is loaded in
    cp_async_wait<kPipe - 2>();
    __syncthreads();

    // prefetch the first rmem from the first k-tile
    cute::copy(smem_tiled_copy_a, tCsA_p(_, _, _0{}), tCrA_cpv(_, _, _0{}));
    cute::copy(smem_tiled_copy_b, tCsB_p(_, _, _0{}), tCrB_cpv(_, _, _0{}));
  }

  CUTE_NO_UNROLL
  while (k_tile_remaining > -(kPipe - 1)) {
    CUTE_UNROLL
    for (int ki = 0; ki < kBlocks; ++ki) {
      if (ki == kBlocks - 1) {
        // advance to next pipe to read from
        tCsA_p = tCsA(_, _, _, pipe_read);
        tCsB_p = tCsB(_, _, _, pipe_read);

        // wait until our next prefetched tile is loaded in
        cp_async_wait<kPipe - 2>();
        __syncthreads();
      }

      // load A, B from smem to registers for next ki
      auto ki_next = (ki + _1{}) % kBlocks;
      copy(smem_tiled_copy_a, tCsA_p(_, _, ki_next), tCrA_cpv(_, _, ki_next));
      copy(smem_tiled_copy_b, tCsB_p(_, _, ki_next), tCrB_cpv(_, _, ki_next));

      if (ki == 0) {
        // copy gmem to smeme for next pipe
        copy(gmem_tiled_copy_ab,
             tAgA(_, _, _, k_tile_next),
             tAsA(_, _, _, pipe_write));
        copy(gmem_tiled_copy_ab,
             tBgB(_, _, _, k_tile_next),
             tBsB(_, _, _, pipe_write));
        cp_async_fence();

        // advance to next k-tile
        if (--k_tile_remaining > 0) {
          ++k_tile_next;
        }

        // advance to next pipe
        pipe_write = pipe_read;
        pipe_read = (pipe_read == kPipe - 1) ? 0 : pipe_read + 1;
      }

      // thread-level gemm for ki
      gemm(tiled_mma, tCrA(_, _, ki), tCrB(_, _, ki), tCrC);
    }
  }

  // ###############  Epilogue  ###############
  // (BLK_M, BLK_N)
  Tensor sC = make_tensor(make_smem_ptr(ss.c_smem.data()), SmemLayoutC{});

  // TODO: fastcast tCrC to DTYPE

  // copy tCrC from registers to smem
  SmemTiledCopyC smem_tiled_copy_c;
  auto smem_thr_copy_c = smem_tiled_copy_c.get_thread_slice(tidx);
  auto tSrC = smem_thr_copy_c.retile_S(tCrC);
  auto tSsC = smem_thr_copy_c.partition_D(sC);
  cute::copy(smem_tiled_copy_c, tSrC, tSsC);

  // wait for smem copy done before gmem copy
  __syncthreads();

  // copy sC from smem to gmem
  GmemTiledCopyC gmem_tiled_copy_c;
  auto gmem_thr_copy_c = gmem_tiled_copy_c.get_thread_slice(tidx);
  auto tGsC = gmem_thr_copy_c.partition_S(sC);
  auto tGgC = gmem_thr_copy_c.partition_D(gC);
  cute::copy(gmem_tiled_copy_c, tGsC, tGgC);
}

template <typename Traits, typename Params>
void launch_grouped_gemm_kernel_sm80(const Params& params,
                                     cudaStream_t stream) {
  const auto smem_size = sizeof(GEMMSharedStorageSM80<Traits>);
  // std::cout << "SMEM size: " << smem_size << " bytes\n";

  auto gemm_kernel = grouped_gemm_kernel_sm80<Traits, Params>;
  cudaFuncSetAttribute(
      gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  // TODO: support persistent kernels
  dim3 grid(params.m_blocks, params.n_blocks);
  dim3 block = Traits::kThreadNum;
  gemm_kernel<<<grid, block, smem_size, stream>>>(params);
}

}  // namespace llm
