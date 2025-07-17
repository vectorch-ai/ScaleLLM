#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cute/config.hpp>
#include <cute/container/array_aligned.hpp>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>

#include "fast_cast.cuh"
#include "layout_convertor.h"
#include "safe_copy.h"

namespace llm {

using namespace cute;

template <class TileShape_,
          class Element_,
          int HeadDim_,
          bool EVEN_K,
          bool ALIBI,
          bool SOFT_CAP,
          bool LOCAL>
struct Sm120CollectiveMhaWs {
  // TODO: multiple stages
  using TileShape = TileShape_;
  using Element = Element_;
  using ElementAccum = float;

  static constexpr int kHeadDim = HeadDim_;
  static constexpr int kBlockM = get<0>(TileShape{});
  static constexpr int kBlockN = get<1>(TileShape{});
  static constexpr int kBlockK = get<2>(TileShape{});

  static constexpr bool kAlibi = ALIBI;
  static constexpr bool kLocal = LOCAL;

  static_assert(kBlockK == 32 || kBlockK == 64);
  static_assert(kHeadDim % kBlockK == 0);

  using BLK_M = Int<kBlockM>;
  using BLK_N = Int<kBlockN>;
  using BLK_K = Int<kBlockK>;
  using HEAD_DIM = Int<kHeadDim>;

  // TiledMMA (64x16x16) for gemm-I and gemm-II
  using MMA_Atom_ =
      std::conditional_t<std::is_same_v<Element, cute::half_t>,
                         MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
                         MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>>;
  using TiledMma = TiledMMA<MMA_Atom_,
                            Layout<Shape<_4, _1, _1>>,  // warp layout 4x1x1
                            Tile<_64, _16, _16>>;       // Tile Shape 64x16x16

  static constexpr int kRowsPerMMA = 2;
  static constexpr int kMmaThreads = size(TiledMma{});

  // Atom layout: (8, BLK_K):(BLK_K, 1) k-major
  using SmemLayoutAtom_ =
      decltype(composition(Swizzle<3, 3, 3>{},
                           Layout<Shape<_8, BLK_K>, Stride<BLK_K, _1>>{}));

  // Q smem: (BLK_M, HEAD_DIM)
  using SmemLayoutQ =
      decltype(tile_to_shape(SmemLayoutAtom_{}, Shape<BLK_M, HEAD_DIM>{}));

  // KV smem: (BLK_N, HEAD_DIM)
  using SmemLayoutK =
      decltype(tile_to_shape(SmemLayoutAtom_{}, Shape<BLK_N, HEAD_DIM>{}));
  using SmemLayoutV =
      decltype(tile_to_shape(SmemLayoutAtom_{}, Shape<BLK_N, HEAD_DIM>{}));

  // V^T smem: (HEAD_DIM, BLK_N)
  using SmemLayoutVt = decltype(select<1, 0>(SmemLayoutV{}));

  // s2r tiled copy for gemm-I
  using SmemTiledCopyQ =
      decltype(make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, Element>{},
                                 TiledMma{}));
  using SmemTiledCopyK =
      decltype(make_tiled_copy_B(Copy_Atom<SM75_U32x4_LDSM_N, Element>{},
                                 TiledMma{}));

  // s2r tiled copy for gemm-II
  using SmemTiledCopyVt =
      decltype(make_tiled_copy_B(Copy_Atom<SM75_U16x8_LDSM_T, Element>{},
                                 TiledMma{}));

  struct SharedStorage : cute::aligned_struct<128> {
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> smem_q;
    struct {
      cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>> smem_k;
      union {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>> smem_v;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutVt>> smem_vt;
      };
    };
  };

  // Host side arguments
  struct Arguments {
    // softcap
    float logits_soft_cap = 0.0;
  };

  // Device side params
  using Params = Arguments;

  // Convert host side arguments to device side params
  static Params to_underlying_arguments(Arguments const& args) {
    // no convertion needed.
    return args;
  }

  // returns false if the block has been skipped
  template <class TensorCMN,
            class FrgTensor,
            class Softmax,
            class Mask,
            class PipelineQ,
            class PipelineKV>
  CUTE_DEVICE void operator()(
      const Params& params,
      const TensorCMN& tScMN_mn,  // ((2, MMA_M), (2, MMA_N), n) => (M, N)
      FrgTensor& tOrO,            // (MMA, MMA_M, MMA_N)
      Softmax& softmax,
      Mask& mask,
      int tidx,
      PipelineQ& q_pipeline,
      typename PipelineQ::PipelineState& q_state,
      PipelineKV& kv_pipeline,
      typename PipelineKV::PipelineState& kv_state,
      int n_block_min,
      int n_block_max,
      char* smem) {
    static_assert(is_rmem<FrgTensor>::value,
                  "Accum tensor must be rmem resident.");

    static constexpr int kBlockM = get<0>(TileShape{});
    static constexpr int kBlockN = get<1>(TileShape{});

    // assert(n_block_min < n_block_max);

    // Construct shared memory tiles
    auto& ss = *reinterpret_cast<SharedStorage*>(smem);

    // (BLK_M, HEAD_DIM), k-major
    Tensor sQ = make_tensor(make_smem_ptr(ss.smem_q.data()), SmemLayoutQ{});
    // (BLK_N, HEAD_DIM), k-major
    Tensor sK = make_tensor(make_smem_ptr(ss.smem_k.data()), SmemLayoutK{});
    // Tensor sV = make_tensor(make_smem_ptr(ss.smem_v.data()), SmemLayoutV{});

    // Tensor for V^t; used in GEMM-II.
    // (HEAD_DIM, BLK_N), k-major
    Tensor sVt = make_tensor(make_smem_ptr(ss.smem_vt.data()), SmemLayoutVt{});

    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_slice(tidx);
    // GEMM-I: S = Q@K.T
    auto tSrQ = thr_mma.partition_fragment_A(sQ);  // (MMA,MMA_M,MMA_K)
    auto tSrK = thr_mma.partition_fragment_B(sK);  // (MMA,MMA_N,MMA_K)

    // s2r tiled copy for qkv
    // copy query to rmem
    SmemTiledCopyQ smem_tiled_copy_Q;
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    auto tSsQ = smem_thr_copy_Q.partition_S(sQ);
    auto tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);

    SmemTiledCopyK smem_tiled_copy_K;
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    auto tSsK = smem_thr_copy_K.partition_S(sK);
    auto tSrK_copy_view = smem_thr_copy_K.retile_D(tSrK);

    // S = Q@K.T
    // tSrAccS: (MMA,MMA_M,MMA_N)
    auto compute_qk = [&](auto& tSrAccS) {
      // prefetch key
      cute::copy(
          smem_tiled_copy_K, tSsK(_, _, _0{}), tSrK_copy_view(_, _, _0{}));

      CUTE_UNROLL
      for (int ki = 0; ki < size<2>(tSrQ); ++ki) {
        // prefetch next key
        if (ki != size<2>(tSrQ) - 1) {
          const auto next_ki = ki + 1;
          cute::copy(smem_tiled_copy_K,
                     tSsK(_, _, next_ki),
                     tSrK_copy_view(_, _, next_ki));
        }
        cute::gemm(tiled_mma, tSrQ(_, _, ki), tSrK(_, _, ki), tSrAccS);
      }
    };

    // GEMM-II: O = softmax(S)@V
    auto tOrVt = thr_mma.partition_fragment_B(sVt);  // (MMA,MMA_K,MMA_N)

    SmemTiledCopyVt smem_tiled_copy_Vt;
    auto smem_thr_copy_Vt = smem_tiled_copy_Vt.get_thread_slice(tidx);
    auto tOsVt = smem_thr_copy_Vt.partition_S(sVt);
    auto tOrVt_copy_view = smem_thr_copy_Vt.retile_D(tOrVt);

    // O = softmax(S)*V
    // tSrAccS: (MMA,MMA_M,MMA_N)
    // tOrAccO: (MMA,MMA_M,MMA_K)
    auto compute_sv = [&](const auto& tSrAccS, auto& tOrAccO) {
      // cast scores from Accumulator to Element
      auto tSrS = make_tensor_like<Element>(tSrAccS);
      fast_cast(tSrAccS, tSrS);

      // convert layout from gemm-I C to gemm-II A
      auto tOrS =
          make_tensor(tSrS.data(), LayoutConvertor::to_mma_a(tSrS.layout()));

      // prefetch V^t
      cute::copy(
          smem_tiled_copy_Vt, tOsVt(_, _, _0{}), tOrVt_copy_view(_, _, _0{}));
      CUTE_UNROLL
      for (int ki = 0; ki < size<2>(tOrS); ++ki) {
        // prefetch next V^t
        if (ki != size<2>(tOrS) - 1) {
          const auto next_ki = ki + 1;
          cute::copy(smem_tiled_copy_Vt,
                     tOsVt(_, _, next_ki),
                     tOrVt_copy_view(_, _, next_ki));
        }
        cute::gemm(tiled_mma, tOrS(_, _, ki), tOrVt(_, _, ki), tOrAccO);
      }
    };

    auto tOrO_mn =
        make_tensor(tOrO.data(), LayoutConvertor::to_mn(tOrO.layout()));

    auto apply_logits_soft_cap = [&](auto& tSrAccS) {
      if constexpr (SOFT_CAP) {
        CUTE_UNROLL
        for (int i = 0; i < size(tSrAccS); ++i) {
          tSrAccS(i) = tanh(tSrAccS(i) * params.logits_soft_cap);
        }
      }
    };

    // ###############  Prologue  ###############
    // wait for query g2s copy done
    q_pipeline.consume_acquire(q_state);
    // copy query from smem to rmem
    cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);

    // release query smem
    q_pipeline.consumer_release(q_state);
    ++q_state;

    // ###############  Mainloop  ###############
    constexpr int n_oob_mask = cute::ceil_div(kBlockM, kBlockN) + 1;
    const int n_blocks = n_block_max - n_block_min;

    // attention score accumulator, (MMA, MMA_M, MMA_N)
    auto tSrS = partition_fragment_C(tiled_mma, Shape<BLK_M, BLK_N>{});
    // ((2, MMA_M), (2, MMA_N))
    auto tSrS_mn =
        make_tensor(tSrS.data(), LayoutConvertor::to_mn(tSrS.layout()));

    CUTE_NO_UNROLL
    for (int i = 0; i < n_blocks; ++i) {
      const int ni = n_block_max - 1 - i;
      clear(tSrS);

      // wait key g2s copy done
      kv_pipeline.consume_acquire(kv_state);

      // 1> S = Q@K.T
      compute_qk(tSrS);

      // release key smem
      kv_pipeline.consumer_release(kv_state);
      ++kv_state;

      if constexpr (SOFT_CAP) {
        apply_logits_soft_cap(tSrS);
      }

      // apply mask
      // ((2, MMA_M), (2, MMA_N)) => (M, N)
      const auto tScS_mn = tScMN_mn(_, _, ni);
      if (i < n_oob_mask) {
        mask.template apply</*OOB_MASK=*/true>(tSrS_mn, tScS_mn);
      } else {
        mask.template apply</*OOB_MASK=*/false>(tSrS_mn, tScS_mn);
      }
      softmax.rescale(tSrS_mn, tOrO_mn);

      // wait value g2s copy done
      kv_pipeline.consume_acquire(kv_state);

      // 2> O = softmax(S)*V
      compute_sv(tSrS, tOrO);

      // release value smem
      kv_pipeline.consumer_release(kv_state);
      ++kv_state;
    }

    // normalize output: o /= rowsum
    softmax.finalize(tOrO_mn);
  }
};

}  // namespace llm
