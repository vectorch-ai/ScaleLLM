#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cute/config.hpp>
#include <cute/container/array_aligned.hpp>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cutlass/pipeline/pipeline.hpp>

#include "common/fast_cast.cuh"
#include "common/layout_convertor.h"
#include "common/mask.h"
#include "common/online_softmax.cuh"
#include "common/safe_copy.h"
#include "sm120_collective_load_cpasync_ws.cuh"
#include "sm120_collective_load_tma_ws.cuh"

namespace llm {

using namespace cute;

template <class TileShape_,
          class Element_,
          int HeadDim_,
          bool EVEN_K,
          bool ALIBI,
          bool SOFT_CAP,
          bool LOCAL,
          bool KV_USE_TMA = false  // whether to use TMA for K/V loading
          >
struct Sm120CollectiveFMhaWs {
  using TileShape = TileShape_;
  using Element = Element_;
  using ElementAccum = float;

  using ClusterShape = Shape<_1, _1, _1>;

  static constexpr int kHeadDim = HeadDim_;
  static constexpr int kBlockM = get<0>(TileShape{});
  static constexpr int kBlockN = get<1>(TileShape{});
  static constexpr int kBlockK = get<2>(TileShape{});

  static constexpr bool kAlibi = ALIBI;
  static constexpr bool kLocal = LOCAL;
  static constexpr bool kKVUseTma = KV_USE_TMA;

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

  // TODO: tune the number of stages based on smem size
  static constexpr int StageCountQ = 1;
  static constexpr int StageCountKV = 3;

  // Atom layout: (8, BLK_K):(BLK_K, 1) k-major
  using SmemLayoutAtom_ =
      decltype(composition(Swizzle<3, 3, 3>{},
                           Layout<Shape<_8, BLK_K>, Stride<BLK_K, _1>>{}));

  // Q smem: (BLK_M, HEAD_DIM)
  using SmemLayoutQ =
      decltype(tile_to_shape(SmemLayoutAtom_{}, Shape<BLK_M, HEAD_DIM>{}));

  // KV smem: (BLK_N, HEAD_DIM, KVStages)
  using SmemLayoutK =
      decltype(tile_to_shape(SmemLayoutAtom_{},
                             Shape<BLK_N, HEAD_DIM, Int<StageCountKV>>{}));
  using SmemLayoutV = SmemLayoutK;

  // V^T smem: (HEAD_DIM, BLK_N, KVStages)
  using SmemLayoutVt = decltype(select<1, 0, 2>(SmemLayoutV{}));

  // tma transaction bytes for (BLK_N, HEAD_DIM)
  static constexpr uint32_t kTmaTransactionBytes =
      size(take<0, 2>(SmemLayoutK{})) * cutlass::sizeof_bits_v<Element> / 8;

  struct TensorStorage {
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> smem_q;
    union {
      cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>> smem_k;
      cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>> smem_v;
      cute::array_aligned<Element, cute::cosize_v<SmemLayoutVt>> smem_vt;
    };
  };

  using PipelineQ = cutlass::PipelineAsync<StageCountQ>;
  using PipelineKV = std::conditional_t<KV_USE_TMA,
                                        cutlass::PipelineTmaAsync<StageCountKV>,
                                        cutlass::PipelineAsync<StageCountKV>>;

  using CpAsyncLoad_ = Sm120CollectiveLoadCpAsyncWs<TileShape,
                                                    Element,
                                                    TensorStorage,
                                                    SmemLayoutQ,
                                                    SmemLayoutK,
                                                    SmemLayoutV,
                                                    PipelineQ,
                                                    PipelineKV,
                                                    EVEN_K>;

  using TmaLoad_ = Sm120CollectiveLoadTmaWs<TileShape,
                                            Element,
                                            TensorStorage,
                                            SmemLayoutQ,
                                            SmemLayoutK,
                                            SmemLayoutV,
                                            PipelineQ,
                                            PipelineKV,
                                            EVEN_K>;

  using Load = std::conditional_t<KV_USE_TMA, TmaLoad_, CpAsyncLoad_>;

  // Host side arguments
  struct Arguments {
    // sliding window attention
    int sliding_window;
    // softcap
    float logits_soft_cap;
    // softmax scale
    float sm_scale;
    // softmax scale in log2
    float sm_scale_log2;
    // group size
    const FastDivmod& group_size;

    // alibi slopes pointer
    const float* alibi_slopes_ptr;
  };

  // Device side params
  using Params = Arguments;

  // Convert host side arguments to device side params
  static Params to_underlying_arguments(Arguments const& args) {
    // no convertion needed.
    return args;
  }

  // load Q/K/V from gmem to smem
  template <class Block>
  CUTE_DEVICE void load(const Block& block,
                        int tidx,
                        PipelineQ& q_pipeline,
                        typename PipelineQ::PipelineState& q_state,
                        PipelineKV& kv_pipeline,
                        typename PipelineKV::PipelineState& kv_state,
                        TensorStorage& ss) {
    // forward to the load implementation
    Load load;
    load(block, tidx, q_pipeline, q_state, kv_pipeline, kv_state, ss);
  }

  template <class Block, class FrgTensor, class PipelineQ, class PipelineKV>
  CUTE_DEVICE void fmha(const Params& params,
                        const Block& block,
                        FrgTensor& tOrO,  // (MMA, MMA_M, MMA_N)
                        int tidx,
                        PipelineQ& q_pipeline,
                        typename PipelineQ::PipelineState& q_state,
                        PipelineKV& kv_pipeline,
                        typename PipelineKV::PipelineState& kv_state,
                        TensorStorage& ss) {
    static_assert(is_rmem<FrgTensor>::value,
                  "Accum tensor must be rmem resident.");

    static constexpr int kBlockM = get<0>(TileShape{});
    static constexpr int kBlockN = get<1>(TileShape{});

    if (!block.is_valid()) {
      // skip invalid block
      return;
    }

    const auto [n_block_min, n_block_max] = block.get_kv_blocks();
    if (n_block_min >= n_block_max) {
      return;  // no kv blocks to process
    }

    const auto [batch_idx, m_block_idx, kv_head_idx] = block.get_block_coord();

    const auto q_packed_len = block.get_packed_len();
    const auto q_len = block.get_q_len();
    const auto kv_len = block.get_kv_len();

    // Construct smem tensors
    // (BLK_M, HEAD_DIM), k-major
    Tensor sQ = make_tensor(make_smem_ptr(ss.smem_q.data()), SmemLayoutQ{});
    // (BLK_N, HEAD_DIM, KVStages), k-major
    Tensor sK = make_tensor(make_smem_ptr(ss.smem_k.data()), SmemLayoutK{});
    // Tensor for V^t; used in GEMM-II.
    // (HEAD_DIM, BLK_N, KVStages), k-major
    Tensor sVt = make_tensor(make_smem_ptr(ss.smem_vt.data()), SmemLayoutVt{});

    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_slice(tidx);
    // GEMM-I: S = Q@K.T
    // (MMA,MMA_M,MMA_K)
    auto tSrQ = thr_mma.partition_fragment_A(sQ);
    // (MMA,MMA_N,MMA_K)
    auto tSrK = thr_mma.partition_fragment_B(sK(_, _, _0{}));

    // s2r tiled copy for qkv
    // copy query to rmem
    auto smem_tiled_copy_Q =
        make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, Element>{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_slice(tidx);
    // (CPY,CPY_M,CPY_K)
    auto tSsQ = smem_thr_copy_Q.partition_S(sQ);
    // (CPY,CPY_M,CPY_K)
    auto tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);

    auto smem_tiled_copy_K =
        make_tiled_copy_B(Copy_Atom<SM75_U32x4_LDSM_N, Element>{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_slice(tidx);
    // (CPY,CPY_N,CPY_K, KVStages)
    auto tSsK = smem_thr_copy_K.partition_S(sK);
    // (CPY,CPY_N,CPY_K)
    auto tSrK_copy_view = smem_thr_copy_K.retile_D(tSrK);

    // S = Q@K.T
    // tSrAccS: (MMA,MMA_M,MMA_N)
    auto compute_qk = [&](auto& tSrAccS, int stage) {
      auto tSsK_s = tSsK(_, _, _, stage);
      // prefetch key
      cute::copy(
          smem_tiled_copy_K, tSsK_s(_, _, _0{}), tSrK_copy_view(_, _, _0{}));

      CUTE_UNROLL
      for (int ki = 0; ki < size<2>(tSrQ); ++ki) {
        // prefetch next key
        if (ki != size<2>(tSrQ) - 1) {
          const auto next_ki = ki + 1;
          cute::copy(smem_tiled_copy_K,
                     tSsK_s(_, _, next_ki),
                     tSrK_copy_view(_, _, next_ki));
        }
        cute::gemm(tiled_mma, tSrQ(_, _, ki), tSrK(_, _, ki), tSrAccS);
      }
    };

    // GEMM-II: O = softmax(S)@V
    // (MMA,MMA_K,MMA_N)
    auto tOrVt = thr_mma.partition_fragment_B(sVt(_, _, _0{}));

    auto smem_tiled_copy_Vt =
        make_tiled_copy_B(Copy_Atom<SM75_U16x8_LDSM_T, Element>{}, tiled_mma);
    auto smem_thr_copy_Vt = smem_tiled_copy_Vt.get_slice(tidx);
    // (CPY,CPY_K,CPY_N, KVStages)
    auto tOsVt = smem_thr_copy_Vt.partition_S(sVt);
    // (CPY,CPY_K,CPY_N)
    auto tOrVt_copy_view = smem_thr_copy_Vt.retile_D(tOrVt);

    // O = softmax(S)*V
    // tSrAccS: (MMA,MMA_M,MMA_N)
    // tOrAccO: (MMA,MMA_M,MMA_K)
    auto compute_sv = [&](const auto& tSrAccS, auto& tOrAccO, int stage) {
      // cast scores from Accumulator to Element
      auto tSrS = make_tensor_like<Element>(tSrAccS);
      fast_cast(tSrAccS, tSrS);

      // convert layout from gemm-I C to gemm-II A
      auto tOrS =
          make_tensor(tSrS.data(), LayoutConvertor::to_mma_a(tSrS.layout()));
      // (CPY,CPY_M,CPY_K)
      auto tOsVt_s = tOsVt(_, _, _, stage);
      // prefetch V^t
      cute::copy(
          smem_tiled_copy_Vt, tOsVt_s(_, _, _0{}), tOrVt_copy_view(_, _, _0{}));
      CUTE_UNROLL
      for (int ki = 0; ki < size<2>(tOrS); ++ki) {
        // prefetch next V^t
        if (ki != size<2>(tOrS) - 1) {
          const auto next_ki = ki + 1;
          cute::copy(smem_tiled_copy_Vt,
                     tOsVt_s(_, _, next_ki),
                     tOrVt_copy_view(_, _, next_ki));
        }
        cute::gemm(tiled_mma, tOrS(_, _, ki), tOrVt(_, _, ki), tOrAccO);
      }
    };

    auto apply_logits_soft_cap = [&](auto& tSrAccS) {
      if constexpr (SOFT_CAP) {
        CUTE_UNROLL
        for (int i = 0; i < size(tSrAccS); ++i) {
          tSrAccS(i) = tanh(tSrAccS(i) * params.logits_soft_cap);
        }
      }
    };

    // wait for query g2s copy done
    q_pipeline.consumer_wait(q_state);
    // copy query from smem to rmem
    cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);

    // release query smem
    q_pipeline.consumer_release(q_state);
    ++q_state;

    // ###############  Mainloop  ###############
    // attention score accumulator, (MMA, MMA_M, MMA_N)
    auto tSrS = partition_fragment_C(tiled_mma, Shape<BLK_M, BLK_N>{});
    // ((2, MMA_M), (2, MMA_N))
    auto tSrS_mn =
        make_tensor(tSrS.data(), LayoutConvertor::to_mn(tSrS.layout()));

    auto tOrO_mn =
        make_tensor(tOrO.data(), LayoutConvertor::to_mn(tOrO.layout()));

    // (BLK_M, BLK_N, n) => (M, N)
    auto cMN =
        local_tile(make_identity_tensor(make_shape(q_packed_len, kv_len)),
                   Shape<BLK_M, BLK_N>{},
                   make_coord(m_block_idx, _));

    // (MMA, MMA_M, MMA_N, n) => (M, N)
    auto tScMN = thr_mma.partition_C(cMN);
    // ((2, MMA_M), (2, MMA_N), n) => (M, N)
    auto tScMN_mn =
        make_tensor(tScMN.data(), LayoutConvertor::to_mn(tScMN.layout()));

    constexpr int kRowsPerThr = kRowsPerMMA * size<1>(tSrS);
    // Create softmax and mask
    OnlineSoftmax<kRowsPerThr> softmax(params.sm_scale_log2);
    Mask<kRowsPerThr, kAlibi, kLocal> mask(
        q_len, kv_len, params.group_size, params.sliding_window);
    if constexpr (kAlibi) {
      const auto tScS_mn = tScMN_mn(_, _, _0{});
      mask.init_alibi(
          tScS_mn, kv_head_idx, params.sm_scale, params.alibi_slopes_ptr);
    }

    const int n_oob_mask_min =
        n_block_max - (cute::ceil_div(kBlockM, kBlockN) + 1);
    CUTE_NO_UNROLL
    for (int ni = n_block_max - 1; ni >= n_block_min; --ni) {
      clear(tSrS);

      // wait key g2s copy done
      kv_pipeline.consumer_wait(kv_state);

      // 1> S = Q@K.T
      compute_qk(tSrS, kv_state.index());

      // release key smem
      kv_pipeline.consumer_release(kv_state);
      ++kv_state;

      if constexpr (SOFT_CAP) {
        apply_logits_soft_cap(tSrS);
      }

      // apply mask
      // ((2, MMA_M), (2, MMA_N)) => (M, N)
      const auto tScS_mn = tScMN_mn(_, _, ni);
      if (ni >= n_oob_mask_min) {
        mask.template apply</*OOB_MASK=*/true>(tSrS_mn, tScS_mn);
      } else {
        mask.template apply</*OOB_MASK=*/false>(tSrS_mn, tScS_mn);
      }
      softmax.rescale(tSrS_mn, tOrO_mn);

      // wait value g2s copy done
      kv_pipeline.consumer_wait(kv_state);

      // 2> O = softmax(S)*V
      compute_sv(tSrS, tOrO, kv_state.index());

      // release value smem
      kv_pipeline.consumer_release(kv_state);
      ++kv_state;
    }

    // normalize output: o /= rowsum
    softmax.finalize(tOrO_mn);
  }
};

}  // namespace llm
