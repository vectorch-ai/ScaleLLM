#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cute/layout.hpp>
#include <cute/tensor.hpp>

#include "gather_tensor.hpp"
#include "mla_params.h"
#include "online_softmax.cuh"

namespace llm {

using namespace cute;

namespace detail {

template <typename Params>
struct MLATile {
  static_assert(cute::dependent_false<Params>, "not implemented");
};

// AttentionTile specialization for AttentionParams
template <>
struct MLATile<MLAParams> {
  const MLAParams& params_;
  const int batch_idx_;

  CUTE_HOST_DEVICE MLATile(const MLAParams& params, int batch_idx)
      : params_(params), batch_idx_(batch_idx) {}

  // return the query/output tile: (q_packed_len, head_dim)
  // return q_rope tile: (q_packed_len, rope_head_dim)
  template <typename Element>
  CUTE_HOST_DEVICE auto get_qo_tile() const {
    // (batch, seq, head, dim)
    const auto q_packed_len = params_.q_len * params_.group_size;
    const auto q_offset = batch_idx_ * get<0>(params_.q_stride);
    auto q =
        make_tensor(make_gmem_ptr((const Element*)params_.q_ptr + q_offset),
                    make_shape(q_packed_len, params_.head_dim),
                    select<2, 3>(params_.q_stride));

    // (batch, seq, head, rope_head_dim)
    const auto q_rope_offset = batch_idx_ * get<0>(params_.q_rope_stride);
    auto q_rope = make_tensor(
        make_gmem_ptr((const Element*)params_.q_rope_ptr + q_rope_offset),
        make_shape(q_packed_len, params_.rope_head_dim),
        select<2, 3>(params_.q_rope_stride));

    // (batch, seq, head, dim)
    const auto o_offset = batch_idx_ * get<0>(params_.o_stride);
    auto o = make_tensor(make_gmem_ptr((Element*)params_.o_ptr + o_offset),
                         make_shape(q_packed_len, params_.head_dim),
                         select<2, 3>(params_.o_stride));
    return make_tuple(q, q_rope, o);
  }

  // return the kv: (kv_len, head_dim)
  // return k_rope: (kv_len, rope_head_dim)
  template <typename Element>
  CUTE_HOST_DEVICE auto get_kv_tile() const {
    // (batch, seq, dim)
    const auto kv_offset = batch_idx_ * get<0>(params_.kv_stride);
    // k[batch_idx, :, :]
    auto kv =
        make_tensor(make_gmem_ptr((const Element*)params_.kv_ptr + kv_offset),
                    make_shape(params_.kv_len, params_.head_dim),
                    select<1, 2>(params_.kv_stride));

    // (batch, seq, rope_head_dim)
    const auto k_rope_offset = batch_idx_ * get<0>(params_.k_rope_stride);
    auto k_rope = make_tensor(
        make_gmem_ptr((const Element*)params_.k_rope_ptr + k_rope_offset),
        make_shape(params_.kv_len, params_.rope_head_dim),
        select<1, 2>(params_.k_rope_stride));
    return make_tuple(kv, k_rope);
  }
};

// paged KV cache + variable length sequence
template <>
struct MLATile<MLAPagedKVParams> {
  // NOLINTNEXTLINE
  const MLAPagedKVParams& params_;
  const int batch_idx_;

  CUTE_HOST_DEVICE MLATile(const MLAPagedKVParams& params, int batch_idx)
      : params_(params), batch_idx_(batch_idx) {}

  // return the query/output tile: (q_packed_len, head_dim)
  // return q_rope tile: (q_packed_len, rope_head_dim)
  template <typename Element>
  CUTE_HOST_DEVICE auto get_qo_tile() const {
    const auto begin = params_.q_cu_lens[batch_idx_];
    const auto qo_len = params_.q_cu_lens[batch_idx_ + 1] - begin;

    // (seq, head, dim)
    const auto q_packed_len = qo_len * params_.group_size;
    const auto q_offset = begin * get<0>(params_.q_stride);

    auto q =
        make_tensor(make_gmem_ptr((const Element*)params_.q_ptr + q_offset),
                    make_shape(q_packed_len, params_.head_dim),
                    select<1, 2>(params_.q_stride));

    // (seq, head, rope_head_dim)
    const auto q_rope_offset = begin * get<0>(params_.q_rope_stride);
    auto q_rope = make_tensor(
        make_gmem_ptr((const Element*)params_.q_rope_ptr + q_rope_offset),
        make_shape(q_packed_len, params_.rope_head_dim),
        select<1, 2>(params_.q_rope_stride));

    // (seq, head, dim)
    const auto o_offset = begin * get<0>(params_.o_stride);
    auto o = make_tensor(make_gmem_ptr((Element*)params_.o_ptr + o_offset),
                         make_shape(q_packed_len, params_.head_dim),
                         select<1, 2>(params_.o_stride));
    return make_tuple(q, q_rope, o);
  }

  // return the kv: (kv_len, head_dim)
  // return k_rope: (kv_len, rope_head_dim)
  template <typename Element>
  CUTE_HOST_DEVICE auto get_kv_tile() const {
    const auto kv_len =
        params_.kv_cu_lens[batch_idx_ + 1] - params_.kv_cu_lens[batch_idx_];

    // map seq_idx to slot_idx
    const int* block_table =
        params_.block_table + params_.block_cu_lens[batch_idx_];
    auto idx_to_slot = [block_table,
                        shr = params_.block_shift_right,
                        mask = params_.block_mask](int idx) {
      return block_table[idx >> shr] + (idx & mask);
    };

    // kv: (seq, dim)
    auto kv = make_gather_tensor(make_gmem_ptr((const Element*)params_.kv_ptr),
                                 make_shape(kv_len, params_.head_dim),
                                 params_.kv_stride,
                                 idx_to_slot);

    // k_rope: (seq, rope_head_dim)
    auto k_rope =
        make_gather_tensor(make_gmem_ptr((const Element*)params_.k_rope_ptr),
                           make_shape(kv_len, params_.rope_head_dim),
                           params_.k_rope_stride,
                           idx_to_slot);
    return make_tuple(kv, k_rope);
  }
};
}  // namespace detail

template <class CollectiveMainloop_,
          class CollectiveEpilogue_,
          class TileScheduler_>
class Sm80KernelMla {
 public:
  using CollectiveMainloop = CollectiveMainloop_;
  using CollectiveEpilogue = CollectiveEpilogue_;
  using TileScheduler = TileScheduler_;

  using TiledMma_PV = typename CollectiveMainloop::TiledMma_PV;

  using Element = typename CollectiveMainloop::Element;
  using BLK_M = typename CollectiveMainloop::BLK_M;
  using BLK_N = typename CollectiveMainloop::BLK_N;
  using BLK_K = typename CollectiveMainloop::BLK_K;
  using HEAD_DIM = typename CollectiveMainloop::HEAD_DIM;
  using ROPE_HEAD_DIM = typename CollectiveMainloop::ROPE_HEAD_DIM;
  using STEPS = typename CollectiveMainloop::STEPS;

  static constexpr int kBlockM = CollectiveMainloop::kBlockM;

  static constexpr int kRowsPerMMA = CollectiveMainloop::kRowsPerMMA;

  static constexpr int kSharedStorageSize =
      cute::max(sizeof(typename CollectiveMainloop::SharedStorage),
                sizeof(typename CollectiveEpilogue::SharedStorage));

  static constexpr int kMmaThreads = CollectiveMainloop::kMmaThreads;

  // Kernel params
  using MainloopParams = typename CollectiveMainloop::Params;
  using EpilogueParams = typename CollectiveEpilogue::Params;
  using TileSchedulerParams = typename TileScheduler::Params;

  // returns grid and block shape for kernel launch
  using TileSchedulerArgs = typename TileScheduler::Arguments;
  static dim3 get_grid_shape(TileSchedulerArgs const& args) {
    return TileScheduler::get_grid_shape(args);
  }
  static dim3 get_block_shape() { return kMmaThreads; }

  template <class Params>
  CUTE_DEVICE void operator()(const Params& params,
                              const TileSchedulerParams& scheduler_params,
                              char* smem) {
    CollectiveMainloop mha;
    CollectiveEpilogue epilogue;
    TileScheduler scheduler(scheduler_params);

    // construct params
    MainloopParams mainloop_params{params.group_size};
    EpilogueParams epilogue_params;

    // process each block
    const auto& group_size = params.group_size;

    for (const auto block_coord : scheduler) {
      // block coord: (batch_idx, m_block_idx, kv_head_idx)
      const auto [batch_idx, m_block_idx, kv_head_idx] = block_coord;
      const auto tidx = threadIdx.x;

      // Q/O: (q_packed_len, HEAD_DIM)
      // Q_ROPE: (q_packed_len, ROPE_HEAD_DIM)
      detail::MLATile<Params> tile(params, batch_idx);
      auto [Q, Q_ROPE, O] = tile.template get_qo_tile<Element>();
      // KV: (kv_len, HEAD_DIM)
      // K_ROPE: (kv_len, ROPE_HEAD_DIM)
      auto [KV, K_ROPE] = tile.template get_kv_tile<Element>();

      // problem shape
      const int q_packed_len = size<0>(Q);
      const int q_len = q_packed_len / group_size;
      const int kv_len = size<0>(KV);

      if (m_block_idx * kBlockM >= size<0>(Q)) {
        // m out of bound, return
        return;
      }

      const auto head_dim = params.head_dim;
      auto problem_shape_mnk = make_shape(q_packed_len, kv_len, head_dim);
      const auto residue_mnk = make_tuple(q_packed_len, kv_len, head_dim);
      const auto rope_residue_mnk =
          make_tuple(q_packed_len, kv_len, ROPE_HEAD_DIM{});

      // (BLK_M, BLK_K, k)
      Tensor gQ =
          local_tile(Q, Shape<BLK_M, BLK_K>{}, make_coord(m_block_idx, _));
      Tensor gO =
          local_tile(O, Shape<BLK_M, BLK_K>{}, make_coord(m_block_idx, _));
      // (BLK_M, BLK_K, k) => (M, K)
      Tensor cQ = local_tile(make_identity_tensor(Q.shape()),
                             Shape<BLK_M, BLK_K>{},
                             make_coord(m_block_idx, _));
      // (BLK_N, BLK_K, n, k)
      Tensor gKV = local_tile(KV, Shape<BLK_N, BLK_K>{}, make_coord(_, _));
      // (BLK_N, BLK_K, n, k) => (N, K)
      Tensor cKV = local_tile(make_identity_tensor(KV.shape()),
                              Shape<BLK_N, BLK_K>{},
                              make_coord(_, _));

      // (BLK_M, ROPE_HEAD_DIM)
      Tensor gQ_rope = local_tile(
          Q_ROPE, Shape<BLK_M, ROPE_HEAD_DIM>{}, make_coord(m_block_idx, _0{}));
      // (BLK_M, ROPE_HEAD_DIM) => (M, K)
      Tensor cQ_rope = local_tile(make_identity_tensor(Q_ROPE.shape()),
                                  Shape<BLK_M, ROPE_HEAD_DIM>{},
                                  make_coord(m_block_idx, _0{}));
      // (BLK_N, ROPE_HEAD_DIM, n)
      Tensor gK_rope = local_tile(
          K_ROPE, Shape<BLK_N, ROPE_HEAD_DIM>{}, make_coord(_, _0{}));
      // (BLK_N, ROPE_HEAD_DIM, n) => (N, K)
      Tensor cK_rope = local_tile(make_identity_tensor(K_ROPE.shape()),
                                  Shape<BLK_N, ROPE_HEAD_DIM>{},
                                  make_coord(_, _0{}));

      TiledMma_PV tiled_mma_pv;
      // accumulator: MMA,MMA_M,MMA_K, k)
      auto tOrAccO =
          partition_fragment_C(tiled_mma_pv, Shape<BLK_M, BLK_K, STEPS>{});
      clear(tOrAccO);

      constexpr int kRowsPerThr = kRowsPerMMA * size<1>(tOrAccO);
      OnlineSoftmax<kRowsPerThr> softmax(params.sm_scale_log2);

      // mainloop
      mha(mainloop_params,
          gQ,
          cQ,
          gKV,
          cKV,
          gQ_rope,
          cQ_rope,
          gK_rope,
          cK_rope,
          tOrAccO,
          softmax,
          tidx,
          block_coord,
          residue_mnk,
          rope_residue_mnk,
          smem);

      // epilogue
      epilogue(epilogue_params,
               tOrAccO,
               tiled_mma_pv,
               gO,
               cQ,
               tidx,
               residue_mnk,
               smem);
    }
  }
};

}  // namespace llm
