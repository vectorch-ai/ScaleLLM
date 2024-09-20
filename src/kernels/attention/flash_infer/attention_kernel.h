//  Adapted from https://github.com/flashinfer-ai/flashinfer/

#pragma once

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <flashinfer/attention/logits_post_hook.cuh>
#include <flashinfer/attention/mask.cuh>
#include <flashinfer/attention/warp_layout.cuh>
#include <flashinfer/cp_async.cuh>
#include <flashinfer/fastdiv.cuh>
#include <flashinfer/frag_layout_swizzle.cuh>
#include <flashinfer/layout.cuh>
#include <flashinfer/math.cuh>
#include <flashinfer/mma.cuh>
#include <flashinfer/permuted_smem.cuh>
#include <flashinfer/pos_enc.cuh>
#include <flashinfer/utils.cuh>

#include "kv_cache.h"
#include "state_merge_kernel.h"

namespace flashinfer {

namespace cg = cooperative_groups;
using cp_async::SharedMemFillMode;
using mma::MMAMode;

constexpr uint32_t warp_size = 32;

// NOLINTNEXTLINE
namespace {

template <PosEncodingMode pos_encoding_mode,
          typename DTypeKV,
          typename DTypeQKAccum>
constexpr bool is_invalid_configuration(uint32_t num_iters_m,
                                        uint32_t num_iters_k,
                                        uint32_t num_iters_n,
                                        uint32_t num_warps_m,
                                        uint32_t num_warps_n) {
  return ((num_iters_k < 4) || (num_iters_k == 4 && num_iters_n % 2 == 1) ||
          (num_iters_k > 4 && num_iters_k % (2 * num_warps_m) != 0) ||
          (num_iters_m *
               (8 * num_iters_k + 2 * sizeof(DTypeQKAccum) * num_iters_n) >=
           256) ||
          (sizeof(DTypeKV) == 1 && num_iters_n * 2 % num_warps_m != 0) ||
          (sizeof(DTypeKV) == 1 &&
           pos_encoding_mode == PosEncodingMode::kRoPELlama));
}

template <uint32_t num_warps_m, uint32_t num_warps_n>
__device__ __forceinline__ uint32_t get_warp_idx_x() {
  if constexpr (num_warps_m == 1) {
    return 0;
  } else {
    return threadIdx.y;
  }
}

template <uint32_t num_warps_m, uint32_t num_warps_n>
__device__ __forceinline__ uint32_t get_warp_idx_z() {
  if constexpr (num_warps_n == 1) {
    return 0;
  } else {
    return threadIdx.z;
  }
}

template <uint32_t num_warps_m, uint32_t num_warps_n>
__device__ __forceinline__ uint32_t get_warp_idx() {
  return get_warp_idx_z<num_warps_m, num_warps_n>() * num_warps_m +
         get_warp_idx_x<num_warps_m, num_warps_n>();
}

template <bool produce_v,
          uint32_t num_warps_m,
          uint32_t num_warps_n,
          uint32_t num_iters_k,
          uint32_t num_iters_n,
          SwizzleMode swizzle_mode,
          typename DType,
          typename IdType>
__device__ __forceinline__ void page_produce_kv(
    smem_t<swizzle_mode> smem,
    uint32_t* smem_offset,
    paged_kv_t<DType, IdType>& paged_kv,
    const uint32_t kv_idx_base,
    const size_t* kv_offset,
    const uint32_t kv_len) {
  // NOTE(Zihao): for fp8, this function doesn't work for head_dim = 64 at the
  // moment
  constexpr SharedMemFillMode fill_mode =
      produce_v ? SharedMemFillMode::kFillZero : SharedMemFillMode::kNoFill;
  constexpr uint32_t head_dim = num_iters_k * 16;
  constexpr uint32_t num_warps = num_warps_m * num_warps_n;
  constexpr uint32_t channel_size_128b_kv =
      head_dim / num_elems_per_128b<DType>();
  const uint32_t warp_idx = get_warp_idx<num_warps_m, num_warps_n>(),
                 lane_idx = threadIdx.x;
  if constexpr (swizzle_mode == SwizzleMode::k128B) {
    uint32_t kv_idx = kv_idx_base + warp_idx * 4 + lane_idx / 8;
    // NOTE(Zihao): num_iters_n * 4 / num_warps_m = num_warps_n * num_iters_n *
    // 4 / num_warps
    static_assert(num_iters_n * 4 % num_warps_m == 0);
#pragma unroll
    for (uint32_t i = 0; i < num_iters_n * 4 / num_warps_m; ++i) {
      DType* gptr = produce_v ? paged_kv.v_data(kv_offset[i])
                              : paged_kv.k_data(kv_offset[i]);
#pragma unroll
      for (uint32_t j = 0; j < num_iters_k / (8 / sizeof(DType)); ++j) {
        smem.load_128b_async<fill_mode>(*smem_offset, gptr, kv_idx < kv_len);
        *smem_offset =
            smem.template advance_offset_by_column<8>(*smem_offset, j);
        gptr += 8 * num_elems_per_128b<DType>();
      }
      kv_idx += num_warps * 4;
      *smem_offset = smem.template advance_offset_by_row<num_warps * 4,
                                                         channel_size_128b_kv>(
                         *smem_offset) -
                     sizeof(DType) * num_iters_k;
    }
    *smem_offset -= num_warps_n * num_iters_n * 16 * channel_size_128b_kv;
  } else {
    uint32_t kv_idx = kv_idx_base + warp_idx * 8 + lane_idx / 4;
    // NOTE(Zihao): num_iters_n * 2 / num_warps_m = num_warps_n * num_iters_n *
    // 2 / num_warps
    static_assert(num_iters_n * 2 % num_warps_m == 0);
#pragma unroll
    for (uint32_t i = 0; i < num_iters_n * 2 / num_warps_m; ++i) {
      DType* gptr = produce_v ? paged_kv.v_data(kv_offset[i])
                              : paged_kv.k_data(kv_offset[i]);
      smem.load_128b_async<fill_mode>(*smem_offset, gptr, kv_idx < kv_len);
      kv_idx += num_warps * 8;
      *smem_offset = smem.template advance_offset_by_row<num_warps * 8,
                                                         channel_size_128b_kv>(
          *smem_offset);
    }
    *smem_offset -= num_warps_n * num_iters_n * 16 * channel_size_128b_kv;
  }
}

template <uint32_t num_iters_m, uint32_t num_iters_k, typename DTypeQKAccum>
__device__ __forceinline__ void init_states(float (*o_frag)[num_iters_k][8],
                                            DTypeQKAccum (*m)[2],
                                            float (*d)[2]) {
#pragma unroll
  for (uint32_t fx = 0; fx < num_iters_m; ++fx) {
#pragma unroll
    for (uint32_t fy = 0; fy < num_iters_k; ++fy) {
#pragma unroll
      for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
        o_frag[fx][fy][reg_id] = 0.f;
      }
    }
  }
#pragma unroll
  for (uint32_t fx = 0; fx < num_iters_m; ++fx) {
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      m[fx][j] = DTypeQKAccum(-5e4);
      d[fx][j] = 1.f;
    }
  }
}

template <uint32_t num_warps_m,
          uint32_t num_warps_n,
          uint32_t num_iters_m,
          uint32_t num_iters_k,
          SwizzleMode swizzle_mode,
          typename DTypeQ>
__device__ __forceinline__ void load_q_global_smem(
    uint32_t packed_offset,
    const uint32_t qo_upper_bound,
    DTypeQ* q_ptr_base,
    const uint32_t q_stride_n,
    const uint32_t q_stride_h,
    const uint_fastdiv group_size,
    smem_t<swizzle_mode>* q_smem) {
  constexpr uint32_t head_dim = num_iters_k * 16;
  constexpr uint32_t channel_size_128b_q =
      head_dim / num_elems_per_128b<DTypeQ>();
  const uint32_t lane_idx = threadIdx.x;
  const uint32_t warp_idx_x = get_warp_idx_x<num_warps_m, num_warps_n>();
  // only let first column warps load q
  // TODO: let all warps load q
  if (get_warp_idx_z<num_warps_m, num_warps_n>() == 0) {
    // rows to load: num_iters_m * 16
    // threads layout in warp: 4 x 8
    // | t0  | t1  | t2  | t3  | t4  | t5  | t6  | t7  |
    // | t8  | t9  | t10 | t11 | t12 | t13 | t14 | t15 |
    // | t16 | t17 | t18 | t19 | t20 | t21 | t22 | t23 |
    // | t24 | t25 | t26 | t27 | t28 | t29 | t30 | t31 |
    //

    // q_smem: [num_iters_m, num_warps_m, 16, head_dim]
    uint32_t q_smem_x = warp_idx_x * num_iters_m * 16 + lane_idx / 8;
    uint32_t q_smem_y = lane_idx % 8;

#pragma unroll
    for (uint32_t fx = 0; fx < num_iters_m; ++fx) {
      // each wrap loads 4 rows, loading 16 rows needs 4(16/4) iters
#pragma unroll
      for (uint32_t j = 0; j < 4; ++j) {
        const uint32_t packed_q_idx =
            packed_offset + fx * 16 + lane_idx / 8 + j * 4;
        uint32_t q, r;
        group_size.divmod(packed_q_idx, q, r);
        // q_idx = packed_q_idx / group_size
        // h_idx = packed_q_idx % group_size
        const uint32_t q_idx = q;
        // q_ptr_base: [n_tokens, n_heads, head_dim]
        // q_ptr for given header: [head_dim]
        DTypeQ* q_ptr = q_ptr_base + q * q_stride_n + r * q_stride_h;

        // load head_dim from global memory to shared memory using 8 threads
        // 8 threads load 8 * 16 bytes columns once
        // iters: head_dim * 2 / (8 * 16) = head_dim / 16 / 4 = num_iters_k / 4
#pragma unroll
        for (uint32_t fyo = 0; fyo < num_iters_k / 4; ++fyo) {
          const uint32_t q_smem_offset_w =
              q_smem->get_permuted_offset<channel_size_128b_q>(q_smem_x,
                                                               q_smem_y);

          // load q fragment from gmem to smem
          q_smem->load_128b_async<SharedMemFillMode::kNoFill>(
              q_smem_offset_w, q_ptr, q_idx < qo_upper_bound);
          // move ahead by 8 int128_t
          q_smem_y += 8;

          // move ahead by 8 * 8 items
          q_ptr += 8 * num_elems_per_128b<DTypeQ>();
        }

        // adjust offset for next iteration
        // move row by 4
        // move columns by -num_iters_k / 4 * 8 = -num_iters_k * 2
        q_smem_x += 4;
        q_smem_y -= num_iters_k * 2;
      }
    }
  }
}

template <uint32_t num_warps_m,
          uint32_t num_warps_n,
          uint32_t num_iters_m,
          uint32_t num_iters_k,
          SwizzleMode swizzle_mode,
          typename DTypeQ>
__device__ __forceinline__ void q_smem_inplace_multiply_sm_scale(
    smem_t<swizzle_mode>* q_smem,
    const float sm_scale) {
  const uint32_t warp_idx = get_warp_idx<num_warps_m, num_warps_n>(),
                 lane_idx = threadIdx.x;
  constexpr uint32_t head_dim = num_iters_k * 16;
  constexpr uint32_t channel_size_128b_q =
      head_dim / num_elems_per_128b<DTypeQ>();
  constexpr uint32_t num_warps = num_warps_m * num_warps_n;
#pragma unroll
  for (uint32_t i = 0; i < num_iters_m * head_dim / (num_warps_n * 16); ++i) {
    vec_t<DTypeQ, 8> tmp;
    tmp.load((DTypeQ*)(q_smem->base) + (i * num_warps + warp_idx) * 256 +
             lane_idx * 8);
#pragma unroll
    for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
      tmp[reg_id] *= sm_scale;
    }
    tmp.store((DTypeQ*)(q_smem->base) + (i * num_warps + warp_idx) * 256 +
              lane_idx * 8);
  }
}

template <LogitsPostHook logits_post_hook,
          uint32_t num_iters_m,
          uint32_t num_iters_k,
          uint32_t num_iters_n,
          SwizzleMode swizzle_mode_q,
          SwizzleMode swizzle_mode_kv,
          typename DTypeQ,
          typename DTypeKV,
          typename DTypeQKAccum>
__device__ __forceinline__ void compute_qk(
    smem_t<swizzle_mode_q>* q_smem,
    uint32_t* q_smem_offset_r,
    smem_t<swizzle_mode_kv>* k_smem,
    uint32_t* k_smem_offset_r,
    DTypeQKAccum (*s_frag)[num_iters_n][8],
    const float soft_cap) {
  constexpr uint32_t head_dim = num_iters_k * 16;
  constexpr uint32_t channel_size_128b_q =
      head_dim / num_elems_per_128b<DTypeQ>();
  constexpr uint32_t channel_size_128b_kv =
      head_dim / num_elems_per_128b<DTypeKV>();
  uint32_t a_frag[num_iters_m][4], b_frag[4];
  // compute q*k^T
#pragma unroll
  for (uint32_t fy = 0; fy < num_iters_k; ++fy) {
#pragma unroll
    for (uint32_t fx = 0; fx < num_iters_m; ++fx) {
      q_smem->ldmatrix_m8n8x4(*q_smem_offset_r, a_frag[fx]);
      *q_smem_offset_r =
          q_smem->template advance_offset_by_row<16, channel_size_128b_q>(
              *q_smem_offset_r);
    }

    *q_smem_offset_r =
        q_smem->template advance_offset_by_column<2>(*q_smem_offset_r, fy) -
        num_iters_m * 16 * channel_size_128b_q;

#pragma unroll
    for (uint32_t fz = 0; fz < num_iters_n; ++fz) {
      if constexpr (sizeof(DTypeKV) == 1) {
        uint32_t b_frag_f8[2];
        if (fy % 2 == 0) {
          k_smem->ldmatrix_m8n8x4_left_half(*k_smem_offset_r, b_frag_f8);
        } else {
          k_smem->ldmatrix_m8n8x4_right_half(*k_smem_offset_r, b_frag_f8);
        }
        b_frag_f8[0] = frag_layout_swizzle_16b_to_8b(b_frag_f8[0]);
        b_frag_f8[1] = frag_layout_swizzle_16b_to_8b(b_frag_f8[1]);
        vec_cast<DTypeQ, DTypeKV>::cast<8>((DTypeQ*)b_frag,
                                           (DTypeKV*)b_frag_f8);
      } else {
        k_smem->ldmatrix_m8n8x4(*k_smem_offset_r, b_frag);
      }
      *k_smem_offset_r =
          k_smem->template advance_offset_by_row<16, channel_size_128b_kv>(
              *k_smem_offset_r);

#pragma unroll
      for (uint32_t fx = 0; fx < num_iters_m; ++fx) {
        if constexpr (std::is_same_v<DTypeQKAccum, float>) {
          if (fy == 0) {
            mma::mma_sync_m16n16k16_row_col_f16f16f32<DTypeQ, MMAMode::kInit>(
                s_frag[fx][fz], a_frag[fx], b_frag);
          } else {
            mma::mma_sync_m16n16k16_row_col_f16f16f32<DTypeQ>(
                s_frag[fx][fz], a_frag[fx], b_frag);
          }
        } else if (std::is_same_v<DTypeQKAccum, half>) {
          if (fy == 0) {
            mma::mma_sync_m16n16k16_row_col_f16f16f16<MMAMode::kInit>(
                (uint32_t*)s_frag[fx][fz], a_frag[fx], b_frag);
          } else {
            mma::mma_sync_m16n16k16_row_col_f16f16f16(
                (uint32_t*)s_frag[fx][fz], a_frag[fx], b_frag);
          }
        }
      }
    }
    if constexpr (sizeof(DTypeKV) == 1) {
      if (fy % 2 == 1) {
        *k_smem_offset_r = k_smem->template advance_offset_by_column<2>(
            *k_smem_offset_r, fy / 2);
      }
      *k_smem_offset_r -= num_iters_n * 16 * channel_size_128b_kv;
    } else {
      *k_smem_offset_r =
          k_smem->template advance_offset_by_column<2>(*k_smem_offset_r, fy) -
          num_iters_n * 16 * channel_size_128b_kv;
    }
  }
  *q_smem_offset_r -= num_iters_k * 2;
  *k_smem_offset_r -= num_iters_k * sizeof(DTypeKV);

  if constexpr (std::is_same<DTypeQKAccum, float>::value) {
#pragma unroll
    for (uint32_t fx = 0; fx < num_iters_m; ++fx) {
#pragma unroll
      for (uint32_t fz = 0; fz < num_iters_n; ++fz) {
#pragma unroll
        for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
          s_frag[fx][fz][reg_id] = apply_logits_post_hook<logits_post_hook>(
              s_frag[fx][fz][reg_id], soft_cap);
        }
      }
    }
  } else {
    static_assert(std::is_same<DTypeQKAccum, half>::value);
#pragma unroll
    for (uint32_t fx = 0; fx < num_iters_m; ++fx) {
#pragma unroll
      for (uint32_t fz = 0; fz < num_iters_n; ++fz) {
#pragma unroll
        for (uint32_t reg_id = 0; reg_id < 4; ++reg_id) {
          *(half2*)(&s_frag[fx][fz][reg_id * 2]) =
              apply_logits_post_hook<logits_post_hook>(
                  *(half2*)(&s_frag[fx][fz][reg_id * 2]), soft_cap);
        }
      }
    }
  }
}

// TODO: move it to a separate file
template <uint32_t num_iters_m, uint32_t num_iters_n, typename T>
__device__ __forceinline__ void apply_alibi_bias(
    const uint32_t qo_packed_idx_base,
    const uint32_t kv_idx_base,
    const int32_t q_offset,
    const uint_fastdiv group_size,
    float (*alibi_slope)[2],
    T (*s_frag)[num_iters_n][8]) {
  const int32_t lane_idx = threadIdx.x;
#pragma unroll
  for (int32_t fx = 0; fx < num_iters_m; ++fx) {
#pragma unroll
    for (int32_t fz = 0; fz < num_iters_n; ++fz) {
#pragma unroll
      for (int32_t reg_id = 0; reg_id < 8; ++reg_id) {
        const int32_t q_idx = (qo_packed_idx_base + fx * 16 + lane_idx / 4 +
                               8 * ((reg_id % 4) / 2)) /
                              group_size,
                      kv_idx = kv_idx_base + fz * 16 + 2 * (lane_idx % 4) +
                               8 * (reg_id / 4) + reg_id % 2;
        s_frag[fx][fz][reg_id] +=
            T(alibi_slope[fx][(reg_id % 4) / 2]) * T(kv_idx - q_idx - q_offset);
      }
    }
  }
}

template <MaskMode mask_mode,
          uint32_t num_iters_m,
          uint32_t num_iters_k,
          uint32_t num_iters_n,
          typename DTypeQKAccum>
__device__ __forceinline__ void mask_s(const uint32_t qo_packed_idx_base,
                                       const uint32_t kv_idx_base,
                                       const uint32_t qo_len,
                                       const uint32_t kv_len,
                                       const uint32_t window_left,
                                       const uint32_t chunk_end,
                                       const uint_fastdiv group_size,
                                       uint8_t* custom_mask,
                                       DTypeQKAccum (*s_frag)[num_iters_n][8]) {
  const uint32_t lane_idx = threadIdx.x;
#pragma unroll
  for (uint32_t fx = 0; fx < num_iters_m; ++fx) {
#pragma unroll
    for (uint32_t fz = 0; fz < num_iters_n; ++fz) {
#pragma unroll
      for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
        const uint32_t q_idx = (qo_packed_idx_base + fx * 16 + lane_idx / 4 +
                                8 * ((reg_id % 4) / 2)) /
                               group_size,
                       kv_idx = kv_idx_base + fz * 16 + 2 * (lane_idx % 4) +
                                8 * (reg_id / 4) + reg_id % 2;
        const bool out_of_boundary =
            (mask_mode == MaskMode::kCausal
                 ? (kv_idx + qo_len > kv_len + q_idx || (kv_idx >= chunk_end) ||
                    kv_idx + qo_len + window_left < kv_len + q_idx)
                 : kv_idx >= chunk_end ||
                       kv_idx + qo_len + window_left < kv_len + q_idx);
        s_frag[fx][fz][reg_id] =
            (out_of_boundary ||
             (mask_mode == MaskMode::kCustom && q_idx < qo_len &&
              !((custom_mask[(q_idx * kv_len + kv_idx) / 8] >>
                 ((q_idx * kv_len + kv_idx) % 8)) &
                1)))
                ? DTypeQKAccum(-5e4)
                : s_frag[fx][fz][reg_id];
      }
    }
  }
}

template <uint32_t num_iters_m,
          uint32_t num_iters_k,
          uint32_t num_iters_n,
          typename DTypeQKAccum>
__device__ __forceinline__ void update_mdo_states(
    DTypeQKAccum (*s_frag)[num_iters_n][8],
    float (*o_frag)[num_iters_k][8],
    DTypeQKAccum (*m)[2],
    float (*d)[2]) {
  if constexpr (std::is_same_v<DTypeQKAccum, float>) {
#pragma unroll
    for (uint32_t fx = 0; fx < num_iters_m; ++fx) {
#pragma unroll
      for (uint32_t j = 0; j < 2; ++j) {
        float m_prev = m[fx][j];
#pragma unroll
        for (uint32_t fz = 0; fz < num_iters_n; ++fz) {
          float m_local =
              max(max(s_frag[fx][fz][j * 2 + 0], s_frag[fx][fz][j * 2 + 1]),
                  max(s_frag[fx][fz][j * 2 + 4], s_frag[fx][fz][j * 2 + 5]));
          m[fx][j] = max(m[fx][j], m_local);
        }
        m[fx][j] = max(m[fx][j], math::shfl_xor_sync(m[fx][j], 0x2));
        m[fx][j] = max(m[fx][j], math::shfl_xor_sync(m[fx][j], 0x1));

        float o_scale = math::ptx_exp2(m_prev - m[fx][j]);
        d[fx][j] *= o_scale;
#pragma unroll
        for (uint32_t fy = 0; fy < num_iters_k; ++fy) {
          o_frag[fx][fy][j * 2 + 0] *= o_scale;
          o_frag[fx][fy][j * 2 + 1] *= o_scale;
          o_frag[fx][fy][j * 2 + 4] *= o_scale;
          o_frag[fx][fy][j * 2 + 5] *= o_scale;
        }
#pragma unroll
        for (uint32_t fz = 0; fz < num_iters_n; ++fz) {
          s_frag[fx][fz][j * 2 + 0] =
              math::ptx_exp2(s_frag[fx][fz][j * 2 + 0] - m[fx][j]);
          s_frag[fx][fz][j * 2 + 1] =
              math::ptx_exp2(s_frag[fx][fz][j * 2 + 1] - m[fx][j]);
          s_frag[fx][fz][j * 2 + 4] =
              math::ptx_exp2(s_frag[fx][fz][j * 2 + 4] - m[fx][j]);
          s_frag[fx][fz][j * 2 + 5] =
              math::ptx_exp2(s_frag[fx][fz][j * 2 + 5] - m[fx][j]);
        }
      }
    }
  } else if constexpr (std::is_same_v<DTypeQKAccum, half>) {
#pragma unroll
    for (uint32_t fx = 0; fx < num_iters_m; ++fx) {
      half m_prev[2];
#pragma unroll
      for (uint32_t j = 0; j < 2; ++j) {
        m_prev[j] = m[fx][j];
#pragma unroll
        for (uint32_t fz = 0; fz < num_iters_n; ++fz) {
          half2 m_local = __hmax2(*(half2*)&s_frag[fx][fz][j * 2],
                                  *(half2*)&s_frag[fx][fz][j * 2 + 4]);
          m[fx][j] = __hmax(m[fx][j], __hmax(m_local.x, m_local.y));
        }
      }
      *(half2*)&m[fx] =
          __hmax2(*(half2*)&m[fx], math::shfl_xor_sync(*(half2*)&m[fx], 0x2));
      *(half2*)&m[fx] =
          __hmax2(*(half2*)&m[fx], math::shfl_xor_sync(*(half2*)&m[fx], 0x1));
#pragma unroll
      for (uint32_t j = 0; j < 2; ++j) {
        float o_scale = math::ptx_exp2(float(m_prev[j] - m[fx][j]));
        d[fx][j] *= o_scale;
#pragma unroll
        for (uint32_t fy = 0; fy < num_iters_k; ++fy) {
          o_frag[fx][fy][j * 2 + 0] *= o_scale;
          o_frag[fx][fy][j * 2 + 1] *= o_scale;
          o_frag[fx][fy][j * 2 + 4] *= o_scale;
          o_frag[fx][fy][j * 2 + 5] *= o_scale;
        }
        half2 m2 = make_half2(m[fx][j], m[fx][j]);
#pragma unroll
        for (uint32_t fz = 0; fz < num_iters_n; ++fz) {
          *(half2*)&s_frag[fx][fz][j * 2] =
              math::ptx_exp2(*(half2*)&s_frag[fx][fz][j * 2] - m2);
          *(half2*)&s_frag[fx][fz][j * 2 + 4] =
              math::ptx_exp2(*(half2*)&s_frag[fx][fz][j * 2 + 4] - m2);
        }
      }
    }
  }
}

template <uint32_t num_iters_m,
          uint32_t num_iters_k,
          uint32_t num_iters_n,
          SwizzleMode swizzle_mode,
          typename DTypeQ,
          typename DTypeKV,
          typename DTypeQKAccum>
__device__ __forceinline__ void compute_sfm_v(
    smem_t<swizzle_mode>* v_smem,
    uint32_t* v_smem_offset_r,
    DTypeQKAccum (*s_frag)[num_iters_n][8],
    float (*o_frag)[num_iters_k][8],
    float (*d)[2]) {
  constexpr uint32_t head_dim = num_iters_k * 16;
  constexpr uint32_t channel_size_128b_kv =
      head_dim / num_elems_per_128b<DTypeKV>();

  DTypeQ s_frag_f16[num_iters_m][num_iters_n][8];
  if constexpr (std::is_same_v<DTypeQKAccum, float>) {
#pragma unroll
    for (uint32_t fx = 0; fx < num_iters_m; ++fx) {
#pragma unroll
      for (uint32_t fz = 0; fz < num_iters_n; ++fz) {
        vec_cast<DTypeQ, float>::cast<8>(s_frag_f16[fx][fz], s_frag[fx][fz]);
      }
    }
  }

#pragma unroll
  for (uint32_t fx = 0; fx < num_iters_m; ++fx) {
#pragma unroll
    for (uint32_t fz = 0; fz < num_iters_n; ++fz) {
      if constexpr (std::is_same<DTypeQKAccum, float>::value) {
        mma::rowsum_f16f16f32(d[fx], s_frag_f16[fx][fz]);
      } else {
        mma::rowsum_f16f16f32(d[fx], s_frag[fx][fz]);
      }
    }
  }

#pragma unroll
  for (uint32_t fz = 0; fz < num_iters_n; ++fz) {
#pragma unroll
    for (uint32_t fy = 0; fy < num_iters_k; ++fy) {
      uint32_t b_frag[4];
      if constexpr (sizeof(DTypeKV) == 1) {
        uint32_t b_frag_f8[2];
        if (fy % 2 == 0) {
          v_smem->ldmatrix_m8n8x4_trans_left_half(*v_smem_offset_r, b_frag_f8);
        } else {
          v_smem->ldmatrix_m8n8x4_trans_right_half(*v_smem_offset_r, b_frag_f8);
        }
        b_frag_f8[0] = frag_layout_swizzle_16b_to_8b_trans(b_frag_f8[0]);
        b_frag_f8[1] = frag_layout_swizzle_16b_to_8b_trans(b_frag_f8[1]);
        vec_cast<DTypeQ, DTypeKV>::cast<8>((DTypeQ*)b_frag,
                                           (DTypeKV*)b_frag_f8);
        swap(b_frag[1], b_frag[2]);
      } else {
        v_smem->ldmatrix_m8n8x4_trans(*v_smem_offset_r, b_frag);
      }
#pragma unroll
      for (uint32_t fx = 0; fx < num_iters_m; ++fx) {
        if constexpr (std::is_same<DTypeQKAccum, float>::value) {
          mma::mma_sync_m16n16k16_row_col_f16f16f32<DTypeQ>(
              o_frag[fx][fy], (uint32_t*)(s_frag_f16[fx][fz]), b_frag);
        } else {
          mma::mma_sync_m16n16k16_row_col_f16f16f32<DTypeQ>(
              o_frag[fx][fy], (uint32_t*)s_frag[fx][fz], b_frag);
        }
      }
      if constexpr (sizeof(DTypeKV) == 1) {
        if (fy % 2 == 1) {
          *v_smem_offset_r = v_smem->template advance_offset_by_column<2>(
              *v_smem_offset_r, fy / 2);
        }
      } else {
        *v_smem_offset_r =
            v_smem->template advance_offset_by_column<2>(*v_smem_offset_r, fy);
      }
    }
    *v_smem_offset_r =
        v_smem->template advance_offset_by_row<16, channel_size_128b_kv>(
            *v_smem_offset_r) -
        sizeof(DTypeKV) * num_iters_k;
  }
  *v_smem_offset_r -= 16 * num_iters_n * channel_size_128b_kv;
}

template <uint32_t num_iters_m, uint32_t num_iters_k, typename DTypeQKAccum>
__device__ __forceinline__ void normalize_d(float (*o_frag)[num_iters_k][8],
                                            DTypeQKAccum (*m)[2],
                                            float (*d)[2]) {
  float d_rcp[num_iters_m][2];
  // compute reciprocal of d
#pragma unroll
  for (uint32_t fx = 0; fx < num_iters_m; ++fx) {
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      d_rcp[fx][j] =
          (m[fx][j] != DTypeQKAccum(-5e4)) ? math::ptx_rcp(d[fx][j]) : 0.f;
    }
  }

#pragma unroll
  for (uint32_t fx = 0; fx < num_iters_m; ++fx) {
#pragma unroll
    for (uint32_t fy = 0; fy < num_iters_k; ++fy) {
#pragma unroll
      for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
        o_frag[fx][fy][reg_id] =
            o_frag[fx][fy][reg_id] * d_rcp[fx][(reg_id % 4) / 2];
      }
    }
  }
}

/*!
 * \brief Synchronize the states of the MDO kernel across the threadblock along
 * threadIdx.z.
 */
template <uint32_t num_warps_m,
          uint32_t num_warps_n,
          uint32_t num_iters_m,
          uint32_t num_iters_k,
          typename DTypeQKAccum>
__device__ __forceinline__ void threadblock_sync_mdo_states(
    float (*o_frag)[num_iters_k][8],
    float* smem_workspace,
    DTypeQKAccum (*m)[2],
    float (*d)[2],
    const uint32_t warp_idx,
    const uint32_t lane_idx) {
  // only necessary when blockDim.z > 1
  if constexpr (num_warps_n > 1) {
    float2* smem_md =
        (float2*)(smem_workspace + num_iters_m * num_iters_k * num_warps_m *
                                       num_warps_n * warp_size * 8);
    // o: [num_warps, num_iters_m, num_iters_k, warp_size(32), 8]
    // md: [num_warps, num_iters_m, 2, warp_size(32), 2 (m/d)]
#pragma unroll
    for (uint32_t fx = 0; fx < num_iters_m; ++fx) {
#pragma unroll
      for (uint32_t fy = 0; fy < num_iters_k; ++fy) {
        vec_t<float, 8>::memcpy(
            smem_workspace +
                (((warp_idx * num_iters_m + fx) * num_iters_k + fy) *
                     warp_size +
                 lane_idx) *
                    8,
            o_frag[fx][fy]);
      }
    }

#pragma unroll
    for (uint32_t fx = 0; fx < num_iters_m; ++fx) {
#pragma unroll
      for (uint32_t j = 0; j < 2; ++j) {
        smem_md[((warp_idx * num_iters_m + fx) * 2 + j) * warp_size +
                lane_idx] = make_float2(float(m[fx][j]), d[fx][j]);
      }
    }

    // synchronize m,d first
    __syncthreads();
#pragma unroll
    for (uint32_t fx = 0; fx < num_iters_m; ++fx) {
      float o_scale[2][num_warps_n];
#pragma unroll
      for (uint32_t j = 0; j < 2; ++j) {
        float m_new = -5e4, d_new = 1.f;
#pragma unroll
        for (uint32_t i = 0; i < num_warps_n; ++i) {
          float2 md = smem_md[(((i * num_warps_m +
                                 get_warp_idx_x<num_warps_m, num_warps_n>()) *
                                    num_iters_m +
                                fx) *
                                   2 +
                               j) *
                                  warp_size +
                              lane_idx];
          float m_prev = m_new, d_prev = d_new;
          m_new = max(m_new, md.x);
          d_new = d_prev * math::ptx_exp2(m_prev - m_new) +
                  md.y * math::ptx_exp2(md.x - m_new);
        }

#pragma unroll
        for (uint32_t i = 0; i < num_warps_n; ++i) {
          float2 md = smem_md[(((i * num_warps_m +
                                 get_warp_idx_x<num_warps_m, num_warps_n>()) *
                                    num_iters_m +
                                fx) *
                                   2 +
                               j) *
                                  warp_size +
                              lane_idx];
          float mi = md.x;
          o_scale[j][i] = math::ptx_exp2(float(mi - m_new));
        }
        m[fx][j] = DTypeQKAccum(m_new);
        d[fx][j] = d_new;
      }

#pragma unroll
      for (uint32_t fy = 0; fy < num_iters_k; ++fy) {
        vec_t<float, 8> o_new;
        o_new.fill(0.f);
#pragma unroll
        for (uint32_t i = 0; i < num_warps_n; ++i) {
          vec_t<float, 8> oi;
          oi.load(smem_workspace +
                  ((((i * num_warps_m +
                      get_warp_idx_x<num_warps_m, num_warps_n>()) *
                         num_iters_m +
                     fx) *
                        num_iters_k +
                    fy) *
                       warp_size +
                   lane_idx) *
                      8);
#pragma unroll
          for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
            o_new[reg_id] += oi[reg_id] * o_scale[(reg_id % 4) / 2][i];
          }
        }
        o_new.store(o_frag[fx][fy]);
      }
    }
  }
}

template <uint32_t num_warps_m,
          uint32_t num_warps_n,
          uint32_t num_iters_m,
          uint32_t num_iters_k,
          SwizzleMode swizzle_mode,
          typename DTypeOut>
__device__ __forceinline__ void write_o_reg_gmem(
    float (*o_frag)[num_iters_k][8],
    smem_t<swizzle_mode>* o_smem,
    DTypeOut* o_ptr_base,
    const uint32_t o_packed_idx_base,
    const uint32_t qo_upper_bound,
    const uint32_t o_stride_n,
    const uint32_t o_stride_h,
    const uint_fastdiv group_size) {
  constexpr uint32_t head_dim = num_iters_k * 16;
  constexpr uint32_t channel_size_128b_out =
      head_dim / num_elems_per_128b<DTypeOut>();
  const uint32_t warp_idx_x = get_warp_idx_x<num_warps_m, num_warps_n>();
  const uint32_t lane_idx = threadIdx.x;

  if (get_warp_idx_z<num_warps_m, num_warps_n>() == 0) {
#pragma unroll
    for (uint32_t fx = 0; fx < num_iters_m; ++fx) {
#pragma unroll
      for (uint32_t fy = 0; fy < num_iters_k; ++fy) {
        uint32_t o_frag_f16[4];
        vec_cast<DTypeOut, float>::cast<8>((DTypeOut*)o_frag_f16,
                                           o_frag[fx][fy]);
#ifdef FLASHINFER_STMATRIX_M8N8X4_ENABLED
        uint32_t o_smem_offset_w =
            o_smem->get_permuted_offset<channel_size_128b_out>(
                (warp_idx_x * num_iters_m + fx) * 16 + lane_idx % 16,
                fy * 2 + lane_idx / 16);
        o_smem->stmatrix_m8n8x4(o_smem_offset_w, o_frag_f16);
#else
        uint32_t o_smem_offset_w =
            o_smem->get_permuted_offset<channel_size_128b_out>(
                (warp_idx_x * num_iters_m + fx) * 16 + lane_idx / 4, fy * 2);
        ((uint32_t*)(o_smem->base + o_smem_offset_w))[lane_idx % 4] =
            o_frag_f16[0];
        ((uint32_t*)(o_smem->base + o_smem_offset_w +
                     8 * channel_size_128b_out))[lane_idx % 4] = o_frag_f16[1];
        ((uint32_t*)(o_smem->base + (o_smem_offset_w ^ 0x1)))[lane_idx % 4] =
            o_frag_f16[2];
        ((uint32_t*)(o_smem->base + (o_smem_offset_w ^ 0x1) +
                     8 * channel_size_128b_out))[lane_idx % 4] = o_frag_f16[3];
#endif
      }
    }

    uint32_t o_smem_offset_w =
        o_smem->get_permuted_offset<channel_size_128b_out>(
            warp_idx_x * num_iters_m * 16 + lane_idx / 8, lane_idx % 8);

#pragma unroll
    for (uint32_t fx = 0; fx < num_iters_m; ++fx) {
#pragma unroll
      for (uint32_t j = 0; j < 4; ++j) {
        uint32_t q, r;
        group_size.divmod(
            o_packed_idx_base + lane_idx / 8 + fx * 16 + j * 4, q, r);
        const uint32_t o_idx = q;
        DTypeOut* o_ptr = o_ptr_base + q * o_stride_n + r * o_stride_h;
#pragma unroll
        for (uint32_t fyo = 0; fyo < num_iters_k / 4; ++fyo) {
          if (o_idx < qo_upper_bound) {
            o_smem->store_128b(o_smem_offset_w, o_ptr);
          }
          o_ptr += 8 * num_elems_per_128b<DTypeOut>();
          o_smem_offset_w = o_smem->template advance_offset_by_column<8>(
              o_smem_offset_w, fyo);
        }
        o_smem_offset_w =
            o_smem->template advance_offset_by_row<4, channel_size_128b_out>(
                o_smem_offset_w) -
            2 * num_iters_k;
      }
    }
  }
}

}  // namespace

// dim3 nblks(n_splits, 1, num_kv_heads);
// dim3 nthrs(32, num_warps_m, num_warps_n);
template <LogitsPostHook logits_post_hook,
          MaskMode mask_mode,
          PosEncodingMode pos_encoding_mode,
          uint32_t num_iters_m,
          uint32_t num_iters_k,
          uint32_t num_iters_n,
          uint32_t num_warps_m,
          uint32_t num_warps_n,
          typename DTypeQ,
          typename DTypeKV,
          typename DTypeQKAccum,
          typename DTypeOut,
          typename IdType>
__global__
__launch_bounds__(num_warps_m* num_warps_n* warp_size) void attention_kernel(
    IdType* __restrict__ request_indices,
    IdType* __restrict__ q_tile_indices,
    IdType* __restrict__ kv_tile_indices,
    DTypeQ* __restrict__ q,
    paged_kv_t<DTypeKV, IdType> paged_kv,
    IdType* __restrict__ q_indptr,
    IdType* __restrict__ kv_indptr,
    uint8_t* __restrict__ custom_mask,
    IdType* __restrict__ qk_indptr,
    IdType* __restrict__ o_indptr,
    DTypeOut* __restrict__ o,
    float* __restrict__ lse,
    bool* __restrict__ block_valid_mask,
    IdType* __restrict__ kv_chunk_size_ptr,
    const bool partition_kv,
    const uint_fastdiv group_size,
    int32_t maybe_window_left,
    const float logits_soft_cap,
    float sm_scale,
    float* __restrict__ alibi_slopes) {
  static_assert(sizeof(DTypeQ) == 2);
  static_assert(sizeof(DTypeOut) == 2);

  // instead of using loge for softmax, we use log2 for better performance
  // exp(x - max) == exp2(x * log_2(e) - max * log_2(e))
  sm_scale *= (logits_post_hook == LogitsPostHook::kNone
                   ? math::log2e
                   : math::ptx_rcp(logits_soft_cap));
  auto block = cg::this_thread_block();
  const uint32_t kv_chunk_size = *kv_chunk_size_ptr;

  const uint32_t bx = blockIdx.x;

  if (block_valid_mask && !block_valid_mask[bx]) {
    return;
  }

  const uint32_t kv_head_idx = blockIdx.z;
  const uint32_t q_head_idx_base = kv_head_idx * group_size;
  const uint32_t lane_idx = threadIdx.x;
  const uint32_t warp_idx = get_warp_idx<num_warps_m, num_warps_n>();
  const uint32_t num_kv_heads = gridDim.z;
  const uint32_t num_qo_heads = num_kv_heads * group_size;

  const uint32_t request_idx = request_indices[bx];
  const uint32_t qo_tile_idx = q_tile_indices[bx];
  const uint32_t kv_tile_idx = kv_tile_indices[bx];

  constexpr uint32_t num_rows_per_cta = num_iters_m * num_warps_m * 16;
  const uint32_t qo_len = q_indptr[request_idx + 1] - q_indptr[request_idx];
  const uint32_t kv_len = kv_indptr[request_idx + 1] - kv_indptr[request_idx];

  // could kv_len be 0?
  const uint32_t kv_len_safe = kv_len > 0 ? kv_len : 1;
  const uint32_t window_left =
      (maybe_window_left >= 0) ? maybe_window_left : kv_len;

  // kv idx range for this chunk
  const uint32_t chunk_start = partition_kv ? kv_tile_idx * kv_chunk_size : 0;
  const uint32_t chunk_end =
      partition_kv ? min((kv_tile_idx + 1) * kv_chunk_size, kv_len) : kv_len;
  const uint32_t chunk_size = chunk_end - chunk_start;
  // heads in query are flattened to first dimension, so we need to div
  // group_size to get original request_idx
  const uint32_t qo_upper_bound =
      min(qo_len, ceil_div((qo_tile_idx + 1) * num_rows_per_cta, group_size));

  constexpr uint32_t head_dim = num_iters_k * 16;
  // TODO: static_assert even division
  constexpr uint32_t channel_size_128b_q =
      head_dim / num_elems_per_128b<DTypeQ>();
  constexpr uint32_t channel_size_128b_kv =
      head_dim / num_elems_per_128b<DTypeKV>();
  constexpr uint32_t channel_size_128b_out =
      head_dim / num_elems_per_128b<DTypeOut>();

  // define shared memory
  extern __shared__ uint8_t smem[];
  // swizzle_mode_q: 128B for 16-bit Q,
  // 64B for 8-bit Q is not supported yet
  constexpr SwizzleMode swizzle_mode_q = SwizzleMode::k128B;
  // [num_iters_m, num_warps_m, MMA_M, head_dim]
  smem_t<swizzle_mode_q> qo_smem(smem);

  constexpr SwizzleMode swizzle_mode_kv =
      (sizeof(DTypeKV) == 1 && head_dim == 64) ? SwizzleMode::k64B
                                               : SwizzleMode::k128B;
  // [num_iters_n, num_warps_n, MMA_M, head_dim]
  smem_t<swizzle_mode_kv> k_smem(
      smem + (num_warps_m * num_iters_m * sizeof(DTypeQ)) * 16 * head_dim);
  // [num_iters_n, num_warps_n, MMA_M, head_dim]
  smem_t<swizzle_mode_kv> v_smem(smem +
                                 (num_warps_m * num_iters_m * sizeof(DTypeQ) +
                                  num_warps_n * num_iters_n * sizeof(DTypeKV)) *
                                     16 * head_dim);

  const uint32_t q_stride_n = num_qo_heads * head_dim;
  const uint32_t q_stride_h = head_dim;

  // [n_tokens, n_heads, head_dim]
  DTypeQ* q_ptr_base =
      q + get_elem_offset_impl(
              /*elem_idx=*/q_indptr[request_idx],
              /*head_idx=*/q_head_idx_base,
              /*feat_idx=*/(lane_idx % 8) * num_elems_per_128b<DTypeQ>(),
              q_stride_n,
              q_stride_h);

  // base index of flattened qo for cur warp,
  // qo_packed: [n_tokens*n_heads, head_dim]
  // per warp rows: num_iters_m * 16
  const uint32_t qo_packed_idx_base =
      (qo_tile_idx * num_warps_m + get_warp_idx_x<num_warps_m, num_warps_n>()) *
      num_iters_m * 16;

  // load q to shared memory once for current wrap
  load_q_global_smem<num_warps_m, num_warps_n, num_iters_m, num_iters_k>(
      qo_packed_idx_base,
      qo_upper_bound,
      q_ptr_base,
      q_stride_n,
      q_stride_h,
      group_size,
      &qo_smem);
  // [] => [q]
  cp_async::commit_group();

  // load first k/v chunk to shared memory
  // calculate q/k/v offsets to reand and write
  uint32_t q_smem_offset_r = qo_smem.get_permuted_offset<channel_size_128b_q>(
      get_warp_idx_x<num_warps_m, num_warps_n>() * num_iters_m * 16 +
          lane_idx % 16,
      lane_idx / 16);

  constexpr uint32_t kv_frag_rows =
      swizzle_mode_kv == SwizzleMode::k128B ? 4 : 8;
  constexpr uint32_t kv_frag_cols =
      swizzle_mode_kv == SwizzleMode::k128B ? 8 : 4;
  size_t kv_offset[num_iters_n *
                   (swizzle_mode_kv == SwizzleMode::k128B ? 4 : 2) /
                   num_warps_m];

  uint32_t k_smem_offset_r = k_smem.get_permuted_offset<channel_size_128b_kv>(
      get_warp_idx_z<num_warps_m, num_warps_n>() * num_iters_n * 16 +
          8 * (lane_idx / 16) + lane_idx % 8,
      (lane_idx % 16) / 8);
  uint32_t v_smem_offset_r = v_smem.get_permuted_offset<channel_size_128b_kv>(
      get_warp_idx_z<num_warps_m, num_warps_n>() * num_iters_n * 16 +
          lane_idx % 16,
      lane_idx / 16);
  uint32_t kv_smem_offset_w = k_smem.get_permuted_offset<channel_size_128b_kv>(
      warp_idx * kv_frag_rows + lane_idx / kv_frag_cols,
      lane_idx % kv_frag_cols);

  // kv_idx of current sequence
  uint32_t kv_idx_base = chunk_start;

#pragma unroll
  for (uint32_t i = 0;
       i < num_iters_n * (swizzle_mode_kv == SwizzleMode::k128B ? 4 : 2) /
               num_warps_m;
       ++i) {
    const uint32_t kv_idx = kv_idx_base + warp_idx * kv_frag_rows +
                            lane_idx / kv_frag_cols +
                            kv_frag_rows * num_warps_m * num_warps_n * i;
    const uint32_t feat_idx =
        (lane_idx % kv_frag_cols) * num_elems_per_128b<DTypeKV>();
    kv_offset[i] =
        kv_idx < kv_len
            ? paged_kv.get_kv_offset(request_idx, kv_idx, kv_head_idx, feat_idx)
            : 0;
  }

  page_produce_kv<false, num_warps_m, num_warps_n, num_iters_k, num_iters_n>(
      k_smem, &kv_smem_offset_w, paged_kv, 0, kv_offset, chunk_size);
  cp_async::commit_group();
  page_produce_kv<true, num_warps_m, num_warps_n, num_iters_k, num_iters_n>(
      v_smem, &kv_smem_offset_w, paged_kv, 0, kv_offset, chunk_size);
  // [q] => [q, k, v]
  cp_async::commit_group();

  // wait for q to be loaded: [q, k, v] => [k, v]
  cp_async::wait_group<2>();
  block.sync();

  // TODO: can we do this in register?
  q_smem_inplace_multiply_sm_scale<num_warps_m,
                                   num_warps_n,
                                   num_iters_m,
                                   num_iters_k,
                                   swizzle_mode_q,
                                   DTypeQ>(&qo_smem, sm_scale);

  const uint32_t num_iterations = ceil_div(
      (mask_mode == MaskMode::kCausal
           ? min(chunk_size,
                 sub_if_greater_or_zero(
                     kv_len - qo_len +
                         ((qo_tile_idx + 1) * num_rows_per_cta) / group_size,
                     chunk_start))
           : chunk_size),
      16 * num_warps_n * num_iters_n);

  const uint32_t window_iteration =
      ceil_div(sub_if_greater_or_zero(kv_len + (bx + 1) * num_rows_per_cta,
                                      qo_len + window_left + chunk_start),
               (16 * num_warps_n * num_iters_n));

  const uint32_t mask_iteration =
      (mask_mode == MaskMode::kCausal
           ? min(chunk_size,
                 sub_if_greater_or_zero(
                     kv_len + (qo_tile_idx * num_rows_per_cta) / group_size -
                         qo_len,
                     chunk_start))
           : chunk_size) /
      (16 * num_warps_n * num_iters_n);

  // alibi slopes for 2 rows 0, 8
  float alibi_slopes_frag[num_iters_m][2];
  if constexpr (pos_encoding_mode == PosEncodingMode::kALiBi) {
#pragma unroll
    for (uint32_t fx = 0; fx < num_iters_m; ++fx) {
#pragma unroll
      for (uint32_t j = 0; j < 2; ++j) {
        const uint32_t qo_head_idx =
            q_head_idx_base +
            (qo_packed_idx_base + lane_idx / 4 + j * 8 + fx * 16) % group_size;
        alibi_slopes_frag[fx][j] = alibi_slopes[qo_head_idx] * math::log2e;
      }
    }
  }

  // define fragments for each thread
  DTypeQKAccum s_frag[num_iters_m][num_iters_n][8];
  // regesters hold a whole data in register? necessary?
  float o_frag[num_iters_m][num_iters_k][8];

  // max and sum for 2 rows 0, 8
  DTypeQKAccum m[num_iters_m][2];
  float d[num_iters_m][2];
  // initialize o = 0, m = -5e4, d = 1
  init_states<num_iters_m, num_iters_k>(o_frag, m, d);

  // iterate over kv chunks
#pragma unroll 1
  for (uint32_t iter = 0; iter < num_iterations; ++iter) {
    kv_idx_base += 16 * num_warps_n * num_iters_n;

    // calculate kv offsets to read before waiting for k ready
#pragma unroll
    for (uint32_t i = 0;
         i < num_iters_n * (swizzle_mode_kv == SwizzleMode::k128B ? 4 : 2) /
                 num_warps_m;
         ++i) {
      const uint32_t kv_idx = kv_idx_base + warp_idx * kv_frag_rows +
                              lane_idx / kv_frag_cols +
                              kv_frag_rows * num_warps_m * num_warps_n * i;
      const uint32_t feat_idx =
          (lane_idx % kv_frag_cols) * num_elems_per_128b<DTypeKV>();
      kv_offset[i] = kv_idx < kv_len
                         ? paged_kv.get_kv_offset(
                               request_idx, kv_idx, kv_head_idx, feat_idx)
                         : 0;
    }

    // wait for k ready: [k, v] => [v]
    cp_async::wait_group<1>();
    block.sync();

    // compute attention score
    compute_qk<logits_post_hook,
               num_iters_m,
               num_iters_k,
               num_iters_n,
               swizzle_mode_q,
               swizzle_mode_kv,
               DTypeQ,
               DTypeKV>(&qo_smem,
                        &q_smem_offset_r,
                        &k_smem,
                        &k_smem_offset_r,
                        s_frag,
                        logits_soft_cap);

    if constexpr (pos_encoding_mode == PosEncodingMode::kALiBi) {
      apply_alibi_bias<num_iters_m, num_iters_n>(
          qo_packed_idx_base,
          chunk_start + (iter * num_warps_n +
                         get_warp_idx_z<num_warps_m, num_warps_n>()) *
                            num_iters_n * 16,
          int(kv_len) - int(qo_len),
          group_size,
          alibi_slopes_frag,
          s_frag);
    }
    // apply mask
    if constexpr (mask_mode == MaskMode::kCustom) {
      mask_s<mask_mode, num_iters_m, num_iters_k, num_iters_n>(
          qo_packed_idx_base,
          chunk_start + (iter * num_warps_n +
                         get_warp_idx_z<num_warps_m, num_warps_n>()) *
                            num_iters_n * 16,
          qo_len,
          kv_len,
          window_left,
          chunk_end,
          group_size,
          custom_mask + qk_indptr[request_idx],
          s_frag);
    } else {
      if (iter >= mask_iteration || iter < window_iteration) {
        mask_s<mask_mode, num_iters_m, num_iters_k, num_iters_n>(
            qo_packed_idx_base,
            chunk_start + (iter * num_warps_n +
                           get_warp_idx_z<num_warps_m, num_warps_n>()) *
                              num_iters_n * 16,
            qo_len,
            kv_len,
            window_left,
            chunk_end,
            group_size,
            nullptr,
            s_frag);
      }
    }

    // compute m,d states in online softmax
    update_mdo_states<num_iters_m, num_iters_k, num_iters_n>(
        s_frag, o_frag, m, d);
    block.sync();

    // produce k for next iteration
    page_produce_kv<false, num_warps_m, num_warps_n, num_iters_k, num_iters_n>(
        k_smem,
        &kv_smem_offset_w,
        paged_kv,
        (iter + 1) * 16 * num_warps_n * num_iters_n,
        kv_offset,
        chunk_size);
    // [v] => [v, k]
    cp_async::commit_group();

    // wait for v ready, [v, k] => [k]
    cp_async::wait_group<1>();
    block.sync();

    // compute sfm*v
    compute_sfm_v<num_iters_m,
                  num_iters_k,
                  num_iters_n,
                  swizzle_mode_kv,
                  DTypeQ,
                  DTypeKV>(&v_smem, &v_smem_offset_r, s_frag, o_frag, d);
    block.sync();

    // produce v for next iteration
    page_produce_kv<true, num_warps_m, num_warps_n, num_iters_k, num_iters_n>(
        v_smem,
        &kv_smem_offset_w,
        paged_kv,
        (iter + 1) * 16 * num_warps_n * num_iters_n,
        kv_offset,
        chunk_size);

    // [k] => [k, v]
    cp_async::commit_group();
  }

  // wait for all data ready
  cp_async::wait_group<0>();
  block.sync();

  // threadblock synchronization
  threadblock_sync_mdo_states<num_warps_m,
                              num_warps_n,
                              num_iters_m,
                              num_iters_k,
                              DTypeQKAccum>(
      o_frag, (float*)smem, m, d, warp_idx, lane_idx);

  // normalize d
  normalize_d<num_iters_m, num_iters_k>(o_frag, m, d);

  const uint32_t num_kv_chunks =
      (kv_len_safe + kv_chunk_size - 1) / kv_chunk_size;

  //  partition_kv: [n_tokens*n_kv_tiles, n_heads, head_dim]
  //  !partition_kv: [n_tokens*1, n_heads, head_dim]
  DTypeOut* o_ptr_base = partition_kv
                             ? o + kv_tile_idx * num_qo_heads * head_dim +
                                   get_elem_offset_impl(
                                       /*elem_idx=*/o_indptr[request_idx],
                                       /*head_idx=*/q_head_idx_base,
                                       /*feat_idx=*/(lane_idx % 8) *
                                           num_elems_per_128b<DTypeOut>(),
                                       num_qo_heads * head_dim,
                                       head_dim)
                             : o + get_elem_offset_impl(
                                       /*elem_idx=*/o_indptr[request_idx],
                                       /*head_idx=*/q_head_idx_base,
                                       /*feat_idx=*/(lane_idx % 8) *
                                           num_elems_per_128b<DTypeOut>(),
                                       num_qo_heads * head_dim,
                                       head_dim);
  // write_back
  write_o_reg_gmem<num_warps_m, num_warps_n, num_iters_m, num_iters_k>(
      o_frag,
      &qo_smem,
      o_ptr_base,
      qo_packed_idx_base,
      qo_len,
      /*o_stride_n=*/
      partition_kv ? num_qo_heads * head_dim * num_kv_chunks
                   : num_qo_heads * head_dim,
      /*o_stride_h=*/head_dim,
      group_size);

  // write lse
  if (lse != nullptr) {
    if (get_warp_idx_z<num_warps_m, num_warps_n>() == 0) {
#pragma unroll
      for (uint32_t fx = 0; fx < num_iters_m; ++fx) {
#pragma unroll
        for (uint32_t j = 0; j < 2; ++j) {
          uint32_t q, r;
          group_size.divmod(
              qo_packed_idx_base + lane_idx / 4 + j * 8 + fx * 16, q, r);
          const uint32_t qo_head_idx = q_head_idx_base + r;
          const uint32_t qo_idx = q;
          if (qo_idx < qo_upper_bound) {
            if (partition_kv) {
              // [n_tokens, n_kv_tiles, n_heads]
              // lse = ((o_indptr[request_idx] + qo_idx * num_kv_chunks +
              // kv_tile_idx) * num_qo_heads + qo_head_idx)
              lse[(o_indptr[request_idx] + qo_idx * num_kv_chunks +
                   kv_tile_idx) *
                      num_qo_heads +
                  qo_head_idx] = math::ptx_log2(d[fx][j]) + float(m[fx][j]);
            } else {
              // [n_tokens, n_heads]
              lse[(o_indptr[request_idx] + qo_idx) * num_qo_heads +
                  qo_head_idx] = math::ptx_log2(d[fx][j]) + float(m[fx][j]);
            }
          }
        }
      }
    }
  }
}

template <WarpLayout WARP_LAYOUT,
          uint32_t HEAD_DIM,
          LogitsPostHook LOGITS_POST_HOOK,
          PosEncodingMode pos_encoding_mode,
          bool ALLOW_FP16_QK_REDUCTION,
          MaskMode MASK_MODE,
          typename DTypeQ,
          typename DTypeKV,
          typename DTypeOut,
          typename IdType>
cudaError_t mha_varlen_dispatch(DTypeQ* q,
                                IdType* request_indices,
                                IdType* q_tile_indices,
                                IdType* kv_tile_indices,
                                IdType* q_indptr,
                                IdType* kv_indptr,
                                paged_kv_t<DTypeKV, IdType> paged_kv,
                                uint8_t* custom_mask,
                                IdType* qk_indptr,
                                IdType* o_indptr,
                                DTypeOut* o,
                                DTypeOut* tmp_v,
                                float* tmp_s,
                                float* lse,
                                IdType* merge_indptr,
                                bool* block_valid_mask,
                                IdType* kv_chunk_size_ptr,
                                uint32_t total_num_rows,
                                uint32_t num_qo_heads,
                                uint32_t num_kv_heads,
                                uint32_t padded_batch_size,
                                int32_t window_left,
                                float logits_soft_cap,
                                float sm_scale,
                                float* alibi_slopes,
                                cudaStream_t stream) {
#if (__CUDA_ARCH__ < 800)
  if constexpr (std::is_same_v<DTypeQ, nv_bfloat16>) {
    FLASHINFER_RUNTIME_ASSERT("Prefill kernels do not support bf16 on sm75.");
  }
#endif

  constexpr uint32_t num_iters_m = get_num_frags_x<WARP_LAYOUT>();
  constexpr uint32_t num_warps_m = get_num_warps_x<WARP_LAYOUT>();
  constexpr uint32_t num_warps_n = get_num_warps_z<WARP_LAYOUT>();
  const uint32_t group_size = num_qo_heads / num_kv_heads;
  const uint_fastdiv group_size_fastdiv(group_size);

  if (padded_batch_size == 0) {
    // No request, skip
    // this won't happen in CUDAGraph mode because we fixed the
    // padded_batch_size
    return cudaSuccess;
  }

  dim3 nblks(padded_batch_size, 1, num_kv_heads);
  dim3 nthrs(32, num_warps_m, num_warps_n);

  constexpr uint32_t num_iters_k = HEAD_DIM / 16;
  using DTypeQKAccum = std::conditional_t<ALLOW_FP16_QK_REDUCTION &&
                                              std::is_same_v<DTypeQ, half>,
                                          half,
                                          float>;

  int dev_id = 0;
  FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
  int max_smem_per_sm = 0;
  FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(
      &max_smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev_id));
  // we expect each sm execute two threadblocks
  // TODO(Zihao): fix the following computation
  const int num_ctas_per_sm =
      max_smem_per_sm > (16 * HEAD_DIM * sizeof(DTypeQ) * 16) ? 2 : 1;
  const int max_smem_per_threadblock = max_smem_per_sm / num_ctas_per_sm;

  const uint32_t max_num_iters_n_reg =
      (HEAD_DIM >= 128 && num_iters_m == 2 &&
       pos_encoding_mode == PosEncodingMode::kRoPELlama &&
       !ALLOW_FP16_QK_REDUCTION)
          ? 2
          : (8 / num_iters_m);
  // TODO(Zihao): fix the following computation
  const uint32_t max_num_iters_n_smem =
      (max_smem_per_threadblock / (16 * HEAD_DIM * sizeof(DTypeQ)) -
       num_iters_m * num_warps_m) /
      (2 * num_warps_n);

  DISPATCH_NUM_FRAGS_Z(
      min(max_num_iters_n_smem, max_num_iters_n_reg), num_iters_n, {
        if constexpr (is_invalid_configuration<pos_encoding_mode,
                                               DTypeKV,
                                               DTypeQKAccum>(num_iters_m,
                                                             num_iters_k,
                                                             num_iters_n,
                                                             num_warps_m,
                                                             num_warps_n)) {
          // Invalid configuration, skip
          std::ostringstream err_msg;
          err_msg << "FlashInfer Internal Error: Invalid configuration : "
                     "num_iters_m="
                  << num_iters_m << " num_iters_k=" << num_iters_k
                  << " num_iters_n=" << num_iters_n
                  << " num_warps_m=" << num_warps_m
                  << " num_warps_n=" << num_warps_n
                  << " please create an issue "
                     "(https://github.com/flashinfer-ai/flashinfer/issues)"
                     " and report the issue to the developers.";
          throw std::invalid_argument(err_msg.str());
        } else {
          // TODO(Zihao): fix the following computation
          uint32_t smem_size =
              (num_iters_m * num_warps_m * sizeof(DTypeQ) +
               num_iters_n * num_warps_n * 2 * sizeof(DTypeQ)) *
              16 * HEAD_DIM;
          auto kernel = attention_kernel<LOGITS_POST_HOOK,
                                         MASK_MODE,
                                         pos_encoding_mode,
                                         num_iters_m,
                                         num_iters_k,
                                         num_iters_n,
                                         num_warps_m,
                                         num_warps_n,
                                         DTypeQ,
                                         DTypeKV,
                                         DTypeQKAccum,
                                         DTypeOut,
                                         IdType>;
          FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(
              kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
          if (tmp_v == nullptr) {
            // do not partition kv
            bool partition_kv = false;
            void* args[] = {(void*)&request_indices,
                            (void*)&q_tile_indices,
                            (void*)&kv_tile_indices,
                            (void*)&q,
                            (void*)&paged_kv,
                            (void*)&q_indptr,
                            (void*)&kv_indptr,
                            (void*)&custom_mask,
                            (void*)&qk_indptr,
                            (void*)&o_indptr,
                            (void*)&o,
                            (void*)&lse,
                            (void*)&block_valid_mask,
                            (void*)&kv_chunk_size_ptr,
                            (void*)&partition_kv,
                            (void*)&group_size_fastdiv,
                            (void*)&window_left,
                            (void*)&logits_soft_cap,
                            (void*)&sm_scale,
                            (void*)&alibi_slopes};
            FLASHINFER_CUDA_CALL(cudaLaunchKernel(
                (void*)kernel, nblks, nthrs, args, smem_size, stream));
          } else {
            bool partition_kv = true;
            void* args[] = {(void*)&request_indices,
                            (void*)&q_tile_indices,
                            (void*)&kv_tile_indices,
                            (void*)&q,
                            (void*)&paged_kv,
                            (void*)&q_indptr,
                            (void*)&kv_indptr,
                            (void*)&custom_mask,
                            (void*)&qk_indptr,
                            (void*)&o_indptr,
                            (void*)&tmp_v,
                            (void*)&tmp_s,
                            (void*)&block_valid_mask,
                            (void*)&kv_chunk_size_ptr,
                            (void*)&partition_kv,
                            (void*)&group_size_fastdiv,
                            (void*)&window_left,
                            (void*)&logits_soft_cap,
                            (void*)&sm_scale,
                            (void*)&alibi_slopes};
            FLASHINFER_CUDA_CALL(cudaLaunchKernel(
                (void*)kernel, nblks, nthrs, args, smem_size, stream));
            FLASHINFER_CUDA_CALL(VariableLengthMergeStates(tmp_v,
                                                           tmp_s,
                                                           merge_indptr,
                                                           o,
                                                           lse,
                                                           total_num_rows,
                                                           num_qo_heads,
                                                           HEAD_DIM,
                                                           stream));
          }
        }
      });
  return cudaSuccess;
}

}  // namespace flashinfer
