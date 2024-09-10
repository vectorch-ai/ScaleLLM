//  Adapted from https://github.com/flashinfer-ai/flashinfer/

#pragma once

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <flashinfer/attention/cascade.cuh>
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
constexpr bool is_invalid_configuration(uint32_t num_frags_x,
                                        uint32_t num_frags_y,
                                        uint32_t num_frags_z,
                                        uint32_t num_warps_x,
                                        uint32_t num_warps_z) {
  return ((num_frags_y < 4) || (num_frags_y == 4 && num_frags_z % 2 == 1) ||
          (num_frags_y > 4 && num_frags_y % (2 * num_warps_x) != 0) ||
          (num_frags_x *
               (8 * num_frags_y + 2 * sizeof(DTypeQKAccum) * num_frags_z) >=
           256) ||
          (sizeof(DTypeKV) == 1 && num_frags_z * 2 % num_warps_x != 0) ||
          (sizeof(DTypeKV) == 1 &&
           pos_encoding_mode == PosEncodingMode::kRoPELlama));
}

template <uint32_t num_warps_x, uint32_t num_warps_z>
__device__ __forceinline__ uint32_t get_warp_idx_x() {
  if constexpr (num_warps_x == 1) {
    return 0;
  } else {
    return threadIdx.y;
  }
}

template <uint32_t num_warps_x, uint32_t num_warps_z>
__device__ __forceinline__ uint32_t get_warp_idx_z() {
  if constexpr (num_warps_z == 1) {
    return 0;
  } else {
    return threadIdx.z;
  }
}

template <uint32_t num_warps_x, uint32_t num_warps_z>
__device__ __forceinline__ uint32_t get_warp_idx() {
  return get_warp_idx_z<num_warps_x, num_warps_z>() * num_warps_x +
         get_warp_idx_x<num_warps_x, num_warps_z>();
}

template <bool produce_v,
          uint32_t num_warps_x,
          uint32_t num_warps_z,
          uint32_t num_frags_y,
          uint32_t num_frags_z,
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
  constexpr uint32_t head_dim = num_frags_y * 16;
  constexpr uint32_t num_warps = num_warps_x * num_warps_z;
  constexpr uint32_t channel_size_128b_kv =
      head_dim / num_elems_per_128b<DType>();
  const uint32_t warp_idx = get_warp_idx<num_warps_x, num_warps_z>(),
                 lane_idx = threadIdx.x;
  if constexpr (swizzle_mode == SwizzleMode::k128B) {
    uint32_t kv_idx = kv_idx_base + warp_idx * 4 + lane_idx / 8;
    // NOTE(Zihao): num_frags_z * 4 / num_warps_x = num_warps_z * num_frags_z *
    // 4 / num_warps
    static_assert(num_frags_z * 4 % num_warps_x == 0);
#pragma unroll
    for (uint32_t i = 0; i < num_frags_z * 4 / num_warps_x; ++i) {
      DType* gptr = produce_v ? paged_kv.v_data(kv_offset[i])
                              : paged_kv.k_data(kv_offset[i]);
#pragma unroll
      for (uint32_t j = 0; j < num_frags_y / (8 / sizeof(DType)); ++j) {
        smem.load_128b_async<fill_mode>(*smem_offset, gptr, kv_idx < kv_len);
        *smem_offset =
            smem.template advance_offset_by_column<8>(*smem_offset, j);
        gptr += 8 * num_elems_per_128b<DType>();
      }
      kv_idx += num_warps * 4;
      *smem_offset = smem.template advance_offset_by_row<num_warps * 4,
                                                         channel_size_128b_kv>(
                         *smem_offset) -
                     sizeof(DType) * num_frags_y;
    }
    *smem_offset -= num_warps_z * num_frags_z * 16 * channel_size_128b_kv;
  } else {
    uint32_t kv_idx = kv_idx_base + warp_idx * 8 + lane_idx / 4;
    // NOTE(Zihao): num_frags_z * 2 / num_warps_x = num_warps_z * num_frags_z *
    // 2 / num_warps
    static_assert(num_frags_z * 2 % num_warps_x == 0);
#pragma unroll
    for (uint32_t i = 0; i < num_frags_z * 2 / num_warps_x; ++i) {
      DType* gptr = produce_v ? paged_kv.v_data(kv_offset[i])
                              : paged_kv.k_data(kv_offset[i]);
      smem.load_128b_async<fill_mode>(*smem_offset, gptr, kv_idx < kv_len);
      kv_idx += num_warps * 8;
      *smem_offset = smem.template advance_offset_by_row<num_warps * 8,
                                                         channel_size_128b_kv>(
          *smem_offset);
    }
    *smem_offset -= num_warps_z * num_frags_z * 16 * channel_size_128b_kv;
  }
}

template <uint32_t num_frags_x, uint32_t num_frags_y, typename DTypeQKAccum>
__device__ __forceinline__ void init_states(float (*o_frag)[num_frags_y][8],
                                            DTypeQKAccum (*m)[2],
                                            float (*d)[2]) {
#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
#pragma unroll
      for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
        o_frag[fx][fy][reg_id] = 0.f;
      }
    }
  }
#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      m[fx][j] = DTypeQKAccum(-5e4);
      d[fx][j] = 1.f;
    }
  }
}

template <uint32_t num_warps_x,
          uint32_t num_warps_z,
          uint32_t num_frags_x,
          uint32_t num_frags_y,
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
  constexpr uint32_t head_dim = num_frags_y * 16;
  constexpr uint32_t channel_size_128b_q =
      head_dim / num_elems_per_128b<DTypeQ>();
  const uint32_t lane_idx = threadIdx.x,
                 warp_idx_x = get_warp_idx_x<num_warps_x, num_warps_z>();

  if (get_warp_idx_z<num_warps_x, num_warps_z>() == 0) {
    uint32_t q_smem_offset_w = q_smem->get_permuted_offset<channel_size_128b_q>(
        warp_idx_x * num_frags_x * 16 + lane_idx / 8, lane_idx % 8);

#pragma unroll
    for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
      for (uint32_t j = 0; j < 4; ++j) {
        uint32_t q, r;
        group_size.divmod(packed_offset + lane_idx / 8 + fx * 16 + j * 4, q, r);
        const uint32_t q_idx = q;
        DTypeQ* q_ptr = q_ptr_base + q * q_stride_n + r * q_stride_h;
#pragma unroll
        for (uint32_t fyo = 0; fyo < num_frags_y / 4; ++fyo) {
          // load q fragment from gmem to smem
          q_smem->load_128b_async<SharedMemFillMode::kNoFill>(
              q_smem_offset_w, q_ptr, q_idx < qo_upper_bound);
          q_smem_offset_w = q_smem->template advance_offset_by_column<8>(
              q_smem_offset_w, fyo);
          q_ptr += 8 * num_elems_per_128b<DTypeQ>();
        }
        q_smem_offset_w =
            q_smem->template advance_offset_by_row<4, channel_size_128b_q>(
                q_smem_offset_w) -
            2 * num_frags_y;
      }
    }
  }
}

template <uint32_t num_warps_x,
          uint32_t num_warps_z,
          uint32_t num_frags_x,
          uint32_t num_frags_y,
          SwizzleMode swizzle_mode,
          typename DTypeQ>
__device__ __forceinline__ void q_smem_inplace_multiply_sm_scale(
    smem_t<swizzle_mode>* q_smem,
    const float sm_scale) {
  const uint32_t warp_idx = get_warp_idx<num_warps_x, num_warps_z>(),
                 lane_idx = threadIdx.x;
  constexpr uint32_t head_dim = num_frags_y * 16;
  constexpr uint32_t channel_size_128b_q =
      head_dim / num_elems_per_128b<DTypeQ>();
  constexpr uint32_t num_warps = num_warps_x * num_warps_z;
#pragma unroll
  for (uint32_t i = 0; i < num_frags_x * head_dim / (num_warps_z * 16); ++i) {
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
          uint32_t num_frags_x,
          uint32_t num_frags_y,
          uint32_t num_frags_z,
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
    DTypeQKAccum (*s_frag)[num_frags_z][8],
    const float soft_cap) {
  constexpr uint32_t head_dim = num_frags_y * 16;
  constexpr uint32_t channel_size_128b_q =
      head_dim / num_elems_per_128b<DTypeQ>();
  constexpr uint32_t channel_size_128b_kv =
      head_dim / num_elems_per_128b<DTypeKV>();
  uint32_t a_frag[num_frags_x][4], b_frag[4];
  // compute q*k^T
#pragma unroll
  for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
#pragma unroll
    for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
      q_smem->ldmatrix_m8n8x4(*q_smem_offset_r, a_frag[fx]);
      *q_smem_offset_r =
          q_smem->template advance_offset_by_row<16, channel_size_128b_q>(
              *q_smem_offset_r);
    }

    *q_smem_offset_r =
        q_smem->template advance_offset_by_column<2>(*q_smem_offset_r, fy) -
        num_frags_x * 16 * channel_size_128b_q;

#pragma unroll
    for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
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
      for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
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
      *k_smem_offset_r -= num_frags_z * 16 * channel_size_128b_kv;
    } else {
      *k_smem_offset_r =
          k_smem->template advance_offset_by_column<2>(*k_smem_offset_r, fy) -
          num_frags_z * 16 * channel_size_128b_kv;
    }
  }
  *q_smem_offset_r -= num_frags_y * 2;
  *k_smem_offset_r -= num_frags_y * sizeof(DTypeKV);

  if constexpr (std::is_same<DTypeQKAccum, float>::value) {
#pragma unroll
    for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
      for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
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
    for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
      for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
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
template <uint32_t num_frags_x, uint32_t num_frags_z, typename T>
__device__ __forceinline__ void apply_alibi_bias(
    const uint32_t qo_packed_idx_base,
    const uint32_t kv_idx_base,
    const int32_t q_offset,
    const uint_fastdiv group_size,
    float (*alibi_slope)[2],
    T (*s_frag)[num_frags_z][8]) {
  const int32_t lane_idx = threadIdx.x;
#pragma unroll
  for (int32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (int32_t fz = 0; fz < num_frags_z; ++fz) {
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
          uint32_t num_frags_x,
          uint32_t num_frags_y,
          uint32_t num_frags_z,
          typename DTypeQKAccum>
__device__ __forceinline__ void mask_s(const uint32_t qo_packed_idx_base,
                                       const uint32_t kv_idx_base,
                                       const uint32_t qo_len,
                                       const uint32_t kv_len,
                                       const uint32_t window_left,
                                       const uint32_t chunk_end,
                                       const uint_fastdiv group_size,
                                       uint8_t* custom_mask,
                                       DTypeQKAccum (*s_frag)[num_frags_z][8]) {
  const uint32_t lane_idx = threadIdx.x;
#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
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

template <uint32_t num_frags_x,
          uint32_t num_frags_y,
          uint32_t num_frags_z,
          typename DTypeQKAccum>
__device__ __forceinline__ void update_mdo_states(
    DTypeQKAccum (*s_frag)[num_frags_z][8],
    float (*o_frag)[num_frags_y][8],
    DTypeQKAccum (*m)[2],
    float (*d)[2]) {
  if constexpr (std::is_same_v<DTypeQKAccum, float>) {
#pragma unroll
    for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
      for (uint32_t j = 0; j < 2; ++j) {
        float m_prev = m[fx][j];
#pragma unroll
        for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
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
        for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
          o_frag[fx][fy][j * 2 + 0] *= o_scale;
          o_frag[fx][fy][j * 2 + 1] *= o_scale;
          o_frag[fx][fy][j * 2 + 4] *= o_scale;
          o_frag[fx][fy][j * 2 + 5] *= o_scale;
        }
#pragma unroll
        for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
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
    for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
      half m_prev[2];
#pragma unroll
      for (uint32_t j = 0; j < 2; ++j) {
        m_prev[j] = m[fx][j];
#pragma unroll
        for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
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
        for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
          o_frag[fx][fy][j * 2 + 0] *= o_scale;
          o_frag[fx][fy][j * 2 + 1] *= o_scale;
          o_frag[fx][fy][j * 2 + 4] *= o_scale;
          o_frag[fx][fy][j * 2 + 5] *= o_scale;
        }
        half2 m2 = make_half2(m[fx][j], m[fx][j]);
#pragma unroll
        for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
          *(half2*)&s_frag[fx][fz][j * 2] =
              math::ptx_exp2(*(half2*)&s_frag[fx][fz][j * 2] - m2);
          *(half2*)&s_frag[fx][fz][j * 2 + 4] =
              math::ptx_exp2(*(half2*)&s_frag[fx][fz][j * 2 + 4] - m2);
        }
      }
    }
  }
}

template <uint32_t num_frags_x,
          uint32_t num_frags_y,
          uint32_t num_frags_z,
          SwizzleMode swizzle_mode,
          typename DTypeQ,
          typename DTypeKV,
          typename DTypeQKAccum>
__device__ __forceinline__ void compute_sfm_v(
    smem_t<swizzle_mode>* v_smem,
    uint32_t* v_smem_offset_r,
    DTypeQKAccum (*s_frag)[num_frags_z][8],
    float (*o_frag)[num_frags_y][8],
    float (*d)[2]) {
  constexpr uint32_t head_dim = num_frags_y * 16;
  constexpr uint32_t channel_size_128b_kv =
      head_dim / num_elems_per_128b<DTypeKV>();

  DTypeQ s_frag_f16[num_frags_x][num_frags_z][8];
  if constexpr (std::is_same_v<DTypeQKAccum, float>) {
#pragma unroll
    for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
      for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
        vec_cast<DTypeQ, float>::cast<8>(s_frag_f16[fx][fz], s_frag[fx][fz]);
      }
    }
  }

#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
      if constexpr (std::is_same<DTypeQKAccum, float>::value) {
        mma::rowsum_f16f16f32(d[fx], s_frag_f16[fx][fz]);
      } else {
        mma::rowsum_f16f16f32(d[fx], s_frag[fx][fz]);
      }
    }
  }

#pragma unroll
  for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
#pragma unroll
    for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
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
      for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
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
        sizeof(DTypeKV) * num_frags_y;
  }
  *v_smem_offset_r -= 16 * num_frags_z * channel_size_128b_kv;
}

template <uint32_t num_frags_x, uint32_t num_frags_y, typename DTypeQKAccum>
__device__ __forceinline__ void normalize_d(float (*o_frag)[num_frags_y][8],
                                            DTypeQKAccum (*m)[2],
                                            float (*d)[2]) {
  float d_rcp[num_frags_x][2];
  // compute reciprocal of d
#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      d_rcp[fx][j] =
          (m[fx][j] != DTypeQKAccum(-5e4)) ? math::ptx_rcp(d[fx][j]) : 0.f;
    }
  }

#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
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
template <uint32_t num_warps_x,
          uint32_t num_warps_z,
          uint32_t num_frags_x,
          uint32_t num_frags_y,
          typename DTypeQKAccum>
__device__ __forceinline__ void threadblock_sync_mdo_states(
    float (*o_frag)[num_frags_y][8],
    float* smem_workspace,
    DTypeQKAccum (*m)[2],
    float (*d)[2],
    const uint32_t warp_idx,
    const uint32_t lane_idx) {
  // only necessary when blockDim.z > 1
  if constexpr (num_warps_z > 1) {
    float2* smem_md =
        (float2*)(smem_workspace + num_frags_x * num_frags_y * num_warps_x *
                                       num_warps_z * warp_size * 8);
    // o: [num_warps, num_frags_x, num_frags_y, warp_size(32), 8]
    // md: [num_warps, num_frags_x, 2, warp_size(32), 2 (m/d)]
#pragma unroll
    for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
      for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
        vec_t<float, 8>::memcpy(
            smem_workspace +
                (((warp_idx * num_frags_x + fx) * num_frags_y + fy) *
                     warp_size +
                 lane_idx) *
                    8,
            o_frag[fx][fy]);
      }
    }

#pragma unroll
    for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
      for (uint32_t j = 0; j < 2; ++j) {
        smem_md[((warp_idx * num_frags_x + fx) * 2 + j) * warp_size +
                lane_idx] = make_float2(float(m[fx][j]), d[fx][j]);
      }
    }

    // synchronize m,d first
    __syncthreads();
#pragma unroll
    for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
      float o_scale[2][num_warps_z];
#pragma unroll
      for (uint32_t j = 0; j < 2; ++j) {
        float m_new = -5e4, d_new = 1.f;
#pragma unroll
        for (uint32_t i = 0; i < num_warps_z; ++i) {
          float2 md = smem_md[(((i * num_warps_x +
                                 get_warp_idx_x<num_warps_x, num_warps_z>()) *
                                    num_frags_x +
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
        for (uint32_t i = 0; i < num_warps_z; ++i) {
          float2 md = smem_md[(((i * num_warps_x +
                                 get_warp_idx_x<num_warps_x, num_warps_z>()) *
                                    num_frags_x +
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
      for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
        vec_t<float, 8> o_new;
        o_new.fill(0.f);
#pragma unroll
        for (uint32_t i = 0; i < num_warps_z; ++i) {
          vec_t<float, 8> oi;
          oi.load(smem_workspace +
                  ((((i * num_warps_x +
                      get_warp_idx_x<num_warps_x, num_warps_z>()) *
                         num_frags_x +
                     fx) *
                        num_frags_y +
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

template <uint32_t num_warps_x,
          uint32_t num_warps_z,
          uint32_t num_frags_x,
          uint32_t num_frags_y,
          SwizzleMode swizzle_mode,
          typename DTypeOut>
__device__ __forceinline__ void write_o_reg_gmem(
    float (*o_frag)[num_frags_y][8],
    smem_t<swizzle_mode>* o_smem,
    DTypeOut* o_ptr_base,
    const uint32_t o_packed_idx_base,
    const uint32_t qo_upper_bound,
    const uint32_t o_stride_n,
    const uint32_t o_stride_h,
    const uint_fastdiv group_size) {
  constexpr uint32_t head_dim = num_frags_y * 16;
  constexpr uint32_t channel_size_128b_out =
      head_dim / num_elems_per_128b<DTypeOut>();
  const uint32_t warp_idx_x = get_warp_idx_x<num_warps_x, num_warps_z>();
  const uint32_t lane_idx = threadIdx.x;

  if (get_warp_idx_z<num_warps_x, num_warps_z>() == 0) {
#pragma unroll
    for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
      for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
        uint32_t o_frag_f16[4];
        vec_cast<DTypeOut, float>::cast<8>((DTypeOut*)o_frag_f16,
                                           o_frag[fx][fy]);
#ifdef FLASHINFER_STMATRIX_M8N8X4_ENABLED
        uint32_t o_smem_offset_w =
            o_smem->get_permuted_offset<channel_size_128b_out>(
                (warp_idx_x * num_frags_x + fx) * 16 + lane_idx % 16,
                fy * 2 + lane_idx / 16);
        o_smem->stmatrix_m8n8x4(o_smem_offset_w, o_frag_f16);
#else
        uint32_t o_smem_offset_w =
            o_smem->get_permuted_offset<channel_size_128b_out>(
                (warp_idx_x * num_frags_x + fx) * 16 + lane_idx / 4, fy * 2);
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
            warp_idx_x * num_frags_x * 16 + lane_idx / 8, lane_idx % 8);

#pragma unroll
    for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
      for (uint32_t j = 0; j < 4; ++j) {
        uint32_t q, r;
        group_size.divmod(
            o_packed_idx_base + lane_idx / 8 + fx * 16 + j * 4, q, r);
        const uint32_t o_idx = q;
        DTypeOut* o_ptr = o_ptr_base + q * o_stride_n + r * o_stride_h;
#pragma unroll
        for (uint32_t fyo = 0; fyo < num_frags_y / 4; ++fyo) {
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
            2 * num_frags_y;
      }
    }
  }
}

}  // namespace

template <LogitsPostHook logits_post_hook,
          MaskMode mask_mode,
          PosEncodingMode pos_encoding_mode,
          uint32_t num_frags_x,
          uint32_t num_frags_y,
          uint32_t num_frags_z,
          uint32_t num_warps_x,
          uint32_t num_warps_z,
          typename DTypeQ,
          typename DTypeKV,
          typename DTypeQKAccum,
          typename DTypeOut,
          typename IdType>
__global__
__launch_bounds__(num_warps_x* num_warps_z* warp_size) void attention_kernel(
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
  sm_scale *= (logits_post_hook == LogitsPostHook::kNone
                   ? math::log2e
                   : math::ptx_rcp(logits_soft_cap));
  auto block = cg::this_thread_block();
  const uint32_t kv_chunk_size = *kv_chunk_size_ptr;

  const uint32_t bx = blockIdx.x, lane_idx = threadIdx.x,
                 warp_idx = get_warp_idx<num_warps_x, num_warps_z>(),
                 kv_head_idx = blockIdx.z;
  if (block_valid_mask && !block_valid_mask[bx]) {
    return;
  }
  const uint32_t num_kv_heads = gridDim.z,
                 num_qo_heads = num_kv_heads * group_size;
  float alibi_slopes_frag[num_frags_x][2];

  const uint32_t request_idx = request_indices[bx],
                 qo_tile_idx = q_tile_indices[bx],
                 kv_tile_idx = kv_tile_indices[bx];
  constexpr uint32_t num_rows_per_cta = num_frags_x * num_warps_x * 16;
  const uint32_t qo_len = q_indptr[request_idx + 1] - q_indptr[request_idx];
  const uint32_t kv_len = kv_indptr[request_idx + 1] - kv_indptr[request_idx];

  const uint32_t kv_len_safe = kv_len > 0 ? kv_len : 1;
  const uint32_t window_left =
      (maybe_window_left >= 0) ? maybe_window_left : kv_len;
  const uint32_t max_chunk_size = partition_kv ? kv_chunk_size : kv_len;
  const uint32_t chunk_start = partition_kv ? kv_tile_idx * max_chunk_size : 0;
  const uint32_t chunk_end =
      partition_kv ? min((kv_tile_idx + 1) * max_chunk_size, kv_len) : kv_len;
  const uint32_t chunk_size = chunk_end - chunk_start;
  const uint32_t qo_upper_bound =
      min(qo_len, ceil_div((qo_tile_idx + 1) * num_rows_per_cta, group_size));

  constexpr uint32_t head_dim = num_frags_y * 16;
  constexpr uint32_t channel_size_128b_q =
      head_dim / num_elems_per_128b<DTypeQ>();
  constexpr uint32_t channel_size_128b_kv =
      head_dim / num_elems_per_128b<DTypeKV>();
  constexpr uint32_t channel_size_128b_out =
      head_dim / num_elems_per_128b<DTypeOut>();

  extern __shared__ uint8_t smem[];

  DTypeQKAccum s_frag[num_frags_x][num_frags_z][8];
  float o_frag[num_frags_x][num_frags_y][8];
  DTypeQKAccum m[num_frags_x][2];
  float d[num_frags_x][2];

  init_states<num_frags_x, num_frags_y>(o_frag, m, d);

  const uint32_t qo_packed_idx_base =
      (qo_tile_idx * num_warps_x + get_warp_idx_x<num_warps_x, num_warps_z>()) *
      num_frags_x * 16;
  const uint32_t q_stride_n = num_qo_heads * head_dim, q_stride_h = head_dim;
  constexpr SwizzleMode swizzle_mode_q = SwizzleMode::k128B;
  smem_t<swizzle_mode_q> qo_smem(smem);
  DTypeQ* q_ptr_base =
      q + get_elem_offset_impl(q_indptr[request_idx],
                               kv_head_idx * group_size,
                               (lane_idx % 8) * num_elems_per_128b<DTypeQ>(),
                               q_stride_n,
                               q_stride_h);
  DTypeOut* o_ptr_base =
      partition_kv ? o + kv_tile_idx * num_qo_heads * head_dim +
                         get_elem_offset_impl(
                             o_indptr[request_idx],
                             kv_head_idx * group_size,
                             (lane_idx % 8) * num_elems_per_128b<DTypeOut>(),
                             num_qo_heads * head_dim,
                             head_dim)
                   : o + get_elem_offset_impl(
                             o_indptr[request_idx],
                             kv_head_idx * group_size,
                             (lane_idx % 8) * num_elems_per_128b<DTypeOut>(),
                             num_qo_heads * head_dim,
                             head_dim);
  uint32_t q_smem_offset_r = qo_smem.get_permuted_offset<channel_size_128b_q>(
      get_warp_idx_x<num_warps_x, num_warps_z>() * num_frags_x * 16 +
          lane_idx % 16,
      lane_idx / 16);

  load_q_global_smem<num_warps_x, num_warps_z, num_frags_x, num_frags_y>(
      qo_packed_idx_base,
      qo_upper_bound,
      q_ptr_base,
      q_stride_n,
      q_stride_h,
      group_size,
      &qo_smem);

  cp_async::commit_group();
  cp_async::wait_group<0>();
  block.sync();

  q_smem_inplace_multiply_sm_scale<num_warps_x,
                                   num_warps_z,
                                   num_frags_x,
                                   num_frags_y,
                                   swizzle_mode_q,
                                   DTypeQ>(&qo_smem, sm_scale);

  if constexpr (pos_encoding_mode == PosEncodingMode::kALiBi) {
#pragma unroll
    for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
      for (uint32_t j = 0; j < 2; ++j) {
        const uint32_t qo_head_idx =
            kv_head_idx * group_size +
            (qo_packed_idx_base + lane_idx / 4 + j * 8 + fx * 16) % group_size;
        alibi_slopes_frag[fx][j] = alibi_slopes[qo_head_idx] * math::log2e;
      }
    }
  }

  constexpr SwizzleMode swizzle_mode_kv =
      (sizeof(DTypeKV) == 1 && head_dim == 64) ? SwizzleMode::k64B
                                               : SwizzleMode::k128B;
  constexpr uint32_t kv_frag_rows =
      swizzle_mode_kv == SwizzleMode::k128B ? 4 : 8;
  constexpr uint32_t kv_frag_cols =
      swizzle_mode_kv == SwizzleMode::k128B ? 8 : 4;
  smem_t<swizzle_mode_kv> k_smem(
      smem + (num_warps_x * num_frags_x * sizeof(DTypeQ)) * 16 * head_dim),
      v_smem(smem + (num_warps_x * num_frags_x * sizeof(DTypeQ) +
                     num_warps_z * num_frags_z * sizeof(DTypeKV)) *
                        16 * head_dim);
  size_t kv_offset[num_frags_z *
                   (swizzle_mode_kv == SwizzleMode::k128B ? 4 : 2) /
                   num_warps_x];

  uint32_t k_smem_offset_r = k_smem.get_permuted_offset<channel_size_128b_kv>(
               get_warp_idx_z<num_warps_x, num_warps_z>() * num_frags_z * 16 +
                   8 * (lane_idx / 16) + lane_idx % 8,
               (lane_idx % 16) / 8),
           v_smem_offset_r = v_smem.get_permuted_offset<channel_size_128b_kv>(
               get_warp_idx_z<num_warps_x, num_warps_z>() * num_frags_z * 16 +
                   lane_idx % 16,
               lane_idx / 16),
           kv_smem_offset_w = k_smem.get_permuted_offset<channel_size_128b_kv>(
               warp_idx * kv_frag_rows + lane_idx / kv_frag_cols,
               lane_idx % kv_frag_cols);

  // kv_idx of current sequence
  uint32_t kv_idx_base = chunk_start;

#pragma unroll
  for (uint32_t i = 0;
       i < num_frags_z * (swizzle_mode_kv == SwizzleMode::k128B ? 4 : 2) /
               num_warps_x;
       ++i) {
    const uint32_t kv_idx = kv_idx_base + warp_idx * kv_frag_rows +
                            lane_idx / kv_frag_cols +
                            kv_frag_rows * num_warps_x * num_warps_z * i;
    const uint32_t feat_idx =
        (lane_idx % kv_frag_cols) * num_elems_per_128b<DTypeKV>();
    kv_offset[i] =
        kv_idx < kv_len
            ? paged_kv.get_kv_offset(request_idx, kv_idx, kv_head_idx, feat_idx)
            : 0;
  }

  page_produce_kv<false, num_warps_x, num_warps_z, num_frags_y, num_frags_z>(
      k_smem, &kv_smem_offset_w, paged_kv, 0, kv_offset, chunk_size);
  cp_async::commit_group();
  page_produce_kv<true, num_warps_x, num_warps_z, num_frags_y, num_frags_z>(
      v_smem, &kv_smem_offset_w, paged_kv, 0, kv_offset, chunk_size);
  cp_async::commit_group();

  const uint32_t num_iterations = ceil_div(
      (mask_mode == MaskMode::kCausal
           ? min(chunk_size,
                 sub_if_greater_or_zero(
                     kv_len - qo_len +
                         ((qo_tile_idx + 1) * num_rows_per_cta) / group_size,
                     chunk_start))
           : chunk_size),
      16 * num_warps_z * num_frags_z);

  const uint32_t window_iteration =
      ceil_div(sub_if_greater_or_zero(kv_len + (bx + 1) * num_rows_per_cta,
                                      qo_len + window_left + chunk_start),
               (16 * num_warps_z * num_frags_z));

  const uint32_t mask_iteration =
      (mask_mode == MaskMode::kCausal
           ? min(chunk_size,
                 sub_if_greater_or_zero(
                     kv_len + (qo_tile_idx * num_rows_per_cta) / group_size -
                         qo_len,
                     chunk_start))
           : chunk_size) /
      (16 * num_warps_z * num_frags_z);

#pragma unroll 1
  for (uint32_t iter = 0; iter < num_iterations; ++iter) {
    kv_idx_base += 16 * num_warps_z * num_frags_z;
#pragma unroll
    for (uint32_t i = 0;
         i < num_frags_z * (swizzle_mode_kv == SwizzleMode::k128B ? 4 : 2) /
                 num_warps_x;
         ++i) {
      const uint32_t kv_idx = kv_idx_base + warp_idx * kv_frag_rows +
                              lane_idx / kv_frag_cols +
                              kv_frag_rows * num_warps_x * num_warps_z * i;
      const uint32_t feat_idx =
          (lane_idx % kv_frag_cols) * num_elems_per_128b<DTypeKV>();
      kv_offset[i] = kv_idx < kv_len
                         ? paged_kv.get_kv_offset(
                               request_idx, kv_idx, kv_head_idx, feat_idx)
                         : 0;
    }
    cp_async::wait_group<1>();
    block.sync();

    // compute attention score
    compute_qk<logits_post_hook,
               num_frags_x,
               num_frags_y,
               num_frags_z,
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
      apply_alibi_bias<num_frags_x, num_frags_z>(
          qo_packed_idx_base,
          chunk_start + (iter * num_warps_z +
                         get_warp_idx_z<num_warps_x, num_warps_z>()) *
                            num_frags_z * 16,
          int(kv_len) - int(qo_len),
          group_size,
          alibi_slopes_frag,
          s_frag);
    }
    // apply mask
    if constexpr (mask_mode == MaskMode::kCustom) {
      mask_s<mask_mode, num_frags_x, num_frags_y, num_frags_z>(
          qo_packed_idx_base,
          chunk_start + (iter * num_warps_z +
                         get_warp_idx_z<num_warps_x, num_warps_z>()) *
                            num_frags_z * 16,
          qo_len,
          kv_len,
          window_left,
          chunk_end,
          group_size,
          custom_mask + qk_indptr[request_idx],
          s_frag);
    } else {
      if (iter >= mask_iteration || iter < window_iteration) {
        mask_s<mask_mode, num_frags_x, num_frags_y, num_frags_z>(
            qo_packed_idx_base,
            chunk_start + (iter * num_warps_z +
                           get_warp_idx_z<num_warps_x, num_warps_z>()) *
                              num_frags_z * 16,
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
    update_mdo_states<num_frags_x, num_frags_y, num_frags_z>(
        s_frag, o_frag, m, d);

    block.sync();
    page_produce_kv<false, num_warps_x, num_warps_z, num_frags_y, num_frags_z>(
        k_smem,
        &kv_smem_offset_w,
        paged_kv,
        (iter + 1) * 16 * num_warps_z * num_frags_z,
        kv_offset,
        chunk_size);
    cp_async::commit_group();
    cp_async::wait_group<1>();
    block.sync();

    // compute sfm*v
    compute_sfm_v<num_frags_x,
                  num_frags_y,
                  num_frags_z,
                  swizzle_mode_kv,
                  DTypeQ,
                  DTypeKV>(&v_smem, &v_smem_offset_r, s_frag, o_frag, d);

    block.sync();
    page_produce_kv<true, num_warps_x, num_warps_z, num_frags_y, num_frags_z>(
        v_smem,
        &kv_smem_offset_w,
        paged_kv,
        (iter + 1) * 16 * num_warps_z * num_frags_z,
        kv_offset,
        chunk_size);
    cp_async::commit_group();
  }
  cp_async::wait_group<0>();
  block.sync();

  // threadblock synchronization
  threadblock_sync_mdo_states<num_warps_x,
                              num_warps_z,
                              num_frags_x,
                              num_frags_y,
                              DTypeQKAccum>(
      o_frag, (float*)smem, m, d, warp_idx, lane_idx);

  // normalize d
  normalize_d<num_frags_x, num_frags_y>(o_frag, m, d);

  const uint32_t num_kv_chunks =
      (kv_len_safe + kv_chunk_size - 1) / kv_chunk_size;

  // write_back
  write_o_reg_gmem<num_warps_x, num_warps_z, num_frags_x, num_frags_y>(
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
    if (get_warp_idx_z<num_warps_x, num_warps_z>() == 0) {
#pragma unroll
      for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
        for (uint32_t j = 0; j < 2; ++j) {
          uint32_t q, r;
          group_size.divmod(
              qo_packed_idx_base + lane_idx / 4 + j * 8 + fx * 16, q, r);
          const uint32_t qo_head_idx = kv_head_idx * group_size + r;
          const uint32_t qo_idx = q;
          if (qo_idx < qo_upper_bound) {
            if (partition_kv) {
              lse[(o_indptr[request_idx] + qo_idx * num_kv_chunks +
                   kv_tile_idx) *
                      num_qo_heads +
                  qo_head_idx] = math::ptx_log2(d[fx][j]) + float(m[fx][j]);
            } else {
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

  constexpr uint32_t num_frags_x = get_num_frags_x<WARP_LAYOUT>();
  constexpr uint32_t num_warps_x = get_num_warps_x<WARP_LAYOUT>();
  constexpr uint32_t num_warps_z = get_num_warps_z<WARP_LAYOUT>();
  const uint32_t group_size = num_qo_heads / num_kv_heads;
  const uint_fastdiv group_size_fastdiv(group_size);

  if (padded_batch_size == 0) {
    // No request, skip
    // this won't happen in CUDAGraph mode because we fixed the
    // padded_batch_size
    return cudaSuccess;
  }

  dim3 nblks(padded_batch_size, 1, num_kv_heads);
  dim3 nthrs(32, num_warps_x, num_warps_z);

  constexpr uint32_t num_frags_y = HEAD_DIM / 16;
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

  const uint32_t max_num_frags_z_reg =
      (HEAD_DIM >= 128 && num_frags_x == 2 &&
       pos_encoding_mode == PosEncodingMode::kRoPELlama &&
       !ALLOW_FP16_QK_REDUCTION)
          ? 2
          : (8 / num_frags_x);
  // TODO(Zihao): fix the following computation
  const uint32_t max_num_frags_z_smem =
      (max_smem_per_threadblock / (16 * HEAD_DIM * sizeof(DTypeQ)) -
       num_frags_x * num_warps_x) /
      (2 * num_warps_z);

  DISPATCH_NUM_FRAGS_Z(
      min(max_num_frags_z_smem, max_num_frags_z_reg), num_frags_z, {
        if constexpr (is_invalid_configuration<pos_encoding_mode,
                                               DTypeKV,
                                               DTypeQKAccum>(num_frags_x,
                                                             num_frags_y,
                                                             num_frags_z,
                                                             num_warps_x,
                                                             num_warps_z)) {
          // Invalid configuration, skip
          std::ostringstream err_msg;
          err_msg << "FlashInfer Internal Error: Invalid configuration : "
                     "num_frags_x="
                  << num_frags_x << " num_frags_y=" << num_frags_y
                  << " num_frags_z=" << num_frags_z
                  << " num_warps_x=" << num_warps_x
                  << " num_warps_z=" << num_warps_z
                  << " please create an issue "
                     "(https://github.com/flashinfer-ai/flashinfer/issues)"
                     " and report the issue to the developers.";
          throw std::invalid_argument(err_msg.str());
        } else {
          // TODO(Zihao): fix the following computation
          uint32_t smem_size =
              (num_frags_x * num_warps_x * sizeof(DTypeQ) +
               num_frags_z * num_warps_z * 2 * sizeof(DTypeQ)) *
              16 * HEAD_DIM;
          auto kernel = attention_kernel<LOGITS_POST_HOOK,
                                         MASK_MODE,
                                         pos_encoding_mode,
                                         num_frags_x,
                                         num_frags_y,
                                         num_frags_z,
                                         num_warps_x,
                                         num_warps_z,
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
