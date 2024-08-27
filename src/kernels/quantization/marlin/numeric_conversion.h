#pragma once

#include "memory.h"
#include "scale_type.h"

namespace marlin {

// Efficiently dequantize an int32 value into a full B-fragment of 4 fp16
// values. We mostly follow the strategy in the link below, with some small
// changes:
// - FP16:
// https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h#L215-L287
// - BF16:
// https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h#L327-L385
template <typename scalar_t>
__device__ inline typename ScalarType<scalar_t>::FragB dequant_4bit(int q) {
  STATIC_ASSERT_SCALAR_TYPE_VALID(scalar_t);
}

template <>
__device__ inline typename ScalarType<half>::FragB dequant_4bit<half>(int q) {
  const int LO = 0x000f000f;
  const int HI = 0x00f000f0;
  const int EX = 0x64006400;
  // Guarantee that the `(a & b) | c` operations are LOP3s.
  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);
  // We want signed int4 outputs, hence we fuse the `-8` symmetric zero point
  // directly into `SUB` and `ADD`.
  const int SUB = 0x64086408;
  const int MUL = 0x2c002c00;
  const int ADD = 0xd480d480;
  typename ScalarType<half>::FragB frag_b;
  frag_b[0] = __hsub2(*reinterpret_cast<half2*>(&lo),
                      *reinterpret_cast<const half2*>(&SUB));
  frag_b[1] = __hfma2(*reinterpret_cast<half2*>(&hi),
                      *reinterpret_cast<const half2*>(&MUL),
                      *reinterpret_cast<const half2*>(&ADD));
  return frag_b;
}

template <>
__device__ inline typename ScalarType<nv_bfloat16>::FragB
dequant_4bit<nv_bfloat16>(int q) {
  static constexpr uint32_t MASK = 0x000f000f;
  static constexpr uint32_t EX = 0x43004300;

  // Guarantee that the `(a & b) | c` operations are LOP3s.

  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);
  q >>= 4;
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);

  typename ScalarType<nv_bfloat16>::FragB frag_b;
  static constexpr uint32_t MUL = 0x3F803F80;
  static constexpr uint32_t ADD = 0xC308C308;

  frag_b[0] = __hfma2(*reinterpret_cast<nv_bfloat162*>(&lo),
                      *reinterpret_cast<const nv_bfloat162*>(&MUL),
                      *reinterpret_cast<const nv_bfloat162*>(&ADD));
  frag_b[1] = __hfma2(*reinterpret_cast<nv_bfloat162*>(&hi),
                      *reinterpret_cast<const nv_bfloat162*>(&MUL),
                      *reinterpret_cast<const nv_bfloat162*>(&ADD));
  return frag_b;
}

// Fast Int8ToFp16/Int8ToBf16: Efficiently dequantize 8bit int values to fp16 or
// bf16 Reference:
// - FP16:
// https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h#L53-L85
// - BF16:
// https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h#L125-L175
template <typename scalar_t>
__device__ inline typename ScalarType<scalar_t>::FragB dequant_8bit(int q) {
  STATIC_ASSERT_SCALAR_TYPE_VALID(scalar_t);
}

template <>
__device__ inline typename ScalarType<half>::FragB dequant_8bit<half>(int q) {
  static constexpr uint32_t mask_for_elt_01 = 0x5250;
  static constexpr uint32_t mask_for_elt_23 = 0x5351;
  static constexpr uint32_t start_byte_for_fp16 = 0x64646464;

  uint32_t lo = prmt<start_byte_for_fp16, mask_for_elt_01>(q);
  uint32_t hi = prmt<start_byte_for_fp16, mask_for_elt_23>(q);

  static constexpr uint32_t I8s_TO_F16s_MAGIC_NUM = 0x64806480;

  typename ScalarType<half>::FragB frag_b;
  frag_b[0] = __hsub2(*reinterpret_cast<half2*>(&lo),
                      *reinterpret_cast<const half2*>(&I8s_TO_F16s_MAGIC_NUM));
  frag_b[1] = __hsub2(*reinterpret_cast<half2*>(&hi),
                      *reinterpret_cast<const half2*>(&I8s_TO_F16s_MAGIC_NUM));
  return frag_b;
}

template <>
__device__ inline typename ScalarType<nv_bfloat16>::FragB
dequant_8bit<nv_bfloat16>(int q) {
  typename ScalarType<nv_bfloat16>::FragB frag_b;

  float fp32_intermediates[4];
  uint32_t* fp32_intermediates_casted =
      reinterpret_cast<uint32_t*>(fp32_intermediates);

  static constexpr uint32_t fp32_base = 0x4B000000;
  fp32_intermediates_casted[0] = __byte_perm(q, fp32_base, 0x7650);
  fp32_intermediates_casted[1] = __byte_perm(q, fp32_base, 0x7652);
  fp32_intermediates_casted[2] = __byte_perm(q, fp32_base, 0x7651);
  fp32_intermediates_casted[3] = __byte_perm(q, fp32_base, 0x7653);

  fp32_intermediates[0] -= 8388736.f;
  fp32_intermediates[1] -= 8388736.f;
  fp32_intermediates[2] -= 8388736.f;
  fp32_intermediates[3] -= 8388736.f;

  uint32_t* bf16_result_ptr = reinterpret_cast<uint32_t*>(&frag_b);
  bf16_result_ptr[0] = __byte_perm(
      fp32_intermediates_casted[0], fp32_intermediates_casted[1], 0x7632);
  bf16_result_ptr[1] = __byte_perm(
      fp32_intermediates_casted[2], fp32_intermediates_casted[3], 0x7632);

  return frag_b;
}

// Zero-point dequantizers

template <typename scalar_t>
__device__ inline typename ScalarType<scalar_t>::FragB dequant_4bit_zp(int q) {
  STATIC_ASSERT_SCALAR_TYPE_VALID(scalar_t);
}

template <>
__device__ inline typename ScalarType<half>::FragB dequant_4bit_zp<half>(
    int q) {
  const int LO = 0x000f000f;
  const int HI = 0x00f000f0;
  const int EX = 0x64006400;
  // Guarantee that the `(a & b) | c` operations are LOP3s.
  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);

  const int SUB = 0x64006400;
  const int MUL = 0x2c002c00;
  const int ADD = 0xd400d400;
  typename ScalarType<half>::FragB frag_b;
  frag_b[0] = __hsub2(*reinterpret_cast<half2*>(&lo),
                      *reinterpret_cast<const half2*>(&SUB));
  frag_b[1] = __hfma2(*reinterpret_cast<half2*>(&hi),
                      *reinterpret_cast<const half2*>(&MUL),
                      *reinterpret_cast<const half2*>(&ADD));
  return frag_b;
}

template <>
__device__ inline typename ScalarType<nv_bfloat16>::FragB
dequant_4bit_zp<nv_bfloat16>(int q) {
  static constexpr uint32_t MASK = 0x000f000f;
  static constexpr uint32_t EX = 0x43004300;

  // Guarantee that the `(a & b) | c` operations are LOP3s.

  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);
  q >>= 4;
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);

  typename ScalarType<nv_bfloat16>::FragB frag_b;
  static constexpr uint32_t MUL = 0x3F803F80;
  static constexpr uint32_t ADD = 0xC300C300;

  frag_b[0] = __hfma2(*reinterpret_cast<nv_bfloat162*>(&lo),
                      *reinterpret_cast<const nv_bfloat162*>(&MUL),
                      *reinterpret_cast<const nv_bfloat162*>(&ADD));
  frag_b[1] = __hfma2(*reinterpret_cast<nv_bfloat162*>(&hi),
                      *reinterpret_cast<const nv_bfloat162*>(&MUL),
                      *reinterpret_cast<const nv_bfloat162*>(&ADD));
  return frag_b;
}

template <typename scalar_t>
__device__ inline typename ScalarType<scalar_t>::FragB dequant_8bit_zp(int q) {
  STATIC_ASSERT_SCALAR_TYPE_VALID(scalar_t);
}

template <>
__device__ inline typename ScalarType<half>::FragB dequant_8bit_zp<half>(
    int q) {
  static constexpr uint32_t mask_for_elt_01 = 0x5250;
  static constexpr uint32_t mask_for_elt_23 = 0x5351;
  static constexpr uint32_t start_byte_for_fp16 = 0x64646464;

  uint32_t lo = prmt<start_byte_for_fp16, mask_for_elt_01>(q);
  uint32_t hi = prmt<start_byte_for_fp16, mask_for_elt_23>(q);

  static constexpr uint32_t I8s_TO_F16s_MAGIC_NUM = 0x64006400;

  typename ScalarType<half>::FragB frag_b;
  frag_b[0] = __hsub2(*reinterpret_cast<half2*>(&lo),
                      *reinterpret_cast<const half2*>(&I8s_TO_F16s_MAGIC_NUM));
  frag_b[1] = __hsub2(*reinterpret_cast<half2*>(&hi),
                      *reinterpret_cast<const half2*>(&I8s_TO_F16s_MAGIC_NUM));
  return frag_b;
}

template <>
__device__ inline typename ScalarType<nv_bfloat16>::FragB
dequant_8bit_zp<nv_bfloat16>(int q) {
  typename ScalarType<nv_bfloat16>::FragB frag_b;

  float fp32_intermediates[4];
  uint32_t* fp32_intermediates_casted =
      reinterpret_cast<uint32_t*>(fp32_intermediates);

  static constexpr uint32_t fp32_base = 0x4B000000;
  fp32_intermediates_casted[0] = __byte_perm(q, fp32_base, 0x7650);
  fp32_intermediates_casted[1] = __byte_perm(q, fp32_base, 0x7652);
  fp32_intermediates_casted[2] = __byte_perm(q, fp32_base, 0x7651);
  fp32_intermediates_casted[3] = __byte_perm(q, fp32_base, 0x7653);

  fp32_intermediates[0] -= 8388608.f;
  fp32_intermediates[1] -= 8388608.f;
  fp32_intermediates[2] -= 8388608.f;
  fp32_intermediates[3] -= 8388608.f;

  uint32_t* bf16_result_ptr = reinterpret_cast<uint32_t*>(&frag_b);
  bf16_result_ptr[0] = __byte_perm(
      fp32_intermediates_casted[0], fp32_intermediates_casted[1], 0x7632);
  bf16_result_ptr[1] = __byte_perm(
      fp32_intermediates_casted[2], fp32_intermediates_casted[3], 0x7632);

  return frag_b;
}

// Multiply dequantized values by the corresponding quantization scale; used
// only for grouped quantization.
template <typename scalar_t>
__device__ inline void scale(typename ScalarType<scalar_t>::FragB& frag_b,
                             typename ScalarType<scalar_t>::FragS& frag_s,
                             int i) {
  using scalar_t2 = typename ScalarType<scalar_t>::scalar_t2;
  scalar_t2 s =
      ScalarType<scalar_t>::num2num2(reinterpret_cast<scalar_t*>(&frag_s)[i]);
  frag_b[0] = __hmul2(frag_b[0], s);
  frag_b[1] = __hmul2(frag_b[1], s);
}

template <typename scalar_t>
__device__ inline void sub_zp(typename ScalarType<scalar_t>::FragB& frag_b,
                              typename ScalarType<scalar_t>::scalar_t2& frag_zp,
                              int i) {
  using scalar_t2 = typename ScalarType<scalar_t>::scalar_t2;
  scalar_t2 zp =
      ScalarType<scalar_t>::num2num2(reinterpret_cast<scalar_t*>(&frag_zp)[i]);
  frag_b[0] = __hsub2(frag_b[0], zp);
  frag_b[1] = __hsub2(frag_b[1], zp);
}

// Same as above, but for act_order (each K is multiplied individually)
template <typename scalar_t>
__device__ inline void scale4(typename ScalarType<scalar_t>::FragB& frag_b,
                              typename ScalarType<scalar_t>::FragS& frag_s_1,
                              typename ScalarType<scalar_t>::FragS& frag_s_2,
                              typename ScalarType<scalar_t>::FragS& frag_s_3,
                              typename ScalarType<scalar_t>::FragS& frag_s_4,
                              int i) {
  using scalar_t2 = typename ScalarType<scalar_t>::scalar_t2;
  scalar_t2 s_val_1_2;
  s_val_1_2.x = reinterpret_cast<scalar_t*>(&frag_s_1)[i];
  s_val_1_2.y = reinterpret_cast<scalar_t*>(&frag_s_2)[i];

  scalar_t2 s_val_3_4;
  s_val_3_4.x = reinterpret_cast<scalar_t*>(&frag_s_3)[i];
  s_val_3_4.y = reinterpret_cast<scalar_t*>(&frag_s_4)[i];

  frag_b[0] = __hmul2(frag_b[0], s_val_1_2);
  frag_b[1] = __hmul2(frag_b[1], s_val_3_4);
}

// Given 2 floats multiply by 2 scales (halves)
template <typename scalar_t>
__device__ inline void scale_float(float* c,
                                   typename ScalarType<scalar_t>::FragS& s) {
  scalar_t* s_ptr = reinterpret_cast<scalar_t*>(&s);
  c[0] = __fmul_rn(c[0], ScalarType<scalar_t>::num2float(s_ptr[0]));
  c[1] = __fmul_rn(c[1], ScalarType<scalar_t>::num2float(s_ptr[1]));
}

}  // namespace marlin