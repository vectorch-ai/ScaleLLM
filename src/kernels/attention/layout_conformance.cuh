#pragma once

#include <cuda.h>

#include <cute/config.hpp>
#include <cute/tensor.hpp>

namespace llm {

namespace detail {
using namespace cute;

CUTE_DEVICE void swap(uint32_t& a, uint32_t& b) {
  auto tmp = a;
  a = b;
  b = tmp;
}

// adapted from https://github.com/flashinfer-ai/flashinfer

// T0(  0,  1,  2,  3) => T0( 0,  1,  8,  9)
// T1(  4,  5,  6,  7) => T1( 2,  3, 10, 11)
// T2(  8,  9, 10, 11) => T2( 4,  5, 12, 13)
// T3( 12, 13, 14, 15) => T3( 6,  7, 14, 15)
CUTE_DEVICE uint32_t frag_B_layout_swizzle_8b(uint32_t x, int tidx) {
  uint32_t tmp = __shfl_xor_sync(0xffffffff, x, 0x1);
  x = __byte_perm(x, tmp, ((tidx & 0x1) == 0) ? 0x5410 : 0x3276);
  tmp = __shfl_xor_sync(0xffffffff, x, 0x2);
  x = __byte_perm(x, tmp, ((tidx & 0x2) == 0) ? 0x5410 : 0x3276);
  return x;
}

// T0: (  0,  16,   1,  17) =>  T0( 0,    1, 128, 129)
// T4: ( 32,  48,  33,  49) =>  T4( 16,  17, 144, 145)
// T8: ( 64,  80,  65,  81) =>  T8( 32,  33, 160, 161)
// T12:( 96, 112,  97, 113) => T12( 48,  49, 176, 177)
// T16:(128, 144, 129, 145) => T16( 64,  65, 192, 193)
// T20:(160, 176, 161, 177) => T20( 80,  81, 208, 209)
// T24:(192, 208, 193, 209) => T24( 96,  97, 224, 225)
// T28:(224, 240, 225, 241) => T28(112, 113, 240, 241)
CUTE_DEVICE uint32_t frag_B_trans_layout_swizzle_8b(uint32_t x, int tidx) {
  uint32_t tmp = __shfl_xor_sync(0xffffffff, x, 0x4);
  x = __byte_perm(x, tmp, ((tidx & 0x4) == 0) ? 0x6420 : 0x3175);
  tmp = __shfl_xor_sync(0xffffffff, x, 0x8);
  x = __byte_perm(x, tmp, ((tidx & 0x8) == 0) ? 0x5410 : 0x3276);
  tmp = __shfl_xor_sync(0xffffffff, x, 0x10);
  x = __byte_perm(x, tmp, ((tidx & 0x10) == 0) ? 0x5410 : 0x3276);
  return x;
}
}  // namespace detail

// TODO: arrange elements for one thread together in quatatization stage to
// avoid shfl cost

// frag: (CPY,CPY_N,CPY_K)
template <class FragmentB>
CUTE_DEVICE void frag_B_layout_swizzle(FragmentB& frag, int tidx) {
  // ? not sure if this cast is expensive ?
  auto frag_32 = cute::recast<uint32_t>(frag);
  CUTE_UNROLL
  for (int i = 0; i < size(frag_32); ++i) {
    frag_32[i] = detail::frag_B_layout_swizzle_8b(frag_32[i], tidx);
  }
}

// frag: (CPY,CPY_K,CPY_N)
template <class FragmentB>
CUTE_DEVICE void frag_B_trans_layout_swizzle(FragmentB& frag, int tidx) {
  auto frag_32 = cute::recast<uint32_t>(frag);
  CUTE_UNROLL
  for (int i = 0; i < size(frag_32); ++i) {
    frag_32[i] = detail::frag_B_trans_layout_swizzle_8b(frag_32[i], tidx);
  }

  auto frag_16 = cute::recast<uint16_t>(frag);
  CUTE_UNROLL
  for (int i = 0; i < cute::size(frag_16); i += 4) {
    // swap 16-bit pair: V0, *V1, *V2, V3 => V0, *V2, *V1, V3
    swap(frag_16(i + 1), frag_16(i + 2));
  }
}

}  // namespace llm