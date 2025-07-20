#pragma once
#include <cute/config.hpp>           // cute::CUTE_HOST_DEVICE
#include <cute/pointer_flagged.hpp>  // cute::smem_ptr_flag
#include <cute/swizzle.hpp>          // cute::Swizzle

namespace llm {
using namespace cute;
// clang-format off
namespace detail {
///////////////////////////////////////////
// Common layouts for GMMA Shared Memory //
///////////////////////////////////////////
// K-major GMMA layouts in units of bits
using Layout_K_INTER_Atom_Bits  = ComposedLayout<Swizzle<0,4,3>, smem_ptr_flag, Layout<Shape<_8, _128>,Stride< _128,_1>>>;
using Layout_K_SW32_Atom_Bits   = ComposedLayout<Swizzle<1,4,3>, smem_ptr_flag, Layout<Shape<_8, _256>,Stride< _256,_1>>>;
using Layout_K_SW64_Atom_Bits   = ComposedLayout<Swizzle<2,4,3>, smem_ptr_flag, Layout<Shape<_8, _512>,Stride< _512,_1>>>;
using Layout_K_SW128_Atom_Bits  = ComposedLayout<Swizzle<3,4,3>, smem_ptr_flag, Layout<Shape<_8,_1024>,Stride<_1024,_1>>>;

// K-major layouts in units of Type
template <class Type>
using Layout_K_INTER_Atom = decltype(upcast<sizeof_bits<Type>::value>(Layout_K_INTER_Atom_Bits{}));
template <class Type>
using Layout_K_SW32_Atom  = decltype(upcast<sizeof_bits<Type>::value>(Layout_K_SW32_Atom_Bits{}));
template <class Type>
using Layout_K_SW64_Atom  = decltype(upcast<sizeof_bits<Type>::value>(Layout_K_SW64_Atom_Bits{}));
template <class Type>
using Layout_K_SW128_Atom = decltype(upcast<sizeof_bits<Type>::value>(Layout_K_SW128_Atom_Bits{}));

} // namespace detail

template <class Element, int kBlockK>
CUTE_HOST_DEVICE constexpr auto smem_layout_atom_selector() {
  if constexpr (kBlockK % size<1>(detail::Layout_K_SW128_Atom<Element>{}) == 0) {
    return detail::Layout_K_SW128_Atom<Element>{};
  }
  else if constexpr (kBlockK % size<1>(detail::Layout_K_SW64_Atom<Element>{}) == 0) {
    return detail::Layout_K_SW64_Atom<Element>{};
  }
  else if constexpr (kBlockK % size<1>(detail::Layout_K_SW32_Atom<Element>{}) == 0) {
    return detail::Layout_K_SW32_Atom<Element>{};
  }
  else if constexpr (kBlockK % size<1>(detail::Layout_K_INTER_Atom<Element>{}) == 0) {
    return detail::Layout_K_INTER_Atom<Element>{};
  }
  else {
    static_assert(kBlockK % size<1>(detail::Layout_K_INTER_Atom<Element>{}) == 0,
                  "kBlockK must be a multiple of size<1>(detail::Layout_K_INTER_Atom<ElementType>{})");
  }
}
// clang-format on

template <class Element, int kThreads, int kBlockK, class Copy_Atom>
CUTE_HOST_DEVICE constexpr auto gmem_tiled_copy_selector(Copy_Atom cp_atom) {
  // maxmize vectorized load (128-bits or 16 bytes per thread)
  constexpr int kElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
  constexpr int kSmemBlockK =
      size<1>(smem_layout_atom_selector<Element, kBlockK>());
  static_assert(kSmemBlockK % kElemsPerLoad == 0,
                "kBlockK must be a multiple of kGmemElemsPerLoad");

  constexpr int kThreadsPerRow = kSmemBlockK / kElemsPerLoad;
  static_assert(kThreads % kThreadsPerRow == 0,
                "kThreads must be a multiple of kThreadsPerRow");
  constexpr int kRows = kThreads / kThreadsPerRow;
  static_assert(kRows <= 64, "kRows must be less than or equal to 64");

  constexpr auto thr_layout = Layout<Shape<Int<kRows>, Int<kThreadsPerRow>>,
                                     Stride<Int<kThreadsPerRow>, _1>>{};
  constexpr auto val_layout = Layout<Shape<_1, Int<kElemsPerLoad>>>{};

  // g2s tiled copy
  return make_tiled_copy(cp_atom, thr_layout, val_layout);
}

}  // namespace llm
