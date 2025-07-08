#pragma once
#include <cute/atom/copy_traits_sm90_tma.hpp>

namespace cute {

namespace detail {

template <class SLayout>
CUTE_HOST_RTC auto get_tma_atom_slayout(
    SLayout const& slayout  // ((ATOM_M, m), (ATOM_N, n))
) {
  return composition(slayout.layout_a(),
                     slayout.offset(),
                     make_layout(get<0, 0>(slayout.layout_b()),
                                 get<1, 0>(slayout.layout_b())));
}

template <class TmaInternalType,
          class CopyOp,
          class GEngine,
          class GLayout,
          class SLayout>
CUTE_HOST_RTC auto make_gather_tma_copy_atom(
    CopyOp copy_op,
    Tensor<GEngine, GLayout> const& gtensor,  // Full GMEM Tensor
    SLayout const& slayout,         // CTA Tile of SMEM, potentially swizzled
    uint32_t const& num_multicast)  // The num of CTAs involved in multicasting)
{
  auto atom_slayout = get_tma_atom_slayout(slayout);
  auto atom_tiler = product_each(shape(atom_slayout));
  auto atom_v_map = make_identity_layout(shape(gtensor)).compose(atom_tiler);
  return make_tma_copy_atom<TmaInternalType>(
      copy_op, gtensor, atom_slayout, num_multicast, atom_v_map);
}

template <class TmaType,
          class CopyOp,
          class GEngine,
          class GLayout,
          class SLayout,
          class TShape,
          class TStride,
          class VShape,
          class VStride>
CUTE_HOST_RTC auto make_gather_tma_copy_tiled(
    CopyOp const& copy_op,
    Tensor<GEngine, GLayout> const& gtensor,   // Full GMEM Tensor
    SLayout const& slayout,                    // CTA Tile of SMEM
    Layout<TShape, TStride> const& cta_t_map,  // T: Thr idx -> logical TMA tid
    Layout<VShape, VStride> const& cta_v_map)  // V: CTA val idx -> gmem mode
{
  // Construct tma copy atom
  auto atom = make_gather_tma_copy_atom<TmaType>(
      copy_op, gtensor, slayout, cosize(cta_t_map));

  // Construct the TiledCopy
  [[maybe_unused]] auto cta_tiler = product_each(shape(cta_v_map));

  auto num_elems_per_tma =
      size<1>(typename decltype(atom)::RefLayout{}) /
      static_value<sizeof_bits<typename GEngine::value_type>>();

  // smem idx -> smem coord
  auto inv_smem_layout = right_inverse(get_nonswizzle_portion(slayout));
  // CTA V -> smem_coord
  auto layout_v = composition(inv_smem_layout, num_elems_per_tma);
  // Scale that up to cover all of the smem_coords
  auto layout_V = tile_to_shape(make_layout(layout_v), size(cta_v_map));
  // CTA T -> smem idx
  auto layout_t = make_layout(cosize(cta_t_map),
                              safe_div(num_elems_per_tma, cosize(cta_t_map)));
  // CTA TID -> smem coord
  auto layout_T =
      composition(inv_smem_layout, composition(layout_t, cta_t_map));
  // Combine with the T mapping
  [[maybe_unused]] auto layout_TV = make_layout(layout_T, layout_V);
  return TiledCopy<decltype(atom), decltype(layout_TV), decltype(cta_tiler)>{
      atom};
}

}  // namespace detail

template <class TmaInternalType = void,
          class CopyOp,
          class GEngine,
          class GLayout,
          class SLayout,
          class CTA_Tiler,
          class Cluster_Size>
CUTE_HOST_RTC auto make_gather_tma_copy(
    CopyOp const& copy_op,
    Tensor<GEngine, GLayout> const& gtensor,
    SLayout const& slayout,  // ((ATOM_M, m), (ATOM_N, n))
    CTA_Tiler const& cta_tiler,
    Cluster_Size const& cluster_size) {
  // Thr idx -> logical TMA tid
  auto cta_t_map = make_layout(cluster_size);
  // CTA val idx -> gmem mode
  auto cta_v_map = make_identity_layout(shape(gtensor)).compose(cta_tiler);
  using TmaType = conditional_t<is_same_v<void, TmaInternalType>,
                                typename GEngine::value_type,
                                TmaInternalType>;
  return detail::make_gather_tma_copy_tiled<TmaType>(
      copy_op, gtensor, slayout, cta_t_map, cta_v_map);
}
}  // namespace cute
