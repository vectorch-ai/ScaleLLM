#include <cute/tensor.hpp>
#include <fstream>

#include "../attention_traits_sm80.h"
#include "print_svg.hpp"

using namespace cute;
using namespace llm;

template <typename Item>
void save_svg(const Item& item, const std::string& filename) {
  std::ofstream os(filename);
  os << item;
}

template <typename AttentionTraits>
void print_attn_traits() {
  // print tiled mma
  save_svg(typename AttentionTraits::TiledMma{}, "tiled_mma.svg");

  // print g2s tiled copy Q
  save_svg(typename AttentionTraits::GmemTiledCopyQKV{}, "g2s_tiled_copy_q.svg");

  // print s2r tiled copy Q
  save_svg(typename AttentionTraits::SmemTiledCopyQ{}, "s2r_tiled_copy_q.svg");
  // print s2r tiled copy K
  save_svg(typename AttentionTraits::SmemTiledCopyK{}, "s2r_tiled_copy_k.svg");
  // print s2r tiled copy Vt
  save_svg(typename AttentionTraits::SmemTiledCopyVT{},
           "s2r_tiled_copy_vt.svg");

  // TODO: print smem layout Q
  // save_svg(typename AttentionTraits::SmemLayoutQ{}, "smem_layout_q.svg");
  // print smem layout KV
  // save_svg(typename AttentionTraits::SmemLayoutKV{}, "smem_layout_kv.svg");
  // print smem layout Vt
  // save_svg(typename AttentionTraits::SmemLayoutVt{}, "smem_layout_vt.svg");
}

int main(int argc, char** argv) {
  // TODO: pass in as parameters
  using Element = cute::half_t;

  constexpr int kHeadDim = 64;
  constexpr int kBlockM = 64;
  constexpr int kBlockN = 64;

  print_attn_traits<AttentionTraitsSM80<Element, kHeadDim, kBlockM, kBlockN>>();

  return 0;
}