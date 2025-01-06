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

  // print tiled copy A
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