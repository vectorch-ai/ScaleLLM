#include <cute/tensor.hpp>
#include <fstream>

#include "../attention_traits_sm80.h"
#include "print_svg.hpp"

using namespace cute;
using namespace llm;

template <typename Traits>
void print_attn_traits() {
  // type alias
  using TiledMma = typename Traits::TiledMma;
  using Layout = typename Traits::LayoutConvertor;

  using SmemLayoutQ = typename Traits::SmemLayoutQ;
  using SmemLayoutK = typename Traits::SmemLayoutK;
  using SmemLayoutV = typename Traits::SmemLayoutV;
  using SmemLayoutVt = typename Traits::SmemLayoutVt;
  using SmemLayoutO = typename Traits::SmemLayoutO;

  using GmemTiledCopyQKV = typename Traits::GmemTiledCopyQKV;
  using GmemTiledCopyO = typename Traits::GmemTiledCopyO;

  using SmemTiledCopyQ = typename Traits::SmemTiledCopyQ;
  using SmemTiledCopyK = typename Traits::SmemTiledCopyK;
  using SmemTiledCopyVt = typename Traits::SmemTiledCopyVt;
  using SmemTiledCopyO = typename Traits::SmemTiledCopyO;

  // print tiled mma
  {
    std::ofstream os("tiled_mma.svg");
    print_svg(os, TiledMma{});
  }

  // print g2s tiled copy Q
  {
    std::ofstream os("g2s_tiled_copy_q.svg");
    print_svg(os, GmemTiledCopyQKV{});
  }
  // print s2g tiled copy O
  {
    std::ofstream os("s2g_tiled_copy_o.svg");
    print_svg(os, GmemTiledCopyO{});
  }

  // print s2r tiled copy Q
  {
    std::ofstream os("s2r_tiled_copy_q.svg");
    print_svg(os, SmemTiledCopyQ{});
  }
  // print s2r tiled copy K
  {
    std::ofstream os("s2r_tiled_copy_k.svg");
    print_svg(os, SmemTiledCopyK{});
  }
  // print s2r tiled copy Vt
  {
    std::ofstream os("s2r_tiled_copy_vt.svg");
    print_svg(os, SmemTiledCopyVt{});
  }
  // print r2s tiled copy O
  {
    std::ofstream os("r2s_tiled_copy_o.svg");
    print_svg(os, SmemTiledCopyO{});
  }

  // print smem layout Q
  {
    std::ofstream os("smem_layout_q.svg");
    print_svg(os, SmemLayoutQ{}, GmemTiledCopyQKV{}, SmemTiledCopyQ{});
  }
  // print smem layout KV
  {
    std::ofstream os("smem_layout_k.svg");
    print_svg(os, SmemLayoutK{}, GmemTiledCopyQKV{}, SmemTiledCopyK{});
  }
  // print smem layout Vt
  {
    std::ofstream os("smem_layout_vt.svg");
    print_svg(os, SmemLayoutV{}, SmemLayoutVt{}, GmemTiledCopyQKV{}, SmemTiledCopyVt{});
  }
  // print smem layout O
  {
    std::ofstream os("smem_layout_o.svg");
    print_svg(os, SmemLayoutO{}, SmemTiledCopyO{}, GmemTiledCopyO{});
  }
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