#include <cute/tensor.hpp>
#include <fstream>

#include "../attention_traits_fp8_kvcache_sm80.h"
#include "../attention_traits_sm80.h"
#include "print_svg.hpp"

using namespace cute;
using namespace llm;

template <typename Traits>
void print_attn_traits() {
  // type alias
  using TiledMma = typename Traits::TiledMma;

  using SmemLayoutQ = typename Traits::SmemLayoutQ;
  using SmemLayoutK = typename Traits::SmemLayoutK;
  using SmemLayoutV = typename Traits::SmemLayoutV;
  using SmemLayoutVt = typename Traits::SmemLayoutVt;
  using SmemLayoutO = typename Traits::SmemLayoutO;

  using GmemTiledCopyQ = typename Traits::GmemTiledCopyQ;
  using GmemTiledCopyKV = typename Traits::GmemTiledCopyKV;
  using GmemTiledCopyO = typename Traits::GmemTiledCopyO;

  using SmemTiledCopyQ = typename Traits::SmemTiledCopyQ;
  using SmemTiledCopyK = typename Traits::SmemTiledCopyK;
  using SmemTiledCopyVt = typename Traits::SmemTiledCopyVt;
  using SmemTiledCopyO = typename Traits::SmemTiledCopyO;

  // print tiled mma
  print("TiledMma: \n");
  print(TiledMma{});
  print("\n\n");
  {
    std::ofstream os("tiled_mma.svg");
    print_svg(os, TiledMma{});
  }

  // print g2s tiled copy Q
  print("GmemTiledCopyQ: \n");
  print(GmemTiledCopyQ{});
  print("\n\n");
  {
    std::ofstream os("g2s_tiled_copy_q.svg");
    print_svg(os, GmemTiledCopyQ{});
  }

  print("GmemTiledCopyKV: \n");
  print(GmemTiledCopyKV{});
  print("\n\n");
  {
    std::ofstream os("g2s_tiled_copy_kv.svg");
    print_svg(os, GmemTiledCopyKV{});
  }

  // print s2g tiled copy O
  print("GmemTiledCopyO: \n");
  print(GmemTiledCopyO{});
  print("\n\n");
  {
    std::ofstream os("s2g_tiled_copy_o.svg");
    print_svg(os, GmemTiledCopyO{});
  }

  // print s2r tiled copy Q
  print("SmemTiledCopyQ: \n");
  print(SmemTiledCopyQ{});
  print("\n\n");
  {
    std::ofstream os("s2r_tiled_copy_q.svg");
    print_svg(os, SmemTiledCopyQ{});
  }
  // print s2r tiled copy K
  print("SmemTiledCopyK: \n");
  print(SmemTiledCopyK{});
  print("\n\n");
  {
    std::ofstream os("s2r_tiled_copy_k.svg");
    print_svg(os, SmemTiledCopyK{});
  }
  // print s2r tiled copy Vt
  print("SmemTiledCopyVt: \n");
  print(SmemTiledCopyVt{});
  print("\n\n");
  {
    std::ofstream os("s2r_tiled_copy_vt.svg");
    print_svg(os, SmemTiledCopyVt{});
  }
  // print r2s tiled copy O
  print("SmemTiledCopyO: \n");
  print(SmemTiledCopyO{});
  print("\n\n");
  {
    std::ofstream os("r2s_tiled_copy_o.svg");
    print_svg(os, SmemTiledCopyO{});
  }

  // print smem layout Q
  print("SmemLayoutQ: \n");
  print(SmemLayoutQ{});
  print("\n\n");
  {
    std::ofstream os("smem_layout_q.svg");
    print_svg(os, SmemLayoutQ{}, GmemTiledCopyQ{}, SmemTiledCopyQ{});
  }
  // print smem layout KV
  print("SmemLayoutK: \n");
  print(SmemLayoutK{});
  print("\n\n");
  {
    std::ofstream os("smem_layout_k.svg");
    print_svg(os, SmemLayoutK{}, GmemTiledCopyKV{}, SmemTiledCopyK{});
  }
  // print smem layout Vt
  print("SmemLayoutVt: \n");
  print(SmemLayoutVt{});
  print("\n\n");
  {
    std::ofstream os("smem_layout_vt.svg");
    print_svg(os,
              SmemLayoutV{},
              SmemLayoutVt{},
              GmemTiledCopyKV{},
              SmemTiledCopyVt{});
  }
  // print smem layout O
  print("SmemLayoutO: \n");
  print(SmemLayoutO{});
  print("\n\n");
  {
    std::ofstream os("smem_layout_o.svg");
    print_svg(os, SmemLayoutO{}, SmemTiledCopyO{}, GmemTiledCopyO{});
  }
}

int main(int argc, char** argv) {
  // TODO: pass in as parameters
  using DTYPE = cute::half_t;
  using KV_DTYPE = cute::int8_t;

  constexpr int kHeadDim = 64;
  constexpr int kBlockM = 64;
  constexpr int kBlockN = 64;
  constexpr int kBlockK = 64;

  using Traits = AttentionTraitsFp8KVCacheSM80<DTYPE,
                                               KV_DTYPE,
                                               kHeadDim,
                                               kBlockM,
                                               kBlockN,
                                               kBlockK>;
  print_attn_traits<Traits>();

  return 0;
}