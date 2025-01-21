#include <cute/tensor.hpp>
#include <fstream>

#include "../attention_traits_fp8_kvcache_sm80.h"
#include "../attention_traits_sm80.h"
#include "print_svg.hpp"
#include "../cute_extensions.cuh"

using namespace cute;
using namespace llm;

template <class... Args>
void save_svg(const std::string& filename, Args&&... args) {
  std::ofstream os(filename);
  print_svg(os, args...);
}

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
  save_svg("tiled_mma.svg", TiledMma{});

  // print g2s tiled copy Q
  print("GmemTiledCopyQ: \n");
  print(GmemTiledCopyQ{});
  print("\n\n");
  save_svg("g2s_tiled_copy_q.svg", GmemTiledCopyQ{});

  // print g2s tiled copy KV
  print("GmemTiledCopyKV: \n");
  print(GmemTiledCopyKV{});
  print("\n\n");
  save_svg("g2s_tiled_copy_kv.svg", GmemTiledCopyKV{});

  // print s2g tiled copy O
  print("GmemTiledCopyO: \n");
  print(GmemTiledCopyO{});
  print("\n\n");
  save_svg("s2g_tiled_copy_o.svg", GmemTiledCopyO{});

  // print s2r tiled copy Q
  print("SmemTiledCopyQ: \n");
  print(SmemTiledCopyQ{});
  print("\n\n");
  save_svg("s2r_tiled_copy_q.svg", SmemTiledCopyQ{});

  // print s2r tiled copy K
  print("SmemTiledCopyK: \n");
  print(SmemTiledCopyK{});
  print("\n\n");
  save_svg("s2r_tiled_copy_k.svg", SmemTiledCopyK{});

  // print s2r tiled copy Vt
  print("SmemTiledCopyVt: \n");
  print(SmemTiledCopyVt{});
  print("\n\n");
  save_svg("s2r_tiled_copy_vt.svg", SmemTiledCopyVt{});

  // print r2s tiled copy O
  print("SmemTiledCopyO: \n");
  print(SmemTiledCopyO{});
  print("\n\n");
  save_svg("r2s_tiled_copy_o.svg", SmemTiledCopyO{});

  // print smem layout Q
  print("SmemLayoutQ: \n");
  print(SmemLayoutQ{});
  print("\n\n");
  save_svg(
      "smem_layout_q.svg", SmemLayoutQ{}, GmemTiledCopyQ{}, SmemTiledCopyQ{});

  // print smem layout KV
  print("SmemLayoutK: \n");
  print(SmemLayoutK{});
  print("\n\n");
  save_svg(
      "smem_layout_k.svg", SmemLayoutK{}, GmemTiledCopyKV{}, SmemTiledCopyK{});

  // print smem layout Vt
  print("SmemLayoutVt: \n");
  print(SmemLayoutVt{});
  print("\n\n");
  save_svg("smem_layout_vt.svg",
           SmemLayoutV{},
           SmemLayoutVt{},
           GmemTiledCopyKV{},
           SmemTiledCopyVt{});

  // print smem layout O
  print("SmemLayoutO: \n");
  print(SmemLayoutO{});
  print("\n\n");
  save_svg(
      "smem_layout_o.svg", SmemLayoutO{}, SmemTiledCopyO{}, GmemTiledCopyO{});
}

template <typename Traits>
void test_attn_traits() {
  // type alias
  using TiledMma = typename Traits::TiledMma;

  using SmemLayoutQ = typename Traits::SmemLayoutQ;
  using SmemLayoutK = typename Traits::SmemLayoutK;
  using SmemLayoutV = typename Traits::SmemLayoutV;
  using SmemLayoutVt = typename Traits::SmemLayoutVt;
  using SmemLayoutO = typename Traits::SmemLayoutO;

  using SmemTiledCopyQ = typename Traits::SmemTiledCopyQ;
  using SmemTiledCopyK = typename Traits::SmemTiledCopyK;
  using SmemTiledCopyVt = typename Traits::SmemTiledCopyVt;
  using SmemTiledCopyO = typename Traits::SmemTiledCopyO;

  // NxK: (64, 64)
  Tensor sK = make_tensor(counting_iterator<int>(0), SmemLayoutK{});
  print(sK);print("\n");

  TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_slice(0);
  // (MMA, MMA_N, MMA_K)
  // ((_2,_2),_8,_4):((_1,_2),_16,_4)
  auto tSrK = thr_mma.partition_fragment_B(sK);
  print(tSrK);print("\n");

  auto tSrK_fp8 = make_fragment_B<cute::int8_t>(thr_mma, sK);
  print(tSrK_fp8);print("\n");

  SmemTiledCopyK smem_tiled_copy_K;
  auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(0);
  print(smem_thr_copy_K);print("\n");

  // (((_4,_2),_1),_4,_4):(((_1,_16),_0),_32,_4)
  auto tSrK_copy_view = smem_thr_copy_K.retile_D(tSrK_fp8);
  print(tSrK_copy_view);print("\n");

}

int main(int argc, char** argv) {
  // TODO: pass in as parameters
  using DTYPE = cute::half_t;
  using KV_DTYPE = cute::int8_t;

  constexpr int kHeadDim = 64;
  constexpr int kBlockM = 64;
  constexpr int kBlockN = 64;
  constexpr int kBlockK = 64;

  using Traits = AttentionTraitsFp8KVCacheSM80<DTYPE, KV_DTYPE,
                                               kHeadDim,
                                               kBlockM,
                                               kBlockN,
                                               kBlockK>;
  // print_attn_traits<Traits>();
  test_attn_traits<Traits>();

  return 0;
}